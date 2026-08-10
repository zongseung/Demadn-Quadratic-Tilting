"""Strict process-isolated sampler benchmark orchestration."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
from importlib.util import find_spec
from pathlib import Path
from typing import Any

from hqrc_v3.provenance import file_sha256

_VERSION = 1
_REQUEST_KEYS = {
    "schema_version",
    "inputs",
    "hashes",
    "model",
    "sampler",
    "request_digest",
}
_RESULT_KEYS = {
    "schema_version",
    "request_digest",
    "backend",
    "pid",
    "wall_seconds",
    "peak_rss_mb",
    "diagnostics",
    "posterior",
    "versions",
    "result_digest",
}
_BENCHMARK_KEYS = {
    "schema_version",
    "benchmarks",
    "maximum_mean_distance_sd",
    "posterior_audit",
    "nutpie_eligible_default",
    "benchmark_digest",
}
_BENCHMARK_WORKER_KEYS = {
    "status",
    "backend",
    "pid",
    "request_digest",
    "wall_seconds",
    "peak_rss_mb",
    "min_bulk_ess_per_second",
    "min_tail_ess_per_second",
    "max_rhat",
    "divergences",
    "versions",
}
_POSTERIOR_AUDIT_ROW_KEYS = {
    "left_mean",
    "right_mean",
    "pooled_sd",
    "distance_sd",
}


class SamplerWorkerError(RuntimeError):
    """Raised when a sampler child or its JSON contract cannot be trusted."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise SamplerWorkerError("worker contract is not finite canonical JSON") from error


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SamplerWorkerError(f"{name} must be a positive integer")
    return value


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise SamplerWorkerError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise SamplerWorkerError(f"{name} must be a SHA-256 digest") from error
    return value.lower()


def write_sampler_request(
    path: Path,
    *,
    hqrc_npz: Path,
    hqrc_metadata: Path,
    approved_ar: Path,
    residual_sha256: str,
    config_sha256: str,
    event_sha256: str,
    variant: str,
    pooling: str,
    options: Mapping[str, object],
    backend: str,
    seed: int,
    draws: int,
    tune: int,
    chains: int,
    profile: str,
    bound_hashes: Mapping[str, str] | None = None,
) -> Path:
    """Write a canonical request binding every worker input to current bytes."""

    input_paths = {
        "hqrc_npz": Path(hqrc_npz).resolve(),
        "hqrc_metadata": Path(hqrc_metadata).resolve(),
        "approved_ar": Path(approved_ar).resolve(),
    }
    if any(not candidate.is_file() for candidate in input_paths.values()):
        raise SamplerWorkerError("worker input artifact is missing")
    inputs = {
        name: {"path": str(candidate), "sha256": file_sha256(candidate)}
        for name, candidate in input_paths.items()
    }
    if bound_hashes is not None:
        for candidate in input_paths.values():
            if bound_hashes.get(candidate.name) != file_sha256(candidate):
                raise SamplerWorkerError("caller-bound worker input hash differs")
    if variant not in {"H1", "H2", "H3", "H4"}:
        raise SamplerWorkerError("model variant is invalid")
    if pooling not in {"complete", "partial", "none"}:
        raise SamplerWorkerError("model pooling is invalid")
    if backend not in {"pymc", "nutpie"}:
        raise SamplerWorkerError("sampler backend is invalid")
    if profile not in {"smoke", "paper"}:
        raise SamplerWorkerError("sampler profile is invalid")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise SamplerWorkerError("seed must be a non-negative integer")
    payload: dict[str, Any] = {
        "schema_version": _VERSION,
        "inputs": inputs,
        "hashes": {
            "residual_sha256": _sha(residual_sha256, "residual_sha256"),
            "config_sha256": _sha(config_sha256, "config_sha256"),
            "event_sha256": _sha(event_sha256, "event_sha256"),
        },
        "model": {"variant": variant, "pooling": pooling, "options": dict(options)},
        "sampler": {
            "backend": backend,
            "seed": seed,
            "draws": _positive_integer(draws, "draws"),
            "tune": _positive_integer(tune, "tune"),
            "chains": _positive_integer(chains, "chains"),
            "profile": profile,
        },
    }
    payload["request_digest"] = _digest(payload)
    return _atomic_json(Path(path), payload)


def load_sampler_request(path: Path) -> dict[str, Any]:
    """Load and fully revalidate a sampler request inside either process."""

    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SamplerWorkerError("sampler request is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != _REQUEST_KEYS:
        raise SamplerWorkerError("sampler request schema differs")
    if raw != _canonical(payload) + b"\n":
        raise SamplerWorkerError("sampler request is not canonical JSON")
    digest = payload.pop("request_digest")
    if not isinstance(digest, str) or digest != _digest(payload):
        raise SamplerWorkerError("sampler request digest differs")
    payload["request_digest"] = digest
    if (
        type(payload.get("schema_version")) is not int
        or payload["schema_version"] != _VERSION
    ):
        raise SamplerWorkerError("sampler request version differs")
    if not isinstance(payload.get("inputs"), dict) or set(payload["inputs"]) != {
        "hqrc_npz",
        "hqrc_metadata",
        "approved_ar",
    }:
        raise SamplerWorkerError("sampler request inputs differ")
    for entry in payload["inputs"].values():
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise SamplerWorkerError("sampler request input entry differs")
        candidate = Path(entry["path"])
        if not candidate.is_absolute() or not candidate.is_file():
            raise SamplerWorkerError("sampler request input path is invalid")
        if _sha(entry["sha256"], "input sha256") != file_sha256(candidate):
            raise SamplerWorkerError("sampler request input digest differs")
    hashes = payload.get("hashes")
    if not isinstance(hashes, dict) or set(hashes) != {
        "residual_sha256",
        "config_sha256",
        "event_sha256",
    }:
        raise SamplerWorkerError("sampler request provenance hashes differ")
    for name, value in hashes.items():
        _sha(value, name)
    model = payload.get("model")
    if not isinstance(model, dict) or set(model) != {"variant", "pooling", "options"}:
        raise SamplerWorkerError("sampler request model schema differs")
    if model["variant"] not in {"H1", "H2", "H3", "H4"} or model["pooling"] not in {
        "complete",
        "partial",
        "none",
    }:
        raise SamplerWorkerError("sampler request model selection differs")
    if not isinstance(model["options"], dict):
        raise SamplerWorkerError("sampler request model options differ")
    sampler = payload.get("sampler")
    if not isinstance(sampler, dict) or set(sampler) != {
        "backend",
        "seed",
        "draws",
        "tune",
        "chains",
        "profile",
    }:
        raise SamplerWorkerError("sampler request sampler schema differs")
    if sampler["backend"] not in {"pymc", "nutpie"} or sampler["profile"] not in {
        "smoke",
        "paper",
    }:
        raise SamplerWorkerError("sampler request backend/profile differs")
    if (
        isinstance(sampler["seed"], bool)
        or not isinstance(sampler["seed"], int)
        or sampler["seed"] < 0
    ):
        raise SamplerWorkerError("sampler request seed differs")
    for name in ("draws", "tune", "chains"):
        _positive_integer(sampler[name], name)
    return payload


@dataclass(frozen=True)
class SamplerWorkerResult:
    backend: str
    pid: int
    request_digest: str
    wall_seconds: float
    peak_rss_mb: float
    max_rhat: float
    min_bulk_ess: float
    min_tail_ess: float
    divergences: int
    posterior: dict[str, dict[str, Any]]
    versions: dict[str, str]

    @property
    def min_bulk_ess_per_second(self) -> float:
        return self.min_bulk_ess / self.wall_seconds

    @property
    def min_tail_ess_per_second(self) -> float:
        return self.min_tail_ess / self.wall_seconds


def _finite_positive(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise SamplerWorkerError(f"{name} must be finite and positive")
    return float(value)


def _finite_nonnegative(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise SamplerWorkerError(f"{name} must be finite and non-negative")
    return float(value)


def sampler_eligible_measurements(
    pymc: Mapping[str, Any],
    nutpie: Mapping[str, Any],
    *,
    mean_distance_sd: float,
) -> bool:
    """Apply the predeclared nutpie gate to validated or dataclass-like measurements."""

    distance = _finite_nonnegative(mean_distance_sd, "posterior mean distance")
    try:
        pymc_status = pymc["status"]
        nutpie_status = nutpie["status"]
        pymc_divergences = pymc["divergences"]
        nutpie_divergences = nutpie["divergences"]
        pymc_rhat = pymc["max_rhat"]
        nutpie_rhat = nutpie["max_rhat"]
        pymc_wall = pymc["wall_seconds"]
        nutpie_wall = nutpie["wall_seconds"]
        pymc_bulk = pymc["min_bulk_ess_per_second"]
        nutpie_bulk = nutpie["min_bulk_ess_per_second"]
    except KeyError as error:
        raise SamplerWorkerError("sampler eligibility measurements are incomplete") from error
    for value, name in (
        (pymc_rhat, "PyMC max_rhat"),
        (nutpie_rhat, "nutpie max_rhat"),
        (pymc_wall, "PyMC wall_seconds"),
        (nutpie_wall, "nutpie wall_seconds"),
        (pymc_bulk, "PyMC min_bulk_ess_per_second"),
        (nutpie_bulk, "nutpie min_bulk_ess_per_second"),
    ):
        _finite_positive(value, name)
    for value, name in (
        (pymc_divergences, "PyMC divergences"),
        (nutpie_divergences, "nutpie divergences"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise SamplerWorkerError(f"{name} must be a non-negative integer")
    return bool(
        pymc_status == nutpie_status == "ok"
        and distance <= 0.1
        and nutpie_divergences <= pymc_divergences
        and nutpie_rhat <= pymc_rhat + 0.01
        and (nutpie_wall <= pymc_wall * 0.8 or nutpie_bulk >= pymc_bulk * 1.2)
    )


def _validate_posterior(value: object) -> dict[str, dict[str, Any]]:
    if not isinstance(value, dict) or not value:
        raise SamplerWorkerError("worker posterior summary is empty")
    normalized: dict[str, dict[str, Any]] = {}
    for name, raw in value.items():
        if not isinstance(name, str) or not name or not isinstance(raw, dict):
            raise SamplerWorkerError("worker posterior parameter is invalid")
        if set(raw) != {"shape", "means", "sds"}:
            raise SamplerWorkerError("worker posterior parameter schema differs")
        shape, means, sds = raw["shape"], raw["means"], raw["sds"]
        if (
            not isinstance(shape, list)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in shape
            )
            or not isinstance(means, list)
            or not isinstance(sds, list)
            or len(means) != len(sds)
            or math.prod(shape) != len(means)
        ):
            raise SamplerWorkerError("worker posterior parameter shape differs")
        if any(
            isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item)
            for item in (*means, *sds)
        ) or any(item < 0 for item in sds):
            raise SamplerWorkerError("worker posterior parameter values are invalid")
        normalized[name] = {"shape": shape, "means": means, "sds": sds}
    return normalized


def load_sampler_result(path: Path, *, request: Mapping[str, Any]) -> SamplerWorkerResult:
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SamplerWorkerError("worker result is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != _RESULT_KEYS:
        raise SamplerWorkerError("worker result schema differs")
    if raw != _canonical(payload) + b"\n":
        raise SamplerWorkerError("worker result is not canonical JSON")
    result_digest = payload.pop("result_digest")
    if not isinstance(result_digest, str) or result_digest != _digest(payload):
        raise SamplerWorkerError("worker result digest differs")
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _VERSION:
        raise SamplerWorkerError("worker result version differs")
    if payload["request_digest"] != request["request_digest"]:
        raise SamplerWorkerError("worker request digest differs")
    if payload["backend"] != request["sampler"]["backend"]:
        raise SamplerWorkerError("worker backend differs")
    pid = payload["pid"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
        raise SamplerWorkerError("worker PID does not identify a child process")
    diagnostics = payload["diagnostics"]
    if not isinstance(diagnostics, dict) or set(diagnostics) != {
        "max_rhat",
        "min_bulk_ess",
        "min_tail_ess",
        "divergences",
    }:
        raise SamplerWorkerError("worker diagnostics schema differs")
    divergences = diagnostics["divergences"]
    if isinstance(divergences, bool) or not isinstance(divergences, int) or divergences < 0:
        raise SamplerWorkerError("worker divergences are invalid")
    versions = payload["versions"]
    if (
        not isinstance(versions, dict)
        or not versions
        or any(
            not isinstance(key, str) or not isinstance(value, str) or not value
            for key, value in versions.items()
        )
    ):
        raise SamplerWorkerError("worker versions are invalid")
    return SamplerWorkerResult(
        backend=payload["backend"],
        pid=pid,
        request_digest=payload["request_digest"],
        wall_seconds=_finite_positive(payload["wall_seconds"], "worker wall_seconds"),
        peak_rss_mb=_finite_positive(payload["peak_rss_mb"], "worker peak_rss_mb"),
        max_rhat=_finite_positive(diagnostics["max_rhat"], "worker max_rhat"),
        min_bulk_ess=_finite_positive(diagnostics["min_bulk_ess"], "worker min_bulk_ess"),
        min_tail_ess=_finite_positive(diagnostics["min_tail_ess"], "worker min_tail_ess"),
        divergences=divergences,
        posterior=_validate_posterior(payload["posterior"]),
        versions=dict(versions),
    )


def run_sampler_worker(
    request_path: Path,
    result_path: Path,
    *,
    timeout_seconds: float,
    worker_command: Sequence[str] | None = None,
) -> SamplerWorkerResult:
    """Launch a fresh child process and fail closed on every transport error."""

    if not isinstance(timeout_seconds, (int, float)) or timeout_seconds <= 0:
        raise SamplerWorkerError("worker timeout must be positive")
    request = load_sampler_request(request_path)
    result = Path(result_path)
    result.unlink(missing_ok=True)
    command = list(worker_command or (sys.executable, "-m", "hqrc_v3.bayes.sampler_worker"))
    command.extend((str(Path(request_path).resolve()), str(result.resolve())))
    environment = os.environ.copy()
    source_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else os.pathsep.join((source_root, existing_pythonpath))
    )
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        _, stderr = process.communicate(timeout=float(timeout_seconds))
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise SamplerWorkerError("sampler worker timeout") from error
    if process.returncode:
        message = stderr.strip()[-500:]
        raise SamplerWorkerError(f"sampler worker exit status {process.returncode}: {message}")
    loaded = load_sampler_result(result, request=request)
    if loaded.pid != process.pid:
        raise SamplerWorkerError("worker result PID differs from the spawned process")
    return loaded


def maximum_posterior_mean_distance(
    left: Mapping[str, Mapping[str, Any]], right: Mapping[str, Mapping[str, Any]]
) -> tuple[float, dict[str, list[dict[str, float]]]]:
    """Return the maximum elementwise mean distance in pooled posterior SD units."""

    left = _validate_posterior(dict(left))
    right = _validate_posterior(dict(right))
    if set(left) != set(right) or not left:
        raise SamplerWorkerError("posterior parameter sets differ")
    maximum = 0.0
    audit: dict[str, list[dict[str, float]]] = {}
    for name in sorted(left):
        a, b = left[name], right[name]
        if a.get("shape") != b.get("shape"):
            raise SamplerWorkerError("posterior parameter shapes differ")
        means_a, means_b, sds_a, sds_b = (
            a.get("means"),
            b.get("means"),
            a.get("sds"),
            b.get("sds"),
        )
        if not all(isinstance(value, list) for value in (means_a, means_b, sds_a, sds_b)):
            raise SamplerWorkerError("posterior audit arrays are invalid")
        if len({len(means_a), len(means_b), len(sds_a), len(sds_b)}) != 1:
            raise SamplerWorkerError("posterior audit array lengths differ")
        rows = []
        for mean_a, mean_b, sd_a, sd_b in zip(means_a, means_b, sds_a, sds_b, strict=True):
            values = (mean_a, mean_b, sd_a, sd_b)
            if any(
                not isinstance(value, (int, float)) or not math.isfinite(value) for value in values
            ):
                raise SamplerWorkerError("posterior audit values are nonfinite")
            pooled = math.sqrt((sd_a**2 + sd_b**2) / 2.0)
            difference = abs(mean_a - mean_b)
            if pooled == 0:
                if difference != 0:
                    raise SamplerWorkerError(
                        "zero pooled SD requires exact posterior mean equality"
                    )
                distance = 0.0
            else:
                distance = difference / pooled
            maximum = max(maximum, distance)
            rows.append(
                {
                    "left_mean": float(mean_a),
                    "right_mean": float(mean_b),
                    "pooled_sd": pooled,
                    "distance_sd": distance,
                }
            )
        audit[name] = rows
    return maximum, audit


def _validate_benchmark_worker(value: object, *, backend: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != _BENCHMARK_WORKER_KEYS:
        raise SamplerWorkerError("sampler benchmark worker schema differs")
    if value["status"] != "ok" or value["backend"] != backend:
        raise SamplerWorkerError("sampler benchmark worker identity differs")
    pid = value["pid"]
    if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
        raise SamplerWorkerError("sampler benchmark worker PID differs")
    _sha(value["request_digest"], "sampler benchmark request digest")
    for name in (
        "wall_seconds",
        "peak_rss_mb",
        "min_bulk_ess_per_second",
        "min_tail_ess_per_second",
        "max_rhat",
    ):
        _finite_positive(value[name], f"sampler benchmark {name}")
    divergences = value["divergences"]
    if isinstance(divergences, bool) or not isinstance(divergences, int) or divergences < 0:
        raise SamplerWorkerError("sampler benchmark divergences differ")
    versions = value["versions"]
    if (
        not isinstance(versions, dict)
        or not versions
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            for name, version in versions.items()
        )
    ):
        raise SamplerWorkerError("sampler benchmark environment versions differ")
    return dict(value)


def _validate_posterior_audit(value: object) -> float:
    if not isinstance(value, dict) or not value:
        raise SamplerWorkerError("sampler benchmark posterior audit is empty")
    maximum = 0.0
    for name, rows in value.items():
        if not isinstance(name, str) or not name or not isinstance(rows, list) or not rows:
            raise SamplerWorkerError("sampler benchmark posterior audit parameter differs")
        for row in rows:
            if not isinstance(row, dict) or set(row) != _POSTERIOR_AUDIT_ROW_KEYS:
                raise SamplerWorkerError("sampler benchmark posterior audit row differs")
            left = row["left_mean"]
            right = row["right_mean"]
            if any(
                isinstance(item, bool)
                or not isinstance(item, (int, float))
                or not math.isfinite(item)
                for item in (left, right)
            ):
                raise SamplerWorkerError("sampler benchmark posterior mean differs")
            pooled = _finite_nonnegative(
                row["pooled_sd"], "sampler benchmark pooled posterior SD"
            )
            distance = _finite_nonnegative(
                row["distance_sd"], "sampler benchmark posterior distance"
            )
            difference = abs(float(left) - float(right))
            if pooled == 0:
                if difference != 0:
                    raise SamplerWorkerError(
                        "sampler benchmark zero pooled SD has unequal posterior means"
                    )
                expected = 0.0
            else:
                expected = difference / pooled
            if not math.isclose(distance, expected, rel_tol=1e-12, abs_tol=1e-15):
                raise SamplerWorkerError("sampler benchmark posterior distance audit differs")
            maximum = max(maximum, expected)
    return maximum


def _validate_sampler_benchmark_payload(payload: dict[str, Any]) -> None:
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _VERSION:
        raise SamplerWorkerError("sampler benchmark version differs")
    benchmarks = payload["benchmarks"]
    if not isinstance(benchmarks, dict) or set(benchmarks) != {"pymc", "nutpie"}:
        raise SamplerWorkerError("sampler benchmark backend mapping differs")
    pymc = _validate_benchmark_worker(benchmarks["pymc"], backend="pymc")
    nutpie = benchmarks["nutpie"]
    unavailable = {"status": "not-installed", "eligible_default": False}
    eligible = payload["nutpie_eligible_default"]
    if not isinstance(eligible, bool):
        raise SamplerWorkerError("sampler benchmark eligibility differs")
    if nutpie == unavailable:
        if (
            payload["maximum_mean_distance_sd"] is not None
            or payload["posterior_audit"] != {}
            or eligible
        ):
            raise SamplerWorkerError("missing nutpie benchmark audit differs")
        return
    nutpie_measurement = _validate_benchmark_worker(nutpie, backend="nutpie")
    maximum = _finite_nonnegative(
        payload["maximum_mean_distance_sd"],
        "sampler benchmark maximum posterior mean distance",
    )
    audited_maximum = _validate_posterior_audit(payload["posterior_audit"])
    if not math.isclose(maximum, audited_maximum, rel_tol=1e-12, abs_tol=1e-15):
        raise SamplerWorkerError("sampler benchmark maximum posterior distance differs")
    expected = sampler_eligible_measurements(
        pymc, nutpie_measurement, mean_distance_sd=maximum
    )
    if eligible is not expected:
        raise SamplerWorkerError("sampler benchmark eligibility differs from its measurements")


def load_sampler_benchmark(
    path: Path, *, bound_sha256: str | None = None
) -> dict[str, Any]:
    """Load the canonical production benchmark and recompute every semantic decision."""

    artifact = Path(path)
    if not artifact.is_file():
        raise SamplerWorkerError("sampler benchmark artifact is missing")
    if bound_sha256 is not None and _sha(
        bound_sha256, "caller-bound sampler benchmark digest"
    ) != file_sha256(artifact):
        raise SamplerWorkerError("caller-bound sampler benchmark hash differs")
    try:
        raw = artifact.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise SamplerWorkerError("sampler benchmark is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != _BENCHMARK_KEYS:
        raise SamplerWorkerError("sampler benchmark schema differs")
    if raw != _canonical(payload) + b"\n":
        raise SamplerWorkerError("sampler benchmark is not canonical JSON")
    digest = payload.pop("benchmark_digest")
    if not isinstance(digest, str) or digest != _digest(payload):
        raise SamplerWorkerError("sampler benchmark digest differs")
    payload["benchmark_digest"] = digest
    _validate_sampler_benchmark_payload(payload)
    return payload


def benchmark_sampler_processes(
    output_path: Path,
    *,
    request_directory: Path,
    timeout_seconds: float,
    request_kwargs: Mapping[str, Any],
    worker_commands: Mapping[str, Sequence[str]] | None = None,
) -> Path:
    """Run required PyMC and installed nutpie workers and publish an audited comparison."""

    directory = Path(request_directory)
    directory.mkdir(parents=True, exist_ok=True)
    if worker_commands is not None and (
        not set(worker_commands).issubset({"pymc", "nutpie"})
        or "pymc" not in worker_commands
        or any(
            isinstance(command, (str, bytes)) or not command
            for command in worker_commands.values()
        )
    ):
        raise SamplerWorkerError("worker command mapping is invalid")
    results: dict[str, SamplerWorkerResult] = {}
    for backend in ("pymc", "nutpie"):
        injected = worker_commands is not None and backend in worker_commands
        if backend == "nutpie" and find_spec("nutpie") is None and not injected:
            continue
        kwargs = dict(request_kwargs)
        kwargs["backend"] = backend
        request = write_sampler_request(directory / f"{backend}-request.json", **kwargs)
        results[backend] = run_sampler_worker(
            request,
            directory / f"{backend}-result.json",
            timeout_seconds=timeout_seconds,
            worker_command=(
                None if worker_commands is None else worker_commands.get(backend)
            ),
        )
    if "pymc" not in results:
        raise SamplerWorkerError("required PyMC worker did not run")
    pymc_result = results["pymc"]
    benchmarks: dict[str, dict[str, Any]] = {"pymc": _worker_audit(pymc_result)}
    distance: float | None = None
    posterior_audit: dict[str, list[dict[str, float]]] = {}
    eligible = False
    if "nutpie" in results:
        nutpie_result = results["nutpie"]
        distance, posterior_audit = maximum_posterior_mean_distance(
            pymc_result.posterior, nutpie_result.posterior
        )
        benchmarks["nutpie"] = _worker_audit(nutpie_result)
        eligible = sampler_eligible_measurements(
            benchmarks["pymc"], benchmarks["nutpie"], mean_distance_sd=distance
        )
    else:
        benchmarks["nutpie"] = {"status": "not-installed", "eligible_default": False}
    payload: dict[str, Any] = {
        "schema_version": _VERSION,
        "benchmarks": benchmarks,
        "maximum_mean_distance_sd": distance,
        "posterior_audit": posterior_audit,
        "nutpie_eligible_default": eligible,
    }
    payload["benchmark_digest"] = ""
    _validate_sampler_benchmark_payload(payload)
    payload.pop("benchmark_digest")
    payload["benchmark_digest"] = _digest(payload)
    return _atomic_json(Path(output_path), payload)


def _worker_audit(result: SamplerWorkerResult) -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": result.backend,
        "pid": result.pid,
        "request_digest": result.request_digest,
        "wall_seconds": result.wall_seconds,
        "peak_rss_mb": result.peak_rss_mb,
        "min_bulk_ess_per_second": result.min_bulk_ess_per_second,
        "min_tail_ess_per_second": result.min_tail_ess_per_second,
        "max_rhat": result.max_rhat,
        "divergences": result.divergences,
        "versions": result.versions,
    }


__all__ = [
    "SamplerWorkerError",
    "SamplerWorkerResult",
    "benchmark_sampler_processes",
    "load_sampler_benchmark",
    "load_sampler_request",
    "load_sampler_result",
    "maximum_posterior_mean_distance",
    "run_sampler_worker",
    "sampler_eligible_measurements",
    "write_sampler_request",
]
