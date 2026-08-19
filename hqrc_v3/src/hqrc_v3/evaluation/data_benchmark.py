"""Measured, process-isolated Polars data-path benchmarks."""

from __future__ import annotations

import json
import math
import os
import statistics
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from hqrc_v3.bayes.benchmark import (
    SamplerWorkerError,
    _atomic_json,
    _canonical,
    _digest,
    load_sampler_benchmark,
)
from hqrc_v3.provenance import file_sha256

_VERSION = 1
_RESULT_VERSION = 2
_WORKLOAD = "polars-numeric-summary-v1"
_REQUEST_KEYS = {
    "schema_version",
    "input",
    "columns",
    "repetitions",
    "workers",
    "seed",
    "workload",
    "request_digest",
}
_RESULT_KEYS = {
    "schema_version",
    "request_digest",
    "pid",
    "parent_pid",
    "threads",
    "dimensions",
    "zero_copy",
    "explicit_copy",
    "workload",
    "peak_rss_mb",
    "versions",
    "result_digest",
}
_FINAL_KEYS = {
    "schema_version",
    "request",
    "sampler",
    "zero_copy",
    "explicit_copy",
    "processing",
    "speed_ratios",
    "derived",
    "decision",
    "environment",
    "benchmark_digest",
}


class DataBenchmarkError(RuntimeError):
    """Raised when a data benchmark or one of its artifacts cannot be trusted."""


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    try:
        return _atomic_json(path, payload)
    except (TypeError, ValueError, RuntimeError) as error:
        raise DataBenchmarkError("data benchmark is not finite canonical JSON") from error


def _read_canonical(path: Path, *, label: str) -> dict[str, Any]:
    try:
        raw = Path(path).read_bytes()
        payload = json.loads(raw)
        canonical = _canonical(payload) + b"\n"
    except (OSError, json.JSONDecodeError, TypeError, ValueError, RuntimeError) as error:
        raise DataBenchmarkError(f"{label} is not valid canonical JSON") from error
    if not isinstance(payload, dict):
        raise DataBenchmarkError(f"{label} schema differs")
    if raw != canonical:
        raise DataBenchmarkError(f"{label} is not canonical JSON")
    return payload


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DataBenchmarkError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise DataBenchmarkError(f"{name} must be a SHA-256 digest") from error
    return value.lower()


def _integer(value: object, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise DataBenchmarkError(f"{name} must be an integer >= {minimum}")
    return value


def _positive_float(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise DataBenchmarkError(f"{name} must be finite and positive")
    return float(value)


def _columns(value: object) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(set(value)) != len(value)
    ):
        raise DataBenchmarkError("columns must be a non-empty unique string list")
    return list(value)


def write_data_benchmark_request(
    path: Path,
    *,
    parquet_path: Path,
    parquet_sha256: str,
    columns: Sequence[str],
    repetitions: int,
    workers: int,
    seed: int,
    workload: str,
) -> Path:
    """Write a canonical request bound to the current Parquet bytes."""

    artifact = Path(parquet_path).resolve()
    if not artifact.is_file() or artifact.suffix.lower() != ".parquet":
        raise DataBenchmarkError("benchmark input must be an existing Parquet artifact")
    expected = _sha(parquet_sha256, "parquet_sha256")
    actual = file_sha256(artifact)
    if expected != actual:
        raise DataBenchmarkError("caller-bound Parquet hash differs")
    if isinstance(columns, (str, bytes)):
        raise DataBenchmarkError("columns must be a non-empty unique string list")
    selected = _columns(list(columns))
    repetitions = _integer(repetitions, "repetitions", minimum=3)
    workers = _integer(workers, "workers", minimum=2)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataBenchmarkError("seed must be a non-negative integer")
    if workload != _WORKLOAD:
        raise DataBenchmarkError(f"workload must be {_WORKLOAD!r}")
    payload: dict[str, Any] = {
        "schema_version": _VERSION,
        "input": {"path": str(artifact), "sha256": actual},
        "columns": selected,
        "repetitions": repetitions,
        "workers": workers,
        "seed": seed,
        "workload": workload,
    }
    payload["request_digest"] = _digest(payload)
    return _write_json(Path(path), payload)


def load_data_benchmark_request(path: Path) -> dict[str, Any]:
    """Load a request and revalidate its schema, digest, and current input bytes."""

    payload = _read_canonical(path, label="data benchmark request")
    if set(payload) != _REQUEST_KEYS:
        raise DataBenchmarkError("data benchmark request schema differs")
    digest = payload.pop("request_digest")
    if not isinstance(digest, str) or digest != _digest(payload):
        raise DataBenchmarkError("data benchmark request digest differs")
    payload["request_digest"] = digest
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _VERSION:
        raise DataBenchmarkError("data benchmark request version differs")
    entry = payload["input"]
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise DataBenchmarkError("data benchmark request input schema differs")
    if not isinstance(entry["path"], str):
        raise DataBenchmarkError("data benchmark request input path differs")
    artifact = Path(entry["path"])
    if (
        not artifact.is_absolute()
        or not artifact.is_file()
        or artifact.suffix.lower() != ".parquet"
    ):
        raise DataBenchmarkError("data benchmark request input path differs")
    if _sha(entry["sha256"], "input sha256") != file_sha256(artifact):
        raise DataBenchmarkError("data benchmark request input digest differs")
    _columns(payload["columns"])
    _integer(payload["repetitions"], "repetitions", minimum=3)
    _integer(payload["workers"], "workers", minimum=2)
    if (
        isinstance(payload["seed"], bool)
        or not isinstance(payload["seed"], int)
        or payload["seed"] < 0
    ):
        raise DataBenchmarkError("data benchmark request seed differs")
    if payload["workload"] != _WORKLOAD:
        raise DataBenchmarkError("data benchmark request workload differs")
    return payload


def _validate_timings(value: object, *, repetitions: int, name: str) -> list[float]:
    if not isinstance(value, list) or len(value) != repetitions:
        raise DataBenchmarkError(f"{name} timing distribution differs")
    return [_positive_float(item, f"{name} timing") for item in value]


def _validate_measurement(
    value: object,
    *,
    repetitions: int,
    rows: int,
    column_count: int,
    copied: bool,
) -> dict[str, Any]:
    base_keys = {
        "timings_seconds",
        "median_seconds",
        "owns_data",
        "writeable",
        "shape",
        "checksum",
    }
    byte_keys = (
        {"allocation_bytes_per_repetition", "allocation_bytes_total"}
        if copied
        else {"owned_bytes_per_repetition"}
    )
    if not isinstance(value, dict) or set(value) != base_keys | byte_keys:
        raise DataBenchmarkError("copy measurement schema differs")
    timings = _validate_timings(
        value["timings_seconds"], repetitions=repetitions, name="copy measurement"
    )
    median = _positive_float(value["median_seconds"], "copy measurement median")
    if median != statistics.median(timings):
        raise DataBenchmarkError("copy measurement median differs")
    expected_shape = [rows, column_count]
    if value["shape"] != expected_shape:
        raise DataBenchmarkError("copy measurement shape differs")
    _sha(value["checksum"], "copy measurement checksum")
    expected_bytes = rows * column_count * 8
    if copied:
        if value["owns_data"] is not True or value["writeable"] is not True:
            raise DataBenchmarkError("explicit copy does not own writeable NumPy memory")
        if value["allocation_bytes_per_repetition"] != expected_bytes:
            raise DataBenchmarkError("explicit copy allocation bytes differ")
        if value["allocation_bytes_total"] != expected_bytes * repetitions:
            raise DataBenchmarkError("explicit copy total allocation bytes differ")
    elif (
        value["owns_data"] is not False
        or value["writeable"] is not False
        or value["owned_bytes_per_repetition"] != 0
    ):
        raise DataBenchmarkError("zero-copy result is not non-owning and read-only")
    return dict(value)


def load_data_benchmark_worker_result(
    path: Path,
    *,
    request: Mapping[str, Any],
    requested_threads: int,
) -> dict[str, Any]:
    """Load and fully validate one child measurement."""

    payload = _read_canonical(path, label="data benchmark worker result")
    if set(payload) != _RESULT_KEYS:
        raise DataBenchmarkError("data benchmark worker result schema differs")
    result_digest = payload.pop("result_digest")
    if not isinstance(result_digest, str) or result_digest != _digest(payload):
        raise DataBenchmarkError("data benchmark worker result digest differs")
    payload["result_digest"] = result_digest
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _RESULT_VERSION:
        raise DataBenchmarkError("data benchmark worker result version differs")
    if payload["request_digest"] != request["request_digest"]:
        raise DataBenchmarkError("data benchmark worker request digest differs")
    pid = _integer(payload["pid"], "worker PID")
    if pid == os.getpid():
        raise DataBenchmarkError("worker PID does not identify a child process")
    parent_pid = _integer(payload["parent_pid"], "worker parent PID")
    if pid == parent_pid:
        raise DataBenchmarkError("worker PID relationship differs")
    threads = payload["threads"]
    if not isinstance(threads, dict) or set(threads) != {"requested", "actual"}:
        raise DataBenchmarkError("worker thread schema differs")
    expected_threads = _integer(requested_threads, "requested_threads")
    if threads["requested"] != expected_threads or threads["actual"] != expected_threads:
        raise DataBenchmarkError("worker actual thread count differs from requested")
    dimensions = payload["dimensions"]
    if not isinstance(dimensions, dict) or set(dimensions) != {"rows", "columns"}:
        raise DataBenchmarkError("worker dimensions schema differs")
    rows = _integer(dimensions["rows"], "worker rows")
    columns = _columns(dimensions["columns"])
    if columns != request["columns"]:
        raise DataBenchmarkError("worker selected columns differ")
    repetitions = request["repetitions"]
    zero_copy = _validate_measurement(
        payload["zero_copy"],
        repetitions=repetitions,
        rows=rows,
        column_count=len(columns),
        copied=False,
    )
    explicit_copy = _validate_measurement(
        payload["explicit_copy"],
        repetitions=repetitions,
        rows=rows,
        column_count=len(columns),
        copied=True,
    )
    if zero_copy["checksum"] != explicit_copy["checksum"]:
        raise DataBenchmarkError("zero-copy and explicit-copy checksums differ")
    workload = payload["workload"]
    if not isinstance(workload, dict) or set(workload) != {
        "identity",
        "seed",
        "timings_seconds",
        "median_seconds",
        "checksum",
    }:
        raise DataBenchmarkError("worker workload schema differs")
    if workload["identity"] != request["workload"] or workload["seed"] != request["seed"]:
        raise DataBenchmarkError("worker workload identity differs")
    workload_timings = _validate_timings(
        workload["timings_seconds"], repetitions=repetitions, name="workload"
    )
    if _positive_float(workload["median_seconds"], "workload median") != statistics.median(
        workload_timings
    ):
        raise DataBenchmarkError("workload median differs")
    _sha(workload["checksum"], "workload checksum")
    _positive_float(payload["peak_rss_mb"], "worker peak_rss_mb")
    versions = payload["versions"]
    if (
        not isinstance(versions, dict)
        or set(versions) != {"python", "numpy", "polars"}
        or any(not isinstance(value, str) or not value for value in versions.values())
    ):
        raise DataBenchmarkError("worker environment versions differ")
    return payload


def run_data_benchmark_worker(
    request_path: Path,
    result_path: Path,
    *,
    requested_threads: int,
    timeout_seconds: float,
    worker_command: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run one Polars measurement with its thread pool fixed before import."""

    requested_threads = _integer(requested_threads, "requested_threads")
    timeout = _positive_float(timeout_seconds, "worker timeout")
    request = load_data_benchmark_request(request_path)
    result = Path(result_path)
    result.unlink(missing_ok=True)
    command = list(
        worker_command
        or (sys.executable, str(Path(__file__).with_name("data_benchmark_worker.py")))
    )
    command.extend(
        (
            str(Path(request_path).resolve()),
            str(result.resolve()),
            str(requested_threads),
        )
    )
    environment = os.environ.copy()
    environment["POLARS_MAX_THREADS"] = str(requested_threads)
    source_root = str(Path(__file__).resolve().parents[2])
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not existing_pythonpath
        else os.pathsep.join((source_root, existing_pythonpath))
    )
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=environment,
        )
    except OSError as error:
        raise DataBenchmarkError("data benchmark worker could not be launched") from error
    try:
        _, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        process.kill()
        process.communicate()
        raise DataBenchmarkError("data benchmark worker timeout") from error
    if process.returncode:
        message = stderr.strip()[-800:]
        raise DataBenchmarkError(
            f"data benchmark worker exit status {process.returncode}: {message}"
        )
    loaded = load_data_benchmark_worker_result(
        result, request=request, requested_threads=requested_threads
    )
    if loaded["pid"] != process.pid and loaded["parent_pid"] != process.pid:
        raise DataBenchmarkError("worker result PID differs from the spawned process")
    return loaded


def _load_bound_sampler_benchmark(path: Path, *, bound_sha256: str) -> dict[str, Any]:
    try:
        return load_sampler_benchmark(path, bound_sha256=bound_sha256)
    except SamplerWorkerError as error:
        raise DataBenchmarkError(str(error)) from error


def _processing_audit(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "pid": result["pid"],
        "requested_threads": result["threads"]["requested"],
        "actual_threads": result["threads"]["actual"],
        "timings_seconds": result["workload"]["timings_seconds"],
        "median_seconds": result["workload"]["median_seconds"],
        "peak_rss_mb": result["peak_rss_mb"],
        "checksum": result["workload"]["checksum"],
    }


def benchmark_polars_data(
    output_path: Path,
    *,
    request_path: Path,
    sampler_benchmark_path: Path,
    sampler_sha256: str,
    worker_directory: Path,
    timeout_seconds: float,
    worker_commands: Mapping[int, Sequence[str]] | None = None,
) -> Path:
    """Measure data conversion and serial/parallel Polars work, then publish a gate."""

    request = load_data_benchmark_request(request_path)
    sampler_path = Path(sampler_benchmark_path).resolve()
    sampler = _load_bound_sampler_benchmark(sampler_path, bound_sha256=sampler_sha256)
    sampler_artifact_sha = file_sha256(sampler_path)
    workers = request["workers"]
    if worker_commands is not None and set(worker_commands) != {1, workers}:
        raise DataBenchmarkError("worker command mapping must cover serial and parallel runs")
    directory = Path(worker_directory)
    directory.mkdir(parents=True, exist_ok=True)
    serial = run_data_benchmark_worker(
        request_path,
        directory / "serial-result.json",
        requested_threads=1,
        timeout_seconds=timeout_seconds,
        worker_command=None if worker_commands is None else worker_commands[1],
    )
    parallel = run_data_benchmark_worker(
        request_path,
        directory / "parallel-result.json",
        requested_threads=workers,
        timeout_seconds=timeout_seconds,
        worker_command=None if worker_commands is None else worker_commands[workers],
    )
    if serial["dimensions"] != parallel["dimensions"]:
        raise DataBenchmarkError("serial and parallel dimensions differ")
    if serial["zero_copy"]["checksum"] != parallel["zero_copy"]["checksum"]:
        raise DataBenchmarkError("serial and parallel input checksums differ")
    if serial["workload"]["checksum"] != parallel["workload"]["checksum"]:
        raise DataBenchmarkError("serial and parallel workload checksums differ")

    zero_median = serial["zero_copy"]["median_seconds"]
    copy_median = serial["explicit_copy"]["median_seconds"]
    serial_median = serial["workload"]["median_seconds"]
    parallel_median = parallel["workload"]["median_seconds"]
    sampler_wall = sampler["benchmarks"]["pymc"]["wall_seconds"]
    data_wall = serial_median + copy_median
    copy_share = copy_median / data_wall
    data_share = data_wall / (data_wall + sampler_wall)
    copy_bottleneck = copy_share >= 0.5
    rust_candidate = data_share >= 0.2 and copy_bottleneck
    rows = serial["dimensions"]["rows"]
    columns = serial["dimensions"]["columns"]
    payload: dict[str, Any] = {
        "schema_version": _VERSION,
        "request": {
            "request_digest": request["request_digest"],
            "artifact_path": request["input"]["path"],
            "artifact_sha256": request["input"]["sha256"],
            "workload": request["workload"],
            "seed": request["seed"],
            "rows": rows,
            "columns": columns,
            "repetitions": request["repetitions"],
            "workers": workers,
        },
        "sampler": {
            "artifact_path": str(sampler_path),
            "artifact_sha256": sampler_artifact_sha,
            "benchmark_digest": sampler["benchmark_digest"],
            "pymc_wall_seconds": sampler_wall,
        },
        "zero_copy": {**serial["zero_copy"], "source_worker_pid": serial["pid"]},
        "explicit_copy": {
            **serial["explicit_copy"],
            "source_worker_pid": serial["pid"],
        },
        "processing": {
            "serial": _processing_audit(serial),
            "parallel": _processing_audit(parallel),
        },
        "speed_ratios": {
            "explicit_copy_over_zero_copy": copy_median / zero_median,
            "serial_over_parallel": serial_median / parallel_median,
        },
        "derived": {
            "measured_data_processing_seconds": data_wall,
            "measured_total_seconds": data_wall + sampler_wall,
            "explicit_copy_share_of_data_processing": copy_share,
            "data_processing_share_of_measured_total": data_share,
            "explicit_copy_real_bottleneck": copy_bottleneck,
        },
        "decision": {
            "data_processing_share_threshold": 0.2,
            "explicit_copy_bottleneck_share_threshold": 0.5,
            "custom_rust_candidate": rust_candidate,
            "custom_rust_implemented": False,
            "polars_already_rust": True,
            "status": (
                "candidate-for-further-benchmarking-only"
                if rust_candidate
                else "deferred-by-measured-gate"
            ),
            "rationale": "Polars already executes in Rust; no custom Rust implementation was made.",
        },
        "environment": {
            "serial": serial["versions"],
            "parallel": parallel["versions"],
        },
    }
    payload["benchmark_digest"] = _digest(payload)
    return _write_json(Path(output_path), payload)


def _validate_final_measurement(
    value: object,
    *,
    repetitions: int,
    rows: int,
    column_count: int,
    copied: bool,
) -> tuple[dict[str, Any], int]:
    if not isinstance(value, dict) or "source_worker_pid" not in value:
        raise DataBenchmarkError("published copy measurement schema differs")
    measurement = dict(value)
    source_pid = _integer(measurement.pop("source_worker_pid"), "copy source worker PID")
    validated = _validate_measurement(
        measurement,
        repetitions=repetitions,
        rows=rows,
        column_count=column_count,
        copied=copied,
    )
    return validated, source_pid


def _validate_processing_output(
    value: object,
    *,
    repetitions: int,
    requested_threads: int,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != {
        "pid",
        "requested_threads",
        "actual_threads",
        "timings_seconds",
        "median_seconds",
        "peak_rss_mb",
        "checksum",
    }:
        raise DataBenchmarkError(f"published {label} processing schema differs")
    _integer(value["pid"], f"{label} worker PID")
    if (
        value["requested_threads"] != requested_threads
        or value["actual_threads"] != requested_threads
    ):
        raise DataBenchmarkError(f"published {label} thread count differs")
    timings = _validate_timings(
        value["timings_seconds"], repetitions=repetitions, name=f"{label} workload"
    )
    median = _positive_float(value["median_seconds"], f"{label} workload median")
    if median != statistics.median(timings):
        raise DataBenchmarkError(f"published {label} workload median differs")
    _positive_float(value["peak_rss_mb"], f"{label} peak_rss_mb")
    _sha(value["checksum"], f"{label} workload checksum")
    return dict(value)


def _validate_environment_versions(value: object, *, label: str) -> dict[str, str]:
    if (
        not isinstance(value, dict)
        or set(value) != {"python", "numpy", "polars"}
        or any(not isinstance(item, str) or not item for item in value.values())
    ):
        raise DataBenchmarkError(f"published {label} environment differs")
    return dict(value)


def load_data_benchmark(path: Path) -> dict[str, Any]:
    """Load and fully revalidate a published benchmark and its bound artifacts."""

    payload = _read_canonical(path, label="data benchmark")
    if set(payload) != _FINAL_KEYS:
        raise DataBenchmarkError("data benchmark schema differs")
    digest = payload.pop("benchmark_digest")
    if not isinstance(digest, str) or digest != _digest(payload):
        raise DataBenchmarkError("data benchmark digest differs")
    payload["benchmark_digest"] = digest
    if type(payload["schema_version"]) is not int or payload["schema_version"] != _VERSION:
        raise DataBenchmarkError("data benchmark version differs")
    request = payload["request"]
    if not isinstance(request, dict) or set(request) != {
        "request_digest",
        "artifact_path",
        "artifact_sha256",
        "workload",
        "seed",
        "rows",
        "columns",
        "repetitions",
        "workers",
    }:
        raise DataBenchmarkError("data benchmark request audit schema differs")
    _sha(request["request_digest"], "request digest")
    artifact_sha = _sha(request["artifact_sha256"], "artifact sha256")
    if not isinstance(request["artifact_path"], str):
        raise DataBenchmarkError("data benchmark artifact path differs")
    artifact = Path(request["artifact_path"])
    if (
        not artifact.is_absolute()
        or not artifact.is_file()
        or artifact.suffix.lower() != ".parquet"
    ):
        raise DataBenchmarkError("data benchmark artifact path differs")
    if file_sha256(artifact) != artifact_sha:
        raise DataBenchmarkError("data benchmark artifact digest differs")
    rows = _integer(request["rows"], "rows")
    columns = _columns(request["columns"])
    repetitions = _integer(request["repetitions"], "repetitions", minimum=3)
    workers = _integer(request["workers"], "workers", minimum=2)
    if (
        isinstance(request["seed"], bool)
        or not isinstance(request["seed"], int)
        or request["seed"] < 0
    ):
        raise DataBenchmarkError("data benchmark seed differs")
    if request["workload"] != _WORKLOAD:
        raise DataBenchmarkError("data benchmark workload differs")
    reconstructed_request = {
        "schema_version": _VERSION,
        "input": {"path": request["artifact_path"], "sha256": artifact_sha},
        "columns": columns,
        "repetitions": repetitions,
        "workers": workers,
        "seed": request["seed"],
        "workload": request["workload"],
    }
    if _digest(reconstructed_request) != request["request_digest"]:
        raise DataBenchmarkError("published request identity differs")
    sampler = payload["sampler"]
    if not isinstance(sampler, dict) or set(sampler) != {
        "artifact_path",
        "artifact_sha256",
        "benchmark_digest",
        "pymc_wall_seconds",
    }:
        raise DataBenchmarkError("data benchmark sampler audit schema differs")
    if not isinstance(sampler["artifact_path"], str):
        raise DataBenchmarkError("sampler artifact path differs")
    sampler_path = Path(sampler["artifact_path"])
    if not sampler_path.is_absolute():
        raise DataBenchmarkError("sampler artifact path differs")
    validated_sampler = _load_bound_sampler_benchmark(
        sampler_path,
        bound_sha256=_sha(sampler["artifact_sha256"], "sampler artifact sha256"),
    )
    if validated_sampler["benchmark_digest"] != _sha(
        sampler["benchmark_digest"], "sampler benchmark digest"
    ):
        raise DataBenchmarkError("published sampler benchmark identity differs")
    sampler_wall = _positive_float(sampler["pymc_wall_seconds"], "sampler wall seconds")
    if sampler_wall != validated_sampler["benchmarks"]["pymc"]["wall_seconds"]:
        raise DataBenchmarkError("published sampler wall time differs")

    zero_copy, zero_pid = _validate_final_measurement(
        payload["zero_copy"],
        repetitions=repetitions,
        rows=rows,
        column_count=len(columns),
        copied=False,
    )
    explicit_copy, copy_pid = _validate_final_measurement(
        payload["explicit_copy"],
        repetitions=repetitions,
        rows=rows,
        column_count=len(columns),
        copied=True,
    )
    if zero_pid != copy_pid or zero_copy["checksum"] != explicit_copy["checksum"]:
        raise DataBenchmarkError("published copy measurement identity differs")

    if not isinstance(payload["processing"], dict) or set(payload["processing"]) != {
        "serial",
        "parallel",
    }:
        raise DataBenchmarkError("data benchmark processing schema differs")
    serial = _validate_processing_output(
        payload["processing"]["serial"],
        repetitions=repetitions,
        requested_threads=1,
        label="serial",
    )
    parallel = _validate_processing_output(
        payload["processing"]["parallel"],
        repetitions=repetitions,
        requested_threads=workers,
        label="parallel",
    )
    if serial["pid"] != zero_pid or serial["pid"] == parallel["pid"]:
        raise DataBenchmarkError("published worker PID identity differs")
    if serial["checksum"] != parallel["checksum"]:
        raise DataBenchmarkError("published serial and parallel checksums differ")

    ratios = payload["speed_ratios"]
    if not isinstance(ratios, dict) or set(ratios) != {
        "explicit_copy_over_zero_copy",
        "serial_over_parallel",
    }:
        raise DataBenchmarkError("data benchmark speed ratio schema differs")
    expected_copy_ratio = explicit_copy["median_seconds"] / zero_copy["median_seconds"]
    expected_parallel_ratio = serial["median_seconds"] / parallel["median_seconds"]
    if not math.isclose(
        _positive_float(ratios["explicit_copy_over_zero_copy"], "copy speed ratio"),
        expected_copy_ratio,
        rel_tol=1e-12,
    ) or not math.isclose(
        _positive_float(ratios["serial_over_parallel"], "parallel speed ratio"),
        expected_parallel_ratio,
        rel_tol=1e-12,
    ):
        raise DataBenchmarkError("data benchmark speed ratio differs")

    derived = payload["derived"]
    if not isinstance(derived, dict) or set(derived) != {
        "measured_data_processing_seconds",
        "measured_total_seconds",
        "explicit_copy_share_of_data_processing",
        "data_processing_share_of_measured_total",
        "explicit_copy_real_bottleneck",
    }:
        raise DataBenchmarkError("data benchmark decision schema differs")
    data_wall = serial["median_seconds"] + explicit_copy["median_seconds"]
    total_wall = data_wall + sampler_wall
    copy_share = explicit_copy["median_seconds"] / data_wall
    data_share = data_wall / total_wall
    measured_values = (
        ("measured_data_processing_seconds", data_wall),
        ("measured_total_seconds", total_wall),
        ("explicit_copy_share_of_data_processing", copy_share),
        ("data_processing_share_of_measured_total", data_share),
    )
    if any(
        not math.isclose(_positive_float(derived[name], f"derived {name}"), expected, rel_tol=1e-12)
        for name, expected in measured_values
    ):
        raise DataBenchmarkError("data benchmark measured share differs")
    copy_bottleneck = copy_share >= 0.5
    if derived["explicit_copy_real_bottleneck"] is not copy_bottleneck:
        raise DataBenchmarkError("data benchmark copy bottleneck decision differs")

    decision = payload["decision"]
    if not isinstance(decision, dict) or set(decision) != {
        "data_processing_share_threshold",
        "explicit_copy_bottleneck_share_threshold",
        "custom_rust_candidate",
        "custom_rust_implemented",
        "polars_already_rust",
        "status",
        "rationale",
    }:
        raise DataBenchmarkError("data benchmark decision schema differs")
    if decision["data_processing_share_threshold"] != 0.2:
        raise DataBenchmarkError("data processing share threshold differs")
    if decision["explicit_copy_bottleneck_share_threshold"] != 0.5:
        raise DataBenchmarkError("copy bottleneck threshold differs")
    rust_candidate = data_share >= 0.2 and copy_bottleneck
    if decision["custom_rust_candidate"] is not rust_candidate:
        raise DataBenchmarkError("custom Rust candidate decision differs")
    if decision["custom_rust_implemented"] is not False:
        raise DataBenchmarkError("custom Rust implementation claim differs")
    if decision["polars_already_rust"] is not True:
        raise DataBenchmarkError("Polars engine claim differs")
    expected_status = (
        "candidate-for-further-benchmarking-only" if rust_candidate else "deferred-by-measured-gate"
    )
    if decision["status"] != expected_status:
        raise DataBenchmarkError("custom Rust status differs")
    if (
        decision["rationale"]
        != "Polars already executes in Rust; no custom Rust implementation was made."
    ):
        raise DataBenchmarkError("custom Rust rationale differs")

    environment = payload["environment"]
    if not isinstance(environment, dict) or set(environment) != {"serial", "parallel"}:
        raise DataBenchmarkError("data benchmark environment schema differs")
    serial_versions = _validate_environment_versions(environment["serial"], label="serial")
    parallel_versions = _validate_environment_versions(environment["parallel"], label="parallel")
    if serial_versions != parallel_versions:
        raise DataBenchmarkError("serial and parallel environment versions differ")
    return payload


__all__ = [
    "DataBenchmarkError",
    "benchmark_polars_data",
    "load_data_benchmark",
    "load_data_benchmark_request",
    "load_data_benchmark_worker_result",
    "run_data_benchmark_worker",
    "write_data_benchmark_request",
]
