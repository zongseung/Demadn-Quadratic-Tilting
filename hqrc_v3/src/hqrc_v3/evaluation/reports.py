"""Strict, provenance-bound publication for materialized HQRC experiment runs."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Literal

import arviz as az
import numpy as np
import polars as pl

from hqrc_v3.bayes.samplers import SamplingError, validate_inference_data
from hqrc_v3.contracts import DataContractError, validate_prediction_frame
from hqrc_v3.diagnostics.ar import (
    calibrate_beta_prior,
    load_approved_calibration,
    write_ar_diagnostics,
)
from hqrc_v3.provenance import file_sha256

_MANIFEST_VERSION = 2
_INPUT_KEYS = frozenset(
    {
        "source_audit",
        "resolved_config",
        "event_registry",
        "standardized_residuals",
        "oof_predictions",
        "approved_ar",
        "posterior",
        "event_metrics",
        "sampler_benchmark",
    }
)
_OUTPUT_KEYS = frozenset(
    {
        "reports/event_metrics.parquet",
        "reports/event_metrics.csv",
        "reports/NON_PAPER.txt",
        "figures/event_metrics.svg",
    }
)
_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "git",
        "source",
        "config",
        "event_registry",
        "residuals",
        "seed",
        "split",
        "evaluation",
        "sampler",
        "model",
        "profile",
        "libraries",
        "hardware",
        "started_at",
        "ended_at",
        "input_artifacts",
        "output_artifacts",
    }
)


class ReportContractError(ValueError):
    """Raised before an incomplete or provenance-incompatible run is published."""


@dataclass(frozen=True)
class SamplerBenchmark:
    backend: str
    wall_seconds: float
    peak_rss_mb: float
    min_bulk_ess_per_second: float
    min_tail_ess_per_second: float
    max_rhat: float
    divergences: int
    status: str = "ok"
    eligible_default: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.backend, str) or self.backend not in {"pymc", "nutpie"}:
            raise ReportContractError("benchmark backend must be pymc or nutpie")
        if self.status not in {"ok", "not-installed"}:
            raise ReportContractError("benchmark status must be ok or not-installed")
        numeric = (
            self.wall_seconds,
            self.peak_rss_mb,
            self.min_bulk_ess_per_second,
            self.min_tail_ess_per_second,
            self.max_rhat,
        )
        if any(
            not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0
            for value in numeric
        ):
            raise ReportContractError("benchmark numbers must be finite and non-negative")
        if (
            not isinstance(self.divergences, int)
            or isinstance(self.divergences, bool)
            or self.divergences < 0
        ):
            raise ReportContractError("benchmark divergences must be a non-negative integer")
        if self.status == "ok" and (
            self.wall_seconds <= 0
            or self.peak_rss_mb <= 0
            or self.min_bulk_ess_per_second <= 0
            or self.min_tail_ess_per_second <= 0
            or self.max_rhat <= 0
        ):
            raise ReportContractError(
                "measured benchmark requires positive timing, RSS, ESS, and R-hat"
            )
        if self.status == "not-installed" and self.eligible_default:
            raise ReportContractError("unavailable nutpie cannot be the default")


def sampler_eligible_default(
    pymc: SamplerBenchmark, nutpie: SamplerBenchmark, *, mean_distance_sd: float
) -> bool:
    """Apply the exact predeclared optional-backend selection rule."""

    if (
        not isinstance(mean_distance_sd, (int, float))
        or not math.isfinite(mean_distance_sd)
        or mean_distance_sd < 0
    ):
        raise ReportContractError("posterior mean distance must be finite and non-negative")
    return bool(
        pymc.status == nutpie.status == "ok"
        and mean_distance_sd <= 0.1
        and nutpie.divergences <= pymc.divergences
        and nutpie.max_rhat <= pymc.max_rhat + 0.01
        and (
            nutpie.wall_seconds <= pymc.wall_seconds * 0.8
            or nutpie.min_bulk_ess_per_second >= pymc.min_bulk_ess_per_second * 1.2
        )
    )


def _safe_path(run: Path, value: object) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise ReportContractError("manifest contains an unsafe artifact path")
    return run / value


def _entry(run: Path, relative: str) -> dict[str, str]:
    path = _safe_path(run, relative)
    if not path.is_file():
        raise ReportContractError(f"required artifact is missing: {relative}")
    return {"path": relative, "sha256": file_sha256(path)}


def _utc(value: object) -> datetime:
    if not isinstance(value, str):
        raise ReportContractError("manifest timestamps must be UTC ISO strings")
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as error:
        raise ReportContractError("manifest timestamps must be UTC ISO strings") from error
    if parsed.tzinfo != UTC:
        raise ReportContractError("manifest timestamps must use UTC")
    return parsed


def _object(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ReportContractError(f"manifest {name} must be an object")
    return value


def _verify_entries(
    run: Path, entries: object, *, required: frozenset[str]
) -> dict[str, dict[str, str]]:
    mapping = _object(entries, "artifact map")
    if set(mapping) != required:
        raise ReportContractError("manifest artifact map has missing or unknown entries")
    result: dict[str, dict[str, str]] = {}
    for name, entry in mapping.items():
        item = _object(entry, f"artifact {name}")
        if set(item) != {"path", "sha256"} or not isinstance(item.get("sha256"), str):
            raise ReportContractError("manifest artifact entry is invalid")
        actual = _entry(run, item["path"])
        if actual["sha256"] != item["sha256"]:
            raise ReportContractError(f"artifact hash mismatch: {item['path']}")
        result[name] = actual
    return result


def _manifest(run: Path) -> dict[str, Any]:
    try:
        value = json.loads((run / "manifest.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReportContractError("missing or invalid manifest.json") from error
    if (
        not isinstance(value, dict)
        or set(value) != _MANIFEST_KEYS
        or value.get("schema_version") != _MANIFEST_VERSION
    ):
        raise ReportContractError("manifest is not the versioned strict schema")
    git, source, config, events, residuals = (
        _object(value[name], name)
        for name in ("git", "source", "config", "event_registry", "residuals")
    )
    if (
        set(git) != {"commit", "dirty"}
        or not isinstance(git["commit"], str)
        or not isinstance(git["dirty"], bool)
    ):
        raise ReportContractError("manifest git identity is invalid")
    for identity in (source, config, events, residuals):
        if set(identity) != {"path", "sha256"} or not isinstance(identity["sha256"], str):
            raise ReportContractError("manifest input identity is invalid")
    if not isinstance(value["seed"], int) or isinstance(value["seed"], bool) or value["seed"] < 0:
        raise ReportContractError("manifest seed is invalid")
    for name in ("split", "evaluation", "sampler", "model", "libraries", "hardware"):
        _object(value[name], name)
    if value["profile"] not in {"smoke", "paper"}:
        raise ReportContractError("manifest profile is invalid")
    if _utc(value["ended_at"]) < _utc(value["started_at"]):
        raise ReportContractError("manifest end time precedes start time")
    inputs = _verify_entries(run, value["input_artifacts"], required=_INPUT_KEYS)
    outputs = _object(value["output_artifacts"], "output artifact map")
    if set(outputs) and set(outputs) != set(_OUTPUT_KEYS):
        raise ReportContractError("manifest output artifact map has missing or unknown entries")
    if outputs:
        _verify_entries(run, outputs, required=_OUTPUT_KEYS)
    for identity, input_name in (
        (source, "source_audit"),
        (config, "resolved_config"),
        (events, "event_registry"),
        (residuals, "standardized_residuals"),
    ):
        if identity != inputs[input_name]:
            raise ReportContractError("manifest identity does not match its bound artifact digest")
    return value


def _validate_posterior(path: Path, manifest: dict[str, Any], *, paper: bool) -> None:
    try:
        idata = az.from_netcdf(path)
        validate_inference_data(idata, paper_profile=paper)
        attrs = json.loads(str(idata.attrs["hqrc_sampler_json"]))
    except (KeyError, OSError, SamplingError, ValueError, TypeError) as error:
        raise ReportContractError(
            "posterior InferenceData or sampler diagnostics are invalid"
        ) from error
    if not isinstance(attrs, dict) or attrs != manifest["sampler"]:
        raise ReportContractError("posterior sampler attrs do not match manifest")
    if paper and (
        attrs.get("chains") != 4 or attrs.get("draws", 0) < 1000 or attrs.get("tune", 0) < 1000
    ):
        raise ReportContractError(
            "paper sampling requires four chains and at least 1000 tune/draws"
        )


def _data_svg(metrics: pl.DataFrame) -> str:
    values = [
        float(value)
        for value in metrics.select(pl.exclude("event_id")).to_numpy().reshape(-1)
        if np.isfinite(value)
    ]
    if not values:
        raise ReportContractError("event metrics contain no finite numeric values for a figure")
    upper = max(values)
    bars = "".join(
        f'<rect x="{10 + index * 24}" y="{80 - 60 * value / upper:.2f}" '
        f'width="16" height="{60 * value / upper:.2f}"/>'
        for index, value in enumerate(values[:12])
    )
    return f'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="100"><g>{bars}</g></svg>\n'


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_report(run_dir: Path, *, profile: Literal["smoke", "paper"] = "smoke") -> Path:
    """Revalidate every bound input, stage data-derived results, then publish atomically."""

    run = Path(run_dir)
    (run / "COMPLETE").unlink(missing_ok=True)
    manifest = _manifest(run)
    if manifest["profile"] != profile:
        raise ReportContractError("requested report profile differs from the run manifest")
    inputs = manifest["input_artifacts"]
    residual, config, events = (
        inputs[name]["sha256"]
        for name in ("standardized_residuals", "resolved_config", "event_registry")
    )
    try:
        load_approved_calibration(
            _safe_path(run, inputs["approved_ar"]["path"]),
            current_residual_sha256=residual,
            current_config_sha256=config,
            current_event_sha256=events,
        )
        validate_prediction_frame(
            pl.read_parquet(_safe_path(run, inputs["oof_predictions"]["path"]))
        )
    except (DataContractError, OSError, pl.exceptions.PolarsError, ValueError) as error:
        raise ReportContractError(
            "run artifacts fail prediction or AR-approval provenance validation"
        ) from error
    _validate_posterior(
        _safe_path(run, inputs["posterior"]["path"]), manifest, paper=profile == "paper"
    )
    try:
        metrics = pl.read_parquet(_safe_path(run, inputs["event_metrics"]["path"]))
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ReportContractError("event metrics are unreadable") from error
    if (
        metrics.is_empty()
        or "event_id" not in metrics.columns
        or not metrics.select(pl.exclude("event_id")).columns
    ):
        raise ReportContractError("event metrics have an incomplete normalized schema")
    if profile == "paper":
        raise ReportContractError(
            "paper reporting requires the full validated manuscript table suite"
        )
    stage = Path(tempfile.mkdtemp(prefix=".report-stage-", dir=run))
    try:
        report_dir, figure_dir = stage / "reports", stage / "figures"
        report_dir.mkdir()
        figure_dir.mkdir()
        metrics.write_parquet(report_dir / "event_metrics.parquet")
        metrics.write_csv(report_dir / "event_metrics.csv")
        (report_dir / "NON_PAPER.txt").write_text(
            "smoke result: never use for manuscript numbers\n"
        )
        (figure_dir / "event_metrics.svg").write_text(_data_svg(metrics))
        destinations = ((report_dir, run / "reports"), (figure_dir, run / "figures"))
        for source, destination in destinations:
            destination.mkdir(exist_ok=True)
            for artifact in source.iterdir():
                os.replace(artifact, destination / artifact.name)
        output_paths = [
            "reports/event_metrics.parquet",
            "reports/event_metrics.csv",
            "reports/NON_PAPER.txt",
            "figures/event_metrics.svg",
        ]
        manifest["output_artifacts"] = {name: _entry(run, name) for name in output_paths}
        manifest["ended_at"] = datetime.now(UTC).isoformat()
        _atomic_json(run / "manifest.json", manifest)
        digest = file_sha256(run / "manifest.json")
        _atomic_json(
            run / "COMPLETE", {"manifest_sha256": digest, "schema_version": _MANIFEST_VERSION}
        )
    finally:
        shutil.rmtree(stage, ignore_errors=True)
    return run / "reports"


def prepare_synthetic_run(*, output_dir: Path, seed: int) -> Path:
    """Materialize smoke inputs and an unapproved proposal; caller controls approval."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ReportContractError("seed must be an integer")
    run = Path(output_dir)
    if run.exists():
        raise ReportContractError("synthetic output directory must not already exist")
    for directory in (
        "inputs",
        "predictions",
        "ar_diagnostics",
        "metrics",
        "posterior",
        "benchmarks",
    ):
        (run / directory).mkdir(parents=True, exist_ok=True)
    (run / "inputs/source_audit.json").write_text(json.dumps({"source": "synthetic", "rows": 96}))
    (run / "inputs/resolved_config.toml").write_text("[experiment]\nprofile = 'smoke'\n")
    (run / "inputs/event_registry.csv").write_text("occurrence_id\nsynthetic-a\nsynthetic-b\n")
    timestamps = [datetime(2020, 1, 1) + np.timedelta64(hour, "h") for hour in range(96)]
    residuals = pl.DataFrame(
        {
            "occurrence_id": ["synthetic-a"] * 48 + ["synthetic-b"] * 48,
            "timestamp": timestamps,
            "standardized_residual": np.linspace(-1.0, 1.0, 96),
            "tau_days": np.repeat(np.arange(48, dtype=float) / 24.0, 2),
            "hour": [time.hour for time in timestamps],
            "model": ["synthetic"] * 96,
            "feature_set": ["B1"] * 96,
            "seed": [seed] * 96,
            "split_id": ["oof-2020"] * 96,
        }
    )
    residual_path = run / "inputs/standardized_residuals.parquet"
    residuals.write_parquet(residual_path)
    prediction = pl.DataFrame(
        {
            "origin": [datetime(2020, 1, 1)] * 24,
            "target_timestamp": [
                datetime(2020, 1, 1) + np.timedelta64(hour, "h") for hour in range(24)
            ],
            "horizon": list(range(1, 25)),
            "observed_mw": [100.0] * 24,
            "predicted_mw": [99.0] * 24,
            "model": ["synthetic"] * 24,
            "feature_set": ["B1"] * 24,
            "seed": [seed] * 24,
            "split_id": ["oof-2020"] * 24,
        }
    )
    prediction.write_parquet(run / "predictions/oof.parquet")
    metrics = pl.DataFrame(
        {"event_id": ["synthetic-a", "synthetic-b"], "rmse": [1.0, 2.0], "crps": [0.5, 0.7]}
    )
    metrics.write_parquet(run / "metrics/event_metrics.parquet")
    source, config, events = (
        file_sha256(run / path)
        for path in (
            "inputs/source_audit.json",
            "inputs/resolved_config.toml",
            "inputs/event_registry.csv",
        )
    )
    proposal = write_ar_diagnostics(
        run / "ar_diagnostics/proposed.json",
        (),
        calibrate_beta_prior(np.array([0.2, 0.4]), event_ids=("synthetic-a", "synthetic-b")),
        residual_sha256=file_sha256(residual_path),
        config_sha256=config,
        event_sha256=events,
        context={
            "model": "synthetic",
            "feature_set": "B1",
            "seed": seed,
            "split_ids": ("oof-2020",),
        },
    )
    idata = az.from_dict(
        posterior={"theta": np.ones((2, 20))},
        sample_stats={"diverging": np.zeros((2, 20), dtype=int)},
    )
    idata.attrs["hqrc_sampler_json"] = json.dumps(
        {"backend": "pymc", "draws": 20, "tune": 10, "chains": 2, "seed": seed}
    )
    idata.to_netcdf(run / "posterior/smoke.nc")
    write_benchmark(
        run / "benchmarks/samplers.json", [SamplerBenchmark("pymc", 1.0, 1.0, 1.0, 1.0, 1.0, 0)]
    )
    return proposal


def run_synthetic_pipeline(*, output_dir: Path, seed: int, approved_path: Path) -> Path:
    """Finalize only a caller-approved synthetic run; never approves calibration itself."""

    run = Path(output_dir)
    approved = Path(approved_path)
    if not run.is_dir() or approved != run / "ar_diagnostics/approved.json":
        raise ReportContractError(
            "synthetic finalization requires its explicit in-run approved artifact"
        )
    paths = {
        "source_audit": "inputs/source_audit.json",
        "resolved_config": "inputs/resolved_config.toml",
        "event_registry": "inputs/event_registry.csv",
        "standardized_residuals": "inputs/standardized_residuals.parquet",
        "oof_predictions": "predictions/oof.parquet",
        "approved_ar": "ar_diagnostics/approved.json",
        "posterior": "posterior/smoke.nc",
        "event_metrics": "metrics/event_metrics.parquet",
        "sampler_benchmark": "benchmarks/samplers.json",
    }
    inputs = {name: _entry(run, path) for name, path in paths.items()}
    manifest: dict[str, Any] = {
        "schema_version": _MANIFEST_VERSION,
        "git": {"commit": _git_commit(), "dirty": True},
        "source": inputs["source_audit"],
        "config": inputs["resolved_config"],
        "event_registry": inputs["event_registry"],
        "residuals": inputs["standardized_residuals"],
        "seed": seed,
        "split": {"id": "oof-2020"},
        "evaluation": {"kind": "synthetic"},
        "sampler": {"backend": "pymc", "draws": 20, "tune": 10, "chains": 2, "seed": seed},
        "model": {"variant": "H3", "pooling": "partial", "feature_set": "B1"},
        "profile": "smoke",
        "libraries": {"arviz": importlib.metadata.version("arviz"), "polars": pl.__version__},
        "hardware": {"platform": platform.platform(), "cpu": platform.processor() or "unknown"},
        "started_at": datetime.now(UTC).isoformat(),
        "ended_at": datetime.now(UTC).isoformat(),
        "input_artifacts": inputs,
        "output_artifacts": {},
    }
    _atomic_json(run / "manifest.json", manifest)
    build_report(run, profile="smoke")
    return run


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except OSError:
        return "unknown"


def write_benchmark(
    path: Path, benchmarks: list[SamplerBenchmark], *, mean_distance_sd: float | None = None
) -> Path:
    """Serialize already measured backend results; production runners must supply measurements."""

    by_backend = {item.backend: item for item in benchmarks}
    if (
        len(by_backend) != len(benchmarks)
        or by_backend.get("pymc", None) is None
        or by_backend["pymc"].status != "ok"
    ):
        raise ReportContractError("benchmark requires one successful measured PyMC result")
    if "nutpie" not in by_backend:
        if find_spec("nutpie") is not None:
            raise ReportContractError("installed nutpie requires a measured benchmark result")
        by_backend["nutpie"] = SamplerBenchmark(
            "nutpie", 0.0, 0.0, 0.0, 0.0, 0.0, 0, "not-installed"
        )
    if by_backend["nutpie"].status == "ok":
        if mean_distance_sd is None:
            raise ReportContractError("measured nutpie requires posterior mean comparison")
        by_backend["nutpie"] = replace(
            by_backend["nutpie"],
            eligible_default=sampler_eligible_default(
                by_backend["pymc"], by_backend["nutpie"], mean_distance_sd=mean_distance_sd
            ),
        )
    payload = {
        "benchmarks": [asdict(by_backend[name]) for name in sorted(by_backend)],
        "mean_distance_sd": mean_distance_sd,
        "arrow_polars": {"status": "not-run"},
        "parallel": {"status": "not-run"},
        "custom_rust_rewrite": "deferred-without-measured-copy-bottleneck",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(Path(path), payload)
    return Path(path)
