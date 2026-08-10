"""Atomic, fail-closed experiment reporting and lightweight benchmark contracts."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import tempfile
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.util import find_spec
from pathlib import Path
from typing import Literal

import numpy as np
import polars as pl

from hqrc_v3.contracts import DataContractError, validate_prediction_frame
from hqrc_v3.diagnostics.ar import approve_calibration, calibrate_beta_prior, write_ar_diagnostics
from hqrc_v3.provenance import file_sha256


class ReportContractError(ValueError):
    """Raised before an incomplete, incompatible, or non-paper run can be reported."""


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


def sampler_eligible_default(
    pymc: SamplerBenchmark, nutpie: SamplerBenchmark, *, mean_distance_sd: float
) -> bool:
    """Apply the predeclared default gate; unavailable nutpie is never eligible."""

    return bool(
        nutpie.status == "ok"
        and mean_distance_sd <= 0.1
        and nutpie.divergences <= pymc.divergences
        and nutpie.max_rhat <= pymc.max_rhat + 0.01
        and (
            nutpie.wall_seconds <= pymc.wall_seconds * 0.8
            or nutpie.min_bulk_ess_per_second >= pymc.min_bulk_ess_per_second * 1.2
        )
    )


def _manifest(run: Path) -> dict[str, object]:
    path = run / "manifest.json"
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReportContractError("missing or invalid manifest.json") from error
    required = {"data_sha256", "config_sha256", "event_sha256", "profile", "artifacts"}
    if (
        not required <= set(value)
        or not isinstance(value["artifacts"], dict)
        or value["profile"] not in {"smoke", "paper"}
        or any(
            not isinstance(value[key], str) or not value[key] for key in required - {"artifacts"}
        )
    ):
        raise ReportContractError("manifest is partial or incompatible")
    return value


def _verify_artifacts(run: Path, manifest: dict[str, object]) -> None:
    artifacts = manifest["artifacts"]
    assert isinstance(artifacts, dict)
    required = (
        "predictions/oof.parquet",
        "ar_diagnostics/approved.json",
        "metrics/event_metrics.parquet",
        "posterior/diagnostics.json",
    )
    for relative in required:
        path = run / relative
        if not path.is_file():
            if relative.endswith("approved.json"):
                raise ReportContractError("approved AR calibration artifact is required")
            raise ReportContractError(f"required artifact is missing: {relative}")
        if artifacts.get(relative) != file_sha256(path):
            raise ReportContractError(f"artifact hash mismatch: {relative}")
    for relative, digest in artifacts.items():
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or not digest
        ):
            raise ReportContractError("manifest contains an invalid artifact identity")
        path = run / relative
        if not path.is_file() or digest != file_sha256(path):
            raise ReportContractError(f"artifact hash mismatch: {relative}")


def _atomic_complete(run: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=".COMPLETE.", dir=run)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write("complete\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, run / "COMPLETE")
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_report(run_dir: Path, *, profile: Literal["smoke", "paper"] = "smoke") -> Path:
    """Validate immutable inputs and emit normalized CSV/Parquet reports atomically last."""

    run = Path(run_dir)
    manifest = _manifest(run)
    if manifest["profile"] != profile:
        raise ReportContractError("requested report profile differs from the run manifest")
    if profile == "paper" and manifest.get("paper_diagnostics_passed") is not True:
        raise ReportContractError("paper diagnostics did not pass; paper-ready output is forbidden")
    _verify_artifacts(run, manifest)
    try:
        diagnostics = json.loads((run / "posterior/diagnostics.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ReportContractError("posterior diagnostics are missing or invalid") from error
    if profile == "paper" and diagnostics.get("paper_diagnostics_passed") is not True:
        raise ReportContractError(
            "posterior diagnostics did not pass; paper-ready output is forbidden"
        )
    try:
        validate_prediction_frame(pl.read_parquet(run / "predictions/oof.parquet"))
    except (DataContractError, OSError, pl.exceptions.PolarsError) as error:
        raise ReportContractError("OOF prediction artifact has an invalid schema") from error
    metrics = pl.read_parquet(run / "metrics/event_metrics.parquet")
    if metrics.is_empty() or "event_id" not in metrics.columns:
        raise ReportContractError("event metrics have an invalid schema")
    output = run / "reports"
    output.mkdir(exist_ok=True)
    metrics.write_parquet(output / "event_metrics.parquet")
    metrics.write_csv(output / "event_metrics.csv")
    pl.DataFrame(
        {"artifact": list(manifest["artifacts"]), "digest": list(manifest["artifacts"].values())}
    ).write_csv(output / "artifact_inventory.csv")
    figures = run / "figures"
    figures.mkdir(exist_ok=True)
    (figures / "report_coverage.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="480" height="90">'
        '<text x="10" y="45">Normalized event metrics report</text></svg>\n'
    )
    if profile == "smoke":
        (output / "NON_PAPER.txt").write_text("smoke result: never use for manuscript numbers\n")
    _atomic_complete(run)
    return output


def run_synthetic_pipeline(*, output_dir: Path, seed: int) -> Path:
    """Create a deterministic small, real-schema run; approval remains explicitly staged."""

    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ReportContractError("seed must be an integer")
    run = Path(output_dir)
    if run.exists():
        raise ReportContractError("synthetic output directory must not already exist")
    for directory in ("predictions", "ar_diagnostics", "metrics", "posterior", "figures"):
        (run / directory).mkdir(parents=True, exist_ok=True)
    origin = datetime(2020, 1, 1)
    prediction = pl.DataFrame(
        {
            "origin": [origin] * 24,
            "target_timestamp": [datetime(2020, 1, 1, hour) for hour in range(24)],
            "horizon": list(range(1, 25)),
            "observed_mw": [100.0] * 24,
            "predicted_mw": [99.0] * 24,
            "model": ["synthetic"] * 24,
            "feature_set": ["B1"] * 24,
            "seed": [seed] * 24,
            "split_id": ["oof-2020"] * 24,
        }
    )
    metric = pl.DataFrame({"event_id": ["synthetic-event"], "rmse": [1.0], "crps": [0.5]})
    prediction_path, metric_path = (
        run / "predictions/oof.parquet",
        run / "metrics/event_metrics.parquet",
    )
    prediction.write_parquet(prediction_path)
    metric.write_parquet(metric_path)
    diagnostics_path = run / "posterior/diagnostics.json"
    diagnostics_path.write_text(
        json.dumps({"paper_diagnostics_passed": False, "profile": "smoke"}, sort_keys=True)
    )
    benchmark_path = write_benchmark(
        run / "benchmarks/samplers.json",
        [
            SamplerBenchmark(
                backend="pymc",
                wall_seconds=0.0,
                peak_rss_mb=0.0,
                min_bulk_ess_per_second=0.0,
                min_tail_ess_per_second=0.0,
                max_rhat=0.0,
                divergences=0,
                status="not-run",
            )
        ],
    )
    proposal = write_ar_diagnostics(
        run / "ar_diagnostics/proposed.json",
        (),
        calibrate_beta_prior(np.array([0.2, 0.4]), event_ids=("a", "b")),
        residual_sha256="synthetic-residual",
        config_sha256="synthetic-config",
        event_sha256="synthetic-events",
        context={
            "model": "synthetic",
            "feature_set": "B1",
            "seed": seed,
            "split_ids": ("oof-2020",),
        },
    )
    approved = approve_calibration(
        proposal,
        run / "ar_diagnostics/approved.json",
        current_residual_sha256="synthetic-residual",
        current_config_sha256="synthetic-config",
        current_event_sha256="synthetic-events",
    )
    artifacts = {
        str(path.relative_to(run)): file_sha256(path)
        for path in (prediction_path, metric_path, diagnostics_path, benchmark_path, approved)
    }
    manifest = {
        "git_commit": _git_commit(),
        "dirty": True,
        "data_sha256": "synthetic-data",
        "config_sha256": "synthetic-config",
        "event_sha256": "synthetic-events",
        "profile": "smoke",
        "seed": seed,
        "split": "synthetic",
        "sampler": "none",
        "started_at": datetime.now(UTC).isoformat(),
        "hardware": platform.platform(),
        "artifacts": artifacts,
    }
    (run / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    build_report(run, profile="smoke")
    return run


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except OSError:
        return "unknown"


def write_benchmark(
    path: Path,
    benchmarks: list[SamplerBenchmark],
    *,
    mean_distance_sd: float | None = None,
    copy_bottleneck: bool = False,
    preprocess_share: float = 0.0,
) -> Path:
    """Write a structured sampler decision, including optional-backend absence."""

    by_backend = {item.backend: item for item in benchmarks}
    if len(by_backend) != len(benchmarks) or "pymc" not in by_backend:
        raise ReportContractError("benchmark must contain one PyMC result per backend")
    if "nutpie" not in by_backend:
        if find_spec("nutpie") is not None:
            raise ReportContractError("installed nutpie requires a measured benchmark result")
        by_backend["nutpie"] = SamplerBenchmark(
            backend="nutpie",
            wall_seconds=0.0,
            peak_rss_mb=0.0,
            min_bulk_ess_per_second=0.0,
            min_tail_ess_per_second=0.0,
            max_rhat=0.0,
            divergences=0,
            status="not-installed",
        )
    pymc, nutpie = by_backend["pymc"], by_backend["nutpie"]
    if nutpie.status == "ok" and mean_distance_sd is not None:
        by_backend["nutpie"] = replace(
            nutpie,
            eligible_default=sampler_eligible_default(
                pymc, nutpie, mean_distance_sd=mean_distance_sd
            ),
        )

    payload = {
        "benchmarks": [asdict(by_backend[name]) for name in sorted(by_backend)],
        "mean_distance_sd": mean_distance_sd,
        "arrow_polars": {
            "zero_copy_vs_explicit_copy": "not-measured",
            "parallel_execution": "not-measured",
        },
        "custom_rust_rewrite": "deferred"
        if not (copy_bottleneck and preprocess_share >= 0.2)
        else "evaluate-after-measurement",
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, sort_keys=True))
    return Path(path)
