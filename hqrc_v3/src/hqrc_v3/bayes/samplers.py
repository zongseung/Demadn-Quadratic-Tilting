"""Sampler selection and transparent diagnostics for HQRC Bayesian models."""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import arviz as az
import numpy as np
import xarray as xr

from hqrc_v3.bayes.model import HQRCData, HQRCModelOptions, Pooling, Variant, build_hqrc_model
from hqrc_v3.diagnostics.ar import ApprovedARCalibration, require_approved_calibration


class SamplingError(RuntimeError):
    """Raised when an unavailable sampler or paper-profile diagnostic gate fails."""


@dataclass(frozen=True)
class SamplingDiagnostics:
    max_rhat: float
    min_bulk_ess: float
    min_tail_ess: float
    divergences: int


def _summary_value(summary, name: str, reducer, default: float) -> float:
    if name not in summary:
        return default
    value = float(reducer(np.asarray(summary[name], dtype=float)))
    return value if math.isfinite(value) else default


def validate_inference_data(idata: az.InferenceData, *, paper_profile: bool) -> SamplingDiagnostics:
    """Return diagnostics and enforce the stricter paper-run thresholds when requested."""

    if not isinstance(idata, az.InferenceData) or not hasattr(idata, "posterior"):
        raise SamplingError("sampler did not return ArviZ InferenceData with posterior draws")
    summary = az.summary(idata, kind="diagnostics")
    if not hasattr(idata, "sample_stats") or "diverging" not in idata.sample_stats:
        if paper_profile:
            raise SamplingError("paper_profile requires sample_stats.diverging diagnostics")
        divergences = 0
    else:
        raw_diverging = np.asarray(idata.sample_stats["diverging"])
        if (
            raw_diverging.dtype.kind not in {"b", "i", "u"}
            or not np.isfinite(raw_diverging.astype(float)).all()
        ):
            raise SamplingError("sample_stats.diverging must be finite boolean/integer diagnostics")
        divergences = int(raw_diverging.sum())
    diagnostics = SamplingDiagnostics(
        max_rhat=_summary_value(summary, "r_hat", np.nanmax, math.inf),
        min_bulk_ess=_summary_value(summary, "ess_bulk", np.nanmin, 0.0),
        min_tail_ess=_summary_value(summary, "ess_tail", np.nanmin, 0.0),
        divergences=divergences,
    )
    if paper_profile and (
        diagnostics.max_rhat > 1.01
        or diagnostics.min_bulk_ess < 400
        or diagnostics.min_tail_ess < 400
        or diagnostics.divergences
    ):
        raise SamplingError(str(diagnostics))
    return diagnostics


def sample_hqrc(
    data: HQRCData,
    calibration: ApprovedARCalibration,
    *,
    variant: Variant = "H3",
    pooling: Pooling = "partial",
    options: HQRCModelOptions | None = None,
    draws: int = 1_000,
    tune: int = 1_000,
    chains: int = 4,
    seed: int = 11,
    backend: Literal["pymc", "nutpie"] = "pymc",
    paper_profile: bool = False,
):
    """Sample a model with PyMC by default, loading nutpie only when selected."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (draws, tune, chains)
    ):
        raise ValueError("draws, tune, and chains must be positive integers")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if paper_profile and (chains != 4 or draws < 1_000 or tune < 1_000):
        raise ValueError("paper_profile requires exactly 4 chains and at least 1000 tune/draws")
    trusted_calibration = require_approved_calibration(calibration)
    model = build_hqrc_model(data, trusted_calibration, variant, pooling, options)
    started = time.perf_counter()
    target_accept = 0.99 if paper_profile else 0.9
    if backend == "pymc":
        import pymc as pm

        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=1,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
                target_accept=target_accept,
            )
    elif backend == "nutpie":
        try:
            import nutpie
        except ImportError as error:
            raise SamplingError("nutpie backend requested but nutpie is not installed") from error
        idata = nutpie.sample_pymc(
            model, draws=draws, tune=tune, chains=chains, seed=seed, target_accept=target_accept
        )
    else:
        raise ValueError("backend must be 'pymc' or 'nutpie'")
    if "event_log_likelihood" not in idata.posterior:
        raise SamplingError("HQRC posterior is missing occurrence-level log likelihood")
    idata.add_groups(
        {
            "log_likelihood": xr.Dataset(
                {"event": idata.posterior["event_log_likelihood"].rename("event")}
            )
        }
    )
    diagnostics = validate_inference_data(idata, paper_profile=paper_profile)
    idata.attrs.update(
        {
            "hqrc_backend": backend,
            "hqrc_elapsed_seconds": time.perf_counter() - started,
            "hqrc_pymc_version": importlib.metadata.version("pymc"),
            "hqrc_arviz_version": importlib.metadata.version("arviz"),
            "hqrc_diagnostics_json": json.dumps(asdict(diagnostics), sort_keys=True),
            "hqrc_sampler_json": json.dumps(
                {
                    "draws": draws,
                    "tune": tune,
                    "chains": chains,
                    "seed": seed,
                    "target_accept": target_accept,
                    "paper_profile": paper_profile,
                },
                sort_keys=True,
            ),
            "hqrc_model_json": json.dumps(
                {
                    "variant": variant,
                    "pooling": pooling,
                    "options": asdict(options or HQRCModelOptions()),
                },
                sort_keys=True,
            ),
            "hqrc_calibration_json": json.dumps(
                {
                    "artifact_path": str(trusted_calibration.artifact_path),
                    "artifact_digest": trusted_calibration.artifact_digest,
                    "residual_sha256": trusted_calibration.residual_sha256,
                    "config_sha256": trusted_calibration.config_sha256,
                    "event_sha256": trusted_calibration.event_sha256,
                    "a": trusted_calibration.a,
                    "b": trusted_calibration.b,
                },
                sort_keys=True,
            ),
        }
    )
    if backend == "nutpie":
        idata.attrs["hqrc_nutpie_version"] = importlib.metadata.version("nutpie")
    return idata
