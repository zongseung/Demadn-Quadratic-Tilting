"""Sampler selection and transparent diagnostics for HQRC Bayesian models."""

from __future__ import annotations

import importlib.metadata
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import arviz as az
import numpy as np

from hqrc_v3.bayes.model import HQRCData, HQRCModelOptions, Pooling, Variant, build_hqrc_model
from hqrc_v3.diagnostics.ar import ARCalibration


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
    diagnostics = SamplingDiagnostics(
        max_rhat=_summary_value(summary, "r_hat", np.nanmax, math.inf),
        min_bulk_ess=_summary_value(summary, "ess_bulk", np.nanmin, 0.0),
        min_tail_ess=_summary_value(summary, "ess_tail", np.nanmin, 0.0),
        divergences=int(np.asarray(idata.sample_stats["diverging"], dtype=int).sum())
        if hasattr(idata, "sample_stats") and "diverging" in idata.sample_stats
        else 0,
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
    calibration: ARCalibration,
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
    model = build_hqrc_model(data, calibration, variant, pooling, options)
    started = time.perf_counter()
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
                target_accept=0.9,
            )
    elif backend == "nutpie":
        try:
            import nutpie
        except ImportError as error:
            raise SamplingError("nutpie backend requested but nutpie is not installed") from error
        idata = nutpie.sample_pymc(model, draws=draws, tune=tune, chains=chains, seed=seed)
    else:
        raise ValueError("backend must be 'pymc' or 'nutpie'")
    diagnostics = validate_inference_data(idata, paper_profile=paper_profile)
    idata.attrs.update(
        {
            "hqrc_backend": backend,
            "hqrc_elapsed_seconds": time.perf_counter() - started,
            "hqrc_pymc_version": importlib.metadata.version("pymc"),
            "hqrc_arviz_version": importlib.metadata.version("arviz"),
            "hqrc_diagnostics": asdict(diagnostics),
        }
    )
    if backend == "nutpie":
        idata.attrs["hqrc_nutpie_version"] = importlib.metadata.version("nutpie")
    return idata
