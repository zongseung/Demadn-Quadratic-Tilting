"""Original-paper hierarchical quadratic tilting without AR or extra terms."""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
from dataclasses import asdict

import arviz as az
import numpy as np
import xarray as xr

from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.bayes.samplers import (
    PYMC_INITIALIZATION,
    PYMC_INITIALIZATION_CHOICES,
    SamplingError,
    validate_inference_data,
)

LEGACY_HQT_MODEL_SPEC = {
    "between_event_covariance": "holiday-specific-lkj-cholesky",
    "between_event_scale_prior": "HalfNormal(1)",
    "hour_profile": False,
    "innovation": "iid-normal",
    "lkj_eta": 2.0,
    "mu_prior": "Normal(0,10)",
    "pandemic_covariate": False,
    "polynomial_order": 2,
    "pooling": "partial-by-holiday-type",
    "sigma_r_prior": "HalfNormal(1)",
    "version": 1,
}


def build_legacy_hqt_model(data: HQRCData):
    """Build the exact H2 model declared in the original HQT manuscript.

    The observation errors are independent Gaussian draws.  In particular,
    this graph deliberately contains no AR coefficient, hour-of-day profile,
    or pandemic-period regression term.
    """

    if not isinstance(data, HQRCData):
        raise TypeError("data must be HQRCData")

    import pymc as pm
    import pytensor.tensor as pt

    event_count = len(data.occurrence_ids)
    occurrence_type = data.occurrence_holiday_type
    design = np.column_stack((np.ones(data.observations.size), data.tau_days, data.tau_days**2))
    with pm.Model() as model:
        mu = pm.Normal("mu", mu=0.0, sigma=10.0, shape=(2, 3))
        offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=(event_count, 3))

        cholesky = []
        scales = []
        for holiday_type in range(2):
            chol, _, stds = pm.LKJCholeskyCov(
                f"between_cov_{holiday_type}",
                n=3,
                eta=2.0,
                sd_dist=pm.HalfNormal.dist(sigma=1.0),
                compute_corr=True,
            )
            cholesky.append(chol)
            scales.append(stds)
        stacked_cholesky = pm.Deterministic("between_cholesky", pt.stack(cholesky))
        pm.Deterministic("between_scale", pt.stack(scales))
        transformed = pt.stack(
            [
                pt.dot(stacked_cholesky[int(occurrence_type[index])], offset[index])
                for index in range(event_count)
            ]
        )
        beta = pm.Deterministic("beta", mu[occurrence_type] + transformed)
        mean = pt.sum(beta[data.occurrence_index] * design, axis=1)
        sigma_r = pm.HalfNormal("sigma_r", sigma=1.0)
        pm.Normal("z_like", mu=mean, sigma=sigma_r, observed=data.observations)

        event_terms = [
            pm.logp(
                pm.Normal.dist(mu=mean[segment], sigma=sigma_r),
                data.observations[segment],
            ).sum()
            for segment in data.segments
        ]
        pm.Deterministic("event_log_likelihood", pt.stack(event_terms))
    return model


def sample_legacy_hqt(
    data: HQRCData,
    *,
    draws: int = 1_000,
    tune: int = 1_000,
    chains: int = 4,
    cores: int = 1,
    seed: int = 11,
    init: str = PYMC_INITIALIZATION,
    target_accept: float = 0.99,
    paper_profile: bool = False,
) -> az.InferenceData:
    """Sample the original HQT model and attach reproducible model metadata."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (draws, tune, chains, cores)
    ):
        raise ValueError("draws, tune, chains, and cores must be positive integers")
    if cores > chains:
        raise ValueError("cores must not exceed chains")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if init not in PYMC_INITIALIZATION_CHOICES:
        raise ValueError("init must be an approved PyMC initialization")
    if (
        isinstance(target_accept, bool)
        or not isinstance(target_accept, (int, float))
        or not math.isfinite(float(target_accept))
        or not 0.0 < float(target_accept) < 1.0
    ):
        raise ValueError("target_accept must be finite and strictly between zero and one")
    if paper_profile and (
        chains < 4 or draws < 1_000 or tune < 1_000 or float(target_accept) != 0.99
    ):
        raise ValueError(
            "paper_profile requires at least 4 chains, at least 1000 tune/draws, "
            "and target_accept=0.99"
        )

    import pymc as pm

    model = build_legacy_hqt_model(data)
    started = time.perf_counter()
    with model:
        idata = pm.sample(
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
            target_accept=float(target_accept),
            init=init,
        )
    if "event_log_likelihood" not in idata.posterior:
        raise SamplingError("legacy HQT posterior is missing event-level log likelihood")
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
            "legacy_hqt_ar": "none",
            "legacy_hqt_arviz_version": importlib.metadata.version("arviz"),
            "legacy_hqt_diagnostics_json": json.dumps(asdict(diagnostics), sort_keys=True),
            "legacy_hqt_elapsed_seconds": time.perf_counter() - started,
            "legacy_hqt_model_json": json.dumps(LEGACY_HQT_MODEL_SPEC, sort_keys=True),
            "legacy_hqt_pymc_version": importlib.metadata.version("pymc"),
            "legacy_hqt_sampler_json": json.dumps(
                {
                    "chains": chains,
                    "cores": cores,
                    "draws": draws,
                    "init": init,
                    "paper_profile": paper_profile,
                    "seed": seed,
                    "target_accept": float(target_accept),
                    "tune": tune,
                },
                sort_keys=True,
            ),
        }
    )
    return idata


def new_event_correction_draws(
    idata: az.InferenceData,
    *,
    holiday_type_index: int,
    tau_days: np.ndarray,
    seed: int,
) -> np.ndarray:
    """Generate the manuscript's new-event quadratic correction draws."""

    if holiday_type_index not in (0, 1):
        raise ValueError("holiday_type_index must be zero or one")
    tau = np.asarray(tau_days, dtype=float)
    if tau.ndim != 1 or tau.size == 0 or not np.isfinite(tau).all():
        raise ValueError("tau_days must be a finite non-empty vector")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if not isinstance(idata, az.InferenceData) or not hasattr(idata, "posterior"):
        raise TypeError("idata must contain posterior draws")
    required = {"mu", "between_cholesky"}
    if not required.issubset(idata.posterior.data_vars):
        raise ValueError("legacy HQT posterior lacks new-event parameters")

    mu = np.asarray(idata.posterior["mu"]).reshape(-1, 2, 3)[:, holiday_type_index]
    chol = np.asarray(idata.posterior["between_cholesky"]).reshape(-1, 2, 3, 3)[
        :, holiday_type_index
    ]
    if mu.shape[0] == 0 or chol.shape[0] != mu.shape[0]:
        raise ValueError("legacy HQT posterior draw dimensions differ")
    rng = np.random.default_rng(seed)
    epsilon = rng.normal(size=(mu.shape[0], 3))
    beta = mu + np.einsum("sij,sj->si", chol, epsilon)
    design = np.column_stack((np.ones(tau.size), tau, tau**2))
    correction = beta @ design.T
    if not np.isfinite(correction).all():
        raise ValueError("legacy HQT correction draws are non-finite")
    return correction


__all__ = [
    "LEGACY_HQT_MODEL_SPEC",
    "build_legacy_hqt_model",
    "new_event_correction_draws",
    "sample_legacy_hqt",
]
