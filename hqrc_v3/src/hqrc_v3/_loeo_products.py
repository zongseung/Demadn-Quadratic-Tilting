"""Deterministic held-out products for one fitted H3 LOEO posterior."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

from hqrc_v3._loeo_contract import derive_loeo_seed
from hqrc_v3._loeo_posterior import posterior_mapping
from hqrc_v3._loeo_types import LOEOFoldError, LOEOFoldInputs, LOEOFoldProducts
from hqrc_v3.bayes.predictive import (
    corrected_predictive_draws,
    draw_new_event_correction,
    posterior_values,
    select_posterior_indices,
    simulate_stationary_ar1,
)
from hqrc_v3.evaluation.metrics import point_metric_frame, probabilistic_metric_frame

_HOLIDAY_INDEX = {"seollal": 0, "chuseok": 1}


def generate_loeo_fold_products(
    inputs: LOEOFoldInputs,
    idata: object,
    *,
    predictive_seed: int,
    predictive_draws: int | None = None,
) -> LOEOFoldProducts:
    """Generate only the held-out event using exactly baseline + sigma * (q + e)."""

    if not isinstance(inputs, LOEOFoldInputs):
        raise TypeError("inputs must be LOEOFoldInputs")
    if inputs.causal is not False:
        raise LOEOFoldError("LOEO products must be causal=false")
    if isinstance(predictive_seed, bool) or not isinstance(predictive_seed, int):
        raise LOEOFoldError("LOEO predictive seed must be an integer")
    posterior = posterior_mapping(idata)
    sample_count = int(posterior.sizes.get("chain", 0)) * int(posterior.sizes.get("draw", 0))
    draws = sample_count if predictive_draws is None else predictive_draws
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise LOEOFoldError("LOEO predictive_draws must be a positive integer")
    specifications = {
        "mu": 2,
        "between_cholesky": 3,
        "gamma": 2,
        "u_phi": 0,
        "phi": 0,
        "sigma_r": 0,
    }
    try:
        indices = select_posterior_indices(
            posterior,
            specifications=specifications,
            draws=draws,
            seed=predictive_seed,
        )
        gamma = posterior_values(posterior, "gamma", trailing=2, indices=indices)
        phi = posterior_values(posterior, "phi", trailing=0, indices=indices)
        sigma_r = posterior_values(posterior, "sigma_r", trailing=0, indices=indices)
        event = inputs.held_out.frame
        holiday_name = str(event["holiday_type"].item(0))
        holiday = _HOLIDAY_INDEX[holiday_name]
        beta = draw_new_event_correction(
            posterior,
            holiday_type=holiday,
            pooling="partial",
            draws=draws,
            seed=derive_loeo_seed(predictive_seed, "new-event-coefficients"),
            restriction=inputs.restriction,
            sample_indices=indices,
        )
        tau = event["tau_days"].to_numpy().astype(float)
        hour = event["hour"].to_numpy().astype(int)
        q = (
            beta[:, 0, None]
            + beta[:, 1, None] * tau[None, :]
            + beta[:, 2, None] * tau[None, :] ** 2
            + gamma[:, holiday, hour]
        )
        e = simulate_stationary_ar1(
            phi=phi,
            sigma=sigma_r,
            horizon=event.height,
            seed=derive_loeo_seed(predictive_seed, "held-out-ar1-reset"),
        )
        baseline = event["predicted_mw"].to_numpy().astype(float)
        predictive = corrected_predictive_draws(baseline, inputs.sigma_eval, q, e)
    except (KeyError, TypeError, ValueError) as error:
        raise LOEOFoldError("LOEO H3 held-out predictive generation failed") from error
    q_mean = q.mean(axis=0)
    point = baseline + inputs.sigma_eval * q_mean
    observed = event["observed_mw"].to_numpy().astype(float)
    quantiles = np.quantile(predictive, [0.025, 0.05, 0.1, 0.5, 0.9, 0.95, 0.975], axis=0)
    hourly = event.select(
        "target_timestamp",
        "occurrence_id",
        "holiday_type",
        "split_id",
        "tau_days",
        "hour",
        "restriction",
    ).with_columns(
        pl.Series("observed_mw", observed),
        pl.Series("baseline_mw", baseline),
        pl.Series("q_mean_standardized", q_mean),
        pl.Series("corrected_point_mw", point),
        pl.Series("predictive_mean_mw", predictive.mean(axis=0)),
        pl.Series("predictive_q025_mw", quantiles[0]),
        pl.Series("predictive_q05_mw", quantiles[1]),
        pl.Series("predictive_q10_mw", quantiles[2]),
        pl.Series("predictive_q50_mw", quantiles[3]),
        pl.Series("predictive_q90_mw", quantiles[4]),
        pl.Series("predictive_q95_mw", quantiles[5]),
        pl.Series("predictive_q975_mw", quantiles[6]),
        pl.lit(inputs.sigma_eval, dtype=pl.Float64).alias("sigma_eval_mw"),
        pl.lit(False, dtype=pl.Boolean).alias("causal"),
    )
    point_metrics = point_metric_frame(inputs.held_out_occurrence_id, observed, point)
    probabilistic_metrics = probabilistic_metric_frame(
        inputs.held_out_occurrence_id, observed, predictive
    )
    metrics = point_metrics.join(probabilistic_metrics, on="event_id", how="inner").with_columns(
        pl.col("event_id").alias("occurrence_id"),
        pl.lit(inputs.sigma_eval, dtype=pl.Float64).alias("sigma_eval_mw"),
        pl.lit(False, dtype=pl.Boolean).alias("causal"),
    )
    phi_all = np.asarray(posterior["phi"], dtype=float).reshape(-1)
    if phi_all.size == 0 or not np.isfinite(phi_all).all() or (np.abs(phi_all) >= 1).any():
        raise LOEOFoldError("LOEO posterior phi draws are invalid")
    phi_interval = np.quantile(phi_all, [0.025, 0.975])
    posterior_summary: dict[str, Any] = {
        "schema_version": 1,
        "held_out_occurrence_id": inputs.held_out_occurrence_id,
        "causal": False,
        "phi": {
            "parameterization": "phi=2*u_phi-1; u_phi~Beta(a_fold,b_fold)",
            "sample_count": int(phi_all.size),
            "mean": float(phi_all.mean()),
            "sd": float(phi_all.std(ddof=1)) if phi_all.size > 1 else 0.0,
            "interval_2_5": float(phi_interval[0]),
            "interval_97_5": float(phi_interval[1]),
            "prior_a": inputs.approved.a,
            "prior_b": inputs.approved.b,
        },
    }
    return LOEOFoldProducts(hourly, metrics, posterior_summary)


__all__ = ["generate_loeo_fold_products", "posterior_mapping"]
