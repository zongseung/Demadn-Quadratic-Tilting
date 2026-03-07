"""틸트 계산 및 적용 (Polars)."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import numpy as np
import polars as pl
from scipy.stats import norm

from .model import HQTResult


def _sigmoid_gate(e_tilt: np.ndarray, sigma_resid: float,
                  threshold_k: float = 0.5, gate_scale_k: float = 0.3) -> np.ndarray:
    """적응형 게이팅: 작은 틸트는 억제, 큰 틸트는 유지.
    w = sigmoid((|e_tilt| - threshold) / scale)
    threshold = threshold_k * sigma_resid
    scale = gate_scale_k * sigma_resid
    """
    threshold = threshold_k * sigma_resid
    scale = gate_scale_k * sigma_resid
    w = 1.0 / (1.0 + np.exp(-(np.abs(e_tilt) - threshold) / scale))
    return w * e_tilt


def _sample_new_event_beta(
    rng: np.random.Generator,
    mu_draws_h: np.ndarray,
    L_draws_h: np.ndarray,
) -> np.ndarray:
    """β_new,s = μ_{h,s} + L_{h,s} @ ε_s, ε_s ~ N(0, I)."""
    S = mu_draws_h.shape[0]
    eps = rng.standard_normal((S, 3))
    return mu_draws_h + np.einsum("sij,sj->si", L_draws_h, eps)


def tilt_from_posterior(
    hqt: HQTResult,
    sigma_resid: float,
    dates: Iterable[datetime],
    tilt_mode: str = "hybrid",
    ci: float = 0.95,
    rng_seed: Optional[int] = 2025,
    gate: bool = False,
    threshold_k: float = 0.5,
    gate_scale_k: float = 0.3,
) -> Tuple[pl.DataFrame, float]:
    """사후 분포에서 틸트 계산.

    Returns
    -------
    tilt_df : pl.DataFrame with columns [datetime, e_tilt, half_width_t]
    half_width_const : float
    """
    dates = list(dates)
    B_draws = hqt.draws_beta
    MU_draws = hqt.draws_mu
    S = B_draws.shape[0]

    sigma_r_sq_mean = float((hqt.draws_sigma_r ** 2).mean())
    zcrit = norm.ppf(0.5 + ci / 2.0)
    half_width_const = float(zcrit * sigma_resid * np.sqrt(1.0 + sigma_r_sq_mean))

    eid_to_pos = {e: i for i, e in enumerate(hqt.event_ids)}
    type_to_pos = {t: i for i, t in enumerate(hqt.type_names)}
    rng = np.random.default_rng(rng_seed)

    zhat_vals: List[float] = []
    hw_vals: List[float] = []

    for d in dates:
        eid = hqt.event_id_of_date.get(d)
        if eid is None:
            zhat_vals.append(0.0)
            hw_vals.append(0.0)
            continue

        tau = hqt.tau_map.get((eid, d), 0)
        tau_s = (float(tau) * hqt.tau_unit_hours) / hqt.tau_scale_hours

        if tilt_mode == "type":
            h = type_to_pos[hqt.type_of_date[d]]
            MU_h = MU_draws[:, h, :]
            z_draws = MU_h[:, 0] + MU_h[:, 1] * tau_s + MU_h[:, 2] * tau_s**2
        elif tilt_mode == "event" and eid in eid_to_pos:
            i = eid_to_pos[eid]
            B_i = B_draws[:, i, :]
            z_draws = B_i[:, 0] + B_i[:, 1] * tau_s + B_i[:, 2] * tau_s**2
        elif eid in eid_to_pos:  # hybrid, known event
            i = eid_to_pos[eid]
            B_i = B_draws[:, i, :]
            z_draws = B_i[:, 0] + B_i[:, 1] * tau_s + B_i[:, 2] * tau_s**2
        else:  # new event
            h = type_to_pos[hqt.type_of_date[d]]
            betas_new = _sample_new_event_beta(rng, MU_draws[:, h, :], hqt.draws_L[h])
            z_draws = betas_new[:, 0] + betas_new[:, 1] * tau_s + betas_new[:, 2] * tau_s**2

        z_mean = float(np.mean(z_draws))
        std_z = float(np.std(z_draws, ddof=1))
        zhat_vals.append(z_mean)

        hw = float(zcrit * np.sqrt(sigma_resid**2 * (1.0 + sigma_r_sq_mean + std_z**2)))
        hw_vals.append(hw)

    e_tilt_vals = np.array([sigma_resid * z for z in zhat_vals])
    if gate:
        e_tilt_vals = _sigmoid_gate(e_tilt_vals, sigma_resid, threshold_k, gate_scale_k)

    tilt_df = pl.DataFrame({
        "datetime": dates,
        "z_hat": zhat_vals,
        "e_tilt": e_tilt_vals.tolist(),
        "half_width_t": hw_vals,
    })
    return tilt_df, half_width_const


def apply_tilt(
    df: pl.DataFrame,
    tilt_df: pl.DataFrame,
    baseline_col: str,
    datetime_col: str = "datetime",
) -> pl.DataFrame:
    """baseline 예측에 틸트를 적용하여 tilted_pred 컬럼 추가."""
    joined = df.join(
        tilt_df.select(["datetime", "e_tilt", "half_width_t"]),
        left_on=datetime_col,
        right_on="datetime",
        how="left",
    ).with_columns(
        pl.col("e_tilt").fill_null(0.0),
        pl.col("half_width_t").fill_null(0.0),
    ).with_columns(
        (pl.col(baseline_col) + pl.col("e_tilt")).alias("tilted_pred"),
    ).with_columns(
        (pl.col("tilted_pred") - pl.col("half_width_t")).alias("lower_t"),
        (pl.col("tilted_pred") + pl.col("half_width_t")).alias("upper_t"),
    )
    return joined
