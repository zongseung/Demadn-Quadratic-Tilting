"""HQT 모델 정의 및 PyMC LKJ 적합."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl
import pymc as pm
import pytensor.tensor as pt


@dataclass
class HQTResult:
    """Hierarchical Quadratic Tilting 사후 결과."""

    draws_beta: np.ndarray       # [S, I, 3]
    draws_mu: np.ndarray         # [S, H, 3]
    draws_L: List[np.ndarray]    # [H] 각 (S, 3, 3)
    draws_sigma_r: np.ndarray    # [S]
    event_ids: List[str]
    event_type: List[str]
    type_names: List[str]
    tau_map: Dict[Tuple[str, datetime], int]
    event_id_of_date: Dict[datetime, str]
    type_of_date: Dict[datetime, str]
    tau_unit_hours: float
    tau_scale_hours: float


def compute_sigma_and_residuals(
    df: pl.DataFrame,
    y_col: str,
    pred_col: str,
    holiday_name_col: str = "holiday_name",
) -> Tuple[float, np.ndarray, List[datetime]]:
    """비명절 잔차에서 σ 추정, 전체 잔차 반환.

    Returns
    -------
    sigma : float
    residuals : np.ndarray
    dates : list[datetime]  (df의 datetime 순서)
    """
    y = df[y_col].cast(pl.Float64).to_numpy()
    yhat = df[pred_col].cast(pl.Float64).to_numpy()
    resid = y - yhat

    mask_non = (df[holiday_name_col].cast(pl.Utf8) == "non-event").to_numpy()
    sigma = float(np.std(resid[mask_non], ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("비명절(non-event) 표본에서 σ를 계산할 수 없습니다.")
    return sigma, resid


def fit_hqt_pymc_lkj(
    df_train: pl.DataFrame,
    sigma_resid: float,
    residuals: np.ndarray,
    tau_map: Dict[Tuple[str, datetime], int],
    event_id_of_date: Dict[datetime, str],
    type_of_date: Dict[datetime, str],
    datetime_col: str = "datetime",
    tau_scale_hours: float = 24.0,
    tau_unit_hours: float = 1.0,
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.95,
    random_seed: int = 2025,
    sampler: str = "nuts",
) -> HQTResult:
    """PyMC LKJ 계층 이차 틸트 적합 (non-centered parametrization)."""
    dt_list = df_train[datetime_col].to_list()

    dates = [d for d in dt_list if d in event_id_of_date]
    if not dates:
        raise ValueError("학습 세트에 추석/설날 윈도우 날짜가 없습니다.")

    # 날짜 -> residual 인덱스 매핑
    dt_to_idx = {d: i for i, d in enumerate(dt_list)}

    ev_ids_sorted = sorted({event_id_of_date[d] for d in dates}, key=str)
    I = len(ev_ids_sorted)
    eid_to_int = {e: i for i, e in enumerate(ev_ids_sorted)}

    types_sorted = ["Chuseok", "Seollal"]
    H = len(types_sorted)
    type_to_int = {t: i for i, t in enumerate(types_sorted)}
    h_of_event = np.array([type_to_int[e.split("_")[1]] for e in ev_ids_sorted], dtype=int)

    z_vec = np.array([residuals[dt_to_idx[d]] / sigma_resid for d in dates], dtype=float)
    tau_vec = np.array([float(tau_map[(event_id_of_date[d], d)]) for d in dates], dtype=float)
    tau_scaled = (tau_vec * float(tau_unit_hours)) / float(tau_scale_hours)
    e_idx_orig = np.array([eid_to_int[event_id_of_date[d]] for d in dates], dtype=int)

    idx_by_type = [np.where(h_of_event == h)[0] for h in range(H)]
    concat_order = np.concatenate([ix for ix in idx_by_type if len(ix) > 0])
    pos_of_event = np.empty(I, dtype=int)
    for pos, orig_idx in enumerate(concat_order):
        pos_of_event[orig_idx] = pos
    e_idx = pos_of_event[e_idx_orig]

    with pm.Model():
        mu = pm.Normal("mu", mu=0.0, sigma=10.0, shape=(H, 3))

        L_list = []
        for h in range(H):
            sd = pm.HalfNormal.dist(1.0, shape=3)
            packed = pm.LKJCholeskyCov(
                f"chol_packed_{h}", n=3, eta=2.0, sd_dist=sd,
                compute_corr=False, store_in_trace=False,
            )
            L_h_raw = pm.expand_packed_triangular(3, packed, lower=True)
            L_h = pm.Deterministic(f"L_chol_{h}", L_h_raw)
            L_list.append(L_h)

        beta_blocks = []
        for h in range(H):
            idx_h = idx_by_type[h]
            n_h = len(idx_h)
            if n_h > 0:
                eps = pm.Normal(f"eps_type{h}", mu=0.0, sigma=1.0, shape=(n_h, 3))
                beta_h = pm.Deterministic(f"beta_type{h}", mu[h] + eps @ L_list[h].T)
                beta_blocks.append(beta_h)

        beta_concat = (
            beta_blocks[0] if len(beta_blocks) == 1
            else pt.concatenate(beta_blocks, axis=0)
        )

        sigma_r = pm.HalfNormal("sigma_r", 1.0)

        mu_z = (
            beta_concat[e_idx, 0]
            + beta_concat[e_idx, 1] * tau_scaled
            + beta_concat[e_idx, 2] * pt.sqr(tau_scaled)
        )
        pm.Normal("z_like", mu=mu_z, sigma=sigma_r, observed=z_vec)

        if sampler == "advi":
            approx = pm.fit(50_000, random_seed=random_seed)
            idata = approx.sample(draws=draws, random_seed=random_seed)
        elif sampler == "numpyro":
            import pymc.sampling.jax as pmjax
            idata = pmjax.sample_numpyro_nuts(
                chains=chains, draws=draws, tune=tune,
                target_accept=target_accept, random_seed=random_seed,
                progressbar=False,
                chain_method="vectorized",
                postprocessing_backend="cpu",
                idata_kwargs={"log_likelihood": False},
            )
        else:
            idata = pm.sample(
                chains=chains, draws=draws, tune=tune,
                target_accept=target_accept, random_seed=random_seed,
                progressbar=False,
            )

    # posterior 정리
    arrays = []
    for h in range(H):
        key = f"beta_type{h}"
        if key in idata.posterior:
            arr = idata.posterior[key].values
            arr = np.moveaxis(arr, 0, 1).reshape(-1, arr.shape[2], 3)
            arrays.append(arr)
    draws_beta_concat = arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)
    draws_beta = draws_beta_concat[:, pos_of_event, :]

    mu_draws = idata.posterior["mu"].values
    mu_draws = np.moveaxis(mu_draws, 0, 1).reshape(-1, H, 3)

    sigma_r_draws = idata.posterior["sigma_r"].values
    sigma_r_draws = np.moveaxis(sigma_r_draws, 0, 1).reshape(-1)

    draws_L: List[np.ndarray] = []
    for h in range(H):
        key = f"L_chol_{h}"
        if key in idata.posterior:
            arr = idata.posterior[key].values
            arr = np.moveaxis(arr, 0, 1).reshape(-1, 3, 3)
        else:
            S = mu_draws.shape[0]
            arr = np.tile(np.eye(3), (S, 1, 1))
        draws_L.append(arr)

    return HQTResult(
        draws_beta=draws_beta,
        draws_mu=mu_draws,
        draws_L=draws_L,
        draws_sigma_r=sigma_r_draws,
        event_ids=ev_ids_sorted,
        event_type=[types_sorted[h] for h in h_of_event],
        type_names=types_sorted,
        tau_map={k: int(v) for k, v in tau_map.items()},
        event_id_of_date=dict(event_id_of_date),
        type_of_date=dict(type_of_date),
        tau_unit_hours=float(tau_unit_hours),
        tau_scale_hours=float(tau_scale_hours),
    )
