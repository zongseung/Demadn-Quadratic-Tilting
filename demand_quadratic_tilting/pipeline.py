"""End-to-End HQT 파이프라인 (Polars)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import polars as pl

from .metrics import evaluate_split
from .model import HQTResult, compute_sigma_and_residuals, fit_hqt_pymc_lkj
from .tilt import apply_tilt, tilt_from_posterior
from .windows import build_holiday_windows_and_tau


def run_hqt_pipeline(
    train_df: pl.DataFrame,
    val_df: pl.DataFrame,
    test_df: pl.DataFrame,
    y_col: str,
    pred_col: str,
    holiday_name_col: str = "holiday_name",
    datetime_col: str = "datetime",
    tilt_mode: str = "hybrid",
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.95,
    sampler: str = "nuts",
    random_seed: int = 2025,
    center_hour: int = 0,
    tau_unit: str = "1h",
    tau_scale_hours: float = 24.0,
    pre_pad_days: int = 1,
    post_pad_days: int = 1,
) -> Dict[str, Any]:
    """학습/검증/테스트 분할에 대해 HQT 적합 및 틸트 적용.

    Parameters
    ----------
    train_df, val_df, test_df : pl.DataFrame
        datetime_col, y_col, pred_col, holiday_name_col 을 포함.
    y_col : str
        실제값 컬럼명.
    pred_col : str
        베이스라인 예측 컬럼명.
    holiday_name_col : str
        공휴일 이름 컬럼 (workalendar 또는 직접 지정).
    tilt_mode : str
        "hybrid" | "event" | "type"
    sampler : str
        "nuts" | "numpyro" | "advi"

    Returns
    -------
    dict with keys: hqt, sigma_resid, windows, train, val, test
    """
    all_df = pl.concat([
        train_df.select([datetime_col, holiday_name_col]),
        val_df.select([datetime_col, holiday_name_col]),
        test_df.select([datetime_col, holiday_name_col]),
    ])

    windows, tau_map, eid_of_date, type_of_date, tau_unit_hours = (
        build_holiday_windows_and_tau(
            all_df,
            holiday_name_col=holiday_name_col,
            datetime_col=datetime_col,
            center_hour=center_hour,
            tau_unit=tau_unit,
            pre_pad_days=pre_pad_days,
            post_pad_days=post_pad_days,
        )
    )

    sigma_resid, residuals = compute_sigma_and_residuals(
        train_df, y_col, pred_col, holiday_name_col,
    )

    hqt = fit_hqt_pymc_lkj(
        df_train=train_df,
        sigma_resid=sigma_resid,
        residuals=residuals,
        tau_map=tau_map,
        event_id_of_date=eid_of_date,
        type_of_date=type_of_date,
        datetime_col=datetime_col,
        tau_scale_hours=tau_scale_hours,
        tau_unit_hours=tau_unit_hours,
        chains=chains,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        sampler=sampler,
        random_seed=random_seed,
    )

    def _process_split(df: pl.DataFrame) -> Dict[str, Any]:
        dt_list = df[datetime_col].to_list()
        tilt_df, half_width_const = tilt_from_posterior(
            hqt, sigma_resid, dt_list, tilt_mode=tilt_mode,
        )
        out = apply_tilt(df, tilt_df, pred_col, datetime_col)
        metrics = evaluate_split(out, y_col, eid_of_date, datetime_col)
        metrics["half_width_const"] = half_width_const
        return {"preds": out, "metrics": metrics}

    return {
        "hqt": hqt,
        "sigma_resid": sigma_resid,
        "windows": windows,
        "train": _process_split(train_df),
        "val": _process_split(val_df),
        "test": _process_split(test_df),
    }
