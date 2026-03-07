"""평가 지표 함수."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import polars as pl


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


def rmse(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y - yhat) ** 2)))


def picp(y: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Prediction Interval Coverage Probability (%)."""
    return float(np.mean((y >= lower) & (y <= upper)) * 100.0)


def aiw(lower: np.ndarray, upper: np.ndarray) -> float:
    """Average Interval Width."""
    return float(np.mean(upper - lower))


def trough_bias_per_event(
    df: pl.DataFrame,
    y_col: str,
    pred_col: str,
    event_id_of_date: dict,
    datetime_col: str = "datetime",
) -> float:
    """이벤트별 trough bias 평균."""
    dt_list = df[datetime_col].to_list()
    y = df[y_col].to_numpy()
    pred = df[pred_col].to_numpy()

    events: Dict[str, List[int]] = {}
    for i, d in enumerate(dt_list):
        eid = event_id_of_date.get(d)
        if eid is not None:
            events.setdefault(eid, []).append(i)

    biases = []
    for eid, idxs in events.items():
        y_sub = y[idxs]
        pred_sub = pred[idxs]
        min_idx = np.argmin(y_sub)
        biases.append(float(pred_sub[min_idx] - y_sub[min_idx]))

    return float(np.mean(biases)) if biases else float("nan")


def evaluate_split(
    df: pl.DataFrame,
    y_col: str,
    event_id_of_date: dict,
    datetime_col: str = "datetime",
) -> dict:
    """tilted_pred, lower_t, upper_t 컬럼이 있는 DataFrame 평가."""
    dt_list = df[datetime_col].to_list()
    y = df[y_col].cast(pl.Float64).to_numpy()
    pred = df["tilted_pred"].to_numpy()
    lo = df["lower_t"].to_numpy()
    hi = df["upper_t"].to_numpy()

    mask_h = np.array([d in event_id_of_date for d in dt_list])

    result = {
        "MAE_all": mae(y, pred),
        "RMSE_all": rmse(y, pred),
    }
    if mask_h.any():
        result["MAE_holiday"] = mae(y[mask_h], pred[mask_h])
        result["RMSE_holiday"] = rmse(y[mask_h], pred[mask_h])
        result["PICP_95"] = picp(y[mask_h], lo[mask_h], hi[mask_h])
        result["AIW"] = aiw(lo[mask_h], hi[mask_h])
    return result
