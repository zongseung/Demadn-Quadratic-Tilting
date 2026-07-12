"""Metrics and HQT-compatible hourly prediction output."""

from __future__ import annotations

from typing import Mapping

import numpy as np
import polars as pl

from .data import WindowedDataset


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if true.shape != pred.shape:
        raise ValueError("true and predicted arrays do not match")
    error = pred - true
    denominator = np.maximum(np.abs(true) + np.abs(pred), 1e-8)
    return {
        "mae_mw": float(np.mean(np.abs(error))),
        "rmse_mw": float(np.sqrt(np.mean(error**2))),
        "wape_pct": float(100 * np.sum(np.abs(error)) / np.sum(np.abs(true))),
        "smape_pct": float(100 * np.mean(2 * np.abs(error) / denominator)),
        "bias_mw": float(np.mean(error)),
    }


def metric_rows(
    model_name: str,
    split_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    if y_true.shape != y_pred.shape:
        raise ValueError("true and predicted multi-horizon arrays do not match")
    rows: list[dict[str, object]] = []
    overall = regression_metrics(y_true, y_pred)
    rows.append(
        {
            "model": model_name,
            "split": split_name,
            "scope": "overall",
            "horizon_hour": None,
            "n": int(y_true.size),
            **overall,
        }
    )
    for horizon in range(y_true.shape[1]):
        rows.append(
            {
                "model": model_name,
                "split": split_name,
                "scope": "horizon",
                "horizon_hour": horizon + 1,
                "n": int(y_true.shape[0]),
                **regression_metrics(y_true[:, horizon], y_pred[:, horizon]),
            }
        )
    return rows


def build_prediction_frame(
    dataset: WindowedDataset,
    predictions: Mapping[str, Mapping[str, np.ndarray]],
) -> pl.DataFrame:
    """Create one row per forecast hour, ready for ``run_hqt_pipeline``."""

    config = dataset.config
    split_frames: list[pl.DataFrame] = []
    for split_name in ("train", "validation", "test"):
        split = dataset.splits[split_name]
        columns: dict[str, object] = {
            "datetime": split.target_datetimes.reshape(-1),
            config.target_col: split.target.reshape(-1),
            "split": [split_name] * split.target.size,
            "horizon_hour": np.tile(
                np.arange(1, config.horizon_hours + 1, dtype=np.int16),
                len(split),
            ),
        }
        for model_name, model_predictions in predictions.items():
            values = model_predictions[split_name]
            if values.shape != split.target.shape:
                raise ValueError(
                    f"{model_name}/{split_name} prediction shape {values.shape} "
                    f"does not match target shape {split.target.shape}"
                )
            columns[model_name] = values.reshape(-1)
        split_frames.append(pl.DataFrame(columns))

    output = pl.concat(split_frames, how="vertical")
    metadata_cols = [
        col for col in config.metadata_cols if col in dataset.frame.columns
    ]
    metadata = dataset.frame.select(
        [pl.col(config.datetime_col).alias("datetime"), *metadata_cols]
    )
    output = output.join(metadata, on="datetime", how="left", validate="1:1")
    preferred = [
        "datetime",
        config.target_col,
        "split",
        "horizon_hour",
        *predictions.keys(),
        *metadata_cols,
    ]
    return output.select(preferred).sort("datetime")
