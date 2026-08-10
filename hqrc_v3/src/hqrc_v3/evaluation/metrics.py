"""Event-unit point and probabilistic forecast metrics."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


class MetricContractError(ValueError):
    """Raised when observed values or predictive draws are not aligned and finite."""


def _aligned(observed: np.ndarray, point: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    actual = np.asarray(observed, dtype=float)
    forecast = np.asarray(point, dtype=float)
    if actual.ndim != 1 or forecast.ndim != 1 or actual.size == 0 or actual.shape != forecast.shape:
        raise MetricContractError("observed and point forecast must be aligned non-empty vectors")
    if not np.isfinite(actual).all() or not np.isfinite(forecast).all():
        raise MetricContractError("observed and point forecast must be finite")
    return actual, forecast


def _draws(observed: np.ndarray, draws: np.ndarray) -> np.ndarray:
    sample = np.asarray(draws, dtype=float)
    if sample.ndim != 2 or sample.shape[0] == 0 or sample.shape[1] != observed.size:
        raise MetricContractError("predictive draws must have shape (draw, timestamp)")
    if not np.isfinite(sample).all():
        raise MetricContractError("predictive draws must be finite")
    return sample


def _maybe(value: float, valid: bool = True) -> float | None:
    return float(value) if valid and math.isfinite(value) else None


def point_metric_frame(event_id: str, observed: np.ndarray, point: np.ndarray) -> pl.DataFrame:
    """Return one guarded point-metric row for one occurrence."""

    if not isinstance(event_id, str) or not event_id.strip():
        raise MetricContractError("event_id must be a nonblank string")
    actual, forecast = _aligned(observed, point)
    error = forecast - actual
    nonzero = np.abs(actual) > np.finfo(float).eps
    mape = np.mean(np.abs(error[nonzero] / actual[nonzero])) * 100 if nonzero.any() else math.nan
    smape_denom = np.abs(actual) + np.abs(forecast)
    smape_valid = smape_denom > np.finfo(float).eps
    smape = (
        np.mean(2.0 * np.abs(error[smape_valid]) / smape_denom[smape_valid]) * 100
        if smape_valid.any()
        else math.nan
    )
    centered = actual - actual.mean()
    total = float(np.dot(centered, centered))
    r2 = 1.0 - float(np.dot(error, error)) / total if total > np.finfo(float).eps else math.nan
    return pl.DataFrame(
        {
            "event_id": [event_id],
            "n_timestamps": [actual.size],
            "rmse": [float(np.sqrt(np.mean(error**2)))],
            "mae": [float(np.mean(np.abs(error)))],
            "mape": [_maybe(mape)],
            "smape": [_maybe(smape)],
            "r2": [_maybe(r2)],
        }
    )


def empirical_crps(observed: np.ndarray, draws: np.ndarray) -> np.ndarray:
    """Compute empirical CRPS independently at each timestamp."""

    actual, _ = _aligned(observed, observed)
    sample = _draws(actual, draws)
    first = np.abs(sample - actual[None, :]).mean(axis=0)
    second = np.abs(sample[:, None, :] - sample[None, :, :]).mean(axis=(0, 1))
    return first - 0.5 * second


def probabilistic_metric_frame(
    event_id: str, observed: np.ndarray, draws: np.ndarray
) -> pl.DataFrame:
    """Return empirical CRPS, .05/.95 pinball, and central interval coverage."""

    if not isinstance(event_id, str) or not event_id.strip():
        raise MetricContractError("event_id must be a nonblank string")
    actual, _ = _aligned(observed, observed)
    sample = _draws(actual, draws)
    quantiles = np.quantile(sample, [0.05, 0.25, 0.75, 0.95], axis=0)

    def pinball(level: float, estimate: np.ndarray) -> float:
        difference = actual - estimate
        return float(np.mean(np.maximum(level * difference, (level - 1.0) * difference)))

    return pl.DataFrame(
        {
            "event_id": [event_id],
            "crps": [float(empirical_crps(actual, sample).mean())],
            "pinball_05": [pinball(0.05, quantiles[0])],
            "pinball_95": [pinball(0.95, quantiles[3])],
            "coverage_50": [float(((actual >= quantiles[1]) & (actual <= quantiles[2])).mean())],
            "coverage_90": [float(((actual >= quantiles[0]) & (actual <= quantiles[3])).mean())],
        }
    )


def event_metric_frame(
    event_id: str,
    timestamps: pl.Series | list[object],
    observed: np.ndarray,
    point: np.ndarray,
    draws: np.ndarray,
) -> pl.DataFrame:
    """Return one per-timestamp event artifact while preserving exact timestamps."""

    actual, forecast = _aligned(observed, point)
    sample = _draws(actual, draws)
    timestamp_values = (
        timestamps.to_list() if isinstance(timestamps, pl.Series) else list(timestamps)
    )
    if len(timestamp_values) != actual.size:
        raise MetricContractError("timestamps must align with event observations")
    crps = empirical_crps(actual, sample)
    quantiles = np.quantile(sample, [0.05, 0.25, 0.75, 0.95], axis=0)
    error = forecast - actual
    return pl.DataFrame(
        {
            "event_id": [event_id] * actual.size,
            "target_timestamp": timestamp_values,
            "observed_mw": actual,
            "point_forecast_mw": forecast,
            "absolute_error": np.abs(error),
            "squared_error": error**2,
            "crps": crps,
            "pinball_05": np.maximum(
                0.05 * (actual - quantiles[0]), -0.95 * (actual - quantiles[0])
            ),
            "pinball_95": np.maximum(
                0.95 * (actual - quantiles[3]), -0.05 * (actual - quantiles[3])
            ),
            "covered_50": (actual >= quantiles[1]) & (actual <= quantiles[2]),
            "covered_90": (actual >= quantiles[0]) & (actual <= quantiles[3]),
        }
    )
