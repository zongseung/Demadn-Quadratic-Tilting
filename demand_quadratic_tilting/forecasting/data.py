"""Hourly data validation, causal window construction, and scaling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import polars as pl
from sklearn.preprocessing import StandardScaler

from .config import ForecastConfig


@dataclass(frozen=True)
class WindowSplit:
    """One chronological split of 168-hour inputs and 24-hour targets."""

    history: np.ndarray
    future_known: np.ndarray
    target: np.ndarray
    target_datetimes: np.ndarray
    target_row_indices: np.ndarray

    def __len__(self) -> int:
        return int(self.history.shape[0])


@dataclass(frozen=True)
class WindowedDataset:
    """Raw hourly frame and window arrays assigned by target timestamps."""

    frame: pl.DataFrame
    splits: Mapping[str, WindowSplit]
    config: ForecastConfig


@dataclass(frozen=True)
class ForecastScalers:
    """Scalers fitted on training-period rows only."""

    target: StandardScaler
    weather: StandardScaler


def load_hourly_data(
    path: str | Path,
    config: ForecastConfig,
) -> pl.DataFrame:
    """Load and strictly validate the hourly forecasting data."""

    config.validate()
    frame = pl.read_csv(path, try_parse_dates=True)
    required = set(config.past_feature_cols) | {
        config.datetime_col,
        *config.metadata_cols,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"missing required columns: {missing}")

    if frame.schema[config.datetime_col] not in (pl.Datetime, pl.Date):
        frame = frame.with_columns(
            pl.col(config.datetime_col).str.to_datetime(strict=True)
        )
    frame = frame.sort(config.datetime_col)
    if frame[config.datetime_col].n_unique() != frame.height:
        raise ValueError("datetime column contains duplicates")

    null_counts = frame.select(pl.col(list(config.past_feature_cols)).null_count()).row(
        0, named=True
    )
    bad_nulls = {key: value for key, value in null_counts.items() if value}
    if bad_nulls:
        raise ValueError(f"forecast feature columns contain nulls: {bad_nulls}")

    if frame.height > 1:
        gaps = (
            frame.select(
                pl.col(config.datetime_col)
                .diff()
                .dt.total_hours()
                .drop_nulls()
                .alias("gap")
            )
            .get_column("gap")
            .unique()
            .to_list()
        )
        if gaps != [1]:
            raise ValueError(f"data must be strictly hourly; observed gaps: {gaps}")

    values = frame.select(config.past_feature_cols).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("forecast feature columns contain non-finite values")

    season_cols = [
        col
        for col in ("spring", "summer", "autoum", "winter")
        if col in config.known_covariate_cols
    ]
    if len(season_cols) == 4:
        invalid = frame.filter(pl.sum_horizontal(season_cols) != 1).height
        if invalid:
            raise ValueError(
                f"season indicators must be one-hot; found {invalid} invalid rows"
            )

    if "holiday_name" in frame.columns:
        frame = frame.with_columns(pl.col("holiday_name").fill_null("non-event"))
    return frame


def _first_aligned_target_index(
    datetimes: np.ndarray,
    history_hours: int,
    origin_hour: int,
) -> int:
    for index in range(history_hours, len(datetimes)):
        value = datetimes[index].astype("datetime64[h]").astype(datetime)
        if value.hour == origin_hour:
            return index
    raise ValueError("no forecast origin matching forecast_origin_hour was found")


def _empty_split(config: ForecastConfig) -> WindowSplit:
    n_features = len(config.past_feature_cols)
    n_known = len(config.known_covariate_cols)
    return WindowSplit(
        history=np.empty((0, config.history_hours, n_features), dtype=np.float32),
        future_known=np.empty((0, config.horizon_hours, n_known), dtype=np.float32),
        target=np.empty((0, config.horizon_hours), dtype=np.float32),
        target_datetimes=np.empty((0, config.horizon_hours), dtype="datetime64[us]"),
        target_row_indices=np.empty((0, config.horizon_hours), dtype=np.int64),
    )


def build_windowed_dataset(
    frame: pl.DataFrame,
    config: ForecastConfig,
) -> WindowedDataset:
    """Build daily windows and assign them using the entire target interval.

    A validation/test window may use history from the immediately preceding
    split, which mirrors real forecasting.  No target is allowed to straddle a
    split boundary.
    """

    config.validate()
    datetimes = frame[config.datetime_col].to_numpy().astype("datetime64[us]")
    past = frame.select(config.past_feature_cols).to_numpy().astype(np.float32)
    future = frame.select(config.known_covariate_cols).to_numpy().astype(np.float32)
    target = frame[config.target_col].to_numpy().astype(np.float32)

    first = _first_aligned_target_index(
        datetimes,
        config.history_hours,
        config.forecast_origin_hour,
    )
    final_start = frame.height - config.horizon_hours
    train_end = np.datetime64(config.train_end)
    validation_end = np.datetime64(config.validation_end)

    buckets: dict[str, dict[str, list[np.ndarray]]] = {
        split: {
            "history": [],
            "future_known": [],
            "target": [],
            "target_datetimes": [],
            "target_row_indices": [],
        }
        for split in ("train", "validation", "test")
    }

    for target_start in range(first, final_start + 1, config.stride_hours):
        target_stop = target_start + config.horizon_hours
        start_time = datetimes[target_start]
        end_time = datetimes[target_stop - 1]
        hour = start_time.astype("datetime64[h]").astype(datetime).hour
        if hour != config.forecast_origin_hour:
            raise ValueError(
                "stride_hours does not preserve the configured forecast origin"
            )

        if end_time <= train_end:
            split = "train"
        elif start_time > train_end and end_time <= validation_end:
            split = "validation"
        elif start_time > validation_end:
            split = "test"
        else:
            continue

        history_start = target_start - config.history_hours
        row_indices = np.arange(target_start, target_stop, dtype=np.int64)
        bucket = buckets[split]
        bucket["history"].append(past[history_start:target_start])
        bucket["future_known"].append(future[target_start:target_stop])
        bucket["target"].append(target[target_start:target_stop])
        bucket["target_datetimes"].append(datetimes[target_start:target_stop])
        bucket["target_row_indices"].append(row_indices)

    splits: dict[str, WindowSplit] = {}
    for split, bucket in buckets.items():
        if not bucket["history"]:
            splits[split] = _empty_split(config)
            continue
        splits[split] = WindowSplit(
            history=np.stack(bucket["history"]).astype(np.float32),
            future_known=np.stack(bucket["future_known"]).astype(np.float32),
            target=np.stack(bucket["target"]).astype(np.float32),
            target_datetimes=np.stack(bucket["target_datetimes"]),
            target_row_indices=np.stack(bucket["target_row_indices"]),
        )

    empty = [name for name, split in splits.items() if not len(split)]
    if empty:
        raise ValueError(f"empty chronological splits: {empty}")
    return WindowedDataset(frame=frame, splits=splits, config=config)


def fit_scalers(
    frame: pl.DataFrame,
    config: ForecastConfig,
) -> ForecastScalers:
    """Fit target and weather scalers without validation/test information."""

    train_rows = frame.filter(
        pl.col(config.datetime_col) <= pl.lit(config.train_end).str.to_datetime()
    )
    if train_rows.is_empty():
        raise ValueError("no rows fall in the configured training interval")
    target_scaler = StandardScaler().fit(
        train_rows.select(config.target_col).to_numpy()
    )
    weather_scaler = StandardScaler().fit(
        train_rows.select(config.weather_cols).to_numpy()
    )
    return ForecastScalers(target=target_scaler, weather=weather_scaler)


def scale_split(
    split: WindowSplit,
    scalers: ForecastScalers,
    config: ForecastConfig,
) -> WindowSplit:
    """Scale continuous inputs and targets; preserve binary covariates."""

    history = split.history.copy()
    shape = history.shape
    history[:, :, 0] = scalers.target.transform(
        history[:, :, 0].reshape(-1, 1)
    ).reshape(shape[0], shape[1])

    weather_stop = 1 + len(config.weather_cols)
    history[:, :, 1:weather_stop] = scalers.weather.transform(
        history[:, :, 1:weather_stop].reshape(-1, len(config.weather_cols))
    ).reshape(shape[0], shape[1], len(config.weather_cols))
    target = scalers.target.transform(split.target.reshape(-1, 1)).reshape(
        split.target.shape
    )
    return WindowSplit(
        history=history.astype(np.float32),
        future_known=split.future_known.astype(np.float32, copy=True),
        target=target.astype(np.float32),
        target_datetimes=split.target_datetimes,
        target_row_indices=split.target_row_indices,
    )


def flatten_model_inputs(split: WindowSplit) -> np.ndarray:
    """Flatten history and append the known 24-hour covariate path."""

    n_windows = len(split)
    return np.concatenate(
        (
            split.history.reshape(n_windows, -1),
            split.future_known.reshape(n_windows, -1),
        ),
        axis=1,
    ).astype(np.float32)
