"""Calendar features and audited daily samples, kept lazy until matrix construction."""

from __future__ import annotations

from datetime import date
from math import pi
from typing import Literal

import numpy as np
import polars as pl

from hqrc_v3.contracts import DataContractError, ForecastMatrix
from hqrc_v3.data import audit_hourly_data
from hqrc_v3.events import EventOccurrence

FeatureSet = Literal["B0", "B1"]
_HISTORY_COLUMNS = ("load_mw", "temperature_c", "relative_humidity")
_B0_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
    "annual_sin",
    "annual_cos",
    "weekly_sin",
    "weekly_cos",
    "heating_degree_hour",
    "cooling_degree_hour",
    "oracle_temperature_c",
    "oracle_relative_humidity",
)
_B1_ONLY_COLUMNS = (
    "is_holiday",
    "event_relative_day",
    "seollal_distance",
    "chuseok_distance",
    "is_bridge_or_substitute_holiday",
    "holiday_type",
)
_HOLIDAY_TOKENS = ("holiday", "event", "seollal", "chuseok", "bridge", "substitute")


def feature_columns(feature_set: FeatureSet) -> tuple[str, ...]:
    """Return only forecast-time covariates allowed for the requested baseline."""

    if feature_set == "B0":
        return _B0_COLUMNS
    if feature_set == "B1":
        return _B0_COLUMNS + _B1_ONLY_COLUMNS
    raise DataContractError(f"unknown feature set: {feature_set}")


def assert_no_holiday_leakage(columns: tuple[str, ...] | list[str] | set[str]) -> None:
    """Fail closed when a B0 feature name can encode event or holiday information."""

    leaking = sorted(
        column for column in columns if any(token in column.lower() for token in _HOLIDAY_TOKENS)
    )
    if leaking:
        raise DataContractError(f"B0 feature set contains holiday-derived columns: {leaking}")


def _date_membership(day: pl.Expr, dates: list[date]) -> pl.Expr:
    return day.is_in(dates) if dates else pl.lit(False)


def _nearest_signed_distance(day: pl.Expr, events: list[EventOccurrence]) -> pl.Expr:
    distances = [(day - pl.lit(event.central_date)).dt.total_days() for event in events]
    if not distances:
        return pl.lit(0, dtype=pl.Int64)
    nearest = distances[0]
    for candidate in distances[1:]:
        nearest = pl.when(candidate.abs() < nearest.abs()).then(candidate).otherwise(nearest)
    return nearest.cast(pl.Int64)


def attach_calendar_features(
    frame: pl.DataFrame | pl.LazyFrame, calendar: tuple[EventOccurrence, ...]
) -> pl.DataFrame | pl.LazyFrame:
    """Attach B0 and B1 candidate features while preserving a lazy input plan."""

    lazy = frame if isinstance(frame, pl.LazyFrame) else frame.lazy()
    day = pl.col("timestamp").dt.date()
    hour = pl.col("timestamp").dt.hour().cast(pl.Float64)
    weekday = pl.col("timestamp").dt.weekday().cast(pl.Float64)
    day_of_year = pl.col("timestamp").dt.ordinal_day().cast(pl.Float64)
    official_days = [
        occurrence.official_start.fromordinal(offset)
        for occurrence in calendar
        for offset in range(
            occurrence.official_start.toordinal(), occurrence.official_end.toordinal() + 1
        )
    ]
    seollal = [occurrence for occurrence in calendar if occurrence.holiday_type == "seollal"]
    chuseok = [occurrence for occurrence in calendar if occurrence.holiday_type == "chuseok"]

    relative_day = pl.lit(0, dtype=pl.Int64)
    holiday_type = pl.lit(0, dtype=pl.Int8)
    for occurrence in calendar:
        in_official_period = day.is_between(occurrence.official_start, occurrence.official_end)
        relative_day = pl.when(in_official_period).then(
            (day - pl.lit(occurrence.central_date)).dt.total_days()
        ).otherwise(relative_day)
        holiday_type = pl.when(in_official_period).then(
            1 if occurrence.holiday_type == "seollal" else 2
        ).otherwise(holiday_type)

    # The calendar contract has no source-of-truth flag for a bridge or substitute
    # date.  Keep it explicit and false rather than inventing an unsupported label.
    featured = lazy.with_columns(
        day.alias("date"),
        hour.cast(pl.Int8).alias("hour"),
        weekday.cast(pl.Int8).alias("day_of_week"),
        (weekday >= 6).cast(pl.Int8).alias("is_weekend"),
        (2 * pi * day_of_year / 365.25).sin().alias("annual_sin"),
        (2 * pi * day_of_year / 365.25).cos().alias("annual_cos"),
        (2 * pi * ((weekday - 1) * 24 + hour) / (7 * 24)).sin().alias("weekly_sin"),
        (2 * pi * ((weekday - 1) * 24 + hour) / (7 * 24)).cos().alias("weekly_cos"),
        (18.0 - pl.col("temperature_c")).clip(lower_bound=0.0).alias("heating_degree_hour"),
        (pl.col("temperature_c") - 24.0).clip(lower_bound=0.0).alias("cooling_degree_hour"),
        pl.col("temperature_c").alias("oracle_temperature_c"),
        pl.col("relative_humidity").alias("oracle_relative_humidity"),
        _date_membership(day, official_days).cast(pl.Int8).alias("is_holiday"),
        relative_day.cast(pl.Int8).alias("event_relative_day"),
        _nearest_signed_distance(day, seollal).alias("seollal_distance"),
        _nearest_signed_distance(day, chuseok).alias("chuseok_distance"),
        pl.lit(0, dtype=pl.Int8).alias("is_bridge_or_substitute_holiday"),
        holiday_type.alias("holiday_type"),
    )
    return featured if isinstance(frame, pl.LazyFrame) else featured.collect()


def build_daily_forecast_matrix(
    frame: pl.DataFrame | pl.LazyFrame, *, feature_set: FeatureSet
) -> ForecastMatrix:
    """Build complete midnight-origin 168-to-24 samples with one NumPy conversion."""

    future_columns = feature_columns(feature_set)
    if feature_set == "B0":
        assert_no_holiday_leakage(future_columns)
    hourly = audit_hourly_data(
        frame, expected_start=None, expected_end=None, expected_rows=None
    )
    required = ("timestamp",) + _HISTORY_COLUMNS + future_columns
    missing = set(required) - set(hourly.columns)
    if missing:
        raise DataContractError(f"featured hourly data is missing columns: {sorted(missing)}")

    # This is the sole Arrow/Polars-to-NumPy boundary.  Slicing thereafter is in-memory.
    packed = hourly.select(required).to_numpy()
    timestamp_dtype = hourly.schema["timestamp"]
    time_unit = getattr(timestamp_dtype, "time_unit", None)
    if time_unit not in {"ms", "us", "ns"}:
        raise DataContractError("timestamp must use a supported Polars datetime unit")
    timestamp_values = np.rint(packed[:, 0]).astype(np.int64)
    timestamps = timestamp_values.astype(f"datetime64[{time_unit}]").astype("datetime64[ns]")
    numeric = np.asarray(packed[:, 1:], dtype=np.float64)
    history_width = len(_HISTORY_COLUMNS)
    future_width = len(future_columns)
    target_starts = np.flatnonzero(
        timestamps == timestamps.astype("datetime64[D]").astype("datetime64[ns]")
    )
    valid_starts = target_starts[(target_starts >= 168) & (target_starts + 24 <= len(timestamps))]
    if not len(valid_starts):
        raise DataContractError("hourly data does not contain a complete 168-to-24 daily sample")

    history = np.stack(
        [numeric[start - 168 : start, :history_width] for start in valid_starts], axis=0
    )
    future = np.stack(
        [
            numeric[
                start : start + 24,
                history_width : history_width + future_width,
            ]
            for start in valid_starts
        ],
        axis=0,
    )
    target = np.stack([numeric[start : start + 24, 0] for start in valid_starts], axis=0)
    target_times = np.stack(
        [timestamps[start : start + 24] for start in valid_starts], axis=0
    )
    return ForecastMatrix(
        origins=timestamps[valid_starts],
        target_times=target_times,
        history=history,
        future=future,
        target=target,
        history_columns=_HISTORY_COLUMNS,
        future_columns=future_columns,
    )
