"""Training-only same-holiday similar-day profiles for the H5 ablation."""

from __future__ import annotations

import math

import numpy as np
import polars as pl


class SimilarDayError(ValueError):
    """Raised when H5 cannot construct an equal-occurrence training profile."""


def _finite_profile(frame: pl.DataFrame, *, value_column: str) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame):
        raise SimilarDayError("profile frame must be a Polars DataFrame")
    required = {"occurrence_id", "holiday_type", "relative_day", "hour", value_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SimilarDayError(f"profile frame is missing required columns: {missing}")
    if frame.is_empty() or frame["occurrence_id"].null_count() or frame[value_column].null_count():
        raise SimilarDayError("profile frame must have non-null occurrence ids and values")
    if (
        frame["holiday_type"].null_count()
        or frame["relative_day"].null_count()
        or frame["hour"].null_count()
    ):
        raise SimilarDayError("profile frame must have complete holiday/day/hour keys")
    values = frame[value_column].cast(pl.Float64, strict=True)
    if not values.is_finite().all():
        raise SimilarDayError("profile values must be finite")
    result = frame.with_columns(values.alias(value_column))
    duplicates = (
        result.group_by("occurrence_id", "relative_day", "hour").len().filter(pl.col("len") != 1)
    )
    if not duplicates.is_empty():
        raise SimilarDayError("each training occurrence/day/hour profile row must be unique")
    return result


def same_holiday_profile(
    training: pl.DataFrame,
    target: pl.DataFrame,
    *,
    value_column: str = "standardized_residual",
) -> np.ndarray:
    """Return an H5 correction with equal-occurrence normalized daily profiles.

    For every relative day, each training occurrence is first divided by its own
    daily mean.  Occurrences then receive equal weight, and a single least-squares
    level is fit from *training rows only*.  Target values never enter that scale.
    """

    if not isinstance(target, pl.DataFrame):
        raise SimilarDayError("target frame must be a Polars DataFrame")
    train = _finite_profile(training, value_column=value_column)
    required_target = {"holiday_type", "relative_day", "hour"}
    missing_target = sorted(required_target - set(target.columns))
    if missing_target or target.is_empty():
        raise SimilarDayError(f"target frame is missing required columns: {missing_target}")
    if any(target[column].null_count() for column in required_target):
        raise SimilarDayError("target frame must have complete holiday/day/hour keys")
    holiday_types = target["holiday_type"].unique().to_list()
    if len(holiday_types) != 1:
        raise SimilarDayError("H5 target must contain one holiday type")
    holiday = holiday_types[0]
    selected = train.filter(pl.col("holiday_type") == holiday)
    if selected.is_empty():
        raise SimilarDayError("H5 requires prior occurrences of the same holiday type")

    rows: list[dict[str, float | int]] = []
    for day in selected["relative_day"].unique().sort().to_list():
        day_rows = selected.filter(pl.col("relative_day") == day)
        occurrences = day_rows["occurrence_id"].unique().sort().to_list()
        normalized: dict[str, dict[int, float]] = {}
        raw_values: list[float] = []
        profile_values: list[float] = []
        for occurrence in occurrences:
            occurrence_day = day_rows.filter(pl.col("occurrence_id") == occurrence).sort("hour")
            raw = occurrence_day[value_column].to_numpy().astype(float)
            mean = float(raw.mean())
            if not math.isfinite(mean) or abs(mean) <= np.finfo(float).eps:
                raise SimilarDayError(
                    "daily means must be finite and non-zero for H5 normalization"
                )
            normalized[occurrence] = dict(zip(occurrence_day["hour"].to_list(), raw / mean))
        common_hours = sorted(set.intersection(*(set(values) for values in normalized.values())))
        if not common_hours:
            raise SimilarDayError("same-holiday occurrences have no common hourly profile")
        mean_profile = {
            hour: float(np.mean([normalized[occurrence][hour] for occurrence in occurrences]))
            for hour in common_hours
        }
        for occurrence in occurrences:
            occurrence_day = day_rows.filter(pl.col("occurrence_id") == occurrence)
            raw_by_hour = dict(
                zip(occurrence_day["hour"].to_list(), occurrence_day[value_column].to_list())
            )
            raw_values.extend(float(raw_by_hour[hour]) for hour in common_hours)
            profile_values.extend(mean_profile[hour] for hour in common_hours)
        design = np.asarray(profile_values, dtype=float)
        scale = float(np.dot(design, raw_values) / np.dot(design, design))
        rows.extend(
            {"relative_day": int(day), "hour": int(hour), "correction": scale * mean_profile[hour]}
            for hour in common_hours
        )

    lookup = pl.DataFrame(rows)
    indexed = target.with_row_index("_row").join(lookup, on=["relative_day", "hour"], how="left")
    if indexed["correction"].null_count():
        raise SimilarDayError("target has a relative-day/hour absent from the training profile")
    return indexed.sort("_row")["correction"].to_numpy().astype(float)
