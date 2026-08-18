from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
from hqrc_v3.events import EventOccurrence, load_holiday_calendar
from hqrc_v3.features import (
    assert_no_holiday_leakage,
    attach_calendar_features,
    build_daily_forecast_matrix,
    feature_columns,
)

B0_FUTURE = (
    "hour",
    "day_of_week",
    "is_weekend",
    "annual_sin",
    "annual_cos",
    "weekly_sin",
    "weekly_cos",
)
B1_ONLY = (
    "is_public_holiday",
    "official_sequence_position",
    "seollal_distance",
    "chuseok_distance",
    "is_substitute_or_temporary_holiday",
    "is_seollal",
    "is_chuseok",
)


def test_b0_has_no_holiday_derived_columns(hourly_frame, holiday_calendar):
    featured = attach_calendar_features(hourly_frame, holiday_calendar)
    b0 = feature_columns("B0")

    assert_no_holiday_leakage(b0)
    assert b0 == B0_FUTURE
    assert not ({"is_public_holiday", "is_seollal", "official_sequence_position"} & set(b0))
    assert not any(
        token in column
        for column in b0
        for token in ("oracle", "temperature", "humidity", "heating", "cooling")
    )
    assert set(b0) <= set(featured.columns)


def test_b1_signed_day_position_has_expected_holiday_boundaries(hourly_frame, holiday_calendar):
    featured = attach_calendar_features(hourly_frame, holiday_calendar)
    boundary_days = (
        featured.filter(featured["timestamp"].dt.date().is_in([date(2023, 1, 8), date(2023, 1, 9)]))
        .group_by("date")
        .agg(pl.col("official_sequence_position").first())
        .sort("date")
    )

    assert boundary_days["official_sequence_position"].to_list() == [-1, 0]
    assert feature_columns("B1") == B0_FUTURE + B1_ONLY


def test_future_matrix_does_not_change_when_target_weather_is_perturbed(
    hourly_frame, holiday_calendar
):
    featured = attach_calendar_features(hourly_frame, holiday_calendar)
    before_b0 = build_daily_forecast_matrix(featured, feature_set="B0")
    before_b1 = build_daily_forecast_matrix(featured, feature_set="B1")
    assert before_b0.history.shape[1:] == (168, 10)
    assert before_b0.future.shape[1:] == (24, 7)
    assert before_b1.history.shape[1:] == (168, 17)
    assert before_b1.future.shape[1:] == (24, 14)
    target_day = date(2023, 1, 8)
    perturbed = hourly_frame.with_columns(
        pl.when(pl.col("timestamp").dt.date() == target_day)
        .then(pl.col("temperature_c") + 1_000.0)
        .otherwise(pl.col("temperature_c"))
        .alias("temperature_c"),
        pl.when(pl.col("timestamp").dt.date() == target_day)
        .then(pl.col("relative_humidity") + 1_000.0)
        .otherwise(pl.col("relative_humidity"))
        .alias("relative_humidity"),
    )
    changed = attach_calendar_features(perturbed, holiday_calendar)
    after_b0 = build_daily_forecast_matrix(changed, feature_set="B0")
    after_b1 = build_daily_forecast_matrix(changed, feature_set="B1")
    origin = np.flatnonzero(before_b0.origins == np.datetime64("2023-01-08", "ns"))[0]

    assert before_b0.future[origin].tobytes() == after_b0.future[origin].tobytes()
    assert before_b1.future[origin].tobytes() == after_b1.future[origin].tobytes()


def test_source_public_holiday_and_official_window_are_distinct(
    hourly_frame, holiday_calendar
):
    featured = attach_calendar_features(hourly_frame, holiday_calendar)
    days = (
        featured.filter(
            pl.col("timestamp").dt.hour() == 0,
            pl.col("timestamp").dt.date().is_in(
                [date(2023, 1, 7), date(2023, 1, 8), date(2023, 1, 10)]
            ),
        )
        .select(
            "date",
            "is_public_holiday",
            "official_sequence_position",
            "is_seollal",
            "is_chuseok",
        )
        .sort("date")
    )

    assert days["is_public_holiday"].to_list() == [0, 1, 1]
    assert days["official_sequence_position"].to_list() == [0, -1, 1]
    assert days["is_seollal"].to_list() == [0, 1, 1]
    assert days["is_chuseok"].to_list() == [0, 0, 0]


def _feature_frame(timestamps: list[datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": [50_000.0] * len(timestamps),
            "temperature_c": [10.0] * len(timestamps),
            "relative_humidity": [50.0] * len(timestamps),
            "source_holiday_name": [""] * len(timestamps),
            "source_public_holiday": [0] * len(timestamps),
        }
    )


def test_signed_distances_use_boundary_support_occurrences() -> None:
    calendar = load_holiday_calendar(
        Path(__file__).resolve().parents[2] / "configs/holiday_calendar.csv"
    )
    featured = attach_calendar_features(
        _feature_frame([datetime(2019, 1, 1), datetime(2024, 10, 31)]), calendar
    )

    assert featured["chuseok_distance"].to_list() == [99, 44]
    assert featured["seollal_distance"].to_list() == [-35, -90]


def test_signed_distance_tie_uses_earlier_center_independent_of_input_order() -> None:
    earlier = EventOccurrence(
        "seollal-2020",
        "seollal",
        date(2020, 1, 1),
        date(2020, 1, 1),
        date(2020, 1, 1),
        0,
    )
    later = EventOccurrence(
        "seollal-2021",
        "seollal",
        date(2020, 1, 5),
        date(2020, 1, 5),
        date(2020, 1, 5),
        0,
    )
    frame = _feature_frame([datetime(2020, 1, 3)])

    forward = attach_calendar_features(frame, (earlier, later))
    reverse = attach_calendar_features(frame, (later, earlier))

    assert forward.item(0, "seollal_distance") == 2
    assert reverse.item(0, "seollal_distance") == 2
