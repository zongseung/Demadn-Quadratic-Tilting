from __future__ import annotations

from datetime import date

import polars as pl
from hqrc_v3.features import (
    assert_no_holiday_leakage,
    attach_calendar_features,
    feature_columns,
)


def test_b0_has_no_holiday_derived_columns(hourly_frame, holiday_calendar):
    featured = attach_calendar_features(hourly_frame, holiday_calendar)
    b0 = feature_columns("B0")

    assert_no_holiday_leakage(b0)
    assert not ({"is_holiday", "holiday_type", "event_relative_day"} & set(b0))
    assert set(b0) <= set(featured.columns)


def test_b1_signed_day_position_has_expected_holiday_boundaries(hourly_frame, holiday_calendar):
    featured = attach_calendar_features(hourly_frame, holiday_calendar)
    boundary_days = (
        featured.filter(featured["timestamp"].dt.date().is_in([date(2023, 1, 8), date(2023, 1, 9)]))
        .group_by("date")
        .agg(pl.col("event_relative_day").first())
        .sort("date")
    )

    assert boundary_days["event_relative_day"].to_list() == [-1, 0]
    assert {"is_holiday", "holiday_type", "event_relative_day"} <= set(feature_columns("B1"))
