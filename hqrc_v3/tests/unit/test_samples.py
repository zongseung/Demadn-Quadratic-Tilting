from __future__ import annotations

from datetime import datetime

import numpy as np
from hqrc_v3.features import (
    attach_calendar_features,
    build_daily_forecast_matrix,
    feature_columns,
    history_columns,
)


def test_daily_matrix_is_168_to_24(hourly_frame, holiday_calendar):
    matrix = build_daily_forecast_matrix(
        attach_calendar_features(hourly_frame, holiday_calendar), feature_set="B1"
    )

    assert matrix.history.shape[1] == 168
    assert matrix.future.shape[1] == 24
    assert matrix.target.shape[1] == 24
    assert matrix.history.shape[1:] == (168, 17)
    assert matrix.future.shape[1:] == (24, 14)
    assert matrix.history_columns == history_columns("B1")
    assert matrix.future_columns == feature_columns("B1")
    assert np.all(np.diff(matrix.target_times.astype("datetime64[h]"), axis=1) == 1)


def test_matrix_keeps_observed_history_and_shared_calendar_schema_distinct(
    hourly_frame, holiday_calendar
):
    matrix = build_daily_forecast_matrix(
        attach_calendar_features(hourly_frame, holiday_calendar), feature_set="B0"
    )

    assert matrix.history.shape[2] == len(matrix.history_columns)
    assert matrix.future.shape[2] == len(matrix.future_columns)
    assert matrix.history_columns == history_columns("B0")
    assert matrix.future_columns == feature_columns("B0")
    assert matrix.history_columns[:3] == (
        "load_mw",
        "temperature_c",
        "relative_humidity",
    )
    assert matrix.history_columns[3:] == matrix.future_columns
    assert matrix.history.shape[1:] == (168, 10)
    assert matrix.future.shape[1:] == (24, 7)


def test_matrix_origin_history_and_target_share_the_midnight_boundary(
    hourly_frame, holiday_calendar
):
    matrix = build_daily_forecast_matrix(
        attach_calendar_features(hourly_frame, holiday_calendar), feature_set="B0"
    )

    expected_origin = np.datetime64(datetime(2023, 1, 8), "ns")
    last_history_time = np.datetime64(hourly_frame.item(167, "timestamp"), "ns")
    assert matrix.origins[0] == expected_origin
    assert matrix.target_times[0, 0] == matrix.origins[0]
    assert last_history_time == matrix.origins[0] - np.timedelta64(1, "h")
    assert matrix.history[0, -1, 0] == hourly_frame.item(167, "load_mw")
    assert matrix.target[0, 0] == hourly_frame.item(168, "load_mw")
    assert matrix.origins.dtype == np.dtype("datetime64[ns]")
    assert matrix.target_times.dtype == np.dtype("datetime64[ns]")
    assert matrix.history.dtype == matrix.future.dtype == matrix.target.dtype == np.dtype("float64")


def test_matrix_take_keeps_each_sample_and_contract_aligned(hourly_frame, holiday_calendar):
    matrix = build_daily_forecast_matrix(
        attach_calendar_features(hourly_frame, holiday_calendar), feature_set="B1"
    )
    selected = matrix.take(np.array([1, 0]))

    np.testing.assert_array_equal(selected.origins, matrix.origins[[1, 0]])
    np.testing.assert_array_equal(selected.target_times, matrix.target_times[[1, 0]])
    np.testing.assert_array_equal(selected.history, matrix.history[[1, 0]])
    np.testing.assert_array_equal(selected.future, matrix.future[[1, 0]])
    np.testing.assert_array_equal(selected.target, matrix.target[[1, 0]])
    assert selected.history_columns == matrix.history_columns
    assert selected.future_columns == matrix.future_columns
