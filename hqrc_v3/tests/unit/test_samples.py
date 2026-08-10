from __future__ import annotations

import numpy as np
from hqrc_v3.features import attach_calendar_features, build_daily_forecast_matrix


def test_daily_matrix_is_168_to_24(hourly_frame, holiday_calendar):
    matrix = build_daily_forecast_matrix(
        attach_calendar_features(hourly_frame, holiday_calendar), feature_set="B1"
    )

    assert matrix.history.shape[1] == 168
    assert matrix.future.shape[1] == 24
    assert matrix.target.shape[1] == 24
    assert np.all(np.diff(matrix.target_times.astype("datetime64[h]"), axis=1) == 1)


def test_matrix_keeps_weather_and_load_history_separate_from_future_covariates(
    hourly_frame, holiday_calendar
):
    matrix = build_daily_forecast_matrix(
        attach_calendar_features(hourly_frame, holiday_calendar), feature_set="B0"
    )

    assert matrix.history.shape[2] == len(matrix.history_columns)
    assert matrix.future.shape[2] == len(matrix.future_columns)
    assert matrix.history_columns == ("load_mw", "temperature_c", "relative_humidity")
