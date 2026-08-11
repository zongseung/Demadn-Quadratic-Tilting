from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from hqrc_v3.baselines.preprocessing import (
    fit_classical_preprocessor,
    fit_sequence_preprocessor,
)
from hqrc_v3.contracts import DataContractError, ForecastMatrix


@pytest.fixture
def matrix() -> ForecastMatrix:
    count = 10
    origins = np.arange("2023-01-01", "2023-01-11", dtype="datetime64[D]").astype(
        "datetime64[ns]"
    )
    target_times = origins[:, None] + np.arange(24).astype("timedelta64[h]")
    sample = np.arange(count, dtype=float)[:, None, None]
    history_hour = np.arange(-168, 0, dtype=float)[None, :, None]
    future_hour = np.arange(24, dtype=float)[None, :, None]
    absolute_history_hour = sample * 24.0 + history_hour
    load = 50_000.0 + absolute_history_hour
    temperature = 10.0 + absolute_history_hour / 168.0
    humidity = 50.0 + absolute_history_hour / 336.0
    history_calendar = np.concatenate(
        (np.mod(absolute_history_hour, 24.0), absolute_history_hour / 24.0), axis=2
    )
    future = np.concatenate(
        (
            np.broadcast_to(future_hour, (count, 24, 1)),
            np.broadcast_to(sample + future_hour / 24.0, (count, 24, 1)),
        ),
        axis=2,
    )
    target = 51_000.0 + sample[:, 0, 0, None] * 100.0 + future_hour[:, :, 0]
    return ForecastMatrix(
        origins=origins,
        target_times=target_times,
        history=np.concatenate((load, temperature, humidity, history_calendar), axis=2),
        future=future,
        target=target,
        history_columns=(
            "load_mw",
            "temperature_c",
            "relative_humidity",
            "hour",
            "annual_sin",
        ),
        future_columns=("hour", "annual_sin"),
    )


def test_classical_preprocessor_uses_full_future_path_and_train_only_statistics(matrix):
    train = matrix.take(np.arange(7))
    validation = matrix.take(np.arange(7, 10))
    fitted = fit_classical_preprocessor(train)
    changed_future = validation.future.copy()
    changed_future[:, 23, 0] += 9_999.0
    changed_validation = replace(validation, future=changed_future)

    transformed = fitted.transform_features(validation)
    changed = fitted.transform_features(changed_validation)

    assert transformed.shape[1] == 168 * 5 + 24 * 2
    assert not np.array_equal(transformed, changed)
    np.testing.assert_allclose(fitted.target_scaler.mean, train.target.mean())
    np.testing.assert_allclose(
        fitted.inverse_target(fitted.transform_target(validation.target)),
        validation.target,
    )
    assert fitted.population_contract() == {
        "x": {
            "count": 7,
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-07T00:00:00",
            "unit": "daily-sample",
        },
        "target": {
            "count": 168,
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-07T23:00:00",
            "unit": "unique-hour",
        },
    }


def test_validation_perturbation_cannot_change_any_fitted_scaler(matrix):
    train = matrix.take(np.arange(7))
    validation = matrix.take(np.arange(7, 10))
    changed_validation = replace(
        validation,
        history=validation.history + 1e9,
        future=validation.future + 1e9,
        target=validation.target + 1e9,
    )

    first_classical = fit_classical_preprocessor(train)
    second_classical = fit_classical_preprocessor(train)
    first_sequence = fit_sequence_preprocessor(train)
    second_sequence = fit_sequence_preprocessor(train)

    first_classical.transform_features(validation)
    second_classical.transform_features(changed_validation)
    first_sequence.transform_inputs(validation)
    second_sequence.transform_inputs(changed_validation)
    np.testing.assert_array_equal(first_classical.x_scaler.mean, second_classical.x_scaler.mean)
    np.testing.assert_array_equal(
        first_classical.target_scaler.mean, second_classical.target_scaler.mean
    )
    np.testing.assert_array_equal(
        first_sequence.weather_scaler.mean, second_sequence.weather_scaler.mean
    )
    np.testing.assert_array_equal(
        first_sequence.future_scaler.mean, second_sequence.future_scaler.mean
    )


def test_sequence_load_and_target_share_one_training_target_scale(matrix):
    train = matrix.take(np.arange(7))
    fitted = fit_sequence_preprocessor(train)
    history, future = fitted.transform_inputs(train)
    target = fitted.transform_target(train.target)

    np.testing.assert_allclose(fitted.target_scaler.mean, train.target.mean())
    np.testing.assert_allclose(fitted.target_scaler.scale, train.target.std())
    np.testing.assert_allclose(
        history[..., 0],
        (train.history[..., 0] - train.target.mean()) / train.target.std(),
    )
    np.testing.assert_allclose(target, (train.target - train.target.mean()) / train.target.std())
    assert future.shape == train.future.shape


def test_sequence_weather_population_deduplicates_overlapping_history_hours(matrix):
    train = matrix.take(np.arange(7))
    fitted = fit_sequence_preprocessor(train)
    unique_hour = np.arange(-168, 6 * 24, dtype=float)
    expected_weather = np.column_stack(
        (10.0 + unique_hour / 168.0, 50.0 + unique_hour / 336.0)
    )

    np.testing.assert_allclose(fitted.weather_scaler.mean, expected_weather.mean(axis=0))
    np.testing.assert_allclose(fitted.weather_scaler.scale, expected_weather.std(axis=0))
    assert expected_weather.shape[0] == 312
    assert fitted.population_contract() == {
        "target": {
            "count": 168,
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-07T23:00:00",
            "unit": "unique-hour",
        },
        "weather": {
            "count": 312,
            "start": "2022-12-25T00:00:00",
            "end": "2023-01-06T23:00:00",
            "unit": "unique-hour",
        },
        "calendar": {
            "count": 168,
            "start": "2023-01-01T00:00:00",
            "end": "2023-01-07T23:00:00",
            "unit": "unique-hour",
        },
    }


def test_sequence_calendar_scaler_is_future_fitted_and_shared_with_history(matrix):
    train = matrix.take(np.arange(7))
    fitted = fit_sequence_preprocessor(train)
    history, future = fitted.transform_inputs(train)
    expected_mean = train.future.mean(axis=(0, 1))
    expected_scale = train.future.reshape(-1, train.future.shape[2]).std(axis=0)

    np.testing.assert_allclose(fitted.calendar_scaler.mean, expected_mean)
    np.testing.assert_allclose(fitted.calendar_scaler.scale, expected_scale)
    np.testing.assert_allclose(
        future,
        (train.future - expected_mean) / expected_scale,
    )
    np.testing.assert_allclose(
        history[..., 3:],
        (train.history[..., 3:] - expected_mean) / expected_scale,
    )


def test_sequence_weather_scaler_rejects_inconsistent_overlapping_hours(matrix):
    train = matrix.take(np.arange(7))
    history = train.history.copy()
    history[1, 0, 1] += 1.0
    inconsistent = replace(train, history=history)

    with pytest.raises(DataContractError, match="inconsistent values"):
        fit_sequence_preprocessor(inconsistent)
