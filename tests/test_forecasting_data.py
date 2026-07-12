from datetime import datetime, timedelta

import numpy as np
import polars as pl

from demand_quadratic_tilting.forecasting.config import ForecastConfig
from demand_quadratic_tilting.forecasting.data import (
    build_windowed_dataset,
    fit_scalers,
    scale_split,
)
from demand_quadratic_tilting.forecasting.evaluation import build_prediction_frame


def _hourly_frame(hours: int = 120) -> pl.DataFrame:
    start = datetime(2020, 1, 1)
    datetimes = [start + timedelta(hours=offset) for offset in range(hours)]
    values = np.arange(hours, dtype=np.float64)
    return pl.DataFrame(
        {
            "일시": datetimes,
            "hm": 50.0 + values,
            "ta": -5.0 + values / 10,
            "power demand(MW)": 10_000.0 + values,
            "holiday_name": ["non-event"] * hours,
            "weekday": [1] * hours,
            "weekend": [0] * hours,
            "spring": [0] * hours,
            "summer": [0] * hours,
            "autoum": [0] * hours,
            "winter": [1] * hours,
            "is_holiday_dummies": [0] * hours,
        }
    )


def _config() -> ForecastConfig:
    return ForecastConfig(
        history_hours=24,
        horizon_hours=24,
        stride_hours=24,
        train_end="2020-01-02 23:00:00",
        validation_end="2020-01-03 23:00:00",
    )


def test_windows_use_prior_split_history_without_target_leakage() -> None:
    frame = _hourly_frame()
    config = _config()
    dataset = build_windowed_dataset(frame, config)

    assert {name: len(split) for name, split in dataset.splits.items()} == {
        "train": 1,
        "validation": 1,
        "test": 2,
    }
    validation = dataset.splits["validation"]
    np.testing.assert_array_equal(validation.target_row_indices[0], np.arange(48, 72))
    np.testing.assert_array_equal(
        validation.history[0, :, 0],
        frame["power demand(MW)"].to_numpy()[24:48],
    )


def test_scalers_are_fit_on_training_rows_only() -> None:
    frame = _hourly_frame()
    config = _config()
    dataset = build_windowed_dataset(frame, config)
    scalers = fit_scalers(frame, config)

    expected_mean = frame["power demand(MW)"].to_numpy()[:48].mean()
    assert scalers.target.mean_[0] == expected_mean
    validation = scale_split(dataset.splits["validation"], scalers, config)
    assert validation.history.dtype == np.float32
    assert validation.future_known.dtype == np.float32


def test_prediction_frame_is_hourly_and_hqt_compatible() -> None:
    frame = _hourly_frame()
    config = _config()
    dataset = build_windowed_dataset(frame, config)
    predictions = {
        "example": {name: split.target.copy() for name, split in dataset.splits.items()}
    }
    output = build_prediction_frame(dataset, predictions)

    assert output.height == 96
    assert output["datetime"].n_unique() == 96
    assert output["example"].null_count() == 0
    assert output["holiday_name"].null_count() == 0
    assert set(output["split"].unique()) == {"train", "validation", "test"}
