from __future__ import annotations

import numpy as np
import pytest
from hqrc_v3.baselines.classical import make_classical_baseline
from hqrc_v3.contracts import ForecastMatrix


@pytest.fixture
def tiny_forecast_matrix() -> ForecastMatrix:
    sample_count = 24
    origins = np.arange(
        np.datetime64("2023-01-01T00:00"),
        np.datetime64("2023-01-25T00:00"),
        dtype="datetime64[D]",
    ).astype("datetime64[ns]")
    target_times = origins[:, None] + np.arange(24).astype("timedelta64[h]")
    sample = np.arange(sample_count, dtype=float)[:, None, None]
    history_hour = np.arange(168, dtype=float)[None, :, None]
    future_hour = np.arange(24, dtype=float)[None, :, None]
    history = np.concatenate(
        [
            sample + history_hour / 100.0,
            np.broadcast_to(history_hour / 10.0, (sample_count, 168, 1)),
        ],
        axis=2,
    )
    future = np.concatenate(
        [
            np.broadcast_to(future_hour / 24.0, (sample_count, 24, 1)),
            sample + future_hour / 50.0,
        ],
        axis=2,
    )
    target = sample[:, 0, 0, None] + future_hour[:, :, 0] + 0.5
    return ForecastMatrix(
        origins=origins,
        target_times=target_times,
        history=history,
        future=future,
        target=target,
        history_columns=("load_mw", "temperature"),
        future_columns=("hour", "known_temperature"),
    )


def tiny_params(name: str) -> dict[str, float | int | str]:
    return {
        "xgboost": {"n_estimators": 3, "max_depth": 2, "learning_rate": 0.2},
        "lightgbm": {"n_estimators": 3, "num_leaves": 4, "learning_rate": 0.2},
        "svr": {"C": 1.0, "epsilon": 0.1, "gamma": "scale"},
    }[name]


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "svr"])
def test_classical_baseline_returns_24_horizons(name, tiny_forecast_matrix):
    baseline = make_classical_baseline(name, tiny_params(name))
    fitted = baseline.fit(tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=7)

    prediction = fitted.predict(tiny_forecast_matrix.take(np.arange(20, 24)))

    assert prediction.shape == (4, 24)
    assert np.isfinite(prediction).all()
