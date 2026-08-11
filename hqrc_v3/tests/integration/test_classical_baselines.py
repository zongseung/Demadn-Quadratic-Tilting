from __future__ import annotations

import hqrc_v3.baselines.classical as classical
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


def stochastic_tiny_params(name: str) -> dict[str, float | int | str]:
    return {
        "xgboost": {
            "colsample_bytree": 0.75,
            "learning_rate": 0.2,
            "max_depth": 2,
            "n_estimators": 5,
            "subsample": 0.75,
        },
        "lightgbm": {
            "colsample_bytree": 0.75,
            "learning_rate": 0.2,
            "min_child_samples": 2,
            "n_estimators": 5,
            "num_leaves": 4,
            "subsample": 0.75,
            "subsample_freq": 1,
        },
    }[name]


@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "svr"])
def test_classical_baseline_returns_24_horizons(name, tiny_forecast_matrix):
    baseline = make_classical_baseline(name, tiny_params(name))
    fitted = baseline.fit(tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=7)

    prediction = fitted.predict(tiny_forecast_matrix.take(np.arange(20, 24)))

    assert prediction.shape == (4, 24)
    assert np.isfinite(prediction).all()


@pytest.mark.parametrize("name", ["xgboost", "lightgbm"])
def test_boosting_repeated_fits_with_the_same_seed_are_reproducible(name, tiny_forecast_matrix):
    train = tiny_forecast_matrix.take(np.arange(20))
    batch = tiny_forecast_matrix.take(np.arange(20, 24))
    first = make_classical_baseline(name, stochastic_tiny_params(name)).fit(
        train, validation=None, seed=17
    )
    second = make_classical_baseline(name, stochastic_tiny_params(name)).fit(
        train, validation=None, seed=17
    )

    first_prediction = first.predict(batch)
    second_prediction = second.predict(batch)

    assert np.isfinite(first_prediction).all()
    assert np.isfinite(second_prediction).all()
    assert {id(model) for model in first.estimators}.isdisjoint(
        {id(model) for model in second.estimators}
    )
    np.testing.assert_allclose(first_prediction, second_prediction, rtol=0.0, atol=1e-12)


def test_svr_receives_full_scaled_path_and_standardized_target_then_returns_mw(
    monkeypatch: pytest.MonkeyPatch, tiny_forecast_matrix: ForecastMatrix
) -> None:
    class RecordingSVR:
        instances: list[RecordingSVR] = []

        def __init__(self, **params: object) -> None:
            self.params = params
            self.fit_features: np.ndarray | None = None
            self.fit_target: np.ndarray | None = None
            self.predict_features: list[np.ndarray] = []
            RecordingSVR.instances.append(self)

        def get_params(self, deep: bool = False) -> dict[str, object]:
            del deep
            return {"C": None, "epsilon": None, "gamma": None, "cache_size": None}

        def fit(self, features: np.ndarray, target: np.ndarray, **_: object) -> RecordingSVR:
            self.fit_features = np.asarray(features).copy()
            self.fit_target = np.asarray(target).copy()
            return self

        def predict(self, features: np.ndarray) -> np.ndarray:
            values = np.asarray(features).copy()
            self.predict_features.append(values)
            return np.zeros(values.shape[0])

    monkeypatch.setattr(classical, "_estimator_class", lambda _: RecordingSVR)
    train = tiny_forecast_matrix.take(np.arange(20))
    validation = tiny_forecast_matrix.take(np.arange(20, 24))
    baseline = make_classical_baseline(
        "svr", {"C": 10.0, "epsilon": 0.05, "gamma": "scale"}
    )

    fitted = baseline.fit(train, validation=validation, seed=7)
    prediction = fitted.predict(validation)

    estimators = [
        estimator for estimator in RecordingSVR.instances if estimator.fit_target is not None
    ]
    assert len(estimators) == 24
    assert all(estimator.params["epsilon"] == 0.05 for estimator in estimators)
    assert all(
        estimator.fit_features.shape[1] == 168 * train.history.shape[2] + 24 * 2
        for estimator in estimators
    )
    for estimator in estimators[1:]:
        np.testing.assert_array_equal(estimator.fit_features, estimators[0].fit_features)
    all_standardized_targets = np.column_stack(
        [estimator.fit_target for estimator in estimators]
    )
    np.testing.assert_allclose(all_standardized_targets.mean(), 0.0, atol=1e-12)
    np.testing.assert_allclose(all_standardized_targets.std(), 1.0, atol=1e-12)
    np.testing.assert_allclose(prediction, train.target.mean())

    changed_future = validation.future.copy()
    changed_future[:, 23, 0] += 9_999.0
    changed = ForecastMatrix(
        origins=validation.origins,
        target_times=validation.target_times,
        history=validation.history,
        future=changed_future,
        target=validation.target,
        history_columns=validation.history_columns,
        future_columns=validation.future_columns,
    )
    fitted.predict(changed)
    assert all(
        not np.array_equal(estimator.predict_features[0], estimator.predict_features[1])
        for estimator in estimators
    )
