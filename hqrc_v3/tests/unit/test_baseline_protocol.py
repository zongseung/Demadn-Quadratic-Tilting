from __future__ import annotations

import builtins
import json
from dataclasses import dataclass, replace

import hqrc_v3.baselines.classical as classical
import numpy as np
import pytest
from hqrc_v3.baselines.classical import (
    HorizonRegressor,
    make_classical_baseline,
    predictions_to_frame,
    select_fixed_baseline_config,
)
from hqrc_v3.contracts import DataContractError, ForecastMatrix
from hqrc_v3.splits import AnnualFold


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
    hour = np.arange(24, dtype=float)[None, :, None]
    history_hour = np.arange(168, dtype=float)[None, :, None]
    history = np.concatenate(
        [
            sample + history_hour / 100.0,
            np.broadcast_to(history_hour / 10.0, (sample_count, 168, 1)),
        ],
        axis=2,
    )
    future = np.concatenate(
        [np.broadcast_to(hour / 24.0, (sample_count, 24, 1)), sample + hour / 50.0], axis=2
    )
    target = sample[:, 0, 0, None] + hour[:, :, 0] + 0.5
    return ForecastMatrix(
        origins=origins,
        target_times=target_times,
        history=history,
        future=future,
        target=target,
        history_columns=("load_mw", "temperature"),
        future_columns=("hour", "known_temperature"),
    )


@dataclass(frozen=True)
class _FakeFitted:
    quality: float

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        return batch.target + self.quality


@dataclass(frozen=True)
class _FakeFactory:
    params: dict[str, float]
    name: str = "fake"

    def fit(
        self, train: ForecastMatrix, validation: ForecastMatrix | None, seed: int
    ) -> _FakeFitted:
        del train, validation, seed
        return _FakeFitted(self.params["quality"])


@pytest.fixture
def fake_candidate_factory():
    return lambda params: _FakeFactory(dict(params))


def test_horizon_regressor_builds_distinct_estimators(tiny_forecast_matrix):
    fitted = make_classical_baseline("svr", {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}).fit(
        tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=7
    )

    assert len({id(model) for model in fitted.estimators}) == 24
    assert {model.n_features_in_ for model in fitted.estimators} == {168 * 2 + 2}


def test_one_shot_selection_returns_lowest_validation_rmse(
    fake_candidate_factory, tiny_forecast_matrix
):
    selected = select_fixed_baseline_config(
        candidates=({"quality": 3.0}, {"quality": 1.0}),
        factory_builder=fake_candidate_factory,
        train=tiny_forecast_matrix.take(np.arange(16)),
        validation=tiny_forecast_matrix.take(np.arange(16, 24)),
        seed=7,
    )

    assert selected.params == {"quality": 1.0}
    assert selected.metric == "rmse"
    assert selected.selected_index == 1
    assert selected.selected_score == 1.0
    assert [(entry.params, entry.rmse) for entry in selected.candidate_scores] == [
        ({"quality": 3.0}, 3.0),
        ({"quality": 1.0}, 1.0),
    ]
    assert json.loads(json.dumps(selected.to_dict(), sort_keys=True)) == {
        "candidate_scores": [
            {"params": {"quality": 3.0}, "rmse": 3.0},
            {"params": {"quality": 1.0}, "rmse": 1.0},
        ],
        "metric": "rmse",
        "params": {"quality": 1.0},
        "selected_index": 1,
        "selected_score": 1.0,
    }


def test_one_shot_selection_breaks_ties_by_candidate_order(
    fake_candidate_factory, tiny_forecast_matrix
):
    selected = select_fixed_baseline_config(
        candidates=({"quality": 1.0}, {"quality": -1.0}),
        factory_builder=fake_candidate_factory,
        train=tiny_forecast_matrix.take(np.arange(16)),
        validation=tiny_forecast_matrix.take(np.arange(16, 24)),
        seed=7,
    )

    assert selected.params == {"quality": 1.0}
    assert selected.selected_index == 0


def test_one_shot_selection_rejects_non_serializable_candidate_params(
    fake_candidate_factory, tiny_forecast_matrix
):
    with pytest.raises(ValueError, match="JSON serializable"):
        select_fixed_baseline_config(
            candidates=({"quality": 1.0, "metadata": {"not-json"}},),
            factory_builder=fake_candidate_factory,
            train=tiny_forecast_matrix.take(np.arange(16)),
            validation=tiny_forecast_matrix.take(np.arange(16, 24)),
            seed=7,
        )


@pytest.mark.parametrize("history_hours", [167, 169])
def test_classical_baseline_requires_exactly_168_history_hours(
    tiny_forecast_matrix, history_hours
):
    history = tiny_forecast_matrix.history
    if history_hours == 167:
        history = history[:, :167, :]
    else:
        history = np.concatenate((history, history[:, -1:, :]), axis=1)
    malformed = replace(
        tiny_forecast_matrix,
        history=history,
    )
    baseline = make_classical_baseline("svr", {"C": 1.0})

    with pytest.raises(ValueError, match="168 history hours"):
        baseline.fit(malformed.take(np.arange(20)), validation=None, seed=7)

    fitted = baseline.fit(tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=7)
    with pytest.raises(ValueError, match="168 history hours"):
        fitted.predict(malformed.take(np.arange(20, 24)))


def test_horizon_regressor_uses_only_its_matching_future_covariates(tiny_forecast_matrix):
    class RecordingEstimator:
        def __init__(self):
            self.inputs: list[np.ndarray] = []

        def predict(self, features: np.ndarray) -> np.ndarray:
            self.inputs.append(features.copy())
            return np.zeros(features.shape[0])

    batch = tiny_forecast_matrix.take(np.arange(2))
    estimators = tuple(RecordingEstimator() for _ in range(24))
    fitted = HorizonRegressor(
        model_name="recording",
        estimators=estimators,
        history_shape=(168, 2),
        future_width=2,
        history_columns=batch.history_columns,
        future_columns=batch.future_columns,
    )

    fitted.predict(batch)

    for horizon, estimator in enumerate(estimators):
        np.testing.assert_array_equal(estimator.inputs[0][:, -2:], batch.future[:, horizon, :])


@pytest.mark.parametrize("stream", ["history", "future"])
@pytest.mark.parametrize("change", ["reordered", "renamed"])
def test_horizon_regressor_rejects_changed_feature_columns(
    tiny_forecast_matrix, stream, change
):
    fitted = make_classical_baseline("svr", {"C": 1.0}).fit(
        tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=7
    )
    batch = tiny_forecast_matrix.take(np.arange(20, 24))
    if change == "renamed":
        columns = getattr(batch, f"{stream}_columns")
        batch = replace(
            batch,
            **{f"{stream}_columns": (f"renamed_{columns[0]}", *columns[1:])},
        )
    elif stream == "history":
        batch = replace(
            batch,
            history=batch.history[:, :, ::-1],
            history_columns=batch.history_columns[::-1],
        )
    else:
        batch = replace(
            batch,
            future=batch.future[:, :, ::-1],
            future_columns=batch.future_columns[::-1],
        )

    with pytest.raises(DataContractError, match="feature columns/order"):
        fitted.predict(batch)


def test_classical_fit_rejects_duplicate_feature_columns(tiny_forecast_matrix):
    malformed = replace(
        tiny_forecast_matrix,
        future_columns=("hour", "hour"),
    )

    with pytest.raises(DataContractError, match="feature column names"):
        make_classical_baseline("svr", {"C": 1.0}).fit(
            malformed, validation=None, seed=7
        )


@pytest.mark.parametrize(
    "name, params",
    [("xgboost", {"random_state": 1}), ("lightgbm", {"n_jobs": 2})],
)
def test_adapter_managed_random_and_thread_params_are_rejected(name, params):
    with pytest.raises(ValueError, match="managed"):
        make_classical_baseline(name, params)


@pytest.mark.parametrize(
    "name, params",
    [("xgboost", {"n_estimators": 1}), ("lightgbm", {"n_estimators": 1})],
)
def test_boosting_models_receive_seed_and_single_thread_defaults(
    tiny_forecast_matrix, name, params
):
    fitted = make_classical_baseline(name, params).fit(
        tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=7
    )

    for estimator in fitted.estimators:
        fitted_params = estimator.get_params(deep=False)
        assert fitted_params["random_state"] == 7
        assert fitted_params["n_jobs"] == 1


def test_boosting_adapter_propagates_the_provided_seed(monkeypatch, tiny_forecast_matrix):
    class SeedRecordingRegressor:
        def __init__(self, **params):
            self.params = params

        def get_params(self, deep=False):
            del deep
            return {"n_jobs": None, "random_state": None}

        def fit(self, features, target):
            del features, target
            return self

        def predict(self, features):
            return np.zeros(features.shape[0])

    monkeypatch.setattr(classical, "_estimator_class", lambda name: SeedRecordingRegressor)

    fitted = classical.make_classical_baseline("xgboost", {}).fit(
        tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=19
    )

    assert {estimator.params["random_state"] for estimator in fitted.estimators} == {19}
    assert {estimator.params["n_jobs"] for estimator in fitted.estimators} == {1}


def test_fixed_fit_does_not_inspect_validation(tiny_forecast_matrix):
    fitted = make_classical_baseline("svr", {"C": 1.0}).fit(
        tiny_forecast_matrix.take(np.arange(20)), validation=object(), seed=7
    )

    assert len(fitted.estimators) == 24


def test_missing_optional_dependency_has_an_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def raise_for_xgboost(name, *args, **kwargs):
        if name == "xgboost":
            raise ImportError("controlled missing dependency")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", raise_for_xgboost)

    with pytest.raises(ImportError, match="xgboost baseline requires"):
        make_classical_baseline("xgboost", {})


def test_predictions_to_frame_emits_validated_task_three_schema(tiny_forecast_matrix):
    batch = tiny_forecast_matrix.take(np.arange(2))
    prediction = batch.target - 0.25

    frame = predictions_to_frame(
        batch,
        prediction,
        model="svr",
        feature_set="B0",
        seed=7,
        fold=AnnualFold.oof(eval_year=2023, first_train_year=2019),
    )

    assert frame.shape == (48, 9)
    assert frame["horizon"].to_list()[:24] == list(range(1, 25))
    assert frame["observed_mw"].to_list()[:24] == pytest.approx(batch.target[0].tolist())
    assert frame["split_id"].unique().to_list() == ["oof-2023"]


def test_classical_adapter_rejects_unknown_model_and_params():
    with pytest.raises(ValueError, match="unknown classical baseline"):
        make_classical_baseline("random_forest", {})
    with pytest.raises(ValueError, match="unsupported parameters"):
        make_classical_baseline("svr", {"not_a_parameter": 1})


@pytest.mark.parametrize("seed", [True, 1.9])
def test_predictions_to_frame_rejects_non_integer_or_boolean_seed(tiny_forecast_matrix, seed):
    batch = tiny_forecast_matrix.take(np.arange(1))

    with pytest.raises((TypeError, ValueError), match="seed"):
        predictions_to_frame(
            batch,
            batch.target,
            model="svr",
            feature_set="B0",
            seed=seed,
            fold=AnnualFold.oof(eval_year=2023, first_train_year=2019),
        )


@pytest.mark.parametrize("model, feature_set", [("", "B0"), ("svr", " ")])
def test_predictions_to_frame_rejects_blank_context(tiny_forecast_matrix, model, feature_set):
    batch = tiny_forecast_matrix.take(np.arange(1))

    with pytest.raises(ValueError, match="nonblank"):
        predictions_to_frame(
            batch,
            batch.target,
            model=model,
            feature_set=feature_set,
            seed=7,
            fold=AnnualFold.oof(eval_year=2023, first_train_year=2019),
        )


def test_predictions_to_frame_rejects_bad_shape_and_fold_context(tiny_forecast_matrix):
    batch = tiny_forecast_matrix.take(np.arange(1))

    with pytest.raises(ValueError, match="shape"):
        predictions_to_frame(
            batch,
            batch.target[:, :23],
            model="svr",
            feature_set="B0",
            seed=7,
            fold=AnnualFold.oof(eval_year=2023, first_train_year=2019),
        )
    with pytest.raises(TypeError, match="AnnualFold"):
        predictions_to_frame(
            batch,
            batch.target,
            model="svr",
            feature_set="B0",
            seed=7,
            fold=object(),
        )
