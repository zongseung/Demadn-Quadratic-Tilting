from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from hqrc_v3.baselines.classical import (
    make_classical_baseline,
    predictions_to_frame,
    select_fixed_baseline_config,
)
from hqrc_v3.contracts import ForecastMatrix
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
    assert selected.candidate_scores == (3.0, 1.0)


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
