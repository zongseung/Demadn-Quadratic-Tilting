from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np
import pytest
import torch

import hqrc_v3.baselines.sequence as sequence
from hqrc_v3.baselines.sequence import SequenceTrainingConfig, fit_seed_ensemble
from hqrc_v3.contracts import DataContractError


@pytest.fixture
def tiny_forecast_matrix():
    from hqrc_v3.contracts import ForecastMatrix

    count = 24
    origins = np.arange("2023-01-01", "2023-01-25", dtype="datetime64[D]").astype("datetime64[ns]")
    target_times = origins[:, None] + np.arange(24).astype("timedelta64[h]")
    sample = np.arange(count, dtype=float)[:, None, None]
    hour = np.arange(24, dtype=float)[None, :, None]
    history_hour = np.arange(-168, 0, dtype=float)[None, :, None]
    absolute_history_hour = sample * 24.0 + history_hour
    history = np.concatenate(
        (
            50_000.0 + absolute_history_hour,
            10.0 + absolute_history_hour / 168.0,
            50.0 + absolute_history_hour / 336.0,
            np.mod(absolute_history_hour, 24.0),
            absolute_history_hour / 24.0,
        ),
        axis=2,
    )
    future = np.concatenate(
        (np.broadcast_to(hour / 24, (count, 24, 1)), sample + hour / 50), axis=2
    )
    target = 51_000.0 + sample[:, 0, 0, None] * 24.0 + hour[:, :, 0]
    return ForecastMatrix(
        origins,
        target_times,
        history,
        future,
        target,
        (
            "load_mw",
            "temperature_c",
            "relative_humidity",
            "hour",
            "annual_sin",
        ),
        ("hour", "annual_sin"),
    )


def _config(*, epochs: int = 2, patience: int = 2) -> SequenceTrainingConfig:
    return SequenceTrainingConfig(
        hidden_size=8,
        layers=1,
        heads=2,
        dropout=0.0,
        learning_rate=1e-2,
        batch_size=8,
        epochs=epochs,
        patience=patience,
        seeds=(3, 5),
    )


def test_seed_ensemble_is_reproducible(tiny_forecast_matrix):
    train = tiny_forecast_matrix.take(np.arange(16))
    validation = tiny_forecast_matrix.take(np.arange(16, 24))

    first = fit_seed_ensemble("lstm", train, validation, _config()).predict(validation)
    second = fit_seed_ensemble("lstm", train, validation, _config()).predict(validation)

    np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)


def test_seed_ensemble_retains_member_predictions_and_mean_identity(tiny_forecast_matrix):
    train = tiny_forecast_matrix.take(np.arange(16))
    ensemble = fit_seed_ensemble("transformer", train, None, _config(epochs=1))

    member_predictions = ensemble.predict_members(tiny_forecast_matrix.take(np.arange(16, 24)))

    assert ensemble.member_seeds == (3, 5)
    assert len(ensemble.members) == 2
    assert member_predictions.shape == (2, 8, 24)
    np.testing.assert_allclose(
        member_predictions.mean(axis=0),
        ensemble.predict(tiny_forecast_matrix.take(np.arange(16, 24))),
    )


def test_seed_ensemble_prediction_apis_reject_reordered_feature_columns(
    tiny_forecast_matrix,
):
    train = tiny_forecast_matrix.take(np.arange(16))
    ensemble = fit_seed_ensemble("lstm", train, None, _config(epochs=1))
    batch = tiny_forecast_matrix.take(np.arange(16, 24))
    reordered = replace(
        batch,
        future=batch.future[:, :, ::-1],
        future_columns=batch.future_columns[::-1],
    )

    with pytest.raises(DataContractError, match="feature columns/order"):
        ensemble.predict_members(reordered)
    with pytest.raises(DataContractError, match="feature columns/order"):
        ensemble.predict(reordered)


def test_validation_restores_best_state_and_no_validation_runs_fixed_epochs(
    monkeypatch, tiny_forecast_matrix
):
    train = tiny_forecast_matrix.take(np.arange(16))
    validation = tiny_forecast_matrix.take(np.arange(16, 24))

    best_state: dict[str, torch.Tensor] = {}

    def controlled_validation_loss(model, *_):
        if not best_state:
            best_state.update(copy.deepcopy(model.state_dict()))
            return 0.0
        return 1.0

    monkeypatch.setattr(sequence, "_validation_loss", controlled_validation_loss)
    config = SequenceTrainingConfig(
        hidden_size=8,
        layers=1,
        heads=2,
        dropout=0.0,
        learning_rate=1e-2,
        batch_size=8,
        epochs=4,
        patience=1,
        seeds=(3,),
    )
    with_validation = fit_seed_ensemble("lstm", train, validation, config)
    without_validation = fit_seed_ensemble("lstm", train, None, _config(epochs=3, patience=1))

    member = with_validation.members[0]
    assert member.best_validation_loss == 0.0
    assert member.epochs_completed == 2
    assert all(
        torch.equal(member.model.state_dict()[name], value) for name, value in best_state.items()
    )
    assert without_validation.members[0].epochs_completed == 3
