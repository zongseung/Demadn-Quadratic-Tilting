from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch
from hqrc_v3.baselines.sequence import (
    Seq2SeqLSTM,
    SequenceTrainingConfig,
    TimeSeriesTransformer,
    TorchBaselineFactory,
)
from hqrc_v3.contracts import DataContractError


@pytest.fixture
def tiny_forecast_matrix():
    from hqrc_v3.contracts import ForecastMatrix

    count = 24
    origins = np.arange("2023-01-01", "2023-01-25", dtype="datetime64[D]").astype("datetime64[ns]")
    target_times = origins[:, None] + np.arange(24).astype("timedelta64[h]")
    sample = np.arange(count, dtype=float)[:, None, None]
    hour = np.arange(24, dtype=float)[None, :, None]
    history_hour = np.arange(168, dtype=float)[None, :, None]
    history = np.concatenate(
        (sample + history_hour / 100, np.broadcast_to(history_hour / 10, (count, 168, 1))), axis=2
    )
    future = np.concatenate(
        (np.broadcast_to(hour / 24, (count, 24, 1)), sample + hour / 50), axis=2
    )
    target = sample[:, 0, 0, None] + hour[:, :, 0] + 0.5
    return ForecastMatrix(
        origins, target_times, history, future, target, ("load", "temperature"), ("hour", "weather")
    )


@pytest.mark.parametrize("model_cls", [Seq2SeqLSTM, TimeSeriesTransformer])
def test_sequence_forward_shape(model_cls):
    model = model_cls(history_features=5, future_features=3, hidden_size=8, dropout=0.0)

    result = model(torch.zeros(2, 168, 5), torch.zeros(2, 24, 3))

    assert result.shape == (2, 24)


@pytest.mark.parametrize("model_cls", [Seq2SeqLSTM, TimeSeriesTransformer])
def test_sequence_forward_uses_known_future_covariates_without_target_input(model_cls):
    torch.manual_seed(3)
    model = model_cls(history_features=2, future_features=2, hidden_size=8, dropout=0.0)
    model.eval()
    history = torch.zeros(2, 168, 2)

    unchanged = model(history, torch.zeros(2, 24, 2))
    changed = model(history, torch.ones(2, 24, 2))

    assert not torch.allclose(unchanged, changed)


@pytest.mark.parametrize(
    "config",
    [
        SequenceTrainingConfig(hidden_size=7, heads=2),
        SequenceTrainingConfig(epochs=0),
        SequenceTrainingConfig(seeds=(True,)),
        SequenceTrainingConfig(seeds=(3, 3)),
    ],
)
def test_sequence_config_rejects_invalid_values(config):
    with pytest.raises((TypeError, ValueError)):
        TorchBaselineFactory("lstm", config)


def test_sequence_factory_rejects_unknown_model_name():
    with pytest.raises(ValueError, match="unknown sequence baseline"):
        TorchBaselineFactory("gru", SequenceTrainingConfig())


def test_sequence_fit_rejects_invalid_matrix_shape(tiny_forecast_matrix):
    malformed = replace(tiny_forecast_matrix, history=tiny_forecast_matrix.history[:, :167, :])

    with pytest.raises(DataContractError, match="168 history hours"):
        TorchBaselineFactory("lstm", SequenceTrainingConfig(epochs=1)).fit(
            malformed, validation=None, seed=3
        )


def test_sequence_fit_uses_train_only_standardization(tiny_forecast_matrix):
    train = tiny_forecast_matrix.take(np.arange(16))
    validation = tiny_forecast_matrix.take(np.arange(16, 24))
    config = SequenceTrainingConfig(
        hidden_size=8, layers=1, heads=2, dropout=0.0, epochs=1, batch_size=8, seeds=(3,)
    )
    fitted = TorchBaselineFactory("lstm", config).fit(train, validation, seed=3)

    np.testing.assert_allclose(fitted.history_scaler.mean, train.history.mean(axis=(0, 1)))
    np.testing.assert_allclose(fitted.future_scaler.mean, train.future.mean(axis=(0, 1)))
    np.testing.assert_allclose(fitted.target_scaler.mean, train.target.mean())
    assert fitted.predict(validation).shape == (8, 24)


def test_sequence_fitted_predict_requires_fitted_feature_width(tiny_forecast_matrix):
    config = SequenceTrainingConfig(
        hidden_size=8, layers=1, heads=2, dropout=0.0, epochs=1, batch_size=8, seeds=(3,)
    )
    fitted = TorchBaselineFactory("transformer", config).fit(
        tiny_forecast_matrix.take(np.arange(16)), validation=None, seed=3
    )
    malformed = replace(
        tiny_forecast_matrix,
        future=tiny_forecast_matrix.future[:, :, :1],
        future_columns=("hour",),
    )

    with pytest.raises(DataContractError, match="feature shape"):
        fitted.predict(malformed)


@pytest.mark.parametrize("stream", ["history", "future"])
def test_sequence_fit_rejects_reordered_validation_feature_columns(
    tiny_forecast_matrix, stream
):
    train = tiny_forecast_matrix.take(np.arange(16))
    validation = tiny_forecast_matrix.take(np.arange(16, 24))
    if stream == "history":
        validation = replace(
            validation,
            history=validation.history[:, :, ::-1],
            history_columns=validation.history_columns[::-1],
        )
    else:
        validation = replace(
            validation,
            future=validation.future[:, :, ::-1],
            future_columns=validation.future_columns[::-1],
        )
    config = SequenceTrainingConfig(
        hidden_size=8, layers=1, heads=2, dropout=0.0, epochs=1, batch_size=8, seeds=(3,)
    )

    with pytest.raises(DataContractError, match="feature columns/order"):
        TorchBaselineFactory("lstm", config).fit(train, validation, seed=3)


@pytest.mark.parametrize("stream", ["history", "future"])
def test_sequence_predict_rejects_reordered_feature_columns(tiny_forecast_matrix, stream):
    train = tiny_forecast_matrix.take(np.arange(16))
    batch = tiny_forecast_matrix.take(np.arange(16, 24))
    config = SequenceTrainingConfig(
        hidden_size=8, layers=1, heads=2, dropout=0.0, epochs=1, batch_size=8, seeds=(3,)
    )
    fitted = TorchBaselineFactory("lstm", config).fit(train, validation=None, seed=3)
    if stream == "history":
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


@pytest.mark.parametrize("columns", [("load", "load"), ("load", " ")])
def test_sequence_fit_rejects_duplicate_or_blank_feature_columns(
    tiny_forecast_matrix, columns
):
    malformed = replace(tiny_forecast_matrix, history_columns=columns)

    with pytest.raises(DataContractError, match="feature column names"):
        TorchBaselineFactory("lstm", SequenceTrainingConfig(epochs=1)).fit(
            malformed, validation=None, seed=3
        )
