from __future__ import annotations

from dataclasses import replace

import hqrc_v3.baselines.sequence as sequence
import numpy as np
import pytest
import torch
from hqrc_v3.baselines.sequence import (
    Seq2SeqLSTM,
    SequenceTrainingConfig,
    TimeSeriesTransformer,
    TorchBaselineFactory,
)
from hqrc_v3.contracts import DataContractError, validate_forecast_feature_columns


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


def _change_feature_schema(matrix, stream, change):
    values = getattr(matrix, stream)
    columns = getattr(matrix, f"{stream}_columns")
    if change == "reordered":
        return replace(
            matrix,
            **{stream: values[:, :, ::-1], f"{stream}_columns": columns[::-1]},
        )
    return replace(
        matrix,
        **{f"{stream}_columns": (f"renamed_{columns[0]}", *columns[1:])},
    )


@pytest.mark.parametrize(("enabled", "warn_only"), [(True, True), (False, False)])
def test_deterministic_context_restores_caller_determinism_mode(enabled, warn_only):
    original_enabled = torch.are_deterministic_algorithms_enabled()
    original_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)
        with sequence._deterministic_cpu_context(seed=3):
            assert torch.are_deterministic_algorithms_enabled()
        assert torch.are_deterministic_algorithms_enabled() is enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() is warn_only
    finally:
        torch.use_deterministic_algorithms(original_enabled, warn_only=original_warn_only)


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

    unique_history_weather = np.concatenate(
        (train.history[0, :, 1:3], train.history[1:, -24:, 1:3].reshape(-1, 2)),
        axis=0,
    )
    np.testing.assert_allclose(
        fitted.weather_scaler.mean, unique_history_weather.mean(axis=0)
    )
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


@pytest.mark.parametrize("change", ["reordered", "renamed"])
@pytest.mark.parametrize("stream", ["history", "future"])
def test_sequence_fit_rejects_changed_validation_feature_columns(
    tiny_forecast_matrix, stream, change
):
    train = tiny_forecast_matrix.take(np.arange(16))
    validation = tiny_forecast_matrix.take(np.arange(16, 24))
    validation = _change_feature_schema(validation, stream, change)
    config = SequenceTrainingConfig(
        hidden_size=8, layers=1, heads=2, dropout=0.0, epochs=1, batch_size=8, seeds=(3,)
    )

    with pytest.raises(DataContractError, match="feature columns/order"):
        TorchBaselineFactory("lstm", config).fit(train, validation, seed=3)


@pytest.mark.parametrize("change", ["reordered", "renamed"])
@pytest.mark.parametrize("stream", ["history", "future"])
def test_sequence_predict_rejects_changed_feature_columns(
    tiny_forecast_matrix, stream, change
):
    train = tiny_forecast_matrix.take(np.arange(16))
    batch = tiny_forecast_matrix.take(np.arange(16, 24))
    config = SequenceTrainingConfig(
        hidden_size=8, layers=1, heads=2, dropout=0.0, epochs=1, batch_size=8, seeds=(3,)
    )
    fitted = TorchBaselineFactory("lstm", config).fit(train, validation=None, seed=3)
    batch = _change_feature_schema(batch, stream, change)

    with pytest.raises(DataContractError, match="feature columns/order"):
        fitted.predict(batch)


@pytest.mark.parametrize(
    "columns",
    [
        ("load_mw", "temperature_c", "relative_humidity", "hour", "hour"),
        ("load_mw", "temperature_c", "relative_humidity", "hour", " "),
    ],
)
def test_sequence_fit_rejects_duplicate_or_blank_feature_columns(
    tiny_forecast_matrix, columns
):
    malformed = replace(tiny_forecast_matrix, history_columns=columns)

    with pytest.raises(DataContractError, match="feature column names"):
        TorchBaselineFactory("lstm", SequenceTrainingConfig(epochs=1)).fit(
            malformed, validation=None, seed=3
        )


@pytest.mark.parametrize("stream", ["history", "future"])
def test_shared_feature_validator_rejects_non_tuple_column_collections(
    tiny_forecast_matrix, stream
):
    columns = list(getattr(tiny_forecast_matrix, f"{stream}_columns"))
    malformed = replace(tiny_forecast_matrix, **{f"{stream}_columns": columns})

    with pytest.raises(DataContractError, match=f"{stream} feature columns must be a tuple"):
        validate_forecast_feature_columns(malformed)
