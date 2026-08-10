"""Deterministic CPU sequence-to-sequence forecasting baseline adapters.

The models deliberately receive only observed history and known future covariates.
Targets are used solely by the training loss, never as decoder inputs.
"""

from __future__ import annotations

import copy
import random
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from hqrc_v3.contracts import DataContractError, ForecastMatrix

_MODEL_NAMES = ("lstm", "transformer")
_HISTORY_HOURS = 168
_HORIZONS = 24


@dataclass(frozen=True)
class SequenceTrainingConfig:
    """Fixed, deliberately small surface for deterministic CPU sequence fitting."""

    hidden_size: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 60
    patience: int = 8
    seeds: tuple[int, ...] = (11, 23, 37, 41, 53)


@dataclass(frozen=True)
class Standardizer:
    """Feature-wise standardizer fitted on the training partition only."""

    mean: np.ndarray | float
    scale: np.ndarray | float

    @classmethod
    def fit(cls, values: np.ndarray, *, axes: tuple[int, ...]) -> Standardizer:
        mean = np.mean(values, axis=axes)
        scale = np.std(values, axis=axes)
        return cls(mean=mean, scale=np.where(np.asarray(scale) == 0.0, 1.0, scale))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean) / self.scale

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.scale + self.mean


class Seq2SeqLSTM(nn.Module):
    """Encode the 168 observed steps, then decode known covariates for 24 hours."""

    def __init__(
        self,
        *,
        history_features: int,
        future_features: int,
        hidden_size: int,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_model_dimensions(history_features, future_features, hidden_size, layers, dropout)
        recurrent_dropout = dropout if layers > 1 else 0.0
        self.history_features = history_features
        self.future_features = future_features
        self.encoder = nn.LSTM(
            history_features,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.decoder = nn.LSTM(
            future_features,
            hidden_size,
            num_layers=layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _validate_torch_inputs(history, future, self.history_features, self.future_features)
        _, state = self.encoder(history)
        decoded, _ = self.decoder(future, state)
        return self.output(decoded).squeeze(-1)


class TimeSeriesTransformer(nn.Module):
    """History memory plus known-future covariate queries, without demand leakage."""

    def __init__(
        self,
        *,
        history_features: int,
        future_features: int,
        hidden_size: int,
        layers: int = 2,
        heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_model_dimensions(history_features, future_features, hidden_size, layers, dropout)
        if isinstance(heads, bool) or not isinstance(heads, int) or heads <= 0:
            raise ValueError("heads must be a positive integer")
        if hidden_size % heads:
            raise ValueError("hidden_size must be divisible by heads")
        self.history_features = history_features
        self.future_features = future_features
        self.history_projection = nn.Linear(history_features, hidden_size)
        self.future_projection = nn.Linear(future_features, hidden_size)
        self.history_position = nn.Parameter(torch.zeros(1, _HISTORY_HOURS, hidden_size))
        self.future_position = nn.Parameter(torch.zeros(1, _HORIZONS, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size, heads, hidden_size * 4, dropout=dropout, batch_first=True
        )
        decoder_layer = nn.TransformerDecoderLayer(
            hidden_size, heads, hidden_size * 4, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=layers)
        self.output = nn.Linear(hidden_size, 1)

    def forward(self, history: torch.Tensor, future: torch.Tensor) -> torch.Tensor:
        _validate_torch_inputs(history, future, self.history_features, self.future_features)
        memory = self.encoder(self.history_projection(history) + self.history_position)
        queries = self.future_projection(future) + self.future_position
        decoded = self.decoder(queries, memory)
        return self.output(decoded).squeeze(-1)


@dataclass
class FittedTorchBaseline:
    """One seed-specific fitted neural baseline and its train-only scalers."""

    model_name: str
    model: nn.Module
    history_scaler: Standardizer
    future_scaler: Standardizer
    target_scaler: Standardizer
    history_shape: tuple[int, int]
    future_width: int
    epochs_completed: int
    best_validation_loss: float | None

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        _validate_matrix(batch)
        if (
            batch.history.shape[1:] != self.history_shape
            or batch.future.shape[2] != self.future_width
        ):
            raise DataContractError(
                "prediction matrix feature shape does not match the fitted baseline"
            )
        history = self.history_scaler.transform(batch.history).astype(np.float32)
        future = self.future_scaler.transform(batch.future).astype(np.float32)
        self.model.eval()
        with torch.inference_mode():
            standardized = (
                self.model(torch.from_numpy(history), torch.from_numpy(future)).cpu().numpy()
            )
        prediction = np.asarray(self.target_scaler.inverse_transform(standardized), dtype=float)
        if prediction.shape != batch.target.shape or not np.isfinite(prediction).all():
            raise DataContractError(
                "sequence baseline predictions must be finite with shape (n_samples, 24)"
            )
        return prediction


@dataclass(frozen=True)
class TorchBaselineFactory:
    """Protocol-compatible factory for a fixed LSTM or Transformer configuration."""

    name: str
    config: SequenceTrainingConfig

    def __post_init__(self) -> None:
        normalized = _validate_config_and_name(self.name, self.config)
        object.__setattr__(self, "name", normalized)

    def fit(
        self, train: ForecastMatrix, validation: ForecastMatrix | None, seed: int
    ) -> FittedTorchBaseline:
        normalized_seed = _require_seed(seed)
        _validate_matrix(train)
        if validation is not None:
            _validate_matrix(validation)
            _require_matching_widths(train, validation)
        return _fit_one(self.name, train, validation, self.config, normalized_seed)


@dataclass(frozen=True)
class SeedEnsembleBaseline:
    """Fitted seed members, their individual predictions, and their mean baseline forecast."""

    model_name: str
    member_seeds: tuple[int, ...]
    members: tuple[FittedTorchBaseline, ...]

    def predict_members(self, batch: ForecastMatrix) -> np.ndarray:
        predictions = np.stack([member.predict(batch) for member in self.members], axis=0)
        if predictions.ndim != 3 or predictions.shape[2] != _HORIZONS:
            raise DataContractError("member predictions must have shape (n_seeds, n_samples, 24)")
        return predictions

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        return self.predict_members(batch).mean(axis=0)


def fit_seed_ensemble(
    name: str,
    train: ForecastMatrix,
    validation: ForecastMatrix | None,
    config: SequenceTrainingConfig,
) -> SeedEnsembleBaseline:
    """Fit one member per declared seed; validation affects stopping, never optimizer batches."""

    factory = TorchBaselineFactory(name, config)
    members = tuple(factory.fit(train, validation, seed) for seed in config.seeds)
    return SeedEnsembleBaseline(factory.name, config.seeds, members)


def _fit_one(
    name: str,
    train: ForecastMatrix,
    validation: ForecastMatrix | None,
    config: SequenceTrainingConfig,
    seed: int,
) -> FittedTorchBaseline:
    history_scaler = Standardizer.fit(train.history, axes=(0, 1))
    future_scaler = Standardizer.fit(train.future, axes=(0, 1))
    target_scaler = Standardizer.fit(train.target, axes=(0, 1))
    train_history = torch.from_numpy(history_scaler.transform(train.history).astype(np.float32))
    train_future = torch.from_numpy(future_scaler.transform(train.future).astype(np.float32))
    train_target = torch.from_numpy(target_scaler.transform(train.target).astype(np.float32))

    with _deterministic_cpu_context(seed):
        model = _build_model(name, train.history.shape[2], train.future.shape[2], config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        loader = DataLoader(
            TensorDataset(train_history, train_future, train_target),
            batch_size=min(config.batch_size, len(train_history)),
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
            num_workers=0,
        )
        best_state: dict[str, torch.Tensor] | None = None
        best_loss: float | None = None
        stale_epochs = 0
        epochs_completed = 0
        for _ in range(config.epochs):
            model.train()
            for history, future, target in loader:
                optimizer.zero_grad(set_to_none=True)
                loss = torch.mean((model(history, future) - target) ** 2)
                loss.backward()
                optimizer.step()
            epochs_completed += 1
            if validation is None:
                continue
            validation_loss = _validation_loss(
                model, validation, history_scaler, future_scaler, target_scaler
            )
            if best_loss is None or validation_loss < best_loss:
                best_loss = validation_loss
                best_state = copy.deepcopy(model.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= config.patience:
                    break
        if best_state is not None:
            model.load_state_dict(best_state)
        fitted_model = copy.deepcopy(model).cpu()

    return FittedTorchBaseline(
        model_name=name,
        model=fitted_model,
        history_scaler=history_scaler,
        future_scaler=future_scaler,
        target_scaler=target_scaler,
        history_shape=train.history.shape[1:],
        future_width=train.future.shape[2],
        epochs_completed=epochs_completed,
        best_validation_loss=best_loss,
    )


def _build_model(
    name: str, history_features: int, future_features: int, config: SequenceTrainingConfig
) -> nn.Module:
    common = {
        "history_features": history_features,
        "future_features": future_features,
        "hidden_size": config.hidden_size,
        "layers": config.layers,
        "dropout": config.dropout,
    }
    if name == "lstm":
        return Seq2SeqLSTM(**common)
    return TimeSeriesTransformer(**common, heads=config.heads)


def _validation_loss(
    model: nn.Module,
    validation: ForecastMatrix,
    history_scaler: Standardizer,
    future_scaler: Standardizer,
    target_scaler: Standardizer,
) -> float:
    history = torch.from_numpy(history_scaler.transform(validation.history).astype(np.float32))
    future = torch.from_numpy(future_scaler.transform(validation.future).astype(np.float32))
    target = torch.from_numpy(target_scaler.transform(validation.target).astype(np.float32))
    model.eval()
    with torch.inference_mode():
        value = float(torch.mean((model(history, future) - target) ** 2).item())
    if not np.isfinite(value):
        raise DataContractError("validation loss must be finite")
    return value


def _validate_config_and_name(name: str, config: SequenceTrainingConfig) -> str:
    if not isinstance(name, str):
        raise TypeError("sequence baseline name must be a string")
    normalized = name.lower()
    if normalized not in _MODEL_NAMES:
        choices = ", ".join(_MODEL_NAMES)
        raise ValueError(f"unknown sequence baseline {name!r}; expected one of: {choices}")
    if not isinstance(config, SequenceTrainingConfig):
        raise TypeError("config must be a SequenceTrainingConfig")
    positive_integers = ("hidden_size", "layers", "heads", "batch_size", "epochs", "patience")
    for field in positive_integers:
        value = getattr(config, field)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if config.hidden_size % config.heads:
        raise ValueError("hidden_size must be divisible by heads")
    if not isinstance(config.dropout, (int, float)) or isinstance(config.dropout, bool):
        raise TypeError("dropout must be a number")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if not isinstance(config.learning_rate, (int, float)) or isinstance(config.learning_rate, bool):
        raise TypeError("learning_rate must be a number")
    if not np.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not isinstance(config.seeds, tuple) or not config.seeds:
        raise ValueError("seeds must be a non-empty tuple")
    seeds = tuple(_require_seed(seed) for seed in config.seeds)
    if len(set(seeds)) != len(seeds):
        raise ValueError("seeds must not contain duplicates")
    return normalized


def _validate_model_dimensions(
    history_features: int, future_features: int, hidden_size: int, layers: int, dropout: float
) -> None:
    for name, value in (
        ("history_features", history_features),
        ("future_features", future_features),
        ("hidden_size", hidden_size),
        ("layers", layers),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if not isinstance(dropout, (int, float)) or isinstance(dropout, bool) or not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")


def _validate_torch_inputs(
    history: torch.Tensor, future: torch.Tensor, history_features: int, future_features: int
) -> None:
    if history.ndim != 3 or future.ndim != 3:
        raise ValueError("history and future must be 3-D tensors")
    if history.shape[0] != future.shape[0] or history.shape[1] != _HISTORY_HOURS:
        raise ValueError("history must have shape (n_samples, 168, history_features)")
    if future.shape[1] != _HORIZONS:
        raise ValueError("future must have shape (n_samples, 24, future_features)")
    if history.shape[2] != history_features or future.shape[2] != future_features:
        raise ValueError("tensor feature width does not match the model")


def _validate_matrix(matrix: ForecastMatrix) -> None:
    if not isinstance(matrix, ForecastMatrix):
        raise TypeError("batch must be a ForecastMatrix")
    if matrix.history.ndim != 3 or matrix.future.ndim != 3 or matrix.target.ndim != 2:
        raise DataContractError("ForecastMatrix arrays must be history/future 3-D and target 2-D")
    count = matrix.history.shape[0]
    if count == 0 or matrix.future.shape[0] != count or matrix.target.shape[0] != count:
        raise DataContractError("ForecastMatrix arrays must contain aligned non-empty samples")
    if matrix.history.shape[1] != _HISTORY_HOURS:
        raise DataContractError("sequence baselines require exactly 168 history hours")
    if matrix.future.shape[1] != _HORIZONS or matrix.target.shape[1] != _HORIZONS:
        raise DataContractError("sequence baselines require exactly 24 future horizons")
    if matrix.history.shape[2] <= 0 or matrix.future.shape[2] <= 0:
        raise DataContractError("ForecastMatrix feature widths must be positive")
    if (
        len(matrix.history_columns) != matrix.history.shape[2]
        or len(matrix.future_columns) != matrix.future.shape[2]
    ):
        raise DataContractError("ForecastMatrix feature columns must match feature widths")
    if matrix.origins.shape != (count,) or matrix.target_times.shape != matrix.target.shape:
        raise DataContractError("ForecastMatrix origins and target times must align with samples")
    for values, description in (
        (matrix.history, "history"),
        (matrix.future, "future"),
        (matrix.target, "target"),
    ):
        if not np.isfinite(np.asarray(values, dtype=float)).all():
            raise DataContractError(f"ForecastMatrix {description} must be finite")


def _require_matching_widths(train: ForecastMatrix, validation: ForecastMatrix) -> None:
    if (
        train.history.shape[1:] != validation.history.shape[1:]
        or train.future.shape[2] != validation.future.shape[2]
    ):
        raise DataContractError("validation matrix feature shape must match the training matrix")


def _require_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer, not a boolean or fractional value")
    return int(seed)


@contextmanager
def _deterministic_cpu_context(seed: int) -> Iterator[None]:
    """Temporarily constrain RNG and Torch CPU execution, restoring process state afterwards."""

    python_state = random.getstate()
    numpy_state = np.random.get_state()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    with torch.random.fork_rng(devices=[]):
        try:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.set_num_threads(1)
            torch.use_deterministic_algorithms(True)
            yield
        finally:
            torch.use_deterministic_algorithms(deterministic)
            torch.set_num_threads(threads)
            random.setstate(python_state)
            np.random.set_state(numpy_state)
