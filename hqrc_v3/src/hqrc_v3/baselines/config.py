"""Strict, immutable configuration for the five manuscript baselines."""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hqrc_v3.provenance import ArtifactMismatch, file_sha256

MODEL_NAMES = (
    "xgboost",
    "lightgbm",
    "svr",
    "seq2seq_lstm",
    "transformer",
)
PAPER_SEEDS = (11, 23, 37, 41, 53)
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_OBSERVED_COLUMNS = ("load_mw", "temperature_c", "relative_humidity")
_B0_FUTURE_COLUMNS = (
    "hour",
    "day_of_week",
    "is_weekend",
    "annual_sin",
    "annual_cos",
    "weekly_sin",
    "weekly_cos",
)
_B1_ONLY_COLUMNS = (
    "is_public_holiday",
    "official_sequence_position",
    "seollal_distance",
    "chuseok_distance",
    "is_substitute_or_temporary_holiday",
    "is_seollal",
    "is_chuseok",
)


class PaperBaselineConfigError(ValueError):
    """Raised when the frozen manuscript registry differs from its contract."""


def _require_exact(value: object, expected: object, field: str) -> None:
    if type(value) is not type(expected) or value != expected:
        raise PaperBaselineConfigError(f"{field} must remain frozen at {expected!r}")


def _require_keys(table: dict[str, Any], expected: set[str], name: str) -> None:
    if set(table) != expected:
        raise PaperBaselineConfigError(
            f"{name} keys differ from the frozen registry: expected {sorted(expected)}"
        )


@dataclass(frozen=True)
class XGBoostPaperConfig:
    objective: str
    eval_metric: str
    n_estimators: int
    learning_rate: float
    max_depth: int
    min_child_weight: int
    subsample: float
    colsample_bytree: float
    reg_lambda: float
    tree_method: str
    early_stopping_rounds: int

    def __post_init__(self) -> None:
        expected = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "n_estimators": 600,
            "learning_rate": 0.03,
            "max_depth": 6,
            "min_child_weight": 3,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "early_stopping_rounds": 35,
        }
        for field, value in expected.items():
            _require_exact(getattr(self, field), value, f"xgboost.{field}")

    def to_estimator_params(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "min_child_weight": self.min_child_weight,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "tree_method": self.tree_method,
            "early_stopping_rounds": self.early_stopping_rounds,
        }


@dataclass(frozen=True)
class LightGBMPaperConfig:
    objective: str
    n_estimators: int
    learning_rate: float
    num_leaves: int
    min_child_samples: int
    subsample: float
    subsample_freq: int
    colsample_bytree: float
    reg_lambda: float
    early_stopping_rounds: int

    def __post_init__(self) -> None:
        expected = {
            "objective": "regression",
            "n_estimators": 800,
            "learning_rate": 0.025,
            "num_leaves": 31,
            "min_child_samples": 20,
            "subsample": 0.9,
            "subsample_freq": 1,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "early_stopping_rounds": 35,
        }
        for field, value in expected.items():
            _require_exact(getattr(self, field), value, f"lightgbm.{field}")

    def to_estimator_params(self) -> dict[str, object]:
        return {
            "objective": self.objective,
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
        }


@dataclass(frozen=True)
class SVRPaperConfig:
    kernel: str
    C: float
    epsilon: float
    gamma: str
    cache_size: float

    def __post_init__(self) -> None:
        expected = {
            "kernel": "rbf",
            "C": 10.0,
            "epsilon": 0.05,
            "gamma": "scale",
            "cache_size": 512.0,
        }
        for field, value in expected.items():
            _require_exact(getattr(self, field), value, f"svr.{field}")

    def to_estimator_params(self) -> dict[str, object]:
        return {
            "kernel": self.kernel,
            "C": self.C,
            "epsilon": self.epsilon,
            "gamma": self.gamma,
            "cache_size": self.cache_size,
        }


@dataclass(frozen=True)
class NeuralPaperConfig:
    model_name: str
    hidden_size: int
    layers: int
    heads: int
    dropout: float
    learning_rate: float
    batch_size: int
    epochs: int
    patience: int
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.model_name not in {"seq2seq_lstm", "transformer"}:
            raise PaperBaselineConfigError("neural model_name must identify a paper model")
        expected = {
            "hidden_size": 64,
            "layers": 2 if self.model_name == "seq2seq_lstm" else 1,
            "heads": 4,
            "dropout": 0.1,
            "learning_rate": 1e-3,
            "batch_size": 64,
            "epochs": 60,
            "patience": 8,
            "seeds": PAPER_SEEDS,
        }
        for field, value in expected.items():
            _require_exact(getattr(self, field), value, f"{self.model_name}.{field}")


@dataclass(frozen=True)
class PreprocessingPaperConfig:
    version: str
    history_hours: int
    future_path_hours: int
    observed_columns: tuple[str, ...]
    b0_future_columns: tuple[str, ...]
    b1_only_columns: tuple[str, ...]
    classical_input_scaler: str
    target_scaler: str
    sequence_weather_scaler: str
    sequence_calendar_scaler: str
    scaler_fit_partition: str
    classical_future_path: str
    classical_x_population: str
    target_population: str
    sequence_weather_population: str
    sequence_calendar_population: str

    def __post_init__(self) -> None:
        expected = {
            "version": "causal-v1",
            "history_hours": 168,
            "future_path_hours": 24,
            "observed_columns": _OBSERVED_COLUMNS,
            "b0_future_columns": _B0_FUTURE_COLUMNS,
            "b1_only_columns": _B1_ONLY_COLUMNS,
            "classical_input_scaler": "standard-population-featurewise",
            "target_scaler": "standard-population",
            "sequence_weather_scaler": "standard-population-featurewise",
            "sequence_calendar_scaler": "standard-population-featurewise",
            "scaler_fit_partition": "estimator-fit-only",
            "classical_future_path": "full",
            "classical_x_population": "daily-sample-rows",
            "target_population": "unique-target-hours",
            "sequence_weather_population": "unique-inferred-history-hours",
            "sequence_calendar_population": "unique-future-hours-shared-history-future",
        }
        for field, value in expected.items():
            _require_exact(getattr(self, field), value, f"preprocessing.{field}")

    def to_manifest(self) -> dict[str, object]:
        return {
            field: list(value) if isinstance(value, tuple) else value
            for field, value in self.__dict__.items()
        }


@dataclass(frozen=True)
class PaperBaselineConfig:
    schema_version: int
    models: tuple[str, ...]
    validation_days: int
    xgboost: XGBoostPaperConfig
    lightgbm: LightGBMPaperConfig
    svr: SVRPaperConfig
    seq2seq_lstm: NeuralPaperConfig
    transformer: NeuralPaperConfig
    preprocessing: PreprocessingPaperConfig
    source_sha256: str

    def __post_init__(self) -> None:
        _require_exact(self.schema_version, 2, "schema_version")
        _require_exact(self.models, MODEL_NAMES, "models")
        _require_exact(self.validation_days, 61, "validation_days")
        if not isinstance(self.source_sha256, str) or _SHA256.fullmatch(self.source_sha256) is None:
            raise PaperBaselineConfigError("source_sha256 must be a lowercase SHA-256 digest")


def _table(document: dict[str, Any], name: str, fields: tuple[str, ...]) -> dict[str, Any]:
    value = document.get(name)
    if not isinstance(value, dict):
        raise PaperBaselineConfigError(f"missing or invalid [{name}] table")
    _require_keys(value, set(fields), f"[{name}]")
    return value


def _models(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PaperBaselineConfigError("models must be an array of strings")
    return tuple(value)


def _seeds(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list) or any(type(item) is not int for item in value):
        raise PaperBaselineConfigError(f"{name}.seeds must be an array of integers")
    return tuple(value)


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise PaperBaselineConfigError(f"{name} must be an array of strings")
    return tuple(value)


def load_paper_baselines(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> PaperBaselineConfig:
    """Load and hash-check the one exact manuscript baseline registry."""

    source = Path(path)
    try:
        actual_sha256 = file_sha256(source)
    except OSError as error:
        raise PaperBaselineConfigError(
            f"unable to load paper baseline config {source}: {error}"
        ) from error
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ArtifactMismatch("model config SHA-256 differs from the requested frozen hash")
    try:
        with source.open("rb") as stream:
            document = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise PaperBaselineConfigError(
            f"unable to load paper baseline config {source}: {error}"
        ) from error

    root_fields = {
        "schema_version",
        "models",
        "validation_days",
        "xgboost",
        "lightgbm",
        "svr",
        "seq2seq_lstm",
        "transformer",
        "preprocessing",
    }
    _require_keys(document, root_fields, "root")
    xgboost_fields = tuple(XGBoostPaperConfig.__dataclass_fields__)
    lightgbm_fields = tuple(LightGBMPaperConfig.__dataclass_fields__)
    svr_fields = tuple(SVRPaperConfig.__dataclass_fields__)
    neural_fields = tuple(
        field for field in NeuralPaperConfig.__dataclass_fields__ if field != "model_name"
    )
    xgboost = _table(document, "xgboost", xgboost_fields)
    lightgbm = _table(document, "lightgbm", lightgbm_fields)
    svr = _table(document, "svr", svr_fields)
    seq2seq_lstm = _table(document, "seq2seq_lstm", neural_fields)
    transformer = _table(document, "transformer", neural_fields)
    preprocessing_fields = tuple(PreprocessingPaperConfig.__dataclass_fields__)
    preprocessing = _table(document, "preprocessing", preprocessing_fields)
    return PaperBaselineConfig(
        schema_version=document.get("schema_version"),
        models=_models(document.get("models")),
        validation_days=document.get("validation_days"),
        xgboost=XGBoostPaperConfig(**xgboost),
        lightgbm=LightGBMPaperConfig(**lightgbm),
        svr=SVRPaperConfig(**svr),
        seq2seq_lstm=NeuralPaperConfig(
            model_name="seq2seq_lstm",
            seeds=_seeds(seq2seq_lstm.pop("seeds"), "seq2seq_lstm"),
            **seq2seq_lstm,
        ),
        transformer=NeuralPaperConfig(
            model_name="transformer",
            seeds=_seeds(transformer.pop("seeds"), "transformer"),
            **transformer,
        ),
        preprocessing=PreprocessingPaperConfig(
            observed_columns=_strings(
                preprocessing.pop("observed_columns"), "preprocessing.observed_columns"
            ),
            b0_future_columns=_strings(
                preprocessing.pop("b0_future_columns"),
                "preprocessing.b0_future_columns",
            ),
            b1_only_columns=_strings(
                preprocessing.pop("b1_only_columns"),
                "preprocessing.b1_only_columns",
            ),
            **preprocessing,
        ),
        source_sha256=actual_sha256,
    )
