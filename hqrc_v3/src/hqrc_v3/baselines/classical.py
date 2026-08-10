"""Fixed-parameter classical 24-horizon forecasting adapters."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from hqrc_v3.contracts import (
    DataContractError,
    ForecastMatrix,
    validate_forecast_feature_columns,
    validate_prediction_frame,
)
from hqrc_v3.splits import AnnualFold

_MODEL_NAMES = ("xgboost", "lightgbm", "svr")
_ADAPTER_MANAGED_PARAMS = frozenset({"random_state", "n_jobs", "nthread", "verbosity"})


def _estimator_class(name: str) -> type[Any]:
    if name == "xgboost":
        try:
            from xgboost import XGBRegressor
        except ImportError as error:  # pragma: no cover - depends on installation
            raise ImportError("xgboost baseline requires `xgboost` to be installed") from error
        return XGBRegressor
    if name == "lightgbm":
        try:
            from lightgbm import LGBMRegressor
        except ImportError as error:  # pragma: no cover - depends on installation
            raise ImportError("lightgbm baseline requires `lightgbm` to be installed") from error
        return LGBMRegressor
    if name == "svr":
        try:
            from sklearn.svm import SVR
        except ImportError as error:  # pragma: no cover - depends on installation
            raise ImportError("svr baseline requires `scikit-learn` to be installed") from error
        return SVR
    choices = ", ".join(_MODEL_NAMES)
    raise ValueError(f"unknown classical baseline {name!r}; expected one of: {choices}")


def _validate_params(name: str, params: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(params, Mapping):
        raise TypeError("classical baseline params must be a mapping")
    estimator = _estimator_class(name)()
    supported = set(estimator.get_params(deep=False))
    unsupported = sorted(set(params) - supported)
    if unsupported:
        raise ValueError(f"unsupported parameters for {name}: {', '.join(unsupported)}")
    managed = sorted(set(params) & _ADAPTER_MANAGED_PARAMS)
    if managed:
        raise ValueError(
            "parameters are managed by the baseline adapter for deterministic fitting: "
            f"{', '.join(managed)}"
        )
    return dict(params)


def _flatten_history(matrix: ForecastMatrix) -> np.ndarray:
    """Flatten the 168-hour observed history once per matrix."""

    if matrix.history.ndim != 3 or matrix.future.ndim != 3 or matrix.target.ndim != 2:
        raise DataContractError("ForecastMatrix arrays must be history/future 3-D and target 2-D")
    if (
        matrix.history.shape[0] != matrix.future.shape[0]
        or matrix.history.shape[0] != matrix.target.shape[0]
    ):
        raise DataContractError("ForecastMatrix arrays must have the same sample count")
    if matrix.history.shape[1] != 168:
        raise DataContractError("classical baselines require exactly 168 history hours")
    if matrix.future.shape[1] != 24 or matrix.target.shape[1] != 24:
        raise DataContractError("classical baselines require exactly 24 future horizons")
    validate_forecast_feature_columns(matrix)
    return matrix.history.reshape(matrix.history.shape[0], -1)


def _features_for_horizon(
    flattened_history: np.ndarray, matrix: ForecastMatrix, horizon: int
) -> np.ndarray:
    """Append only covariates known at the requested forecast horizon."""

    return np.concatenate((flattened_history, matrix.future[:, horizon, :]), axis=1)


def _validate_finite(values: np.ndarray, *, description: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise DataContractError(f"{description} must be finite")
    return array


def _estimator_input(model_name: str, features: np.ndarray) -> Any:
    """Match LightGBM's named fit representation at prediction time."""

    if model_name != "lightgbm":
        return features
    try:
        import pandas as pd
    except ImportError as error:  # pragma: no cover - depends on installation
        raise ImportError("lightgbm baseline requires `pandas` for stable feature names") from error
    columns = [f"feature_{index}" for index in range(features.shape[1])]
    return pd.DataFrame(features, columns=columns, copy=False)


def _require_integer_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise TypeError("seed must be an integer, not a boolean or fractional value")
    return int(seed)


def _require_nonblank_string(value: str, *, description: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{description} must be a nonblank string")
    return value


def _json_copy(value: Mapping[str, Any], *, description: str) -> dict[str, Any]:
    """Return a detached JSON-compatible copy, failing before an artifact is ambiguous."""

    try:
        encoded = json.dumps(deepcopy(dict(value)), sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} must be JSON serializable") from error
    decoded = json.loads(encoded)
    if not isinstance(decoded, dict):  # pragma: no cover - ``dict(value)`` guarantees this
        raise ValueError(f"{description} must serialize to an object")
    return decoded


@dataclass(frozen=True)
class HorizonRegressor:
    """Twenty-four independently fitted estimators, one for each target hour."""

    model_name: str
    estimators: tuple[Any, ...]
    history_shape: tuple[int, int]
    future_width: int
    history_columns: tuple[str, ...]
    future_columns: tuple[str, ...]

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        if len(self.estimators) != 24:
            raise RuntimeError("a horizon regressor must contain exactly 24 estimators")
        flattened_history = _flatten_history(batch)
        if (
            batch.history.shape[1:] != self.history_shape
            or batch.future.shape[2] != self.future_width
        ):
            raise DataContractError(
                "prediction matrix feature shape does not match the fitted baseline"
            )
        if (
            batch.history_columns != self.history_columns
            or batch.future_columns != self.future_columns
        ):
            raise DataContractError(
                "prediction matrix feature columns/order do not match the fitted baseline"
            )
        prediction = np.empty((batch.target.shape[0], 24), dtype=float)
        for horizon, estimator in enumerate(self.estimators):
            features = _features_for_horizon(flattened_history, batch, horizon)
            prediction[:, horizon] = estimator.predict(
                _estimator_input(self.model_name, features)
            )
        return _validate_finite(prediction, description="baseline predictions")


@dataclass(frozen=True)
class ClassicalBaseline:
    """Factory for a fixed-parameter, independent-horizon classical baseline."""

    name: str
    params: dict[str, Any]

    def fit(
        self, train: ForecastMatrix, validation: ForecastMatrix | None, seed: int
    ) -> HorizonRegressor:
        del validation  # These fixed adapters never inspect an evaluation partition while fitting.
        normalized_seed = _require_integer_seed(seed)
        target = _validate_finite(train.target, description="training target")
        estimator_class = _estimator_class(self.name)
        estimators: list[Any] = []
        flattened_history = _flatten_history(train)
        for horizon in range(24):
            features = _validate_finite(
                _features_for_horizon(flattened_history, train, horizon),
                description="training features",
            )
            estimator_params = dict(self.params)
            available = estimator_class().get_params(deep=False)
            if "random_state" in available:
                estimator_params["random_state"] = normalized_seed
            if "n_jobs" in available:
                estimator_params["n_jobs"] = 1
            if "nthread" in available:
                estimator_params["nthread"] = 1
            if self.name == "xgboost" and "verbosity" in available:
                estimator_params["verbosity"] = 0
            if self.name == "lightgbm":
                estimator_params["verbosity"] = -1
            estimator = estimator_class(**estimator_params)
            estimator.fit(_estimator_input(self.name, features), target[:, horizon])
            estimators.append(estimator)
        return HorizonRegressor(
            model_name=self.name,
            estimators=tuple(estimators),
            history_shape=train.history.shape[1:],
            future_width=train.future.shape[2],
            history_columns=tuple(train.history_columns),
            future_columns=tuple(train.future_columns),
        )


def make_classical_baseline(name: str, params: Mapping[str, Any]) -> ClassicalBaseline:
    """Create a fixed-config adapter without any search or tuning behaviour."""

    if not isinstance(name, str):
        raise TypeError("classical baseline name must be a string")
    normalized_name = name.lower()
    return ClassicalBaseline(name=normalized_name, params=_validate_params(normalized_name, params))


@dataclass(frozen=True)
class CandidateScore:
    """One declared candidate and its independently interpretable validation RMSE."""

    params: dict[str, Any]
    rmse: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "params": _json_copy(self.params, description="candidate params"),
            "rmse": self.rmse,
        }


@dataclass(frozen=True)
class FixedBaselineSelection:
    """Serializable result of the single allowed 2019--2022/2023 configuration choice."""

    params: dict[str, Any]
    candidate_scores: tuple[CandidateScore, ...]
    selected_index: int
    selected_score: float
    metric: str = "rmse"

    def to_dict(self) -> dict[str, Any]:
        """Export an ordered, deterministic artifact record suitable for JSON serialization."""

        return {
            "params": _json_copy(self.params, description="selected params"),
            "metric": self.metric,
            "selected_index": self.selected_index,
            "selected_score": self.selected_score,
            "candidate_scores": [entry.to_dict() for entry in self.candidate_scores],
        }


def select_fixed_baseline_config(
    *,
    candidates: tuple[Mapping[str, Any], ...] | list[Mapping[str, Any]],
    factory_builder: Callable[[Mapping[str, Any]], Any],
    train: ForecastMatrix,
    validation: ForecastMatrix,
    seed: int,
) -> FixedBaselineSelection:
    """Score declared candidates once, choosing the first candidate with the lowest validation RMSE.

    Callers are responsible for supplying the immutable 2019--2022 training matrix and the
    2023 validation matrix.  The function intentionally has no relation to OOF fitting.
    """

    if not candidates:
        raise ValueError("at least one fixed baseline candidate is required")
    serialized_candidates = tuple(
        _json_copy(params, description="candidate params") for params in candidates
    )
    scores: list[CandidateScore] = []
    for params in serialized_candidates:
        fitted = factory_builder(params).fit(train, validation=None, seed=seed)
        prediction = _validate_finite(
            fitted.predict(validation), description="validation predictions"
        )
        if prediction.shape != validation.target.shape:
            raise DataContractError("validation prediction shape must match the validation target")
        validation_target = _validate_finite(
            validation.target, description="validation target"
        )
        score = float(np.sqrt(np.mean((prediction - validation_target) ** 2)))
        scores.append(
            CandidateScore(
                params=_json_copy(params, description="candidate params"),
                rmse=score,
            )
        )
    best_index = int(np.argmin([entry.rmse for entry in scores]))
    return FixedBaselineSelection(
        params=_json_copy(serialized_candidates[best_index], description="selected params"),
        candidate_scores=tuple(scores),
        selected_index=best_index,
        selected_score=scores[best_index].rmse,
        metric="rmse",
    )


def predictions_to_frame(
    batch: ForecastMatrix,
    prediction: np.ndarray,
    *,
    model: str,
    feature_set: str,
    seed: int,
    fold: AnnualFold,
) -> pl.DataFrame:
    """Convert a dense 24-horizon prediction into the validated Task 3 long-form schema."""

    normalized_seed = _require_integer_seed(seed)
    model = _require_nonblank_string(model, description="model")
    feature_set = _require_nonblank_string(feature_set, description="feature_set")
    prediction_array = _validate_finite(prediction, description="baseline predictions")
    observed = _validate_finite(batch.target, description="observed target")
    if (
        prediction_array.shape != observed.shape
        or prediction_array.ndim != 2
        or prediction_array.shape[1] != 24
    ):
        raise DataContractError("prediction must have shape (n_samples, 24) matching the target")
    if batch.target_times.shape != observed.shape or batch.origins.shape != (observed.shape[0],):
        raise DataContractError(
            "ForecastMatrix origins and target times must align with the target"
        )
    if not isinstance(fold, AnnualFold):
        raise TypeError("fold must be an AnnualFold with an immutable split_id")
    count = observed.shape[0]
    frame = pl.DataFrame(
        {
            "origin": np.repeat(batch.origins, 24),
            "target_timestamp": batch.target_times.reshape(-1),
            "horizon": np.tile(np.arange(1, 25, dtype=np.int64), count),
            "observed_mw": observed.reshape(-1),
            "predicted_mw": prediction_array.reshape(-1),
            "model": [model] * (count * 24),
            "feature_set": [feature_set] * (count * 24),
            "seed": np.full(count * 24, normalized_seed, dtype=np.int64),
            "split_id": [fold.split_id] * (count * 24),
        }
    )
    return validate_prediction_frame(frame, fold=fold)
