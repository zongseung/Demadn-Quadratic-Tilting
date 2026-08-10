"""Fixed-parameter classical 24-horizon forecasting adapters."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import polars as pl

from hqrc_v3.contracts import DataContractError, ForecastMatrix, validate_prediction_frame
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
    if matrix.future.shape[1] != 24 or matrix.target.shape[1] != 24:
        raise DataContractError("classical baselines require exactly 24 future horizons")
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


@dataclass(frozen=True)
class HorizonRegressor:
    """Twenty-four independently fitted estimators, one for each target hour."""

    model_name: str
    estimators: tuple[Any, ...]
    history_shape: tuple[int, int]
    future_width: int

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        if len(self.estimators) != 24:
            raise RuntimeError("a horizon regressor must contain exactly 24 estimators")
        if (
            batch.history.shape[1:] != self.history_shape
            or batch.future.shape[2] != self.future_width
        ):
            raise DataContractError(
                "prediction matrix feature shape does not match the fitted baseline"
            )
        prediction = np.empty((batch.target.shape[0], 24), dtype=float)
        flattened_history = _flatten_history(batch)
        for horizon, estimator in enumerate(self.estimators):
            prediction[:, horizon] = estimator.predict(
                _features_for_horizon(flattened_history, batch, horizon)
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
        if not isinstance(seed, (int, np.integer)):
            raise TypeError("seed must be an integer")
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
                estimator_params["random_state"] = int(seed)
            if "n_jobs" in available:
                estimator_params["n_jobs"] = 1
            if "nthread" in available:
                estimator_params["nthread"] = 1
            if self.name == "xgboost" and "verbosity" in available:
                estimator_params["verbosity"] = 0
            if self.name == "lightgbm" and "verbosity" in available:
                estimator_params["verbosity"] = -1
            estimator = estimator_class(**estimator_params)
            estimator.fit(features, target[:, horizon])
            estimators.append(estimator)
        return HorizonRegressor(
            model_name=self.name,
            estimators=tuple(estimators),
            history_shape=train.history.shape[1:],
            future_width=train.future.shape[2],
        )


def make_classical_baseline(name: str, params: Mapping[str, Any]) -> ClassicalBaseline:
    """Create a fixed-config adapter without any search or tuning behaviour."""

    normalized_name = name.lower()
    return ClassicalBaseline(name=normalized_name, params=_validate_params(normalized_name, params))


@dataclass(frozen=True)
class FixedBaselineSelection:
    """Serializable result of the single allowed 2019--2022/2023 configuration choice."""

    params: dict[str, Any]
    candidate_scores: tuple[float, ...]
    metric: str = "rmse"


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
    scores: list[float] = []
    for params in candidates:
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
        scores.append(score)
    best_index = int(np.argmin(scores))
    return FixedBaselineSelection(
        params=dict(candidates[best_index]), candidate_scores=tuple(scores), metric="rmse"
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
            "seed": np.full(count * 24, seed, dtype=np.int64),
            "split_id": [fold.split_id] * (count * 24),
        }
    )
    return validate_prediction_frame(frame, fold=fold)
