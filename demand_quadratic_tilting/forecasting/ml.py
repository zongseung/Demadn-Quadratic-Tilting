"""Tree, boosting, and kernel multi-horizon baselines."""

from __future__ import annotations

from dataclasses import dataclass
from os import cpu_count
from typing import Any
from warnings import catch_warnings, filterwarnings

import numpy as np
from joblib import Parallel, delayed
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


ML_MODEL_NAMES = ("xgboost", "lightgbm", "random_forest", "svr")


@dataclass
class DirectMultiHorizonModel:
    """One direct regressor per forecast horizon."""

    name: str
    feature_scaler: StandardScaler
    estimators: list[RegressorMixin]
    output_horizon: int | None = None

    @property
    def horizon_hours(self) -> int:
        return self.output_horizon or len(self.estimators)

    def predict(self, features: np.ndarray) -> np.ndarray:
        scaled = self.feature_scaler.transform(features)
        if self.name == "random_forest":
            return self.estimators[0].predict(scaled).astype(np.float32)
        if self.name == "lightgbm":
            with catch_warnings():
                filterwarnings(
                    "ignore",
                    message="X does not have valid feature names.*",
                    category=UserWarning,
                )
                columns = [
                    estimator.predict(scaled, validate_features=False)
                    for estimator in self.estimators
                ]
        else:
            columns = [estimator.predict(scaled) for estimator in self.estimators]
        return np.column_stack(columns).astype(np.float32)


def _make_estimator(
    name: str,
    seed: int,
    quick: bool,
) -> RegressorMixin:
    if name == "xgboost":
        return XGBRegressor(
            objective="reg:squarederror",
            eval_metric="rmse",
            n_estimators=40 if quick else 600,
            learning_rate=0.08 if quick else 0.03,
            max_depth=4 if quick else 6,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            early_stopping_rounds=10 if quick else 35,
            random_state=seed,
            n_jobs=1,
            tree_method="hist",
        )
    if name == "lightgbm":
        return LGBMRegressor(
            objective="regression",
            n_estimators=50 if quick else 800,
            learning_rate=0.08 if quick else 0.025,
            num_leaves=31,
            max_depth=-1,
            min_child_samples=20,
            subsample=0.9,
            subsample_freq=1,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=seed,
            n_jobs=1,
            verbosity=-1,
        )
    if name == "svr":
        return SVR(
            kernel="rbf",
            C=10.0,
            epsilon=0.05,
            gamma="scale",
            cache_size=512,
        )
    raise ValueError(f"unknown ML model {name!r}; expected one of {ML_MODEL_NAMES}")


def _fit_horizon(
    name: str,
    horizon: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    seed: int,
    quick: bool,
) -> RegressorMixin:
    estimator = _make_estimator(name, seed + horizon, quick)
    if name == "xgboost":
        estimator.fit(
            x_train,
            y_train[:, horizon],
            eval_set=[(x_validation, y_validation[:, horizon])],
            verbose=False,
        )
    elif name == "lightgbm":
        rounds = 10 if quick else 35
        estimator.fit(
            x_train,
            np.ascontiguousarray(y_train[:, horizon]),
            eval_set=[(x_validation, np.ascontiguousarray(y_validation[:, horizon]))],
            eval_metric="rmse",
            callbacks=[
                early_stopping(rounds, verbose=False),
                log_evaluation(period=0),
            ],
        )
    else:
        estimator.fit(x_train, y_train[:, horizon])
    return estimator


def fit_direct_multi_horizon(
    name: str,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
    *,
    random_seed: int = 2025,
    n_jobs: int | None = None,
    quick: bool = False,
) -> DirectMultiHorizonModel:
    """Fit direct horizon regressors or a native multi-output forest.

    A single input scaler is fitted on training windows only.  Validation
    targets are used solely for early stopping in the two boosting models.
    """

    if name not in ML_MODEL_NAMES:
        raise ValueError(f"unknown ML model {name!r}")
    if y_train.ndim != 2 or y_validation.ndim != 2:
        raise ValueError("multi-horizon targets must be two-dimensional")
    if y_train.shape[1] != y_validation.shape[1]:
        raise ValueError("training and validation horizons do not match")

    feature_scaler = StandardScaler().fit(x_train)
    train_scaled = feature_scaler.transform(x_train).astype(np.float32)
    validation_scaled = feature_scaler.transform(x_validation).astype(np.float32)
    if n_jobs is None:
        n_jobs = min(4, cpu_count() or 1)

    if name == "random_forest":
        forest = RandomForestRegressor(
            n_estimators=40 if quick else 500,
            criterion="squared_error",
            max_depth=12 if quick else 24,
            min_samples_split=2,
            min_samples_leaf=2,
            max_features=0.5,
            bootstrap=True,
            random_state=random_seed,
            n_jobs=n_jobs,
        )
        forest.fit(train_scaled, y_train)
        return DirectMultiHorizonModel(
            name=name,
            feature_scaler=feature_scaler,
            estimators=[forest],
            output_horizon=y_train.shape[1],
        )

    estimators = Parallel(n_jobs=n_jobs, prefer="threads")(
        delayed(_fit_horizon)(
            name,
            horizon,
            train_scaled,
            y_train,
            validation_scaled,
            y_validation,
            random_seed,
            quick,
        )
        for horizon in range(y_train.shape[1])
    )
    return DirectMultiHorizonModel(
        name=name,
        feature_scaler=feature_scaler,
        estimators=list(estimators),
    )


def model_summary(model: DirectMultiHorizonModel) -> dict[str, Any]:
    """Return serializable fitted-tree/iteration diagnostics."""

    if model.name == "random_forest":
        forest = model.estimators[0]
        return {
            "name": model.name,
            "horizon_hours": model.horizon_hours,
            "native_multi_output": True,
            "n_estimators": int(forest.n_estimators),
            "max_depth": forest.max_depth,
            "min_samples_leaf": int(forest.min_samples_leaf),
            "max_features": forest.max_features,
        }

    best_iterations: list[int | None] = []
    for estimator in model.estimators:
        value = getattr(estimator, "best_iteration", None)
        if value is None:
            value = getattr(estimator, "best_iteration_", None)
        best_iterations.append(None if value is None else int(value))
    finite = [value for value in best_iterations if value is not None]
    return {
        "name": model.name,
        "horizon_hours": model.horizon_hours,
        "best_iterations": best_iterations,
        "mean_best_iteration": (float(np.mean(finite)) if finite else None),
    }
