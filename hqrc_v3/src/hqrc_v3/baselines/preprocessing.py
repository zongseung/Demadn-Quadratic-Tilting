"""Immutable train-only preprocessing for the manuscript baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hqrc_v3.contracts import (
    DataContractError,
    ForecastMatrix,
    validate_forecast_feature_columns,
)

HISTORY_HOURS = 168
FUTURE_PATH_HOURS = 24
TARGET_SCALER_KIND = "standard-population"
FEATURE_SCALER_KIND = "standard-population-featurewise"
SCALER_FIT_PARTITION = "estimator-fit-only"


@dataclass(frozen=True)
class ScalerPopulation:
    """Auditable population used to fit one scaler."""

    count: int
    start: str
    end: str
    unit: str

    def __post_init__(self) -> None:
        if isinstance(self.count, bool) or not isinstance(self.count, int) or self.count <= 0:
            raise DataContractError("scaler population count must be a positive integer")
        if not self.start or not self.end or self.start > self.end:
            raise DataContractError("scaler population range must be non-empty and ordered")
        if self.unit not in {"daily-sample", "unique-hour"}:
            raise DataContractError("unknown scaler population unit")

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "start": self.start,
            "end": self.end,
            "unit": self.unit,
        }


def _timestamp_string(value: np.datetime64) -> str:
    if np.isnat(value):
        raise DataContractError("scaler population timestamps must be complete")
    return np.datetime_as_string(value.astype("datetime64[ns]"), unit="s")


def _population(times: np.ndarray, *, unit: str) -> ScalerPopulation:
    timestamps = np.asarray(times).reshape(-1).astype("datetime64[ns]")
    if not timestamps.size or np.isnat(timestamps).any():
        raise DataContractError("scaler population timestamps must be non-empty and complete")
    unique = np.unique(timestamps)
    return ScalerPopulation(
        count=int(unique.size),
        start=_timestamp_string(unique[0]),
        end=_timestamp_string(unique[-1]),
        unit=unit,
    )


def _readonly_float(values: np.ndarray, *, description: str) -> np.ndarray:
    array = np.array(values, dtype=np.float64, copy=True)
    if not np.isfinite(array).all():
        raise DataContractError(f"{description} must be finite")
    array.setflags(write=False)
    return array


def _finite_2d(values: np.ndarray, *, description: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not array.shape[0] or not array.shape[1]:
        raise DataContractError(f"{description} must be a non-empty 2-D array")
    if not np.isfinite(array).all():
        raise DataContractError(f"{description} must be finite")
    return array


@dataclass(frozen=True)
class FittedStandardScaler:
    """A frozen population-standardization state with an exact column contract."""

    mean: np.ndarray
    scale: np.ndarray
    columns: tuple[str, ...]
    kind: str = FEATURE_SCALER_KIND
    fit_partition: str = SCALER_FIT_PARTITION

    def __post_init__(self) -> None:
        mean = _readonly_float(self.mean, description="scaler mean").reshape(-1)
        scale = _readonly_float(self.scale, description="scaler scale").reshape(-1)
        if mean.shape != scale.shape or mean.shape != (len(self.columns),):
            raise DataContractError("scaler statistics must match its exact column schema")
        if any(not isinstance(column, str) or not column.strip() for column in self.columns):
            raise DataContractError("scaler columns must be nonblank strings")
        if len(set(self.columns)) != len(self.columns):
            raise DataContractError("scaler columns must be unique")
        if (scale <= 0.0).any():
            raise DataContractError("scaler scale must be strictly positive")
        if self.fit_partition != SCALER_FIT_PARTITION:
            raise DataContractError("scaler fit partition must remain estimator-fit-only")
        if self.kind not in {TARGET_SCALER_KIND, FEATURE_SCALER_KIND}:
            raise DataContractError("unknown frozen scaler kind")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(
        cls,
        values: np.ndarray,
        *,
        columns: tuple[str, ...],
        kind: str = FEATURE_SCALER_KIND,
    ) -> FittedStandardScaler:
        matrix = _finite_2d(values, description="scaler fit values")
        if matrix.shape[1] != len(columns):
            raise DataContractError("scaler fit values do not match the declared columns")
        mean = matrix.mean(axis=0)
        raw_scale = matrix.std(axis=0, ddof=0)
        scale = np.where(raw_scale == 0.0, 1.0, raw_scale)
        return cls(mean=mean, scale=scale, columns=columns, kind=kind)

    def transform(
        self, values: np.ndarray, *, columns: tuple[str, ...] | None = None
    ) -> np.ndarray:
        if columns is not None and columns != self.columns:
            raise DataContractError("transform columns/order differ from the fitted scaler")
        matrix = _finite_2d(values, description="scaler transform values")
        if matrix.shape[1] != len(self.columns):
            raise DataContractError("transform width differs from the fitted scaler")
        transformed = (matrix - self.mean) / self.scale
        if not np.isfinite(transformed).all():
            raise DataContractError("standardized values must be finite")
        return transformed

    def inverse_transform(
        self, values: np.ndarray, *, columns: tuple[str, ...] | None = None
    ) -> np.ndarray:
        if columns is not None and columns != self.columns:
            raise DataContractError("inverse-transform columns/order differ from the fitted scaler")
        matrix = _finite_2d(values, description="scaler inverse-transform values")
        if matrix.shape[1] != len(self.columns):
            raise DataContractError("inverse-transform width differs from the fitted scaler")
        restored = matrix * self.scale + self.mean
        if not np.isfinite(restored).all():
            raise DataContractError("inverse-transformed values must be finite")
        return restored


def _validate_matrix(matrix: ForecastMatrix) -> None:
    if not isinstance(matrix, ForecastMatrix):
        raise TypeError("batch must be a ForecastMatrix")
    if matrix.history.ndim != 3 or matrix.future.ndim != 3 or matrix.target.ndim != 2:
        raise DataContractError("ForecastMatrix arrays must be history/future 3-D and target 2-D")
    count = matrix.history.shape[0]
    if count <= 0 or matrix.future.shape[0] != count or matrix.target.shape[0] != count:
        raise DataContractError("ForecastMatrix arrays must contain aligned non-empty samples")
    if matrix.history.shape[1] != HISTORY_HOURS:
        raise DataContractError("preprocessing requires exactly 168 history hours")
    if matrix.future.shape[1] != FUTURE_PATH_HOURS or matrix.target.shape[1] != FUTURE_PATH_HOURS:
        raise DataContractError("preprocessing requires exactly 24 future/target hours")
    if matrix.origins.shape != (count,) or matrix.target_times.shape != (count, 24):
        raise DataContractError("ForecastMatrix timestamps must align with its samples")
    validate_forecast_feature_columns(matrix)
    for values, description in (
        (matrix.history, "history"),
        (matrix.future, "future"),
        (matrix.target, "target"),
    ):
        if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
            raise DataContractError(f"ForecastMatrix {description} must be finite")


def _require_matching_schema(
    matrix: ForecastMatrix,
    *,
    history_shape: tuple[int, int],
    future_shape: tuple[int, int],
    history_columns: tuple[str, ...],
    future_columns: tuple[str, ...],
) -> None:
    _validate_matrix(matrix)
    if matrix.history.shape[1:] != history_shape or matrix.future.shape[1:] != future_shape:
        raise DataContractError("matrix feature shape differs from fitted preprocessing")
    if matrix.history_columns != history_columns or matrix.future_columns != future_columns:
        raise DataContractError("matrix feature columns/order differ from fitted preprocessing")


def full_path_design(matrix: ForecastMatrix) -> np.ndarray:
    """Return one history-plus-complete-future design row per daily sample."""

    _validate_matrix(matrix)
    design = np.concatenate(
        (
            np.asarray(matrix.history, dtype=np.float64).reshape(matrix.history.shape[0], -1),
            np.asarray(matrix.future, dtype=np.float64).reshape(matrix.future.shape[0], -1),
        ),
        axis=1,
    )
    if not np.isfinite(design).all():
        raise DataContractError("full-path classical design must be finite")
    return design


def _design_columns(matrix: ForecastMatrix) -> tuple[str, ...]:
    return tuple(
        [
            f"history[{offset:03d}].{column}"
            for offset in range(HISTORY_HOURS)
            for column in matrix.history_columns
        ]
        + [
            f"future[{offset:02d}].{column}"
            for offset in range(FUTURE_PATH_HOURS)
            for column in matrix.future_columns
        ]
    )


def _target_scaler(matrix: ForecastMatrix) -> FittedStandardScaler:
    _validate_unique_target_hours(matrix)
    return FittedStandardScaler.fit(
        np.asarray(matrix.target, dtype=np.float64).reshape(-1, 1),
        columns=("target_mw",),
        kind=TARGET_SCALER_KIND,
    )


def _validate_unique_target_hours(matrix: ForecastMatrix) -> None:
    times = np.asarray(matrix.target_times).reshape(-1).astype("datetime64[ns]")
    if np.isnat(times).any() or np.unique(times).size != times.size:
        raise DataContractError("estimator-fit target paths must contain unique complete hours")
    expected = np.asarray(matrix.origins)[:, None].astype("datetime64[ns]") + np.arange(
        FUTURE_PATH_HOURS
    ).astype("timedelta64[h]")
    if not np.array_equal(np.asarray(matrix.target_times).astype("datetime64[ns]"), expected):
        raise DataContractError("target timestamps must match midnight origins and horizons")


@dataclass(frozen=True)
class FittedClassicalPreprocessor:
    """Complete-path X and common target scale fitted on estimator rows only."""

    x_scaler: FittedStandardScaler
    target_scaler: FittedStandardScaler
    history_shape: tuple[int, int]
    future_shape: tuple[int, int]
    history_columns: tuple[str, ...]
    future_columns: tuple[str, ...]
    x_population: ScalerPopulation
    target_population: ScalerPopulation

    def population_contract(self) -> dict[str, dict[str, object]]:
        return {
            "x": self.x_population.to_dict(),
            "target": self.target_population.to_dict(),
        }

    def transform_features(self, matrix: ForecastMatrix) -> np.ndarray:
        self._require_schema(matrix)
        return self.x_scaler.transform(
            full_path_design(matrix), columns=self.x_scaler.columns
        )

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != FUTURE_PATH_HOURS:
            raise DataContractError("target values must have shape (n_samples, 24)")
        return self.target_scaler.transform(array.reshape(-1, 1)).reshape(array.shape)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != FUTURE_PATH_HOURS:
            raise DataContractError("target values must have shape (n_samples, 24)")
        return self.target_scaler.inverse_transform(array.reshape(-1, 1)).reshape(array.shape)

    def _require_schema(self, matrix: ForecastMatrix) -> None:
        _require_matching_schema(
            matrix,
            history_shape=self.history_shape,
            future_shape=self.future_shape,
            history_columns=self.history_columns,
            future_columns=self.future_columns,
        )


def fit_classical_preprocessor(train: ForecastMatrix) -> FittedClassicalPreprocessor:
    _validate_matrix(train)
    design = full_path_design(train)
    return FittedClassicalPreprocessor(
        x_scaler=FittedStandardScaler.fit(design, columns=_design_columns(train)),
        target_scaler=_target_scaler(train),
        history_shape=train.history.shape[1:],
        future_shape=train.future.shape[1:],
        history_columns=train.history_columns,
        future_columns=train.future_columns,
        x_population=_population(train.origins, unit="daily-sample"),
        target_population=_population(train.target_times, unit="unique-hour"),
    )


def _deduplicated_history_values(
    matrix: ForecastMatrix, indices: tuple[int, ...]
) -> tuple[np.ndarray, np.ndarray]:
    if not indices:
        return np.empty((0, 0), dtype=np.float64), np.empty(0, dtype="datetime64[ns]")
    offsets = np.arange(-HISTORY_HOURS, 0).astype("timedelta64[h]")
    timestamps = (
        np.asarray(matrix.origins)[:, None].astype("datetime64[ns]") + offsets[None, :]
    ).reshape(-1)
    values = np.asarray(matrix.history[:, :, indices], dtype=np.float64).reshape(
        -1, len(indices)
    )
    order = np.argsort(timestamps, kind="stable")
    timestamps = timestamps[order]
    values = values[order]
    unique, first, counts = np.unique(timestamps, return_index=True, return_counts=True)
    if np.isnat(unique).any():
        raise DataContractError("history timestamps must be complete datetimes")
    for start, count in zip(first, counts, strict=True):
        if count > 1 and not np.array_equal(
            values[start : start + count], np.broadcast_to(values[start], (count, len(indices)))
        ):
            raise DataContractError("overlapping history timestamps contain inconsistent values")
    return values[first], unique


@dataclass(frozen=True)
class FittedSequencePreprocessor:
    """Shared target/load, weather, and calendar coordinate contract."""

    target_scaler: FittedStandardScaler
    weather_scaler: FittedStandardScaler
    calendar_scaler: FittedStandardScaler
    history_shape: tuple[int, int]
    future_shape: tuple[int, int]
    history_columns: tuple[str, ...]
    future_columns: tuple[str, ...]
    load_index: int
    weather_indices: tuple[int, ...]
    calendar_indices: tuple[int, ...]
    target_population: ScalerPopulation
    weather_population: ScalerPopulation
    calendar_population: ScalerPopulation

    @property
    def future_scaler(self) -> FittedStandardScaler:
        """Compatibility alias: the future scaler is the shared calendar scaler."""

        return self.calendar_scaler

    def population_contract(self) -> dict[str, dict[str, object]]:
        return {
            "target": self.target_population.to_dict(),
            "weather": self.weather_population.to_dict(),
            "calendar": self.calendar_population.to_dict(),
        }

    def transform_inputs(self, matrix: ForecastMatrix) -> tuple[np.ndarray, np.ndarray]:
        _require_matching_schema(
            matrix,
            history_shape=self.history_shape,
            future_shape=self.future_shape,
            history_columns=self.history_columns,
            future_columns=self.future_columns,
        )
        history = np.array(matrix.history, dtype=np.float64, copy=True)
        history[..., self.load_index] = self.target_scaler.transform(
            history[..., self.load_index].reshape(-1, 1)
        ).reshape(history.shape[:2])
        if self.weather_indices:
            weather = history[:, :, self.weather_indices].reshape(-1, len(self.weather_indices))
            history[:, :, self.weather_indices] = self.weather_scaler.transform(
                weather, columns=self.weather_scaler.columns
            ).reshape(history.shape[0], HISTORY_HOURS, len(self.weather_indices))
        if self.calendar_indices:
            calendar = history[:, :, self.calendar_indices].reshape(
                -1, len(self.calendar_indices)
            )
            history[:, :, self.calendar_indices] = self.calendar_scaler.transform(
                calendar, columns=self.calendar_scaler.columns
            ).reshape(history.shape[0], HISTORY_HOURS, len(self.calendar_indices))
        future = self.calendar_scaler.transform(
            np.asarray(matrix.future, dtype=np.float64).reshape(-1, matrix.future.shape[2]),
            columns=matrix.future_columns,
        ).reshape(matrix.future.shape)
        return history, future

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != FUTURE_PATH_HOURS:
            raise DataContractError("target values must have shape (n_samples, 24)")
        return self.target_scaler.transform(array.reshape(-1, 1)).reshape(array.shape)

    def inverse_target(self, values: np.ndarray) -> np.ndarray:
        array = np.asarray(values, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != FUTURE_PATH_HOURS:
            raise DataContractError("target values must have shape (n_samples, 24)")
        return self.target_scaler.inverse_transform(array.reshape(-1, 1)).reshape(array.shape)


def fit_sequence_preprocessor(train: ForecastMatrix) -> FittedSequencePreprocessor:
    _validate_matrix(train)
    _validate_unique_target_hours(train)
    try:
        load_index = train.history_columns.index("load_mw")
    except ValueError as error:
        raise DataContractError("sequence history must contain one load_mw channel") from error
    weather_columns = tuple(
        column
        for column in ("temperature_c", "relative_humidity")
        if column in train.history_columns
    )
    weather_indices = tuple(train.history_columns.index(column) for column in weather_columns)
    calendar_indices = tuple(
        train.history_columns.index(column)
        for column in train.future_columns
        if column in train.history_columns
    )
    expected_history = ("load_mw", *weather_columns, *train.future_columns)
    if train.history_columns != expected_history:
        raise DataContractError(
            "sequence history columns must be load_mw, observed weather, then future calendar"
        )
    if tuple(train.history_columns[index] for index in calendar_indices) != train.future_columns:
        raise DataContractError("history/future calendar columns must match in exact order")
    weather_values, weather_times = _deduplicated_history_values(train, weather_indices)
    if not weather_columns:
        raise DataContractError("sequence history must contain observed weather channels")
    calendar_values = np.asarray(train.future, dtype=np.float64).reshape(
        -1, train.future.shape[2]
    )
    return FittedSequencePreprocessor(
        target_scaler=_target_scaler(train),
        weather_scaler=FittedStandardScaler.fit(
            weather_values, columns=weather_columns
        ),
        calendar_scaler=FittedStandardScaler.fit(
            calendar_values, columns=train.future_columns
        ),
        history_shape=train.history.shape[1:],
        future_shape=train.future.shape[1:],
        history_columns=train.history_columns,
        future_columns=train.future_columns,
        load_index=load_index,
        weather_indices=weather_indices,
        calendar_indices=calendar_indices,
        target_population=_population(train.target_times, unit="unique-hour"),
        weather_population=_population(weather_times, unit="unique-hour"),
        calendar_population=_population(train.target_times, unit="unique-hour"),
    )


__all__ = [
    "FEATURE_SCALER_KIND",
    "FUTURE_PATH_HOURS",
    "FittedClassicalPreprocessor",
    "FittedSequencePreprocessor",
    "FittedStandardScaler",
    "HISTORY_HOURS",
    "SCALER_FIT_PARTITION",
    "ScalerPopulation",
    "TARGET_SCALER_KIND",
    "fit_classical_preprocessor",
    "fit_sequence_preprocessor",
    "full_path_design",
]
