"""Shared data contracts for the hourly forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, time
from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    from hqrc_v3.splits import AnnualFold


class DataContractError(ValueError):
    """Raised when hourly input or its derived forecast samples are invalid."""


PREDICTION_COLUMNS = (
    "origin",
    "target_timestamp",
    "horizon",
    "observed_mw",
    "predicted_mw",
    "model",
    "feature_set",
    "seed",
    "split_id",
)

_INTEGER_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}
_NUMERIC_DTYPES = _INTEGER_DTYPES | {pl.Float32, pl.Float64}


def _is_naive_datetime(dtype: pl.DataType) -> bool:
    return dtype.base_type() == pl.Datetime and dtype.time_zone is None


def _require_prediction_input_types(frame: pl.DataFrame) -> None:
    timestamp_columns = ("origin", "target_timestamp")
    if not all(_is_naive_datetime(frame.schema[column]) for column in timestamp_columns):
        raise DataContractError("prediction timestamps must be naive Polars Datetime values")
    integer_columns = ("horizon", "seed")
    if any(frame.schema[column] not in _INTEGER_DTYPES for column in integer_columns):
        raise DataContractError("prediction horizon and seed must be integer values")
    mw_columns = ("observed_mw", "predicted_mw")
    if any(frame.schema[column] not in _NUMERIC_DTYPES for column in mw_columns):
        raise DataContractError("prediction MW values must be numeric")
    identifier_columns = ("model", "feature_set", "split_id")
    identifier_types = {pl.String, pl.Categorical, pl.Enum}
    if any(frame.schema[column] not in identifier_types for column in identifier_columns):
        raise DataContractError("prediction identifiers must be strings")


def validate_prediction_frame(
    frame: pl.DataFrame, *, fold: AnnualFold | None = None
) -> pl.DataFrame:
    """Validate and normalize a single long-form baseline prediction artifact.

    A frame is deliberately one model/feature/seed/fold context.  This prevents
    downstream residual calculations from accidentally combining incompatible
    fitted models or expanding-origin folds.
    """

    if not isinstance(frame, pl.DataFrame):
        raise DataContractError("prediction frame must be a Polars DataFrame")
    if tuple(frame.columns) != PREDICTION_COLUMNS:
        raise DataContractError("prediction frame must contain exactly the required schema columns")
    if frame.is_empty():
        raise DataContractError("prediction frame must not be empty")
    _require_prediction_input_types(frame)
    if any(frame[column].null_count() for column in PREDICTION_COLUMNS):
        raise DataContractError("prediction frame must not contain null values")

    validated = frame.with_columns(
        pl.col("origin").cast(pl.Datetime("ns"), strict=True),
        pl.col("target_timestamp").cast(pl.Datetime("ns"), strict=True),
        pl.col("horizon").cast(pl.Int64, strict=True),
        pl.col("observed_mw").cast(pl.Float64, strict=True),
        pl.col("predicted_mw").cast(pl.Float64, strict=True),
        pl.col("model").cast(pl.String, strict=True),
        pl.col("feature_set").cast(pl.String, strict=True),
        pl.col("seed").cast(pl.Int64, strict=True),
        pl.col("split_id").cast(pl.String, strict=True),
    )
    finite_mw = validated.select(
        (pl.col("observed_mw").is_finite() & pl.col("predicted_mw").is_finite()).all()
    ).item()
    if not finite_mw:
        raise DataContractError("prediction MW values must be finite")
    for column in ("model", "feature_set", "seed", "split_id"):
        if validated[column].n_unique() != 1:
            raise DataContractError("prediction identifiers must be stable within one frame")
    if any(
        validated[column].str.strip_chars().eq("").any()
        for column in ("model", "feature_set", "split_id")
    ):
        raise DataContractError("prediction identifiers must not be blank")

    prediction_key = ("model", "feature_set", "seed", "split_id", "target_timestamp")
    if not validated.group_by(*prediction_key).len().filter(pl.col("len") > 1).is_empty():
        raise DataContractError("duplicate prediction key")

    horizon_checks = validated.group_by("origin").agg(
        pl.len().alias("rows"),
        pl.col("horizon").min().alias("minimum"),
        pl.col("horizon").max().alias("maximum"),
        pl.col("horizon").n_unique().alias("unique_horizons"),
    )
    if not horizon_checks.filter(
        (pl.col("rows") != 24)
        | (pl.col("minimum") != 1)
        | (pl.col("maximum") != 24)
        | (pl.col("unique_horizons") != 24)
    ).is_empty():
        raise DataContractError(
            "each prediction origin must contain 24 horizons numbered 1 through 24 exactly once"
        )

    expected_targets = pl.col("origin") + (pl.col("horizon") - 1) * pl.duration(hours=1)
    if not validated.filter(pl.col("target_timestamp") != expected_targets).is_empty():
        raise DataContractError("target timestamp is inconsistent with origin and horizon")
    if fold is not None:
        earliest_eval_target = datetime.combine(fold.eval_start, time.min)
        if validated.filter(pl.col("target_timestamp") < earliest_eval_target).height:
            raise DataContractError("evaluation target is at or before the training end")
        if validated["split_id"].item(0) != fold.split_id:
            raise DataContractError("prediction split_id does not match its fold")
    return validated


@dataclass(frozen=True)
class ForecastMatrix:
    """Daily samples issued at 00:00 with 168 observed and 24 target hours."""

    origins: np.ndarray
    target_times: np.ndarray
    history: np.ndarray
    future: np.ndarray
    target: np.ndarray
    history_columns: tuple[str, ...]
    future_columns: tuple[str, ...]

    def take(self, indices: np.ndarray) -> ForecastMatrix:
        """Return a sample-aligned subset without changing feature contracts."""

        return replace(
            self,
            origins=self.origins[indices],
            target_times=self.target_times[indices],
            history=self.history[indices],
            future=self.future[indices],
            target=self.target[indices],
        )
