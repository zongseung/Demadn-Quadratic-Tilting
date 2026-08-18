from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from hqrc_v3.contracts import PREDICTION_COLUMNS, DataContractError, validate_prediction_frame
from hqrc_v3.splits import AnnualFold


def _prediction_frame(
    origin: datetime = datetime(2023, 1, 1), split_id: str = "oof-2023"
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin": [origin] * 24,
            "target_timestamp": [origin + timedelta(hours=hour) for hour in range(24)],
            "horizon": list(range(1, 25)),
            "observed_mw": [100.0 + hour for hour in range(24)],
            "predicted_mw": [99.0 + hour for hour in range(24)],
            "model": ["xgb"] * 24,
            "feature_set": ["B0"] * 24,
            "seed": [7] * 24,
            "split_id": [split_id] * 24,
        }
    )


def test_prediction_frame_has_exact_schema_and_safe_numeric_casts():
    frame = _prediction_frame().with_columns(
        pl.col("horizon").cast(pl.Int8),
        pl.col("observed_mw").cast(pl.Int32),
        pl.col("predicted_mw").cast(pl.Int32),
        pl.col("seed").cast(pl.Int8),
    )

    validated = validate_prediction_frame(frame)

    assert validated.columns == list(PREDICTION_COLUMNS)
    assert validated.schema == {
        "origin": pl.Datetime("ns"),
        "target_timestamp": pl.Datetime("ns"),
        "horizon": pl.Int64,
        "observed_mw": pl.Float64,
        "predicted_mw": pl.Float64,
        "model": pl.String,
        "feature_set": pl.String,
        "seed": pl.Int64,
        "split_id": pl.String,
    }


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.slice(0, 23), "24 horizons"),
        (lambda frame: frame.with_columns(pl.lit(float("nan")).alias("predicted_mw")), "finite"),
        (
            lambda frame: frame.with_columns(
                pl.when(pl.col("horizon") == 2)
                .then(pl.lit("other"))
                .otherwise(pl.col("split_id"))
                .alias("split_id")
            ),
            "stable",
        ),
        (
            lambda frame: frame.with_columns(
                pl.when(pl.col("horizon") == 2)
                .then(pl.col("target_timestamp") + pl.duration(days=1))
                .otherwise(pl.col("target_timestamp"))
                .alias("target_timestamp")
            ),
            "target timestamp",
        ),
    ],
)
def test_prediction_frame_rejects_invalid_horizons_values_and_identifiers(mutation, message):
    with pytest.raises(DataContractError, match=message):
        validate_prediction_frame(mutation(_prediction_frame()))


def test_prediction_frame_rejects_duplicate_target_key_and_extra_columns():
    frame = _prediction_frame()
    duplicate = pl.concat([frame, frame.head(1)])

    with pytest.raises(DataContractError, match="duplicate"):
        validate_prediction_frame(duplicate)
    with pytest.raises(DataContractError, match="exactly"):
        validate_prediction_frame(frame.with_columns(pl.lit(False).alias("is_event")))


def test_prediction_frame_rejects_targets_at_or_before_its_training_end():
    fold = AnnualFold.oof(eval_year=2023, first_train_year=2019)
    origin = datetime(2022, 12, 31)
    frame = _prediction_frame().with_columns(
        pl.lit(origin).alias("origin"),
        (pl.lit(origin) + (pl.col("horizon") - 1) * pl.duration(hours=1)).alias("target_timestamp"),
    )

    with pytest.raises(DataContractError, match="training end"):
        validate_prediction_frame(frame, fold=fold)


@pytest.mark.parametrize(
    ("origin", "split_id", "message"),
    [
        (datetime(2024, 1, 1), "oof-2023", "evaluation range"),
        (datetime(2024, 1, 1), "not-a-split", "unknown immutable split_id"),
    ],
)
def test_prediction_frame_resolves_immutable_fold_from_its_split_id(origin, split_id, message):
    with pytest.raises(DataContractError, match=message):
        validate_prediction_frame(_prediction_frame(origin, split_id))


def test_prediction_frame_rejects_targets_past_its_resolved_eval_end():
    frame = _prediction_frame(datetime(2023, 12, 31, 12), "oof-2023")

    with pytest.raises(DataContractError, match="evaluation range"):
        validate_prediction_frame(frame)
