from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest

from hqrc_v3.contracts import DataContractError
from hqrc_v3.residuals import compute_fold_scale, standardize_event_residuals


def _prediction_frame() -> pl.DataFrame:
    origin = datetime(2023, 1, 1)
    return pl.DataFrame(
        {
            "origin": [origin] * 24,
            "target_timestamp": [origin + timedelta(hours=hour) for hour in range(24)],
            "horizon": list(range(1, 25)),
            "observed_mw": [100.0] * 24,
            "predicted_mw": [99.0] * 24,
            "model": ["xgb"] * 24,
            "feature_set": ["B0"] * 24,
            "seed": [7] * 24,
            "split_id": ["oof-2023"] * 24,
        }
    )


def test_scale_uses_only_non_event_rows():
    frame = (
        _prediction_frame()
        .head(3)
        .with_columns(
            pl.Series("is_event", [False, False, True]),
            pl.Series("residual_mw", [3.0, 4.0, 1000.0]),
        )
    )

    assert compute_fold_scale(frame) == pytest.approx((12.5) ** 0.5)


def test_scale_rejects_empty_non_event_nonfinite_and_nonpositive_values():
    base = _prediction_frame().head(3)
    for events, residuals in (
        ([True, True, True], [1.0, 1.0, 1.0]),
        ([False, False, False], [float("nan"), 1.0, 1.0]),
        ([False, False, False], [0.0, 0.0, 0.0]),
    ):
        frame = base.with_columns(
            pl.Series("is_event", events), pl.Series("residual_mw", residuals)
        )
        with pytest.raises(DataContractError):
            compute_fold_scale(frame)


@pytest.mark.parametrize("split_id", ["oof-2024", "oof-x", "final-2024"])
def test_scale_rejects_non_immutable_oof_contexts(split_id):
    frame = (
        _prediction_frame()
        .head(3)
        .with_columns(
            pl.lit(split_id).alias("split_id"),
            pl.Series("is_event", [False, False, False]),
            pl.Series("residual_mw", [1.0, 1.0, 1.0]),
        )
    )

    with pytest.raises(DataContractError, match="OOF"):
        compute_fold_scale(frame)


def test_event_standardization_keeps_the_scale_context_and_refuses_mixed_folds():
    base = _prediction_frame().head(3)
    scale_source = base.with_columns(
        pl.Series("is_event", [False, False, False]),
        pl.Series("residual_mw", [3.0, 4.0, 5.0]),
    )
    scale = compute_fold_scale(scale_source)
    event_frame = base.with_columns(
        pl.Series("is_event", [True, True, True]),
        pl.Series("residual_mw", [3.0, 4.0, 5.0]),
    )

    standardized = standardize_event_residuals(event_frame, scale)

    assert standardized["standardized_residual"].to_list() == pytest.approx(
        [3.0 / scale, 4.0 / scale, 5.0 / scale]
    )
    mixed_fold = event_frame.with_columns(pl.lit("oof-2022").alias("split_id"))
    with pytest.raises(DataContractError, match="context"):
        standardize_event_residuals(mixed_fold, scale)
