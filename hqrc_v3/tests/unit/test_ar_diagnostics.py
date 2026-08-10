from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hqrc_v3.diagnostics.ar import (
    ARCalibrationError,
    calibrate_beta_prior,
    detrend_event_residuals,
    diagnose_event_residuals,
    estimate_event_phi,
    estimate_event_phis,
    validate_event_residual_context,
)


def _residual_frame(*, events: int = 3, rows: int = 120) -> pl.DataFrame:
    records: list[dict[str, object]] = []
    for event_index in range(events):
        start = datetime(2023, 1, 1) + timedelta(days=14 * event_index)
        tau = np.arange(rows, dtype=float) / 24.0 - 2.0
        hour = np.arange(rows) % 24
        values = 0.4 * tau - 0.08 * tau**2
        for position in range(rows):
            records.append(
                {
                    "occurrence_id": f"event-{event_index}",
                    "timestamp": start + timedelta(hours=position),
                    "tau_days": tau[position],
                    "hour": int(hour[position]),
                    "standardized_residual": float(values[position]),
                    "model": "baseline",
                    "feature_set": "B1",
                    "seed": 17,
                    "split_id": f"oof-{2020 + event_index}",
                }
            )
    return pl.DataFrame(records)


def test_context_rejects_duplicate_gap_unsorted_nonfinite_and_mixed_rows():
    frame = _residual_frame(events=1)
    cases = (
        frame.vstack(frame.head(1)),
        frame.filter(pl.col("timestamp") != frame["timestamp"][20]),
        frame.with_columns(
            pl.when(pl.int_range(pl.len()) == 1)
            .then(pl.lit("other"))
            .otherwise(pl.col("model"))
            .alias("model")
        ),
    )
    for invalid in cases:
        with pytest.raises(ARCalibrationError):
            validate_event_residual_context(invalid)
    with pytest.raises(ARCalibrationError, match="ordered"):
        validate_event_residual_context(frame[[1, 0, *range(2, frame.height)]])
    with pytest.raises(ARCalibrationError, match="finite"):
        validate_event_residual_context(
            frame.with_columns(
                pl.when(pl.int_range(pl.len()) == 1)
                .then(float("nan"))
                .otherwise(pl.col("standardized_residual"))
                .alias("standardized_residual")
            )
        )


def test_context_allows_distinct_oof_splits_but_rejects_unknown_or_mixed_event_splits():
    frame = _residual_frame(events=3)

    context = validate_event_residual_context(frame)

    assert context.split_ids == ("oof-2020", "oof-2021", "oof-2022")
    with pytest.raises(ARCalibrationError, match="unknown split"):
        validate_event_residual_context(frame.with_columns(pl.lit("oof-2025").alias("split_id")))
    with pytest.raises(ARCalibrationError, match="one split"):
        validate_event_residual_context(
            frame.with_columns(
                pl.when(pl.int_range(pl.len()) == 1)
                .then(pl.lit("oof-2021"))
                .otherwise(pl.col("split_id"))
                .alias("split_id")
            )
        )
    with pytest.raises(ARCalibrationError, match="integer"):
        validate_event_residual_context(frame.with_columns(pl.lit(17.5).alias("seed")))


def test_detrending_removes_each_event_quadratic_and_hour_harmonics_independently():
    frame = _residual_frame(events=2)
    enriched = frame.with_columns(
        (
            pl.col("standardized_residual")
            + 1.2 * (2 * np.pi * pl.col("hour") / 24).sin()
            - 0.7 * (4 * np.pi * pl.col("hour") / 24).cos()
        ).alias("standardized_residual")
    )

    detrended = detrend_event_residuals(enriched)

    assert detrended["detrended_residual"].to_numpy() == pytest.approx(0.0, abs=1e-10)


def test_phi_never_pairs_two_events_and_clips_or_rejects_zero_lag_variance():
    residuals = {
        "event-a": np.array([1.0, 0.5]),
        "event-b": np.array([100.0, 50.0]),
    }

    assert estimate_event_phis(residuals) == {
        "event-a": pytest.approx(0.5),
        "event-b": pytest.approx(0.5),
    }
    assert estimate_event_phi(np.array([1.0, 10.0, 100.0])) == pytest.approx(0.98)
    with pytest.raises(ARCalibrationError, match="zero lag variance"):
        estimate_event_phi(np.zeros(120))


def test_diagnostics_save_exact_48_lag_arrays_and_safe_ljung_box_lag():
    frame = _residual_frame()
    frame = frame.with_columns(
        (pl.col("standardized_residual") + 0.2 * (pl.col("hour") % 2)).alias(
            "standardized_residual"
        )
    )

    diagnostics = diagnose_event_residuals(frame)

    assert [diagnostic.occurrence_id for diagnostic in diagnostics] == [
        "event-0",
        "event-1",
        "event-2",
    ]
    for diagnostic in diagnostics:
        assert len(diagnostic.raw_acf) == len(diagnostic.raw_pacf) == 48
        assert len(diagnostic.detrended_acf) == len(diagnostic.detrended_pacf) == 48
        assert len(diagnostic.innovation_acf) == len(diagnostic.innovation_pacf) == 48
        assert diagnostic.ljung_box_lag == 24
        assert np.isfinite(np.asarray(diagnostic.raw_acf)).all()
        assert len(diagnostic.raw_acf_lower) == len(diagnostic.raw_acf_upper) == 48
        assert len(diagnostic.detrended_acf_lower) == len(diagnostic.detrended_acf_upper) == 48
        assert len(diagnostic.innovation_acf_lower) == len(diagnostic.innovation_acf_upper) == 48
        assert all(
            lower < 0 < upper
            for lower, upper in zip(diagnostic.raw_acf_lower, diagnostic.raw_acf_upper)
        )
        assert diagnostic.pacf_reference_half_width > 0


def test_beta_calibration_is_finite_capped_and_order_invariant():
    phi = np.array([0.84, 0.88, 0.91, 0.86])

    calibration = calibrate_beta_prior(phi, event_ids=("d", "a", "c", "b"))
    reordered = calibrate_beta_prior(phi[::-1], event_ids=("b", "c", "a", "d"))

    assert calibration.a > 1.0
    assert calibration.b > 1.0
    assert 8.0 <= calibration.a + calibration.b <= 40.0
    assert calibration.phi_center > 0.8
    assert (calibration.a, calibration.b, calibration.phi_center) == pytest.approx(
        (reordered.a, reordered.b, reordered.phi_center)
    )
    assert calibration.event_ids == ("a", "b", "c", "d")
    with pytest.raises(ARCalibrationError, match="at least"):
        calibrate_beta_prior(np.array([0.5]))
    with pytest.raises(ARCalibrationError, match="finite"):
        calibrate_beta_prior(np.array([np.nan, 0.5]))
    with pytest.raises(ARCalibrationError, match="range"):
        calibrate_beta_prior(np.array([1.01, 0.5]))
