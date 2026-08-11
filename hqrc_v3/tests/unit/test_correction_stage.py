from __future__ import annotations

from datetime import datetime, time, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hqrc_v3.correction_stage import (
    CausalCorrectionError,
    build_causal_hqrc_data,
    build_causal_prediction_data,
    select_latest_oof_scale,
)
from hqrc_v3.diagnostics.ar import (
    EventResidualContext,
    approve_calibration,
    calibrate_beta_prior,
    load_approved_calibration,
    write_ar_diagnostics,
)
from hqrc_v3.events import load_event_registry

EVENTS = Path(__file__).parents[2] / "configs/events.csv"
SPLITS = tuple(f"oof-{year}" for year in range(2020, 2024))


def _approved(tmp_path: Path, event_ids: tuple[str, ...]):
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibrate_beta_prior(np.linspace(0.2, 0.55, len(event_ids)), event_ids=event_ids),
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="events",
        context=EventResidualContext("lightgbm", "B1", 7, SPLITS),
    )
    approved = approve_calibration(
        proposal,
        tmp_path / "approved.json",
        current_residual_sha256="residual",
        current_config_sha256="config",
        current_event_sha256="events",
    )
    return load_approved_calibration(
        approved,
        current_residual_sha256="residual",
        current_config_sha256="config",
        current_event_sha256="events",
    )


def _training_frame() -> pl.DataFrame:
    rows = []
    for event in load_event_registry(EVENTS):
        if event.central_date.year == 2024:
            continue
        start = datetime.combine(event.window_start, time.min)
        hours = (event.window_end - event.window_start).days * 24 + 24
        center = datetime.combine(event.central_date, time.min)
        for offset in range(hours):
            timestamp = start + timedelta(hours=offset)
            rows.append(
                {
                    "target_timestamp": timestamp,
                    "standardized_residual": float(offset + 1) / 100.0,
                    "model": "lightgbm",
                    "feature_set": "B1",
                    "seed": 7,
                    "split_id": f"oof-{event.central_date.year}",
                    "occurrence_id": event.occurrence_id,
                    "holiday_type": event.holiday_type,
                    "tau_days": (timestamp - center).total_seconds() / 86_400.0,
                    "hour": timestamp.hour,
                    "restriction": event.restriction,
                }
            )
    return pl.DataFrame(rows).sort("target_timestamp")


def _final_stream() -> pl.DataFrame:
    rows = []
    start = datetime(2024, 1, 1)
    for day_offset in range(305):
        origin = start + timedelta(days=day_offset)
        for horizon in range(1, 25):
            timestamp = origin + timedelta(hours=horizon - 1)
            rows.append(
                {
                    "origin": origin,
                    "target_timestamp": timestamp,
                    "horizon": horizon,
                    "observed_mw": 50_000.0 + horizon,
                    "predicted_mw": 49_000.0 + horizon,
                    "model": "lightgbm",
                    "feature_set": "B1",
                    "seed": 7,
                    "split_id": "final-2024",
                }
            )
    return pl.DataFrame(rows)


def test_training_adapter_binds_exact_approval_context_and_registered_events(tmp_path):
    events = load_event_registry(EVENTS)
    frame = _training_frame()
    event_ids = tuple(sorted(frame["occurrence_id"].unique().to_list()))
    approved = _approved(tmp_path, event_ids)

    data = build_causal_hqrc_data(frame, approved, events=events)

    assert data.occurrence_ids == tuple(
        event.occurrence_id for event in events if event.central_date.year < 2024
    )
    assert data.observations.size == 1_032
    assert tuple(np.unique(data.holiday_type_index)) == (0, 1)
    assert np.array_equal(data.hour, frame.sort("target_timestamp")["hour"].to_numpy())


@pytest.mark.parametrize("mutation", ["missing", "extra", "final", "context", "nonhourly"])
def test_training_adapter_rejects_noncausal_or_incomplete_inputs(tmp_path, mutation):
    events = load_event_registry(EVENTS)
    frame = _training_frame()
    approved = _approved(tmp_path, tuple(sorted(frame["occurrence_id"].unique().to_list())))
    if mutation == "missing":
        frame = frame.slice(1)
    elif mutation == "extra":
        frame = pl.concat(
            [frame, frame.tail(1).with_columns(pl.lit("extra").alias("occurrence_id"))]
        )
    elif mutation == "final":
        frame = frame.with_columns(pl.lit("final-2024").alias("split_id"))
    elif mutation == "context":
        frame = frame.with_columns(pl.lit("B0").alias("feature_set"))
    else:
        frame = frame.with_columns(
            pl.when(pl.arange(0, pl.len()) == 1)
            .then(pl.col("target_timestamp") + pl.duration(hours=1))
            .otherwise(pl.col("target_timestamp"))
            .alias("target_timestamp")
        )

    with pytest.raises(CausalCorrectionError):
        build_causal_hqrc_data(frame, approved, events=events)


def test_latest_scale_is_only_finite_positive_oof_2023_context_value(tmp_path):
    frame = _training_frame()
    approved = _approved(tmp_path, tuple(sorted(frame["occurrence_id"].unique().to_list())))
    record = {
        "model": "lightgbm",
        "feature_set": "B1",
        "seed": 7,
        "latest_complete_oof_scale": {"split_id": "oof-2023", "sigma_n_mw": 123.5},
    }
    manifest = {"contexts": [record]}

    assert select_latest_oof_scale(manifest, approved) == 123.5
    record["latest_complete_oof_scale"]["split_id"] = "final-2024"
    with pytest.raises(CausalCorrectionError, match="oof-2023"):
        select_latest_oof_scale(manifest, approved)


def test_final_stream_selects_exact_two_2024_events_and_excludes_october_first():
    prediction = build_causal_prediction_data(
        _final_stream(),
        context=EventResidualContext("lightgbm", "B1", 7, SPLITS),
        events=load_event_registry(EVENTS),
    )

    assert prediction.full_frame.height == 7_320
    assert prediction.event_frame.height == 264
    assert prediction.occurrence_ids == ("seollal-2024", "chuseok-2024")
    counts = prediction.event_frame.group_by("occurrence_id").len().sort("occurrence_id")
    assert dict(counts.iter_rows()) == {"chuseok-2024": 120, "seollal-2024": 144}
    assert datetime(2024, 10, 1) not in prediction.event_frame["target_timestamp"].to_list()


def test_final_stream_rejects_context_or_coverage_change():
    frame = _final_stream().with_columns(pl.lit("B0").alias("feature_set"))
    with pytest.raises(CausalCorrectionError):
        build_causal_prediction_data(
            frame,
            context=EventResidualContext("lightgbm", "B1", 7, SPLITS),
            events=load_event_registry(EVENTS),
        )
