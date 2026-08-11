"""Opt-in real-source proof of the Task 12 causal baseline contract."""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hqrc_v3.baselines.config import load_paper_baselines
from hqrc_v3.baselines.paper import run_paper_oof_stage
from hqrc_v3.data import (
    FIXED_PUBLIC_HOLIDAY_DATES,
    FIXED_ROWS,
    FIXED_SUBSTITUTE_OR_TEMPORARY_DATES,
    audit_hourly_data,
    load_temporary_holiday_availability,
    read_hourly_data,
)
from hqrc_v3.events import load_event_registry, load_holiday_calendar
from hqrc_v3.features import (
    attach_calendar_features,
    build_daily_forecast_matrix,
    feature_columns,
    history_columns,
)
from hqrc_v3.provenance import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SOURCE = REPOSITORY_ROOT / "power_demand_final.csv"
MODEL_CONFIG = PROJECT_ROOT / "configs/model_spaces.toml"
EXPERIMENT_CONFIG = PROJECT_ROOT / "configs/experiment.toml"
EVENT_REGISTRY = PROJECT_ROOT / "configs/events.csv"
HOLIDAY_CALENDAR = PROJECT_ROOT / "configs/holiday_calendar.csv"
TEMPORARY_AVAILABILITY = PROJECT_ROOT / "configs/temporary_holiday_availability.csv"


@pytest.mark.slow
def test_real_causal_matrix_lightgbm_inverse_mw_and_manifest_reload(tmp_path: Path) -> None:
    if not SOURCE.is_file():
        pytest.skip("repository power_demand_final.csv source is absent")

    availability = load_temporary_holiday_availability(TEMPORARY_AVAILABILITY)
    audited = audit_hourly_data(
        read_hourly_data(SOURCE),
        expected_start=datetime(2019, 1, 1),
        expected_end=datetime(2024, 10, 31, 23),
        expected_rows=FIXED_ROWS,
        expected_public_holiday_dates=FIXED_PUBLIC_HOLIDAY_DATES,
        expected_substitute_or_temporary_dates=(
            FIXED_SUBSTITUTE_OR_TEMPORARY_DATES
        ),
        temporary_holiday_availability=availability,
    )
    calendar = load_holiday_calendar(HOLIDAY_CALENDAR)
    events = load_event_registry(EVENT_REGISTRY)
    featured = attach_calendar_features(audited, calendar)
    assert isinstance(featured, pl.DataFrame)

    public_dates = featured.filter(pl.col("is_public_holiday") == 1)["date"].n_unique()
    exceptional_dates = featured.filter(
        pl.col("is_substitute_or_temporary_holiday") == 1
    )["date"].n_unique()
    assert (audited.height, public_dates, exceptional_dates) == (51_144, 107, 14)
    assert featured.select((pl.col("is_seollal") + pl.col("is_chuseok")).max()).item() <= 1
    armed_forces_day = featured.filter(pl.col("date") == date(2024, 10, 1))
    assert armed_forces_day.select(
        "is_public_holiday",
        "is_substitute_or_temporary_holiday",
        "is_seollal",
        "is_chuseok",
    ).unique().row(0) == (1, 1, 0, 0)
    assert not any(
        event.window_start <= date(2024, 10, 1) <= event.window_end for event in events
    )

    b0 = build_daily_forecast_matrix(featured, feature_set="B0")
    b1 = build_daily_forecast_matrix(featured, feature_set="B1")
    assert len(calendar) == 14
    assert len(events) == 10
    assert len(b0.origins) == len(b1.origins) == 2_124
    assert b0.history.shape == (2_124, 168, 10)
    assert b0.future.shape == (2_124, 24, 7)
    assert b1.history.shape == (2_124, 168, 17)
    assert b1.future.shape == (2_124, 24, 14)
    assert b0.history_columns == history_columns("B0")
    assert b0.future_columns == feature_columns("B0")
    assert b1.history_columns == history_columns("B1")
    assert b1.future_columns == feature_columns("B1")
    forbidden = ("oracle", "temperature", "humidity", "heating", "cooling")
    assert not any(
        token in column.lower()
        for column in (*b0.future_columns, *b1.future_columns)
        for token in forbidden
    )
    assert np.array_equal(b0.origins, b1.origins)
    assert np.array_equal(b0.target_times, b1.target_times)
    assert np.array_equal(b0.target, b1.target)

    hashes = {
        "data_sha256": file_sha256(SOURCE),
        "experiment_sha256": file_sha256(EXPERIMENT_CONFIG),
        "model_config_sha256": file_sha256(MODEL_CONFIG),
        "event_registry_sha256": file_sha256(EVENT_REGISTRY),
        "holiday_calendar_sha256": file_sha256(HOLIDAY_CALENDAR),
        "temporary_holiday_availability_sha256": file_sha256(
            TEMPORARY_AVAILABILITY
        ),
    }
    arguments = {
        "matrices": {"B1": b1},
        "config": load_paper_baselines(MODEL_CONFIG),
        "run_dir": tmp_path / "run",
        "cache_dir": tmp_path / "cache",
        "artifact_hashes": hashes,
        "classical_seed": 7,
        "models": ("lightgbm",),
        "feature_sets": ("B1",),
        "profile": "smoke",
        "oof_years": (2020,),
        "smoke_boosting_rounds": 3,
    }
    first = run_paper_oof_stage(**arguments)
    point = pl.read_parquet(first.point_path)
    assert first.fit_count == 1
    assert first.cache_hit_count == 0
    assert point.height == 366 * 24
    assert point["model"].unique().to_list() == ["lightgbm"]
    assert point["feature_set"].unique().to_list() == ["B1"]
    assert point["split_id"].unique().to_list() == ["oof-2020"]
    assert point["predicted_mw"].is_finite().all()
    assert point["predicted_mw"].mean() > 10_000.0

    manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    assert manifest["preprocessing"]["version"] == "causal-v1"
    assert manifest["input_hashes"]["temporary_holiday_availability_sha256"] == hashes[
        "temporary_holiday_availability_sha256"
    ]
    assert manifest["stages"]["oof"]["preprocessing_populations"] == [
        {
            "model": "lightgbm",
            "feature_set": "B1",
            "split_id": "oof-2020",
            "scalers": {
                "x": {
                    "count": 297,
                    "start": "2019-01-08T00:00:00",
                    "end": "2019-10-31T00:00:00",
                    "unit": "daily-sample",
                },
                "target": {
                    "count": 7_128,
                    "start": "2019-01-08T00:00:00",
                    "end": "2019-10-31T23:00:00",
                    "unit": "unique-hour",
                },
            },
        }
    ]

    reloaded = run_paper_oof_stage(**arguments)
    assert reloaded.fit_count == 0
    assert reloaded.cache_hit_count == 1
