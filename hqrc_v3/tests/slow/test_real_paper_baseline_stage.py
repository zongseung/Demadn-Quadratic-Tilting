"""Real one-stream smoke for the concrete paper-baseline execution path."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from hqrc_v3.baselines.config import load_paper_baselines
from hqrc_v3.baselines.paper import run_paper_oof_stage
from hqrc_v3.config import load_config
from hqrc_v3.data import audit_hourly_data, read_hourly_data
from hqrc_v3.events import load_event_registry, load_holiday_calendar
from hqrc_v3.features import attach_calendar_features, build_daily_forecast_matrix
from hqrc_v3.provenance import file_sha256

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.slow
def test_real_lightgbm_b1_one_fold_uses_non_paper_stage_path(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[3] / "power_demand_final.csv"
    if not source.is_file():
        pytest.skip("repository power_demand_final.csv source is absent")
    experiment_path = PROJECT_ROOT / "configs/experiment.toml"
    model_path = PROJECT_ROOT / "configs/model_spaces.toml"
    event_path = PROJECT_ROOT / "configs/events.csv"
    calendar_path = PROJECT_ROOT / "configs/holiday_calendar.csv"
    load_config(experiment_path)
    load_event_registry(event_path)
    calendar = load_holiday_calendar(calendar_path)
    config = load_paper_baselines(model_path, expected_sha256=file_sha256(model_path))
    audited = audit_hourly_data(
        read_hourly_data(source),
        expected_start=datetime(2019, 1, 1),
        expected_end=datetime(2024, 10, 31, 23),
        expected_rows=51_144,
    )
    matrix = build_daily_forecast_matrix(
        attach_calendar_features(audited, calendar), feature_set="B1"
    )

    result = run_paper_oof_stage(
        matrices={"B1": matrix},
        config=config,
        run_dir=tmp_path,
        cache_dir=tmp_path / "stream-cache",
        artifact_hashes={
            "data_sha256": file_sha256(source),
            "experiment_sha256": file_sha256(experiment_path),
            "model_config_sha256": file_sha256(model_path),
            "event_registry_sha256": file_sha256(event_path),
            "holiday_calendar_sha256": file_sha256(calendar_path),
        },
        classical_seed=7,
        models=("lightgbm",),
        feature_sets=("B1",),
        profile="smoke",
        oof_years=(2020,),
        smoke_boosting_rounds=3,
    )

    point = pl.read_parquet(result.point_path)
    members = pl.read_parquet(result.members_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert result.fit_count == 1
    assert point["model"].unique().to_list() == ["lightgbm"]
    assert point["feature_set"].unique().to_list() == ["B1"]
    assert point["split_id"].unique().to_list() == ["oof-2020"]
    assert point.height > 0
    assert members.is_empty()
    assert manifest["profile"] == "smoke"
    assert manifest["execution_overrides"] == {"boosting_rounds": 3}
