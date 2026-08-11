from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import hqrc_v3.residual_stage as residual_stage
from hqrc_v3.baselines.config import MODEL_NAMES, PAPER_SEEDS
from hqrc_v3.events import EventOccurrence
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
from hqrc_v3.residual_stage import (
    STANDARDIZED_RESIDUAL_COLUMNS,
    build_standardized_residuals,
    load_standardized_residual_manifest,
    prepare_standardized_residual_artifact,
)


def _daily_predictions(
    days: list[date],
    *,
    model: str = "lightgbm",
    feature_set: str = "B1",
    seed: int = 7,
    split_id: str,
    residual_by_day: dict[date, float],
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for day in days:
        origin = datetime.combine(day, datetime.min.time())
        residual = residual_by_day[day]
        for horizon in range(1, 25):
            rows.append(
                {
                    "origin": origin,
                    "target_timestamp": origin + timedelta(hours=horizon - 1),
                    "horizon": horizon,
                    "observed_mw": 100.0,
                    "predicted_mw": 100.0 - residual,
                    "model": model,
                    "feature_set": feature_set,
                    "seed": seed,
                    "split_id": split_id,
                }
            )
    return pl.DataFrame(rows)


def test_build_standardized_residuals_uses_fold_local_non_event_scale_and_exact_windows():
    events = (
        EventOccurrence(
            "seollal-2020",
            "seollal",
            date(2020, 1, 25),
            date(2020, 1, 25),
            date(2020, 1, 25),
            0,
        ),
        EventOccurrence(
            "chuseok-2021",
            "chuseok",
            date(2021, 9, 21),
            date(2021, 9, 21),
            date(2021, 9, 21),
            1,
        ),
    )
    frames: list[pl.DataFrame] = []
    for event, split_id, non_event_residual in (
        (events[0], "oof-2020", 2.0),
        (events[1], "oof-2021", 4.0),
    ):
        event_days = [event.window_start + timedelta(days=offset) for offset in range(3)]
        non_event_day = event.window_end + timedelta(days=2)
        days = [*event_days, non_event_day]
        frames.append(
            _daily_predictions(
                days,
                split_id=split_id,
                residual_by_day={
                    **{day: 10.0 for day in event_days},
                    non_event_day: non_event_residual,
                },
            )
        )

    result = build_standardized_residuals(pl.concat(frames), events)

    assert tuple(result.frame.columns) == STANDARDIZED_RESIDUAL_COLUMNS
    assert result.frame.height == 2 * 3 * 24
    assert [(item.split_id, item.sigma_n_mw, item.non_event_rows) for item in result.scales] == [
        ("oof-2020", pytest.approx(2.0), 24),
        ("oof-2021", pytest.approx(4.0), 24),
    ]
    by_event = result.frame.partition_by("occurrence_id", as_dict=True)
    seollal = by_event[("seollal-2020",)]
    chuseok = by_event[("chuseok-2021",)]
    assert seollal["standardized_residual"].unique().item() == pytest.approx(5.0)
    assert chuseok["standardized_residual"].unique().item() == pytest.approx(2.5)
    assert (seollal["tau_days"].min(), seollal["tau_days"].max()) == pytest.approx(
        (-1.0, 1.0 + 23.0 / 24.0)
    )
    assert seollal["hour"].to_list() == list(range(24)) * 3
    assert seollal["restriction"].unique().to_list() == [0]
    assert chuseok["restriction"].unique().to_list() == [1]
    assert result.frame.select((pl.col("residual_mw") == 10.0).all()).item()


def test_scale_is_separate_for_every_model_feature_seed_and_fold_context():
    event = EventOccurrence(
        "seollal-2020",
        "seollal",
        date(2020, 1, 25),
        date(2020, 1, 25),
        date(2020, 1, 25),
        0,
    )
    event_days = [event.window_start + timedelta(days=offset) for offset in range(3)]
    non_event_day = event.window_end + timedelta(days=2)
    days = [*event_days, non_event_day]
    first = _daily_predictions(
        days,
        model="xgboost",
        feature_set="B0",
        seed=7,
        split_id="oof-2020",
        residual_by_day={**{day: 10.0 for day in event_days}, non_event_day: 2.0},
    )
    second = _daily_predictions(
        days,
        model="transformer",
        feature_set="B1",
        seed=0,
        split_id="oof-2020",
        residual_by_day={**{day: 12.0 for day in event_days}, non_event_day: 3.0},
    )

    result = build_standardized_residuals(pl.concat([first, second]), (event,))

    assert [item.sigma_n_mw for item in result.scales] == pytest.approx([3.0, 2.0])
    standardized = {
        (model, feature): value
        for model, feature, value in result.frame.group_by("model", "feature_set").agg(
            pl.col("standardized_residual").first()
        ).iter_rows()
    }
    assert standardized == {
        ("transformer", "B1"): pytest.approx(4.0),
        ("xgboost", "B0"): pytest.approx(5.0),
    }


def _full_year_prediction_frame(
    *, year: int, model: str = "lightgbm", feature_set: str = "B1", seed: int = 7
) -> pl.DataFrame:
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31, 23)
    timestamps = pl.datetime_range(start, end, interval="1h", eager=True, time_unit="ns")
    return pl.DataFrame(
        {
            "target_timestamp": timestamps,
            "observed_mw": pl.Series([100.0], dtype=pl.Float64)
            .repeat_by(len(timestamps))
            .explode(),
        }
    ).with_columns(
        pl.col("target_timestamp").dt.truncate("1d").alias("origin"),
        (pl.col("target_timestamp").dt.hour() + 1).cast(pl.Int64).alias("horizon"),
        pl.lit(98.0, dtype=pl.Float64).alias("predicted_mw"),
        pl.lit(model).alias("model"),
        pl.lit(feature_set).alias("feature_set"),
        pl.lit(seed, dtype=pl.Int64).alias("seed"),
        pl.lit(f"oof-{year}").alias("split_id"),
    ).select(
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


def _write_smoke_baseline_publication(run: Path, config: Path, events: Path) -> None:
    predictions = run / "predictions"
    predictions.mkdir(parents=True)
    point_path = predictions / "oof.parquet"
    _full_year_prediction_frame(year=2020).write_parquet(point_path)
    manifest = {
        "schema_version": 2,
        "profile": "smoke",
        "input_hashes": {
            "data_sha256": "a" * 64,
            "experiment_sha256": file_sha256(config),
            "model_config_sha256": "b" * 64,
            "event_registry_sha256": file_sha256(events),
            "holiday_calendar_sha256": "c" * 64,
            "temporary_holiday_availability_sha256": "d" * 64,
        },
        "models": ["lightgbm"],
        "feature_sets": ["B1"],
        "classical_seed": 7,
        "neural_seeds": list(PAPER_SEEDS),
        "ensemble_seed": 0,
        "feature_schemas": {},
        "preprocessing": {},
        "execution_overrides": {"boosting_rounds": 3},
        "stages": {
            "oof": {
                "eval_years": [2020],
                "expected_coverage": {"count": 366 * 24, "sha256": "c" * 64},
                "preprocessing_populations": [],
                "split_ids": ["oof-2020"],
                "streams": [{"model": "lightgbm", "feature_set": "B1", "seeds": [7]}],
                "artifacts": {
                    "members": {
                        "path": "predictions/oof_members.parquet",
                        "sha256": "e" * 64,
                    },
                    "point": {
                        "path": "predictions/oof.parquet",
                        "sha256": file_sha256(point_path),
                    },
                },
            }
        },
    }
    (predictions / "baseline_manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def test_prepare_artifact_is_hash_bound_atomic_and_reusable(tmp_path: Path):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_smoke_baseline_publication(run, config, events)

    first = prepare_standardized_residual_artifact(
        run_dir=run,
        config_path=config,
        event_registry_path=events,
        profile="smoke",
    )
    artifact_sha = file_sha256(first.residual_path)
    first_manifest = load_standardized_residual_manifest(
        first.manifest_path,
        run_dir=run,
        config_sha256=file_sha256(config),
        event_sha256=file_sha256(events),
    )

    assert first.reused is False
    assert first.frame.height == 264
    assert first_manifest["outputs"]["standardized_residuals"]["sha256"] == artifact_sha
    assert first_manifest["contexts"] == [
        {
            "feature_set": "B1",
            "fold_scales": [
                {
                    "non_event_rows": 8520,
                    "sigma_n_mw": 2.0,
                    "split_id": "oof-2020",
                }
            ],
            "latest_complete_oof_scale": {
                "sigma_n_mw": 2.0,
                "split_id": "oof-2020",
            },
            "model": "lightgbm",
            "occurrence_ids": ["chuseok-2020", "seollal-2020"],
            "rows": 264,
            "seed": 7,
            "split_ids": ["oof-2020"],
        }
    ]

    second = prepare_standardized_residual_artifact(
        run_dir=run,
        config_path=config,
        event_registry_path=events,
        profile="smoke",
    )
    assert second.reused is True
    assert file_sha256(second.residual_path) == artifact_sha

    point = run / "predictions/oof.parquet"
    changed = pl.read_parquet(point).with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.col("predicted_mw") + 1.0)
        .otherwise(pl.col("predicted_mw"))
        .alias("predicted_mw")
    )
    changed.write_parquet(point)
    with pytest.raises(ArtifactMismatch, match="prediction"):
        prepare_standardized_residual_artifact(
            run_dir=run,
            config_path=config,
            event_registry_path=events,
            profile="smoke",
        )
    assert file_sha256(first.residual_path) == artifact_sha


def test_prepare_recovers_an_interrupted_first_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_smoke_baseline_publication(run, config, events)
    original_replace = residual_stage.os.replace
    calls = 0

    def fail_manifest_replace(source: str | Path, destination: str | Path) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected manifest publication failure")
        original_replace(source, destination)

    monkeypatch.setattr(residual_stage.os, "replace", fail_manifest_replace)
    with pytest.raises(OSError, match="injected"):
        prepare_standardized_residual_artifact(
            run_dir=run,
            config_path=config,
            event_registry_path=events,
            profile="smoke",
        )
    assert (run / "inputs/standardized_residuals.parquet").is_file()
    assert not (run / "inputs/standardized_residuals_manifest.json").exists()

    monkeypatch.setattr(residual_stage.os, "replace", original_replace)
    recovered = prepare_standardized_residual_artifact(
        run_dir=run,
        config_path=config,
        event_registry_path=events,
        profile="smoke",
    )
    assert recovered.reused is False
    assert recovered.manifest_path.is_file()


def test_paper_profile_requires_all_five_models_and_both_feature_sets(tmp_path: Path):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_smoke_baseline_publication(run, config, events)

    with pytest.raises(ArtifactMismatch, match="profile"):
        prepare_standardized_residual_artifact(
            run_dir=run,
            config_path=config,
            event_registry_path=events,
            profile="paper",
        )
    assert tuple(MODEL_NAMES) == (
        "xgboost",
        "lightgbm",
        "svr",
        "seq2seq_lstm",
        "transformer",
    )
