from __future__ import annotations

import json
from datetime import date, datetime, timedelta
from pathlib import Path

import hqrc_v3.residual_stage as residual_stage
import polars as pl
import pytest
from hqrc_v3.baselines.config import MODEL_NAMES, PAPER_SEEDS, load_paper_baselines
from hqrc_v3.baselines.paper import derive_oof_source_truth, prediction_coverage_record
from hqrc_v3.data import (
    audit_hourly_data,
    load_temporary_holiday_availability,
    read_hourly_data,
)
from hqrc_v3.events import EventOccurrence, load_holiday_calendar
from hqrc_v3.features import (
    attach_calendar_features,
    build_daily_forecast_matrix,
    feature_columns,
    history_columns,
)
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


def _write_smoke_source(run: Path) -> Path:
    source = run / "source.csv"
    if source.is_file():
        return source
    run.mkdir(parents=True, exist_ok=True)
    timestamps = pl.datetime_range(
        datetime(2019, 1, 1),
        datetime(2020, 12, 31, 23),
        interval="1h",
        eager=True,
        time_unit="ns",
    )
    pl.DataFrame({"일시": timestamps}).with_columns(
        pl.lit(50.0).alias("hm"),
        pl.lit(15.0).alias("ta"),
        pl.lit(100.0).alias("power demand(MW)"),
        pl.when(pl.col("일시").dt.date() == date(2020, 8, 17))
        .then(pl.lit("Temporary Public Holiday"))
        .otherwise(pl.lit(""))
        .alias("holiday_name"),
        (pl.col("일시").dt.date() == date(2020, 8, 17))
        .cast(pl.Int8)
        .alias("is_holiday_dummies"),
    ).write_csv(source)
    return source


def _source_arguments(run: Path, config: Path, events: Path) -> dict[str, Path]:
    return {
        "data_path": _write_smoke_source(run),
        "config_path": config,
        "model_config_path": config.parent / "model_spaces.toml",
        "event_registry_path": events,
        "holiday_calendar_path": config.parent / "holiday_calendar.csv",
        "temporary_holiday_availability_path": (
            config.parent / "temporary_holiday_availability.csv"
        ),
    }


def _prepare(
    run: Path, config: Path, events: Path, *, profile: str = "smoke"
) -> residual_stage.StandardizedResidualArtifact:
    return prepare_standardized_residual_artifact(
        run_dir=run,
        **_source_arguments(run, config, events),
        profile=profile,
    )


def _source_truth(run: Path, config: Path, model: str) -> dict[str, object]:
    sources = _source_arguments(run, config, config.parent / "events.csv")
    availability = load_temporary_holiday_availability(
        sources["temporary_holiday_availability_path"]
    )
    audited = audit_hourly_data(
        read_hourly_data(sources["data_path"]),
        expected_start=None,
        expected_end=None,
        expected_rows=None,
        temporary_holiday_availability=availability,
    )
    featured = attach_calendar_features(
        audited, load_holiday_calendar(sources["holiday_calendar_path"])
    )
    matrix = build_daily_forecast_matrix(featured, feature_set="B1")
    return derive_oof_source_truth(
        matrices={"B1": matrix},
        config=load_paper_baselines(sources["model_config_path"]),
        models=(model,),
        feature_sets=("B1",),
        split_ids=("oof-2020",),
        eval_years=(2020,),
    )


def _write_smoke_baseline_publication(run: Path, config: Path, events: Path) -> None:
    predictions = run / "predictions"
    predictions.mkdir(parents=True)
    point_path = predictions / "oof.parquet"
    point_frame = _full_year_prediction_frame(year=2020)
    point_frame.write_parquet(point_path)
    member_path = predictions / "oof_members.parquet"
    pl.DataFrame(schema=point_frame.schema).select(point_frame.columns).write_parquet(
        member_path
    )
    sources = _source_arguments(run, config, events)
    baseline_config = load_paper_baselines(sources["model_config_path"])
    truth = _source_truth(run, config, "lightgbm")
    manifest = {
        "schema_version": 2,
        "profile": "smoke",
        "input_hashes": {
            "data_sha256": file_sha256(sources["data_path"]),
            "experiment_sha256": file_sha256(config),
            "model_config_sha256": file_sha256(sources["model_config_path"]),
            "event_registry_sha256": file_sha256(events),
            "holiday_calendar_sha256": file_sha256(sources["holiday_calendar_path"]),
            "temporary_holiday_availability_sha256": file_sha256(
                sources["temporary_holiday_availability_path"]
            ),
        },
        "models": ["lightgbm"],
        "feature_sets": ["B1"],
        "classical_seed": 7,
        "neural_seeds": list(PAPER_SEEDS),
        "ensemble_seed": 0,
        "feature_schemas": {
            "B1": {
                "history": list(history_columns("B1")),
                "future": list(feature_columns("B1")),
            }
        },
        "preprocessing": baseline_config.preprocessing.to_manifest(),
        "execution_overrides": {"boosting_rounds": 3},
        "stages": {
            "oof": {
                "eval_years": [2020],
                "expected_coverage": truth["expected_coverage"],
                "preprocessing_populations": truth["preprocessing_populations"],
                "split_ids": ["oof-2020"],
                "streams": [{"model": "lightgbm", "feature_set": "B1", "seeds": [7]}],
                "artifacts": {
                    "members": {
                        "path": "predictions/oof_members.parquet",
                        "sha256": file_sha256(member_path),
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


def _rewrite_manifest(path: Path, mutate) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    mutate(manifest)
    path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    return manifest


def _write_neural_smoke_baseline_publication(run: Path, config: Path, events: Path) -> None:
    _write_smoke_baseline_publication(run, config, events)
    predictions = run / "predictions"
    point_path = predictions / "oof.parquet"
    point = _full_year_prediction_frame(
        year=2020, model="transformer", feature_set="B1", seed=0
    )
    point.write_parquet(point_path)
    member_path = predictions / "oof_members.parquet"
    members = pl.concat(
        [
            _full_year_prediction_frame(
                year=2020,
                model="transformer",
                feature_set="B1",
                seed=seed,
            )
            for seed in PAPER_SEEDS
        ]
    )
    members.write_parquet(member_path)
    manifest_path = predictions / "baseline_manifest.json"

    def update(manifest: dict[str, object]) -> None:
        manifest["models"] = ["transformer"]
        truth = _source_truth(run, config, "transformer")
        manifest["stages"]["oof"]["preprocessing_populations"] = truth[
            "preprocessing_populations"
        ]
        stage = manifest["stages"]["oof"]
        stage["streams"] = [
            {"model": "transformer", "feature_set": "B1", "seeds": list(PAPER_SEEDS)}
        ]
        stage["artifacts"]["point"]["sha256"] = file_sha256(point_path)
        stage["artifacts"]["members"]["sha256"] = file_sha256(member_path)

    _rewrite_manifest(manifest_path, update)


def test_prepare_artifact_is_hash_bound_atomic_and_reusable(tmp_path: Path):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_smoke_baseline_publication(run, config, events)

    first = _prepare(run, config, events)
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

    second = _prepare(run, config, events)
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
        _prepare(run, config, events)
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
        _prepare(run, config, events)
    assert (run / "inputs/standardized_residuals.parquet").is_file()
    assert not (run / "inputs/standardized_residuals_manifest.json").exists()

    monkeypatch.setattr(residual_stage.os, "replace", original_replace)
    recovered = _prepare(run, config, events)
    assert recovered.reused is False
    assert recovered.manifest_path.is_file()


def test_prepare_rejects_tampered_neural_seed_contract(tmp_path: Path):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_smoke_baseline_publication(run, config, events)
    manifest_path = run / "predictions/baseline_manifest.json"
    _rewrite_manifest(manifest_path, lambda manifest: manifest.__setitem__("neural_seeds", [1]))

    with pytest.raises(ArtifactMismatch, match="neural seed"):
        _prepare(run, config, events)


def test_prepare_rejects_neural_point_that_is_not_exact_member_mean(tmp_path: Path):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_neural_smoke_baseline_publication(run, config, events)
    member_path = run / "predictions/oof_members.parquet"
    changed = pl.read_parquet(member_path).with_columns(
        pl.when(pl.arange(0, pl.len()) == 0)
        .then(pl.col("predicted_mw") + 5.0)
        .otherwise(pl.col("predicted_mw"))
        .alias("predicted_mw")
    )
    changed.write_parquet(member_path)
    manifest_path = run / "predictions/baseline_manifest.json"

    def rehash(manifest: dict[str, object]) -> None:
        manifest["stages"]["oof"]["artifacts"]["members"]["sha256"] = file_sha256(
            member_path
        )

    _rewrite_manifest(manifest_path, rehash)

    with pytest.raises(ArtifactMismatch, match="five-seed mean"):
        _prepare(run, config, events)


@pytest.mark.parametrize("tamper", ("count", "range"))
def test_prepare_rejects_self_consistent_but_false_scaler_population(
    tmp_path: Path, tamper: str
):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_smoke_baseline_publication(run, config, events)
    manifest_path = run / "predictions/baseline_manifest.json"

    def falsify_population(manifest: dict[str, object]) -> None:
        target = manifest["stages"]["oof"]["preprocessing_populations"][0][
            "scalers"
        ]["target"]
        if tamper == "count":
            target["count"] += 24
        else:
            target["start"] = "2019-01-02T00:00:00"

    _rewrite_manifest(manifest_path, falsify_population)

    with pytest.raises(ArtifactMismatch, match="preprocessing population"):
        _prepare(run, config, events)


def test_prepare_rejects_rehashed_observed_values_that_disagree_with_raw_data(
    tmp_path: Path,
):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_neural_smoke_baseline_publication(run, config, events)
    point_path = run / "predictions/oof.parquet"
    member_path = run / "predictions/oof_members.parquet"
    point = pl.read_parquet(point_path).with_columns(
        (pl.col("observed_mw") + 1_000.0).alias("observed_mw")
    )
    members = pl.read_parquet(member_path).with_columns(
        (pl.col("observed_mw") + 1_000.0).alias("observed_mw")
    )
    point.write_parquet(point_path)
    members.write_parquet(member_path)
    manifest_path = run / "predictions/baseline_manifest.json"

    def rebind_to_false_observations(manifest: dict[str, object]) -> None:
        stage = manifest["stages"]["oof"]
        stage["artifacts"]["point"]["sha256"] = file_sha256(point_path)
        stage["artifacts"]["members"]["sha256"] = file_sha256(member_path)
        stage["expected_coverage"] = prediction_coverage_record(point)

    _rewrite_manifest(manifest_path, rebind_to_false_observations)

    with pytest.raises(ArtifactMismatch, match="observed|source|coverage"):
        _prepare(run, config, events)


def test_residual_manifest_survives_legitimate_final_stage_append(tmp_path: Path):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_smoke_baseline_publication(run, config, events)
    prepared = _prepare(run, config, events)
    baseline_manifest = run / "predictions/baseline_manifest.json"
    _rewrite_manifest(
        baseline_manifest,
        lambda manifest: manifest["stages"].__setitem__("final", {"legitimate": "append"}),
    )

    reloaded = load_standardized_residual_manifest(
        prepared.manifest_path,
        run_dir=run,
        config_sha256=file_sha256(config),
        event_sha256=file_sha256(events),
    )
    assert reloaded["outputs"]["standardized_residuals"]["rows"] == 264


def test_diagnostic_context_rejects_fake_2024_occurrence_labeled_as_oof_2023():
    project = Path(__file__).resolve().parents[2]
    events = residual_stage.load_event_registry(project / "configs/events.csv")
    frames: list[pl.DataFrame] = []
    for year in range(2020, 2024):
        year_events = [event for event in events if event.central_date.year == year]
        event_days = sorted(
            {
                event.window_start + timedelta(days=offset)
                for event in year_events
                for offset in range((event.window_end - event.window_start).days + 1)
            }
        )
        non_event_day = date(year, 7, 1)
        frames.append(
            _daily_predictions(
                [*event_days, non_event_day],
                split_id=f"oof-{year}",
                residual_by_day={
                    **{day: 10.0 for day in event_days},
                    non_event_day: 2.0,
                },
            )
        )
    frame = build_standardized_residuals(pl.concat(frames), events).frame
    fake = frame.with_columns(
        pl.when(pl.col("occurrence_id") == "chuseok-2023")
        .then(pl.lit("fake-2024"))
        .otherwise(pl.col("occurrence_id"))
        .alias("occurrence_id")
    )
    context_record = {
        "model": "lightgbm",
        "feature_set": "B1",
        "seed": 7,
        "split_ids": [f"oof-{year}" for year in range(2020, 2024)],
        "occurrence_ids": sorted(fake["occurrence_id"].unique().to_list()),
    }

    with pytest.raises(ArtifactMismatch, match="occurrence"):
        residual_stage.select_diagnostic_residual_context(
            fake,
            {"profile": "paper", "contexts": [context_record]},
            events=events,
            model="lightgbm",
            feature_set="B1",
            seed=None,
            through=2023,
        )


def test_paper_profile_requires_all_five_models_and_both_feature_sets(tmp_path: Path):
    project = Path(__file__).resolve().parents[2]
    config = project / "configs/experiment.toml"
    events = project / "configs/events.csv"
    run = tmp_path / "run"
    _write_smoke_baseline_publication(run, config, events)

    with pytest.raises(ArtifactMismatch, match="profile"):
        _prepare(run, config, events, profile="paper")
    assert tuple(MODEL_NAMES) == (
        "xgboost",
        "lightgbm",
        "svr",
        "seq2seq_lstm",
        "transformer",
    )
