from __future__ import annotations

import json
from dataclasses import FrozenInstanceError, replace
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import hqrc_v3.correction_source as source_module
from hqrc_v3.correction_source import (
    CorrectionSourceError,
    validate_correction_source,
)
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.events import EventOccurrence, EventRegistryError
from hqrc_v3.provenance import file_sha256
from hqrc_v3.residual_stage import build_standardized_residuals

CONTEXT = EventResidualContext(
    "lightgbm",
    "B1",
    7,
    tuple(f"oof-{year}" for year in range(2020, 2024)),
)


def _prediction_frame(
    *,
    split_id: str,
    observed: float = 100.0,
    context: EventResidualContext = CONTEXT,
) -> pl.DataFrame:
    year = 2024 if split_id == "final-2024" else int(split_id.removeprefix("oof-"))
    origin = datetime(year, 1, 1)
    return pl.DataFrame(
        {
            "origin": [origin] * 24,
            "target_timestamp": [origin + timedelta(hours=index) for index in range(24)],
            "horizon": list(range(1, 25)),
            "observed_mw": [observed] * 24,
            "predicted_mw": [90.0] * 24,
            "model": [context.model] * 24,
            "feature_set": [context.feature_set] * 24,
            "seed": [context.seed] * 24,
            "split_id": [split_id] * 24,
        }
    )


def _canonical_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def _standardized_fixture(
    context: EventResidualContext, events: tuple[EventOccurrence, ...]
) -> tuple[pl.DataFrame, list[dict[str, object]]]:
    frames = []
    for event in events:
        rows = []
        days = [
            event.window_start + timedelta(days=offset)
            for offset in range((event.window_end - event.window_start).days + 1)
        ]
        non_event_day = event.window_end + timedelta(days=7)
        for day in (*days, non_event_day):
            origin = datetime.combine(day, datetime.min.time())
            residual = 10.0 if day in days else 2.0
            for hour in range(24):
                rows.append(
                    {
                        "origin": origin,
                        "target_timestamp": origin + timedelta(hours=hour),
                        "horizon": hour + 1,
                        "observed_mw": 100.0,
                        "predicted_mw": 100.0 - residual,
                        "model": context.model,
                        "feature_set": context.feature_set,
                        "seed": context.seed,
                        "split_id": f"oof-{event.central_date.year}",
                    }
                )
        frames.append(pl.DataFrame(rows))
    build = build_standardized_residuals(pl.concat(frames), events)
    scales = [
        {
            "split_id": scale.split_id,
            "sigma_n_mw": scale.sigma_n_mw,
            "non_event_rows": scale.non_event_rows,
        }
        for scale in build.scales
    ]
    return build.frame, scales


def _install_source_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    feature_set: str = "B1",
):
    context = EventResidualContext(
        CONTEXT.model, feature_set, CONTEXT.seed, CONTEXT.split_ids
    )
    events = tuple(
        EventOccurrence(
            occurrence_id=f"seollal-{year}",
            holiday_type="seollal",
            central_date=datetime(year, 1, 25).date(),
            official_start=datetime(year, 1, 24).date(),
            official_end=datetime(year, 1, 26).date(),
            restriction=0,
        )
        for year in range(2020, 2024)
    )
    run = tmp_path / "run"
    inputs = run / "inputs"
    predictions = run / "predictions"
    inputs.mkdir(parents=True)
    predictions.mkdir(parents=True)
    (inputs / ".standardized-residuals.lock").touch()
    (predictions / ".baseline-publication.lock").touch()

    source_paths = {
        "data": tmp_path / "data.csv",
        "experiment_config": tmp_path / "experiment.toml",
        "model_config": tmp_path / "model_spaces.toml",
        "event_registry": tmp_path / "events.csv",
        "holiday_calendar": tmp_path / "holiday_calendar.csv",
        "temporary_holiday_availability": tmp_path
        / "temporary_holiday_availability.csv",
    }
    for name, path in source_paths.items():
        path.write_text(name, encoding="utf-8")
    source_hashes = {name: file_sha256(path) for name, path in source_paths.items()}

    oof_point = predictions / "oof.parquet"
    oof_members = predictions / "oof_members.parquet"
    final_point = predictions / "final_2024.parquet"
    final_members = predictions / "final_2024_members.parquet"
    oof_frame = pl.concat(
        [
            _prediction_frame(split_id=split_id, context=context)
            for split_id in context.split_ids
        ]
    )
    oof_frame.write_parquet(oof_point)
    pl.DataFrame(schema=oof_frame.schema).write_parquet(oof_members)
    _prediction_frame(split_id="final-2024", context=context).write_parquet(final_point)
    pl.DataFrame(
        schema=_prediction_frame(split_id="final-2024", context=context).schema
    ).write_parquet(
        final_members
    )
    residual_path = inputs / "standardized_residuals.parquet"
    residual_frame, fold_scales = _standardized_fixture(context, events)
    residual_frame.write_parquet(residual_path)

    residual_manifest = {
        "profile": "smoke",
        "inputs": {
            **{
                name: {"path": path.resolve().as_posix(), "sha256": source_hashes[name]}
                for name, path in source_paths.items()
            },
            "baseline_oof_semantics": {
                "path": "predictions/baseline_manifest.json",
                "sha256": "b" * 64,
            },
            "oof_members": {
                "path": "predictions/oof_members.parquet",
                "sha256": file_sha256(oof_members),
            },
            "oof_predictions": {
                "path": "predictions/oof.parquet",
                "sha256": file_sha256(oof_point),
            },
        },
        "outputs": {
            "standardized_residuals": {
                "path": "inputs/standardized_residuals.parquet",
                "sha256": file_sha256(residual_path),
                "rows": residual_frame.height,
            }
        },
        "contexts": [
            {
                "model": context.model,
                "feature_set": context.feature_set,
                "seed": context.seed,
                "split_ids": list(context.split_ids),
                "occurrence_ids": [event.occurrence_id for event in events],
                "rows": residual_frame.height,
                "fold_scales": fold_scales,
                "latest_complete_oof_scale": {
                    "split_id": "oof-2023",
                    "sigma_n_mw": fold_scales[-1]["sigma_n_mw"],
                },
            }
        ],
    }
    residual_manifest_path = inputs / "standardized_residuals_manifest.json"
    _canonical_write(residual_manifest_path, residual_manifest)

    baseline_manifest = {
        "profile": "smoke",
        "models": [context.model],
        "feature_sets": [context.feature_set],
        "classical_seed": context.seed,
        "execution_overrides": {"boosting_rounds": 3},
        "stages": {
            "oof": {
                "artifacts": {
                    "members": {
                        "path": "predictions/oof_members.parquet",
                        "sha256": file_sha256(oof_members),
                    },
                    "point": {
                        "path": "predictions/oof.parquet",
                        "sha256": file_sha256(oof_point),
                    },
                }
            },
            "final": {
                "artifacts": {
                    "members": {
                        "path": "predictions/final_2024_members.parquet",
                        "sha256": file_sha256(final_members),
                    },
                    "point": {
                        "path": "predictions/final_2024.parquet",
                        "sha256": file_sha256(final_point),
                    },
                }
            },
        },
    }
    baseline_manifest_path = predictions / "baseline_manifest.json"
    _canonical_write(baseline_manifest_path, baseline_manifest)

    target_times = np.asarray(
        [[np.datetime64(datetime(2024, 1, 1) + timedelta(hours=index)) for index in range(24)]]
    )
    matrix = SimpleNamespace(
        origins=np.asarray([np.datetime64(datetime(2024, 1, 1))]),
        target_times=target_times,
        target=np.full((1, 24), 100.0),
    )
    matrix.take = lambda _indices: matrix

    monkeypatch.setattr(source_module, "load_config", lambda _path: object())
    monkeypatch.setattr(source_module, "load_event_registry", lambda _path: events)
    monkeypatch.setattr(source_module, "load_holiday_calendar", lambda _path: events)
    monkeypatch.setattr(
        source_module, "load_temporary_holiday_availability", lambda _path: object()
    )
    monkeypatch.setattr(
        source_module,
        "load_standardized_residual_manifest",
        lambda *_args, **_kwargs: json.loads(residual_manifest_path.read_bytes()),
    )
    monkeypatch.setattr(source_module, "load_paper_baselines", lambda _path: object())
    monkeypatch.setattr(source_module, "read_hourly_data", lambda _path: object())
    monkeypatch.setattr(source_module, "audit_hourly_data", lambda value, **_kwargs: value)
    monkeypatch.setattr(
        source_module, "attach_calendar_features", lambda value, _calendar: value
    )
    monkeypatch.setattr(
        source_module, "build_daily_forecast_matrix", lambda _value, *, feature_set: matrix
    )
    runner_state = SimpleNamespace(calls=[], fit_count=0)

    def run_paper_final_stage(**kwargs):
        runner_state.calls.append(kwargs)
        return SimpleNamespace(
            members_path=final_members,
            point_path=final_point,
            manifest_path=baseline_manifest_path,
            fit_count=runner_state.fit_count,
        )

    monkeypatch.setattr(source_module, "run_paper_final_stage", run_paper_final_stage)
    monkeypatch.setattr(source_module, "final_fold", lambda: object())
    monkeypatch.setattr(
        source_module,
        "select_fold_samples",
        lambda _matrix, _fold: (np.asarray([], dtype=np.int64), np.asarray([0])),
    )
    monkeypatch.setattr(
        source_module,
        "build_standardized_residuals",
        lambda _frame, _events: SimpleNamespace(frame=residual_frame.clone()),
    )
    return SimpleNamespace(
        run=run,
        sources=source_paths,
        residual_manifest=residual_manifest,
        residual_manifest_path=residual_manifest_path,
        baseline_manifest=baseline_manifest,
        baseline_manifest_path=baseline_manifest_path,
        residual_path=residual_path,
        oof_point=oof_point,
        final_members=final_members,
        final_point=final_point,
        runner_state=runner_state,
        context=context,
        event=events[0],
    )


def test_validated_source_is_immutable_and_loads_canonical_context_streams(
    tmp_path, monkeypatch
):
    fixture = _install_source_fixture(tmp_path, monkeypatch)

    source = validate_correction_source(
        run_dir=fixture.run,
        config_path=fixture.sources["experiment_config"],
        profile="smoke",
    )

    with pytest.raises(FrozenInstanceError):
        source.run_dir = tmp_path  # type: ignore[misc]
    with pytest.raises(TypeError):
        source.source_hashes["data"] = "changed"  # type: ignore[index]
    changed_manifest = source.residual_manifest
    changed_manifest["profile"] = "changed"
    assert source.residual_manifest["profile"] == "smoke"
    assert source.available_contexts == (CONTEXT,)
    assert source.load_standardized_context(CONTEXT, through=2023).equals(
        pl.read_parquet(fixture.residual_path)
    )
    assert source.load_oof_point_context(CONTEXT).equals(pl.read_parquet(fixture.oof_point))
    assert source.load_final_point_context(CONTEXT).equals(pl.read_parquet(fixture.final_point))
    assert len(fixture.runner_state.calls) == 1


def test_validated_source_accepts_b1w_context_without_aliasing_b1(
    tmp_path, monkeypatch
) -> None:
    fixture = _install_source_fixture(tmp_path, monkeypatch, feature_set="B1W")

    source = validate_correction_source(
        run_dir=fixture.run,
        config_path=fixture.sources["experiment_config"],
        profile="smoke",
    )

    assert source.available_contexts == (fixture.context,)
    assert fixture.runner_state.calls[0]["feature_sets"] == ("B1W",)
    assert source.load_standardized_context(
        fixture.context, through=2023
    ).equals(pl.read_parquet(fixture.residual_path))
    with pytest.raises(CorrectionSourceError, match="requested context"):
        source.load_oof_point_context(CONTEXT)


def test_validated_source_rejects_feature_correction_registry_misalignment(
    tmp_path, monkeypatch
) -> None:
    fixture = _install_source_fixture(tmp_path, monkeypatch, feature_set="B1W")
    changed = replace(
        fixture.event,
        official_start=fixture.event.official_start + timedelta(days=1),
    )
    monkeypatch.setattr(source_module, "load_holiday_calendar", lambda _path: (changed,))

    with pytest.raises(EventRegistryError, match="feature/correction window registry"):
        validate_correction_source(
            run_dir=fixture.run,
            config_path=fixture.sources["experiment_config"],
            profile="smoke",
        )
    assert fixture.runner_state.calls == []


@pytest.mark.parametrize(
    ("condition", "artifact"),
    [
        ("missing", "members"),
        ("missing", "point"),
        ("corrupt", "members"),
        ("corrupt", "point"),
        ("symlink", "members"),
        ("symlink", "point"),
        ("hash", "members"),
        ("hash", "point"),
        ("namespace", "point"),
    ],
)
def test_final_artifact_preflight_rejects_before_baseline_runner(
    tmp_path, monkeypatch, condition, artifact
):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    path = fixture.final_members if artifact == "members" else fixture.final_point
    entry = fixture.baseline_manifest["stages"]["final"]["artifacts"][artifact]
    if condition == "missing":
        path.unlink()
    elif condition == "corrupt":
        path.write_bytes(b"not a parquet publication")
        entry["sha256"] = file_sha256(path)
        _canonical_write(fixture.baseline_manifest_path, fixture.baseline_manifest)
    elif condition == "symlink":
        target = tmp_path / f"{artifact}-copy.parquet"
        target.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(target)
    elif condition == "hash":
        entry["sha256"] = "0" * 64
        _canonical_write(fixture.baseline_manifest_path, fixture.baseline_manifest)
    else:
        (fixture.run / "predictions/unknown.bin").write_bytes(b"unknown")

    with pytest.raises(CorrectionSourceError):
        validate_correction_source(
            run_dir=fixture.run,
            config_path=fixture.sources["experiment_config"],
            profile="smoke",
        )

    assert fixture.runner_state.calls == []


def test_valid_final_artifacts_call_reuse_runner_and_reject_any_fit(tmp_path, monkeypatch):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    fixture.runner_state.fit_count = 1

    with pytest.raises(CorrectionSourceError, match="unexpectedly fitted"):
        validate_correction_source(
            run_dir=fixture.run,
            config_path=fixture.sources["experiment_config"],
            profile="smoke",
        )

    assert len(fixture.runner_state.calls) == 1


def test_validated_source_rejects_missing_requested_context(tmp_path, monkeypatch):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    source = validate_correction_source(
        run_dir=fixture.run,
        config_path=fixture.sources["experiment_config"],
        profile="smoke",
    )
    missing = EventResidualContext("xgboost", "B1", 7, CONTEXT.split_ids)

    with pytest.raises(CorrectionSourceError, match="requested context"):
        source.load_final_point_context(missing)


@pytest.mark.parametrize(
    ("mutation", "split_ids"),
    [
        ("subset", CONTEXT.split_ids[:-1]),
        ("reordered", (CONTEXT.split_ids[1], CONTEXT.split_ids[0], *CONTEXT.split_ids[2:])),
        ("duplicate", (*CONTEXT.split_ids, CONTEXT.split_ids[-1])),
        ("substituted", (*CONTEXT.split_ids[:-1], "oof-2024")),
    ],
)
@pytest.mark.parametrize(
    "loader",
    ["standardized", "oof-point", "final-point"],
)
def test_all_context_loaders_reject_changed_split_identity(
    tmp_path, monkeypatch, mutation, split_ids, loader
):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    source = validate_correction_source(
        run_dir=fixture.run,
        config_path=fixture.sources["experiment_config"],
        profile="smoke",
    )
    changed = EventResidualContext(
        CONTEXT.model,
        CONTEXT.feature_set,
        CONTEXT.seed,
        split_ids,
    )

    with pytest.raises(CorrectionSourceError, match="requested context"):
        if loader == "standardized":
            source.load_standardized_context(changed, through=2023)
        elif loader == "oof-point":
            source.load_oof_point_context(changed)
        else:
            source.load_final_point_context(changed)


def test_source_preflight_rejects_same_content_config_path_substitution(tmp_path, monkeypatch):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    substitute = tmp_path / "substitute.toml"
    substitute.write_bytes(fixture.sources["experiment_config"].read_bytes())

    with pytest.raises(CorrectionSourceError, match="requested experiment config"):
        validate_correction_source(
            run_dir=fixture.run,
            config_path=substitute,
            profile="smoke",
        )


def test_source_preflight_rejects_symlinked_requested_config(tmp_path, monkeypatch):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    substitute = tmp_path / "config-link.toml"
    substitute.symlink_to(fixture.sources["experiment_config"])

    with pytest.raises(CorrectionSourceError, match="requested experiment config.*unsafe"):
        validate_correction_source(
            run_dir=fixture.run,
            config_path=substitute,
            profile="smoke",
        )


def test_source_preflight_rejects_rehashed_semantic_target_mutation(tmp_path, monkeypatch):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    changed = pl.read_parquet(fixture.final_point).with_columns(
        (pl.col("observed_mw") + 1.0).alias("observed_mw")
    )
    changed.write_parquet(fixture.final_point)
    fixture.baseline_manifest["stages"]["final"]["artifacts"]["point"][
        "sha256"
    ] = file_sha256(fixture.final_point)
    _canonical_write(fixture.baseline_manifest_path, fixture.baseline_manifest)

    with pytest.raises(CorrectionSourceError, match="source-derived truth"):
        validate_correction_source(
            run_dir=fixture.run,
            config_path=fixture.sources["experiment_config"],
            profile="smoke",
        )


@pytest.mark.parametrize("mutation", ["unknown", "symlink"])
def test_source_preflight_rejects_unknown_or_symlinked_publication_entries(
    tmp_path, monkeypatch, mutation
):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    if mutation == "unknown":
        (fixture.run / "inputs/unknown.bin").write_bytes(b"unknown")
    else:
        target = fixture.run / "final-copy.parquet"
        target.write_bytes(fixture.final_point.read_bytes())
        fixture.final_point.unlink()
        fixture.final_point.symlink_to(target)

    with pytest.raises(CorrectionSourceError, match="unknown|unsafe"):
        validate_correction_source(
            run_dir=fixture.run,
            config_path=fixture.sources["experiment_config"],
            profile="smoke",
        )


def test_validated_source_rechecks_artifact_hash_before_loading_context(tmp_path, monkeypatch):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    source = validate_correction_source(
        run_dir=fixture.run,
        config_path=fixture.sources["experiment_config"],
        profile="smoke",
    )
    pl.read_parquet(fixture.oof_point).with_columns(
        (pl.col("predicted_mw") + 1.0).alias("predicted_mw")
    ).write_parquet(fixture.oof_point)

    with pytest.raises(CorrectionSourceError, match="hash differs"):
        source.load_oof_point_context(CONTEXT)


def test_validated_source_rechecks_namespace_before_loading_context(tmp_path, monkeypatch):
    fixture = _install_source_fixture(tmp_path, monkeypatch)
    source = validate_correction_source(
        run_dir=fixture.run,
        config_path=fixture.sources["experiment_config"],
        profile="smoke",
    )
    (fixture.run / "predictions/unknown.bin").write_bytes(b"unknown")

    with pytest.raises(CorrectionSourceError, match="unknown entries"):
        source.load_final_point_context(CONTEXT)
