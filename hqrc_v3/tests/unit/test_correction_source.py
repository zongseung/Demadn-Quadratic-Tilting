from __future__ import annotations

import json
from dataclasses import FrozenInstanceError
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import hqrc_v3.correction_source as source_module
import numpy as np
import polars as pl
import pytest
from hqrc_v3.correction_source import (
    CorrectionSourceError,
    validate_correction_source,
)
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.events import EventOccurrence
from hqrc_v3.provenance import file_sha256

CONTEXT = EventResidualContext(
    "lightgbm",
    "B1",
    7,
    tuple(f"oof-{year}" for year in range(2020, 2024)),
)


def _prediction_frame(*, split_id: str, observed: float = 100.0) -> pl.DataFrame:
    year = 2024 if split_id == "final-2024" else int(split_id.removeprefix("oof-"))
    origin = datetime(year, 1, 1)
    return pl.DataFrame(
        {
            "origin": [origin] * 24,
            "target_timestamp": [origin + timedelta(hours=index) for index in range(24)],
            "horizon": list(range(1, 25)),
            "observed_mw": [observed] * 24,
            "predicted_mw": [90.0] * 24,
            "model": [CONTEXT.model] * 24,
            "feature_set": [CONTEXT.feature_set] * 24,
            "seed": [CONTEXT.seed] * 24,
            "split_id": [split_id] * 24,
        }
    )


def _canonical_write(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def _install_source_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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
        [_prediction_frame(split_id=split_id) for split_id in CONTEXT.split_ids]
    )
    oof_frame.write_parquet(oof_point)
    pl.DataFrame(schema=oof_frame.schema).write_parquet(oof_members)
    _prediction_frame(split_id="final-2024").write_parquet(final_point)
    pl.DataFrame(schema=_prediction_frame(split_id="final-2024").schema).write_parquet(
        final_members
    )
    residual_path = inputs / "standardized_residuals.parquet"
    residual_frame = pl.DataFrame(
        {
            "model": [CONTEXT.model],
            "feature_set": [CONTEXT.feature_set],
            "seed": [CONTEXT.seed],
            "split_id": ["oof-2023"],
            "standardized_residual": [1.0],
        }
    )
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
                "rows": 1,
            }
        },
        "contexts": [
            {
                "model": CONTEXT.model,
                "feature_set": CONTEXT.feature_set,
                "seed": CONTEXT.seed,
                "split_ids": list(CONTEXT.split_ids),
                "occurrence_ids": ["seollal-2023"],
                "rows": 1,
                "fold_scales": [],
                "latest_complete_oof_scale": {
                    "split_id": "oof-2023",
                    "sigma_n_mw": 10.0,
                },
            }
        ],
    }
    residual_manifest_path = inputs / "standardized_residuals_manifest.json"
    _canonical_write(residual_manifest_path, residual_manifest)

    baseline_manifest = {
        "profile": "smoke",
        "models": [CONTEXT.model],
        "feature_sets": [CONTEXT.feature_set],
        "classical_seed": CONTEXT.seed,
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

    event = EventOccurrence(
        occurrence_id="seollal-2023",
        holiday_type="seollal",
        central_date=datetime(2023, 1, 22).date(),
        official_start=datetime(2023, 1, 21).date(),
        official_end=datetime(2023, 1, 24).date(),
        restriction=0,
    )
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
    monkeypatch.setattr(source_module, "load_event_registry", lambda _path: (event,))
    monkeypatch.setattr(source_module, "load_holiday_calendar", lambda _path: (event,))
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
    monkeypatch.setattr(
        source_module,
        "run_paper_final_stage",
        lambda **_kwargs: SimpleNamespace(
            members_path=final_members,
            point_path=final_point,
            manifest_path=baseline_manifest_path,
            fit_count=0,
        ),
    )
    monkeypatch.setattr(source_module, "final_fold", lambda: object())
    monkeypatch.setattr(
        source_module,
        "select_fold_samples",
        lambda _matrix, _fold: (np.asarray([], dtype=np.int64), np.asarray([0])),
    )
    monkeypatch.setattr(
        source_module,
        "select_diagnostic_residual_context",
        lambda frame, _manifest, **_kwargs: frame.clone(),
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
        final_point=final_point,
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
