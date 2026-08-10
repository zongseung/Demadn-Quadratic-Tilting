"""Concrete all-model paper OOF/final publication contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import hqrc_v3.baselines.paper as paper
import numpy as np
import polars as pl
import pytest
from hqrc_v3.baselines.config import MODEL_NAMES, PAPER_SEEDS, load_paper_baselines
from hqrc_v3.baselines.paper import run_paper_final_stage, run_paper_oof_stage
from hqrc_v3.contracts import DataContractError, ForecastMatrix
from hqrc_v3.features import feature_columns
from hqrc_v3.oof import cache_key
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
from hqrc_v3.splits import expanding_oof_folds

from hqrc_v3 import cli

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs/model_spaces.toml"
HASHES = {
    "data_sha256": "a" * 64,
    "experiment_sha256": "b" * 64,
    "model_config_sha256": file_sha256(MODEL_CONFIG),
    "event_registry_sha256": "d" * 64,
    "holiday_calendar_sha256": "e" * 64,
}


class AbruptPublicationStop(BaseException):
    """Simulate process death, which ordinary Exception cleanup cannot catch."""


def _matrix(feature_set: str) -> ForecastMatrix:
    origins = np.concatenate(
        [
            np.arange(
                np.datetime64(f"{year}-10-31"),
                np.datetime64(f"{year + 1}-01-01"),
                dtype="datetime64[D]",
            )
            for year in range(2019, 2024)
        ]
        + [np.array([np.datetime64("2024-01-01")])]
    ).astype("datetime64[ns]")
    count = len(origins)
    target_times = origins[:, None] + np.arange(24).astype("timedelta64[h]")
    columns = feature_columns(feature_set)  # type: ignore[arg-type]
    values = np.arange(count, dtype=float)
    return ForecastMatrix(
        origins=origins,
        target_times=target_times,
        history=np.broadcast_to(values[:, None, None], (count, 168, 1)).copy(),
        future=np.zeros((count, 24, len(columns)), dtype=float),
        target=values[:, None] + np.arange(24, dtype=float)[None, :],
        history_columns=("load_mw",),
        future_columns=columns,
    )


def _write_short_cli_source(tmp_path: Path) -> Path:
    timestamps = pl.datetime_range(
        pl.datetime(2019, 1, 1),
        pl.datetime(2019, 1, 10, 23),
        interval="1h",
        eager=True,
    )
    source = tmp_path / "source.csv"
    pl.DataFrame(
        {
            "일시": timestamps,
            "hm": np.full(len(timestamps), 50.0),
            "ta": np.full(len(timestamps), 15.0),
            "power demand(MW)": np.full(len(timestamps), 100.0),
        }
    ).write_csv(source)
    return source


def _baseline_cli_arguments(source: Path, tmp_path: Path) -> list[str]:
    return [
        "generate-oof",
        "--data",
        str(source),
        "--config",
        str(PROJECT_ROOT / "configs/experiment.toml"),
        "--frozen-model-config",
        str(MODEL_CONFIG),
        "--frozen-model-hash",
        file_sha256(MODEL_CONFIG),
        "--run-dir",
        str(tmp_path / "run"),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--feature-set",
        "all",
        "--model",
        "all",
        "--seed",
        "7",
    ]


@dataclass
class RecordingFactory:
    name: str
    calls: list[dict[str, object]] = field(default_factory=list)

    @property
    def uses_validation_tail(self) -> bool:
        return self.name != "svr"

    def fit(
        self,
        train: ForecastMatrix,
        validation: ForecastMatrix | None,
        seed: int,
    ) -> RecordingFitted:
        self.calls.append(
            {
                "seed": seed,
                "train_max": train.target_times.max(),
                "train_rows": len(train.origins),
                "validation_max": None if validation is None else validation.target_times.max(),
                "validation_rows": 0 if validation is None else len(validation.origins),
            }
        )
        return RecordingFitted(self.name, seed)


@dataclass(frozen=True)
class RecordingFitted:
    model_name: str
    seed: int

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        model_offset = MODEL_NAMES.index(self.model_name) * 1000.0
        return np.full(batch.target.shape, model_offset + float(self.seed), dtype=float)


def _recording_builder() -> tuple[dict[str, RecordingFactory], Any]:
    factories = {model: RecordingFactory(model) for model in MODEL_NAMES}

    def build(config: object, model: str, **_: object) -> RecordingFactory:
        assert config.models == MODEL_NAMES
        return factories[model]

    return factories, build


def _run_oof(run: Path, *, factory_builder: Any | None = None):
    _, default_builder = _recording_builder()
    return run_paper_oof_stage(
        matrices={"B0": _matrix("B0"), "B1": _matrix("B1")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=run,
        cache_dir=run / "stream-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        factory_builder=factory_builder or default_builder,
    )


def _rebind_artifact_hash(run: Path, *, stage: str, artifact: str, path: Path) -> None:
    manifest_path = run / "predictions/baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"][stage]["artifacts"][artifact]["sha256"] = file_sha256(path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _mutate_all_oof_coverage(run: Path, mutation: str) -> None:
    paths = {
        "members": run / "predictions/oof_members.parquet",
        "point": run / "predictions/oof.parquet",
    }
    for artifact, path in paths.items():
        frame = pl.read_parquet(path)
        first_origin = frame["origin"].min()
        if mutation == "missing":
            changed = frame.filter(pl.col("origin") != first_origin)
        elif mutation == "extra":
            new_origin = datetime(2020, 1, 1)
            extra = frame.filter(pl.col("origin") == first_origin).with_columns(
                pl.lit(new_origin).cast(pl.Datetime("ns")).alias("origin"),
                (
                    pl.lit(new_origin).cast(pl.Datetime("ns"))
                    + (pl.col("horizon") - 1) * pl.duration(hours=1)
                ).alias("target_timestamp"),
            )
            changed = pl.concat((frame, extra), how="vertical_relaxed")
        elif mutation == "duplicate":
            duplicate = frame.filter(pl.col("origin") == first_origin)
            changed = pl.concat((frame, duplicate), how="vertical")
        elif mutation == "misaligned":
            changed = frame.with_columns(
                pl.when((pl.col("origin") == first_origin) & (pl.col("horizon") == 24))
                .then(pl.col("target_timestamp") + pl.duration(minutes=30))
                .otherwise(pl.col("target_timestamp"))
                .alias("target_timestamp")
            )
        elif mutation == "observed":
            changed = frame.with_columns(
                (pl.col("observed_mw") + 0.25).alias("observed_mw")
            )
        else:  # pragma: no cover - test helper has a closed call set
            raise AssertionError(mutation)
        changed.write_parquet(path)
        _rebind_artifact_hash(run, stage="oof", artifact=artifact, path=path)


def test_oof_stage_publishes_exact_years_members_and_pointwise_means(tmp_path: Path) -> None:
    factories, builder = _recording_builder()

    result = _run_oof(tmp_path, factory_builder=builder)

    assert result.members_path == tmp_path / "predictions/oof_members.parquet"
    assert result.point_path == tmp_path / "predictions/oof.parquet"
    members = pl.read_parquet(result.members_path)
    point = pl.read_parquet(result.point_path)
    assert tuple(point["model"].unique(maintain_order=True)) == MODEL_NAMES
    assert set(point["feature_set"]) == {"B0", "B1"}
    assert set(point["target_timestamp"].dt.year()) == {2020, 2021, 2022, 2023}
    assert set(members["model"]) == {"seq2seq_lstm", "transformer"}
    assert set(members["seed"]) == set(PAPER_SEEDS)
    assert set(
        point.filter(pl.col("model").is_in(["seq2seq_lstm", "transformer"]))["seed"]
    ) == {0}

    member_mean = members.group_by(
        "origin", "target_timestamp", "horizon", "observed_mw", "model", "feature_set", "split_id"
    ).agg(pl.col("predicted_mw").mean().alias("member_mean"))
    neural_point = point.filter(
        pl.col("model").is_in(["seq2seq_lstm", "transformer"])
    ).join(
        member_mean,
        on=[
            "origin",
            "target_timestamp",
            "horizon",
            "observed_mw",
            "model",
            "feature_set",
            "split_id",
        ],
        how="inner",
    )
    assert neural_point.height == member_mean.height
    np.testing.assert_array_equal(neural_point["predicted_mw"], neural_point["member_mean"])

    for model, factory in factories.items():
        expected_calls = 8 if model in {"xgboost", "lightgbm", "svr"} else 40
        assert len(factory.calls) == expected_calls
        if model == "svr":
            assert {call["validation_rows"] for call in factory.calls} == {0}
        else:
            assert {call["validation_rows"] for call in factory.calls} == {61}
            assert all(
                call["validation_max"] < np.datetime64(f"{year}-01-01")
                for call, year in zip(factory.calls[:4], range(2020, 2024), strict=True)
            )


def test_final_stage_publishes_only_2024_and_exact_neural_mean(tmp_path: Path) -> None:
    factories, builder = _recording_builder()

    result = run_paper_final_stage(
        matrices={"B0": _matrix("B0"), "B1": _matrix("B1")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=tmp_path,
        cache_dir=tmp_path / "stream-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        factory_builder=builder,
    )

    assert result.members_path.name == "final_2024_members.parquet"
    assert result.point_path.name == "final_2024.parquet"
    members = pl.read_parquet(result.members_path)
    point = pl.read_parquet(result.point_path)
    assert set(point["target_timestamp"].dt.year()) == {2024}
    assert set(point["split_id"]) == {"final-2024"}
    assert set(members["seed"]) == set(PAPER_SEEDS)
    for model, factory in factories.items():
        assert len(factory.calls) == (2 if model in {"xgboost", "lightgbm", "svr"} else 10)


def test_completed_oof_is_a_validated_cache_hit_without_fitting(tmp_path: Path) -> None:
    _run_oof(tmp_path)
    factories, builder = _recording_builder()

    result = _run_oof(tmp_path, factory_builder=builder)

    assert result.fit_count == 0
    assert result.cache_hit_count == 104
    assert all(factory.calls == [] for factory in factories.values())


def test_stream_caches_resume_publication_without_refitting(tmp_path: Path) -> None:
    first = _run_oof(tmp_path)
    first.members_path.unlink()
    first.point_path.unlink()
    first.manifest_path.unlink()
    factories, builder = _recording_builder()

    resumed = _run_oof(tmp_path, factory_builder=builder)

    assert resumed.fit_count == 0
    assert resumed.cache_hit_count == 104
    assert resumed.members_path.is_file()
    assert resumed.point_path.is_file()
    assert all(factory.calls == [] for factory in factories.values())


def test_manifest_binds_inputs_models_seeds_feature_schemas_and_streams(tmp_path: Path) -> None:
    result = _run_oof(tmp_path)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 1
    assert manifest["input_hashes"] == HASHES
    assert tuple(manifest["models"]) == MODEL_NAMES
    assert tuple(manifest["neural_seeds"]) == PAPER_SEEDS
    assert manifest["ensemble_seed"] == 0
    assert set(manifest["feature_schemas"]) == {"B0", "B1"}
    assert len(manifest["stages"]["oof"]["streams"]) == 10
    assert manifest["stages"]["oof"]["expected_coverage"]["count"] == 5_952
    assert len(manifest["stages"]["oof"]["expected_coverage"]["sha256"]) == 64


@pytest.mark.parametrize(
    "mutation", ["missing", "extra", "duplicate", "misaligned", "observed"]
)
def test_hash_rebound_uniform_coverage_tampering_never_becomes_a_cache_hit(
    tmp_path: Path,
    mutation: str,
) -> None:
    _run_oof(tmp_path)
    _mutate_all_oof_coverage(tmp_path, mutation)

    with pytest.raises(
        ArtifactMismatch, match="coverage|duplicate|target timestamp"
    ):
        _run_oof(tmp_path)


def test_reload_recomputes_expected_coverage_from_the_supplied_matrix(tmp_path: Path) -> None:
    _run_oof(tmp_path)
    b0 = _matrix("B0")
    b1 = _matrix("B1")
    changed = {
        "B0": replace(b0, target=b0.target + 1.0),
        "B1": replace(b1, target=b1.target + 1.0),
    }

    with pytest.raises(ArtifactMismatch, match="expected coverage"):
        run_paper_oof_stage(
            matrices=changed,
            config=load_paper_baselines(MODEL_CONFIG),
            run_dir=tmp_path,
            cache_dir=tmp_path / "stream-cache",
            artifact_hashes=HASHES,
            classical_seed=7,
        )


def test_reload_rejects_feature_matrix_with_different_coverage_keys(tmp_path: Path) -> None:
    _run_oof(tmp_path)
    b1 = _matrix("B1")
    changed = replace(b1, origins=b1.origins + np.timedelta64(1, "h"))

    with pytest.raises(ArtifactMismatch, match="coverage"):
        run_paper_oof_stage(
            matrices={"B0": _matrix("B0"), "B1": changed},
            config=load_paper_baselines(MODEL_CONFIG),
            run_dir=tmp_path,
            cache_dir=tmp_path / "stream-cache",
            artifact_hashes=HASHES,
            classical_seed=7,
        )


@pytest.mark.parametrize("changed_hash", HASHES)
def test_completed_stage_rejects_any_changed_bound_input(
    tmp_path: Path,
    changed_hash: str,
) -> None:
    _run_oof(tmp_path)
    changed = dict(HASHES)
    changed[changed_hash] = "f" * 64

    with pytest.raises(ArtifactMismatch, match=changed_hash):
        run_paper_oof_stage(
            matrices={"B0": _matrix("B0"), "B1": _matrix("B1")},
            config=load_paper_baselines(MODEL_CONFIG),
            run_dir=tmp_path,
            cache_dir=tmp_path / "stream-cache",
            artifact_hashes=changed,
            classical_seed=7,
        )


def test_partial_publication_fails_closed(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    pl.DataFrame({"not": ["a prediction"]}).write_parquet(predictions / "oof.parquet")

    with pytest.raises(ArtifactMismatch, match="partial"):
        _run_oof(tmp_path)


def test_partial_stream_cache_fails_closed(tmp_path: Path) -> None:
    cache = tmp_path / "stream-cache"
    cache.mkdir()
    key = cache_key("xgboost", "B0", 7, expanding_oof_folds()[0])
    pl.DataFrame({"partial": [1]}).write_parquet(cache / f"{key}.parquet")

    with pytest.raises(ArtifactMismatch, match="partial prediction cache"):
        _run_oof(tmp_path)


def test_tampered_published_parquet_fails_closed(tmp_path: Path) -> None:
    result = _run_oof(tmp_path)
    point = pl.read_parquet(result.point_path).with_columns(
        (pl.col("predicted_mw") + 1.0).alias("predicted_mw")
    )
    point.write_parquet(result.point_path)

    with pytest.raises(ArtifactMismatch, match="oof.*SHA-256"):
        _run_oof(tmp_path)


def test_hash_rebound_wrong_model_name_still_fails_semantically(tmp_path: Path) -> None:
    result = _run_oof(tmp_path)
    point = pl.read_parquet(result.point_path).with_columns(
        pl.when(pl.col("model") == "xgboost")
        .then(pl.lit("wrong-model"))
        .otherwise(pl.col("model"))
        .alias("model")
    )
    point.write_parquet(result.point_path)
    _rebind_artifact_hash(tmp_path, stage="oof", artifact="point", path=result.point_path)

    with pytest.raises(ArtifactMismatch, match="model"):
        _run_oof(tmp_path)


def test_hash_rebound_changed_member_seed_set_still_fails_semantically(tmp_path: Path) -> None:
    result = _run_oof(tmp_path)
    members = pl.read_parquet(result.members_path).with_columns(
        pl.when(pl.col("seed") == 53)
        .then(pl.lit(59))
        .otherwise(pl.col("seed"))
        .cast(pl.Int64)
        .alias("seed")
    )
    members.write_parquet(result.members_path)
    _rebind_artifact_hash(tmp_path, stage="oof", artifact="members", path=result.members_path)

    with pytest.raises(ArtifactMismatch, match="seed"):
        _run_oof(tmp_path)


def test_changed_feature_schema_fails_before_cache_reuse(tmp_path: Path) -> None:
    _run_oof(tmp_path)
    b0 = _matrix("B0")
    changed_b0 = replace(
        b0,
        future_columns=("renamed_hour", *b0.future_columns[1:]),
    )

    with pytest.raises(ArtifactMismatch, match="feature schema"):
        run_paper_oof_stage(
            matrices={"B0": changed_b0, "B1": _matrix("B1")},
            config=load_paper_baselines(MODEL_CONFIG),
            run_dir=tmp_path,
            cache_dir=tmp_path / "stream-cache",
            artifact_hashes=HASHES,
            classical_seed=7,
        )


def test_failed_group_publication_leaves_no_partial_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_after_first_file(boundary: str) -> None:
        if boundary == "oof-members-published":
            raise OSError("injected publication failure")

    monkeypatch.setattr(paper, "_publication_boundary", fail_after_first_file)

    with pytest.raises(OSError, match="injected publication failure"):
        _run_oof(tmp_path)

    assert not (tmp_path / "predictions/oof_members.parquet").exists()
    assert not (tmp_path / "predictions/oof.parquet").exists()
    assert not (tmp_path / "predictions/baseline_manifest.json").exists()


@pytest.mark.parametrize(
    "boundary",
    [
        "oof-members-published",
        "oof-point-published",
        "oof-manifest-published",
    ],
)
def test_abrupt_publication_is_recovered_from_durable_journal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    def terminate_at(selected: str) -> None:
        if selected == boundary:
            raise AbruptPublicationStop(boundary)

    monkeypatch.setattr(paper, "_publication_boundary", terminate_at)
    with pytest.raises(AbruptPublicationStop, match=boundary):
        _run_oof(tmp_path)
    journal = tmp_path / "predictions/.baseline-transaction.json"
    assert journal.is_file()
    factories, builder = _recording_builder()
    monkeypatch.setattr(paper, "_publication_boundary", lambda _: None)

    recovered = _run_oof(tmp_path, factory_builder=builder)

    assert recovered.fit_count == 0
    assert recovered.cache_hit_count == 104
    assert recovered.members_path.is_file()
    assert recovered.point_path.is_file()
    assert recovered.manifest_path.is_file()
    assert not journal.exists()
    assert all(factory.calls == [] for factory in factories.values())


def test_unrecoverable_interrupted_transaction_rolls_back_and_republishes_from_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def terminate_after_members(boundary: str) -> None:
        if boundary == "oof-members-published":
            raise AbruptPublicationStop(boundary)

    monkeypatch.setattr(paper, "_publication_boundary", terminate_after_members)
    with pytest.raises(AbruptPublicationStop):
        _run_oof(tmp_path)
    journal_path = tmp_path / "predictions/.baseline-transaction.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    (journal_path.parent / journal["point"]["temp"]).unlink()
    factories, builder = _recording_builder()
    monkeypatch.setattr(paper, "_publication_boundary", lambda _: None)

    recovered = _run_oof(tmp_path, factory_builder=builder)

    assert recovered.fit_count == 0
    assert recovered.cache_hit_count == 104
    assert recovered.members_path.is_file()
    assert recovered.point_path.is_file()
    assert recovered.manifest_path.is_file()
    assert not journal_path.exists()
    assert all(factory.calls == [] for factory in factories.values())


def test_abrupt_oof_recovery_preserves_a_completed_final_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, builder = _recording_builder()
    final = run_paper_final_stage(
        matrices={"B0": _matrix("B0"), "B1": _matrix("B1")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=tmp_path,
        cache_dir=tmp_path / "stream-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        factory_builder=builder,
    )
    final_hashes = (file_sha256(final.members_path), file_sha256(final.point_path))

    def terminate_after_members(boundary: str) -> None:
        if boundary == "oof-members-published":
            raise AbruptPublicationStop(boundary)

    monkeypatch.setattr(paper, "_publication_boundary", terminate_after_members)
    with pytest.raises(AbruptPublicationStop):
        _run_oof(tmp_path)
    monkeypatch.setattr(paper, "_publication_boundary", lambda _: None)

    _run_oof(tmp_path)

    manifest = json.loads(final.manifest_path.read_text(encoding="utf-8"))
    assert set(manifest["stages"]) == {"oof", "final"}
    assert (file_sha256(final.members_path), file_sha256(final.point_path)) == final_hashes


def test_orphan_final_products_are_not_ignored_by_oof_cache_hit(tmp_path: Path) -> None:
    result = _run_oof(tmp_path)
    predictions = tmp_path / "predictions"
    (predictions / "final_2024_members.parquet").write_bytes(result.members_path.read_bytes())
    (predictions / "final_2024.parquet").write_bytes(result.point_path.read_bytes())

    with pytest.raises(ArtifactMismatch, match="orphan final"):
        _run_oof(tmp_path)


def test_publication_journal_rejects_path_traversal(tmp_path: Path) -> None:
    predictions = tmp_path / "predictions"
    predictions.mkdir()
    intended_manifest = {
        "schema_version": 1,
        "profile": "paper",
        "input_hashes": HASHES,
        "models": list(MODEL_NAMES),
        "feature_sets": ["B0", "B1"],
        "classical_seed": 7,
        "neural_seeds": list(PAPER_SEEDS),
        "ensemble_seed": 0,
        "feature_schemas": {},
        "execution_overrides": {},
        "stages": {},
    }
    journal = {
        "schema_version": 1,
        "stage": "oof",
        "members": {
            "temp": "../escape.parquet",
            "final": "oof_members.parquet",
            "sha256": "a" * 64,
        },
        "point": {
            "temp": ".baseline.safe.point.parquet",
            "final": "oof.parquet",
            "sha256": "b" * 64,
        },
        "manifest": {
            "temp": ".baseline.safe.manifest.json",
            "final": "baseline_manifest.json",
            "sha256": "c" * 64,
        },
        "intended_manifest": intended_manifest,
        "prior_manifest": None,
    }
    (predictions / ".baseline-transaction.json").write_text(
        json.dumps(journal, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ArtifactMismatch, match="unsafe.*journal path"):
        _run_oof(tmp_path)


def test_concrete_cli_loads_both_feature_matrices_and_hash_bound_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_short_cli_source(tmp_path)
    captured: list[dict[str, object]] = []

    def record_stage(**kwargs: object) -> object:
        captured.append(kwargs)
        return object()

    monkeypatch.setattr(cli, "run_paper_oof_stage", record_stage)

    result = cli.main([*_baseline_cli_arguments(source, tmp_path), "--profile", "smoke"])

    assert result == 0
    assert len(captured) == 1
    matrices = captured[0]["matrices"]
    assert set(matrices) == {"B0", "B1"}
    assert matrices["B0"].future_columns == feature_columns("B0")
    assert matrices["B1"].future_columns == feature_columns("B1")
    assert captured[0]["artifact_hashes"] == {
        "data_sha256": file_sha256(source),
        "experiment_sha256": file_sha256(PROJECT_ROOT / "configs/experiment.toml"),
        "model_config_sha256": file_sha256(MODEL_CONFIG),
        "event_registry_sha256": file_sha256(PROJECT_ROOT / "configs/events.csv"),
        "holiday_calendar_sha256": file_sha256(
            PROJECT_ROOT / "configs/holiday_calendar.csv"
        ),
    }


def test_paper_cli_rejects_short_source_before_feature_construction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source = _write_short_cli_source(tmp_path)
    feature_calls: list[object] = []
    stage_calls: list[object] = []

    def forbidden_features(*args: object, **kwargs: object) -> object:
        feature_calls.append((args, kwargs))
        raise AssertionError("feature construction must follow the fixed paper audit")

    monkeypatch.setattr(cli, "attach_calendar_features", forbidden_features)
    monkeypatch.setattr(cli, "run_paper_oof_stage", stage_calls.append)

    assert cli.main(_baseline_cli_arguments(source, tmp_path)) == 2
    assert feature_calls == []
    assert stage_calls == []
    assert "expected 51144 hourly rows" in capsys.readouterr().err


def test_paper_cli_passes_exact_fixed_bounds_to_the_audit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_short_cli_source(tmp_path)
    received: list[dict[str, object]] = []

    def record_and_stop(frame: object, **expected: object) -> object:
        del frame
        received.append(expected)
        raise DataContractError("audit boundary observed")

    monkeypatch.setattr(cli, "audit_hourly_data", record_and_stop)

    assert cli.main(_baseline_cli_arguments(source, tmp_path)) == 2
    assert received == [
        {
            "expected_start": datetime(2019, 1, 1),
            "expected_end": datetime(2024, 10, 31, 23),
            "expected_rows": 51_144,
        }
    ]


def test_readme_hqrc_commands_are_executable_from_the_repository_root() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Run these commands from the repository root" in readme
    for command in (
        "generate-oof",
        "fit-final-baselines",
        "audit-data",
        "diagnose-ar",
        "approve-ar-calibration",
        "report",
    ):
        assert f"uv run --project hqrc_v3 hqrc {command}" in readme
    assert "uv run hqrc " not in readme
    assert "\nhqrc " not in readme
