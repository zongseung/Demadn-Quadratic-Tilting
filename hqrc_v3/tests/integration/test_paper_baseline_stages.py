"""Concrete all-model paper OOF/final publication contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

import hqrc_v3.baselines.paper as paper
from hqrc_v3 import cli
from hqrc_v3.baselines.config import MODEL_NAMES, PAPER_SEEDS, load_paper_baselines
from hqrc_v3.baselines.paper import run_paper_final_stage, run_paper_oof_stage
from hqrc_v3.contracts import DataContractError, ForecastMatrix
from hqrc_v3.features import feature_columns, history_columns
from hqrc_v3.oof import cache_key
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
from hqrc_v3.splits import expanding_oof_folds

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_CONFIG = PROJECT_ROOT / "configs/model_spaces.toml"
HASHES = {
    "data_sha256": "a" * 64,
    "experiment_sha256": "b" * 64,
    "model_config_sha256": file_sha256(MODEL_CONFIG),
    "event_registry_sha256": "d" * 64,
    "holiday_calendar_sha256": "e" * 64,
    "temporary_holiday_availability_sha256": "6" * 64,
}


def _population(times: np.ndarray, *, unit: str) -> dict[str, object]:
    unique = np.unique(np.asarray(times).reshape(-1).astype("datetime64[ns]"))
    return {
        "count": int(unique.size),
        "start": np.datetime_as_string(unique[0], unit="s"),
        "end": np.datetime_as_string(unique[-1], unit="s"),
        "unit": unit,
    }


def _actual_population_contract(
    model: str, train: ForecastMatrix
) -> dict[str, dict[str, object]]:
    target = _population(train.target_times, unit="unique-hour")
    if model in {"seq2seq_lstm", "transformer"}:
        history_times = train.origins[:, None] + np.arange(-168, 0).astype(
            "timedelta64[h]"
        )
        return {
            "target": target,
            "weather": _population(history_times, unit="unique-hour"),
            "calendar": target,
        }
    return {
        "x": _population(train.origins, unit="daily-sample"),
        "target": target,
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
        history=np.broadcast_to(
            values[:, None, None], (count, 168, len(history_columns(feature_set)))
        ).copy(),
        future=np.zeros((count, 24, len(columns)), dtype=float),
        target=values[:, None] + np.arange(24, dtype=float)[None, :],
        history_columns=history_columns(feature_set),  # type: ignore[arg-type]
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
            "holiday_name": [""] * len(timestamps),
            "is_holiday_dummies": np.zeros(len(timestamps), dtype=np.int8),
        }
    ).write_csv(source)
    return source


def _baseline_cli_arguments(
    source: Path, tmp_path: Path, *, feature_set: str = "all"
) -> list[str]:
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
        feature_set,
        "--model",
        "all",
        "--seed",
        "7",
    ]


@dataclass
class RecordingFactory:
    name: str
    calls: list[dict[str, object]] = field(default_factory=list)
    population_mutator: Any | None = None

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
                "future_columns": train.future_columns,
            }
        )
        population = _actual_population_contract(self.name, train)
        if self.population_mutator is not None:
            population = self.population_mutator(population, seed)
        return RecordingFitted(self.name, seed, population)


@dataclass(frozen=True)
class RecordingFitted:
    model_name: str
    seed: int
    population: dict[str, dict[str, object]]

    def population_contract(self) -> dict[str, dict[str, object]]:
        return self.population

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        model_offset = MODEL_NAMES.index(self.model_name) * 1000.0
        return np.full(batch.target.shape, model_offset + float(self.seed), dtype=float)


def _recording_builder() -> tuple[dict[str, RecordingFactory], Any]:
    factories = {model: RecordingFactory(model) for model in MODEL_NAMES}

    def build(config: object, model: str, **_: object) -> RecordingFactory:
        assert config.models == MODEL_NAMES
        return factories[model]

    return factories, build


def test_paper_accepts_exact_reviewer_feature_suite(tmp_path: Path) -> None:
    _, builder = _recording_builder()

    result = run_paper_oof_stage(
        matrices={"B0": _matrix("B0"), "B1W": _matrix("B1W")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=tmp_path / "reviewer",
        cache_dir=tmp_path / "shared-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        feature_sets=("B0", "B1W"),
        profile="paper",
        factory_builder=builder,
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["feature_sets"] == ["B0", "B1W"]
    assert manifest["feature_schemas"]["B0"] == {
        "history": list(history_columns("B0")),
        "future": list(feature_columns("B0")),
    }
    assert manifest["feature_schemas"]["B1W"]["window_definition"] == [
        "official-sequence-buffer-v1"
    ]


def test_paper_rejects_mixed_three_feature_suite(tmp_path: Path) -> None:
    _, builder = _recording_builder()

    with pytest.raises(DataContractError, match="paper feature suite"):
        run_paper_oof_stage(
            matrices={name: _matrix(name) for name in ("B0", "B1", "B1W")},
            config=load_paper_baselines(MODEL_CONFIG),
            run_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            artifact_hashes=HASHES,
            classical_seed=7,
            feature_sets=("B0", "B1", "B1W"),
            profile="paper",
            factory_builder=builder,
        )


def test_reviewer_run_reuses_b0_from_distinct_legacy_run_directory(
    tmp_path: Path,
) -> None:
    _, legacy_builder = _recording_builder()
    run_paper_oof_stage(
        matrices={"B0": _matrix("B0"), "B1": _matrix("B1")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=tmp_path / "legacy-run",
        cache_dir=tmp_path / "shared-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        feature_sets=("B0", "B1"),
        profile="paper",
        factory_builder=legacy_builder,
    )
    reviewer_factories, reviewer_builder = _recording_builder()

    reviewer = run_paper_oof_stage(
        matrices={"B0": _matrix("B0"), "B1W": _matrix("B1W")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=tmp_path / "reviewer-run",
        cache_dir=tmp_path / "shared-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        feature_sets=("B0", "B1W"),
        profile="paper",
        factory_builder=reviewer_builder,
    )

    assert reviewer.fit_count == 52
    assert reviewer.cache_hit_count == 52
    for model, factory in reviewer_factories.items():
        expected_calls = 4 if model in {"xgboost", "lightgbm", "svr"} else 20
        assert len(factory.calls) == expected_calls
        assert {call["future_columns"] for call in factory.calls} == {
            feature_columns("B1W")
        }


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


def _run_final(run: Path, *, factory_builder: Any | None = None):
    _, default_builder = _recording_builder()
    return run_paper_final_stage(
        matrices={"B0": _matrix("B0"), "B1": _matrix("B1")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=run,
        cache_dir=run / "stream-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        factory_builder=factory_builder or default_builder,
    )


def _run_smoke_oof(run: Path, *, years: tuple[int, ...] = (2020, 2021)):
    _, builder = _recording_builder()
    return run_paper_oof_stage(
        matrices={"B0": _matrix("B0")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=run,
        cache_dir=run / "stream-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        factory_builder=builder,
        models=("xgboost",),
        feature_sets=("B0",),
        profile="smoke",
        oof_years=years,
    )


def _run_smoke_final(run: Path):
    _, builder = _recording_builder()
    return run_paper_final_stage(
        matrices={"B0": _matrix("B0")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=run,
        cache_dir=run / "stream-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        factory_builder=builder,
        models=("xgboost",),
        feature_sets=("B0",),
        profile="smoke",
    )


def _rebind_artifact_hash(run: Path, *, stage: str, artifact: str, path: Path) -> None:
    manifest_path = run / "predictions/baseline_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["stages"][stage]["artifacts"][artifact]["sha256"] = file_sha256(path)
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _truncate_stage_uniformly(run: Path, *, stage: str) -> None:
    predictions = run / "predictions"
    paths = (
        {
            "members": predictions / "oof_members.parquet",
            "point": predictions / "oof.parquet",
        }
        if stage == "oof"
        else {
            "members": predictions / "final_2024_members.parquet",
            "point": predictions / "final_2024.parquet",
        }
    )
    for artifact, path in paths.items():
        frame = pl.read_parquet(path)
        frame.filter(pl.col("origin") != frame["origin"].min()).write_parquet(path)
        _rebind_artifact_hash(run, stage=stage, artifact=artifact, path=path)


def _rewrite_interrupted_manifest_binding(run: Path, *, sibling_stage: str) -> None:
    predictions = run / "predictions"
    journal_path = predictions / ".baseline-transaction.json"
    manifest_path = predictions / "baseline_manifest.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    filenames = (
        {"members": "oof_members.parquet", "point": "oof.parquet"}
        if sibling_stage == "oof"
        else {
            "members": "final_2024_members.parquet",
            "point": "final_2024.parquet",
        }
    )
    for artifact, filename in filenames.items():
        path = predictions / filename
        frame = pl.read_parquet(path)
        frame.filter(pl.col("origin") != frame["origin"].min()).write_parquet(path)
        digest = file_sha256(path)
        journal["prior_manifest"]["stages"][sibling_stage]["artifacts"][artifact][
            "sha256"
        ] = digest
        journal["intended_manifest"]["stages"][sibling_stage]["artifacts"][artifact][
            "sha256"
        ] = digest
    intended_payload = json.dumps(
        journal["intended_manifest"], sort_keys=True, separators=(",", ":")
    ).encode()
    journal["manifest"]["sha256"] = hashlib.sha256(intended_payload).hexdigest()
    manifest_path.write_bytes(intended_payload)
    journal_path.write_text(
        json.dumps(journal, sort_keys=True, separators=(",", ":")), encoding="utf-8"
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
    first_populations = json.loads(first.manifest_path.read_text(encoding="utf-8"))[
        "stages"
    ]["oof"]["preprocessing_populations"]
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
    resumed_populations = json.loads(
        resumed.manifest_path.read_text(encoding="utf-8")
    )["stages"]["oof"]["preprocessing_populations"]
    assert resumed_populations == first_populations


def test_manifest_binds_inputs_models_seeds_feature_schemas_and_streams(tmp_path: Path) -> None:
    result = _run_oof(tmp_path)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["schema_version"] == 2
    assert manifest["input_hashes"] == HASHES
    assert tuple(manifest["models"]) == MODEL_NAMES
    assert tuple(manifest["neural_seeds"]) == PAPER_SEEDS
    assert manifest["ensemble_seed"] == 0
    assert set(manifest["feature_schemas"]) == {"B0", "B1"}
    assert manifest["preprocessing"]["version"] == "causal-v1"
    populations = manifest["stages"]["oof"]["preprocessing_populations"]
    assert len(populations) == 40
    assert populations[0] == {
        "model": "xgboost",
        "feature_set": "B0",
        "split_id": "oof-2020",
        "scalers": {
            "x": {
                "count": 1,
                "start": "2019-10-31T00:00:00",
                "end": "2019-10-31T00:00:00",
                "unit": "daily-sample",
            },
            "target": {
                "count": 24,
                "start": "2019-10-31T00:00:00",
                "end": "2019-10-31T23:00:00",
                "unit": "unique-hour",
            },
        },
    }
    sequence_population = next(
        row
        for row in populations
        if row["model"] == "seq2seq_lstm"
        and row["feature_set"] == "B0"
        and row["split_id"] == "oof-2020"
    )
    assert sequence_population["scalers"] == {
        "target": {
            "count": 24,
            "start": "2019-10-31T00:00:00",
            "end": "2019-10-31T23:00:00",
            "unit": "unique-hour",
        },
        "weather": {
            "count": 168,
            "start": "2019-10-24T00:00:00",
            "end": "2019-10-30T23:00:00",
            "unit": "unique-hour",
        },
        "calendar": {
            "count": 24,
            "start": "2019-10-31T00:00:00",
            "end": "2019-10-31T23:00:00",
            "unit": "unique-hour",
        },
    }
    assert len(manifest["stages"]["oof"]["streams"]) == 10
    assert manifest["stages"]["oof"]["expected_coverage"]["count"] == 5_952
    assert len(manifest["stages"]["oof"]["expected_coverage"]["sha256"]) == 64
    first_cache_metadata = json.loads(
        (
            tmp_path
            / "stream-cache"
            / f"{cache_key('xgboost', 'B0', 7, expanding_oof_folds()[0])}.json"
        ).read_text(encoding="utf-8")
    )
    assert first_cache_metadata["population_contract"] == populations[0]["scalers"]


def test_stage_rejects_fitted_adapter_that_misreports_population(tmp_path: Path) -> None:
    def misreport(
        population: dict[str, dict[str, object]], seed: int
    ) -> dict[str, dict[str, object]]:
        del seed
        return {
            **population,
            "x": {**population["x"], "count": int(population["x"]["count"]) + 1},
        }

    factory = RecordingFactory("xgboost", population_mutator=misreport)

    def builder(config: object, model: str, **_: object) -> RecordingFactory:
        del config
        assert model == "xgboost"
        return factory

    with pytest.raises(DataContractError, match="actual preprocessing population.*expected"):
        run_paper_oof_stage(
            matrices={"B0": _matrix("B0")},
            config=load_paper_baselines(MODEL_CONFIG),
            run_dir=tmp_path,
            cache_dir=tmp_path / "stream-cache",
            artifact_hashes=HASHES,
            classical_seed=7,
            factory_builder=builder,
            models=("xgboost",),
            feature_sets=("B0",),
            profile="smoke",
            oof_years=(2020,),
        )

    assert not (tmp_path / "predictions/baseline_manifest.json").exists()


def test_stage_requires_all_neural_seeds_to_report_same_population(tmp_path: Path) -> None:
    def disagree(
        population: dict[str, dict[str, object]], seed: int
    ) -> dict[str, dict[str, object]]:
        if seed != PAPER_SEEDS[-1]:
            return population
        return {
            **population,
            "weather": {
                **population["weather"],
                "count": int(population["weather"]["count"]) + 1,
            },
        }

    factory = RecordingFactory("seq2seq_lstm", population_mutator=disagree)

    def builder(config: object, model: str, **_: object) -> RecordingFactory:
        del config
        assert model == "seq2seq_lstm"
        return factory

    with pytest.raises(DataContractError, match="neural seeds.*population"):
        run_paper_oof_stage(
            matrices={"B0": _matrix("B0")},
            config=load_paper_baselines(MODEL_CONFIG),
            run_dir=tmp_path,
            cache_dir=tmp_path / "stream-cache",
            artifact_hashes=HASHES,
            classical_seed=7,
            factory_builder=builder,
            models=("seq2seq_lstm",),
            feature_sets=("B0",),
            profile="smoke",
            oof_years=(2020,),
        )

    assert not (tmp_path / "predictions/baseline_manifest.json").exists()


@pytest.mark.parametrize("mutation", ["missing", "version", "future_path", "scaler"])
def test_completed_stage_rejects_changed_or_omitted_preprocessing_identity(
    tmp_path: Path,
    mutation: str,
) -> None:
    result = _run_oof(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    if mutation == "missing":
        del manifest["preprocessing"]
    elif mutation == "version":
        manifest["preprocessing"]["version"] = "causal-v2"
    elif mutation == "future_path":
        manifest["preprocessing"]["future_path_hours"] = 23
    else:
        manifest["preprocessing"]["target_scaler"] = "different-scaler"
    result.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ArtifactMismatch, match="manifest schema|preprocessing"):
        _run_oof(tmp_path)


def test_completed_stage_rejects_rewritten_scaler_population(tmp_path: Path) -> None:
    result = _run_oof(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["oof"]["preprocessing_populations"][0]["scalers"]["x"][
        "count"
    ] += 1
    result.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ArtifactMismatch, match="preprocessing scaler populations"):
        _run_oof(tmp_path)


def test_completed_stage_rejects_omitted_temporary_availability_hash(
    tmp_path: Path,
) -> None:
    result = _run_oof(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    del manifest["input_hashes"]["temporary_holiday_availability_sha256"]
    result.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(
        ArtifactMismatch, match="temporary_holiday_availability_sha256"
    ):
        _run_oof(tmp_path)


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


def test_rehashed_uniformly_truncated_oof_sibling_blocks_final_stage(
    tmp_path: Path,
) -> None:
    _run_oof(tmp_path)
    _truncate_stage_uniformly(tmp_path, stage="oof")

    with pytest.raises(ArtifactMismatch, match="coverage"):
        _run_final(tmp_path)


def test_rehashed_wrong_model_final_sibling_blocks_oof_stage(tmp_path: Path) -> None:
    final = _run_final(tmp_path)
    changed = pl.read_parquet(final.point_path).with_columns(
        pl.when(pl.col("model") == "xgboost")
        .then(pl.lit("wrong-model"))
        .otherwise(pl.col("model"))
        .alias("model")
    )
    changed.write_parquet(final.point_path)
    _rebind_artifact_hash(
        tmp_path, stage="final", artifact="point", path=final.point_path
    )

    with pytest.raises(ArtifactMismatch, match="model"):
        _run_oof(tmp_path)


def test_paper_final_rejects_nonexact_recorded_oof_fold_contract(tmp_path: Path) -> None:
    result = _run_oof(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["oof"]["split_ids"] = [
        "oof-2020",
        "oof-2021",
        "oof-2022",
    ]
    manifest["stages"]["oof"]["eval_years"] = [2020, 2021, 2022]
    result.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ArtifactMismatch, match="fold identities"):
        _run_final(tmp_path)


def test_paper_oof_rejects_nonfinal_recorded_final_fold_contract(tmp_path: Path) -> None:
    result = _run_final(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["final"]["split_ids"] = ["oof-2023"]
    manifest["stages"]["final"]["eval_years"] = [2023]
    result.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ArtifactMismatch, match="fold identities"):
        _run_oof(tmp_path)


@pytest.mark.parametrize(
    ("split_ids", "eval_years"),
    [
        (["oof-2020", "oof-2099"], [2020, 2099]),
        (["oof-2020", "oof-2020"], [2020, 2020]),
        (["oof-2021", "oof-2020"], [2021, 2020]),
    ],
    ids=("unknown", "duplicate", "reordered"),
)
def test_smoke_final_rejects_invalid_recorded_oof_split_identity_or_order(
    tmp_path: Path,
    split_ids: list[str],
    eval_years: list[int],
) -> None:
    result = _run_smoke_oof(tmp_path)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["stages"]["oof"]["split_ids"] = split_ids
    manifest["stages"]["oof"]["eval_years"] = eval_years
    result.manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ArtifactMismatch, match="split identities"):
        _run_smoke_final(tmp_path)


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


def test_journal_recovery_semantically_validates_a_rehashed_sibling_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _run_oof(tmp_path)

    def terminate_after_manifest(boundary: str) -> None:
        if boundary == "final-manifest-published":
            raise AbruptPublicationStop(boundary)

    monkeypatch.setattr(paper, "_publication_boundary", terminate_after_manifest)
    with pytest.raises(AbruptPublicationStop):
        _run_final(tmp_path)
    journal = tmp_path / "predictions/.baseline-transaction.json"
    assert journal.is_file()
    _rewrite_interrupted_manifest_binding(tmp_path, sibling_stage="oof")
    monkeypatch.setattr(paper, "_publication_boundary", lambda _: None)

    with pytest.raises(ArtifactMismatch, match="coverage"):
        _run_final(tmp_path)

    assert not journal.exists()


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
        "temporary_holiday_availability_sha256": file_sha256(
            PROJECT_ROOT / "configs/temporary_holiday_availability.csv"
        ),
    }


def test_concrete_cli_maps_reviewer_to_b0_and_b1w(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_short_cli_source(tmp_path)
    captured: list[dict[str, object]] = []
    monkeypatch.setattr(
        cli,
        "run_paper_oof_stage",
        lambda **kwargs: captured.append(kwargs),
    )

    result = cli.main(
        [
            *_baseline_cli_arguments(source, tmp_path, feature_set="reviewer"),
            "--profile",
            "smoke",
        ]
    )

    assert result == 0
    assert captured[0]["feature_sets"] == ("B0", "B1W")
    assert set(captured[0]["matrices"]) == {"B0", "B1W"}
    assert captured[0]["matrices"]["B1W"].future_columns == feature_columns("B1W")


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
    assert len(received) == 1
    expected = received[0]
    assert expected["expected_start"] == datetime(2019, 1, 1)
    assert expected["expected_end"] == datetime(2024, 10, 31, 23)
    assert expected["expected_rows"] == 51_144
    assert expected["expected_public_holiday_dates"] == 107
    assert expected["expected_substitute_or_temporary_dates"] == 14
    assert len(expected["temporary_holiday_availability"]) == 3


def test_readme_hqrc_commands_are_executable_from_the_repository_root() -> None:
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    assert "Run these commands from the repository root" in readme
    assert "uv sync --project hqrc_v3 --locked" in readme
    assert "export UV_LOCKED" not in readme
    assert "repository-root `uv.lock`" in readme
    for command in (
        "generate-oof",
        "fit-final-baselines",
        "prepare-residuals",
        "audit-data",
        "diagnose-ar",
        "approve-ar-calibration",
        "report",
    ):
        assert f"uv run --project hqrc_v3 --locked hqrc {command}" in readme
    assert "uv run --project hqrc_v3 hqrc " not in readme
    assert "uv run hqrc " not in readme
    assert "\nhqrc " not in readme
    assert "MODEL_SHA256=<" not in readme
    assert (
        "MODEL_SHA256=\"$(openssl dgst -sha256 "
        "hqrc_v3/configs/model_spaces.toml | awk '{print $NF}')\""
    ) in readme
    assert readme.count('--frozen-model-hash "$MODEL_SHA256"') == 4
    assert "--frozen-model-hash MODEL_SHA256" not in readme
