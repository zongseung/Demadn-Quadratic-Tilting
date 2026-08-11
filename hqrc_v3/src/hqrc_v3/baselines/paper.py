"""Factories and immutable publication stages for the paper baselines."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import struct
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import polars as pl

from hqrc_v3.baselines.classical import (
    ClassicalBaseline,
    make_classical_baseline,
    predictions_to_frame,
)
from hqrc_v3.baselines.config import (
    MODEL_NAMES,
    PAPER_SEEDS,
    NeuralPaperConfig,
    PaperBaselineConfig,
    PaperBaselineConfigError,
)
from hqrc_v3.baselines.protocol import BaselineFactory
from hqrc_v3.baselines.sequence import SequenceTrainingConfig, TorchBaselineFactory
from hqrc_v3.contracts import (
    PREDICTION_COLUMNS,
    DataContractError,
    ForecastMatrix,
    validate_forecast_feature_columns,
    validate_prediction_frame,
)
from hqrc_v3.features import feature_columns, history_columns
from hqrc_v3.oof import (
    FeatureSet,
    chronological_validation_tail,
    generate_cached_final,
    generate_oof_stream,
)
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
from hqrc_v3.residuals import PredictionCache
from hqrc_v3.splits import (
    AnnualFold,
    expanding_oof_folds,
    final_fold,
    select_fold_samples,
)

ENSEMBLE_SEED = 0
PAPER_HASH_KEYS = (
    "data_sha256",
    "experiment_sha256",
    "model_config_sha256",
    "event_registry_sha256",
    "holiday_calendar_sha256",
    "temporary_holiday_availability_sha256",
)
_FEATURE_SETS: tuple[FeatureSet, ...] = ("B0", "B1")
_NEURAL_MODELS = frozenset(("seq2seq_lstm", "transformer"))
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_TRANSACTION_NAME = ".baseline-transaction.json"
_COVERAGE_COLUMNS = (
    "origin",
    "target_timestamp",
    "horizon",
    "split_id",
    "observed_mw",
)
_MANIFEST_IDENTITY_KEYS = {
    "schema_version",
    "profile",
    "input_hashes",
    "models",
    "feature_sets",
    "classical_seed",
    "neural_seeds",
    "ensemble_seed",
    "feature_schemas",
    "preprocessing",
    "execution_overrides",
}
_FactoryBuilder = Callable[..., BaselineFactory]


@dataclass(frozen=True)
class PaperStageResult:
    """Published products and fit/cache accounting for one baseline stage."""

    stage: Literal["oof", "final"]
    members_path: Path
    point_path: Path
    manifest_path: Path
    fit_count: int
    cache_hit_count: int
    profile: Literal["paper", "smoke"]


def _publication_boundary(_: str) -> None:
    """No-op failure-injection hook at group-publication boundaries."""


def _sequence_config(config: NeuralPaperConfig) -> SequenceTrainingConfig:
    return SequenceTrainingConfig(
        hidden_size=config.hidden_size,
        layers=config.layers,
        heads=config.heads,
        dropout=config.dropout,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        epochs=config.epochs,
        patience=config.patience,
        seeds=config.seeds,
    )


def make_paper_factory(
    config: PaperBaselineConfig,
    model: str,
    *,
    profile: Literal["paper", "smoke"] = "paper",
    smoke_boosting_rounds: int | None = None,
) -> BaselineFactory:
    """Create exactly one immutable paper factory without search or substitution."""

    if not isinstance(config, PaperBaselineConfig):
        raise TypeError("config must be a PaperBaselineConfig")
    if model not in MODEL_NAMES or model not in config.models:
        raise PaperBaselineConfigError(f"unknown paper model {model!r}")
    if profile not in {"paper", "smoke"}:
        raise PaperBaselineConfigError("factory profile must be exactly 'paper' or 'smoke'")
    if smoke_boosting_rounds is not None:
        if profile != "smoke":
            raise PaperBaselineConfigError("paper factories cannot override boosting rounds")
        if model not in {"xgboost", "lightgbm"}:
            raise PaperBaselineConfigError("smoke boosting rounds apply only to boosting models")
        if (
            isinstance(smoke_boosting_rounds, bool)
            or not isinstance(smoke_boosting_rounds, int)
            or smoke_boosting_rounds <= 0
        ):
            raise PaperBaselineConfigError("smoke boosting rounds must be a positive integer")
    if model == "xgboost":
        params = config.xgboost.to_estimator_params()
        if smoke_boosting_rounds is not None:
            params["n_estimators"] = smoke_boosting_rounds
        return make_classical_baseline(model, params)
    if model == "lightgbm":
        params = config.lightgbm.to_estimator_params()
        if smoke_boosting_rounds is not None:
            params["n_estimators"] = smoke_boosting_rounds
        return make_classical_baseline(
            model,
            params,
            early_stopping_rounds=config.lightgbm.early_stopping_rounds,
        )
    if model == "svr":
        return make_classical_baseline(model, config.svr.to_estimator_params())
    neural = config.seq2seq_lstm if model == "seq2seq_lstm" else config.transformer
    return TorchBaselineFactory(model, _sequence_config(neural))


def _require_artifact_hashes(
    hashes: Mapping[str, str], config: PaperBaselineConfig
) -> dict[str, str]:
    if not isinstance(hashes, Mapping) or set(hashes) != set(PAPER_HASH_KEYS):
        raise DataContractError(
            "paper artifact hashes must contain exactly data, experiment, model config, "
            "event registry, holiday calendar, and temporary-holiday availability "
            "SHA-256 values"
        )
    normalized = dict(hashes)
    for name, value in normalized.items():
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise DataContractError(f"{name} must be a lowercase SHA-256 digest")
    if normalized["model_config_sha256"] != config.source_sha256:
        raise ArtifactMismatch("model_config_sha256 differs from the loaded frozen config")
    return {name: normalized[name] for name in PAPER_HASH_KEYS}


def _require_stage_selection(
    *,
    config: PaperBaselineConfig,
    models: tuple[str, ...] | None,
    feature_sets: tuple[FeatureSet, ...] | None,
    profile: Literal["paper", "smoke"],
) -> tuple[tuple[str, ...], tuple[FeatureSet, ...]]:
    if profile not in {"paper", "smoke"}:
        raise DataContractError("baseline profile must be exactly 'paper' or 'smoke'")
    selected_models = config.models if models is None else models
    selected_features = _FEATURE_SETS if feature_sets is None else feature_sets
    if (
        not isinstance(selected_models, tuple)
        or not selected_models
        or len(set(selected_models)) != len(selected_models)
        or any(model not in config.models for model in selected_models)
    ):
        raise DataContractError("selected models must be a unique non-empty paper-model subset")
    if (
        not isinstance(selected_features, tuple)
        or not selected_features
        or len(set(selected_features)) != len(selected_features)
        or any(feature_set not in _FEATURE_SETS for feature_set in selected_features)
    ):
        raise DataContractError("selected feature sets must be a unique B0/B1 subset")
    if profile == "paper" and (
        selected_models != config.models or selected_features != _FEATURE_SETS
    ):
        raise DataContractError("paper publication requires all five models and both B0/B1")
    return selected_models, selected_features


def _feature_schema(matrix: ForecastMatrix, feature_set: FeatureSet) -> dict[str, list[str]]:
    if not isinstance(matrix, ForecastMatrix):
        raise TypeError(f"{feature_set} matrix must be a ForecastMatrix")
    validate_forecast_feature_columns(matrix)
    if (
        matrix.history_columns != history_columns(feature_set)
        or matrix.future_columns != feature_columns(feature_set)
    ):
        raise DataContractError(f"{feature_set} matrix does not match its frozen feature schema")
    return {
        "history": list(matrix.history_columns),
        "future": list(matrix.future_columns),
    }


def _matrix_contracts(
    matrices: Mapping[str, ForecastMatrix], feature_sets: tuple[FeatureSet, ...]
) -> tuple[dict[FeatureSet, ForecastMatrix], dict[str, dict[str, list[str]]]]:
    if not isinstance(matrices, Mapping) or set(matrices) != set(feature_sets):
        raise DataContractError("matrices must contain exactly the selected feature sets")
    normalized: dict[FeatureSet, ForecastMatrix] = {}
    schemas: dict[str, dict[str, list[str]]] = {}
    reference_origins: np.ndarray | None = None
    reference_times: np.ndarray | None = None
    reference_target: np.ndarray | None = None
    for feature_set in feature_sets:
        matrix = matrices[feature_set]
        schemas[feature_set] = _feature_schema(matrix, feature_set)
        if reference_times is None:
            reference_origins = matrix.origins
            reference_times = matrix.target_times
            reference_target = matrix.target
        elif (
            not np.array_equal(matrix.origins, reference_origins)
            or not np.array_equal(matrix.target_times, reference_times)
            or not np.array_equal(matrix.target, reference_target)
        ):
            raise DataContractError(
                "B0 and B1 matrix coverage keys or observed values differ"
            )
        normalized[feature_set] = matrix
    return normalized, schemas


def _schema_digest(schema: Mapping[str, list[str]]) -> str:
    payload = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _cache_hashes(
    hashes: Mapping[str, str],
    schema: Mapping[str, list[str]],
    execution_overrides: Mapping[str, int],
) -> dict[str, str]:
    combined_config = hashlib.sha256(
        (
            hashes["experiment_sha256"] + ":" + hashes["model_config_sha256"]
        ).encode()
    ).hexdigest()
    return {
        "config": combined_config,
        "data": hashes["data_sha256"],
        "events": hashes["event_registry_sha256"],
        "experiment_sha256": hashes["experiment_sha256"],
        "model_config_sha256": hashes["model_config_sha256"],
        "holiday_calendar_sha256": hashes["holiday_calendar_sha256"],
        "temporary_holiday_availability_sha256": hashes[
            "temporary_holiday_availability_sha256"
        ],
        "feature_schema_sha256": _schema_digest(schema),
        "execution_profile_sha256": hashlib.sha256(
            json.dumps(
                execution_overrides, sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
    }


def _empty_prediction_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "origin": pl.Datetime("ns"),
            "target_timestamp": pl.Datetime("ns"),
            "horizon": pl.Int64,
            "observed_mw": pl.Float64,
            "predicted_mw": pl.Float64,
            "model": pl.String,
            "feature_set": pl.String,
            "seed": pl.Int64,
            "split_id": pl.String,
        }
    ).select(PREDICTION_COLUMNS)


def _validated_groups(frame: pl.DataFrame, *, description: str) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame) or tuple(frame.columns) != PREDICTION_COLUMNS:
        raise ArtifactMismatch(f"{description} prediction schema differs")
    if frame.is_empty():
        return frame
    try:
        for group in frame.partition_by(
            ["model", "feature_set", "seed", "split_id"], maintain_order=True
        ):
            validate_prediction_frame(group)
    except (DataContractError, TypeError, ValueError) as error:
        raise ArtifactMismatch(f"{description} prediction frame is invalid: {error}") from error
    return frame


def _ensemble_frame(
    member_frames: list[pl.DataFrame],
    *,
    model: str,
    feature_set: FeatureSet,
    expected_seeds: tuple[int, ...],
) -> pl.DataFrame:
    members = _validated_groups(
        pl.concat(member_frames, how="vertical"), description="neural member"
    )
    if set(members["model"]) != {model} or set(members["feature_set"]) != {feature_set}:
        raise DataContractError("neural member frame context differs from its stream")
    if set(members["seed"]) != set(expected_seeds):
        raise DataContractError("neural member seed set differs from the frozen five seeds")
    keys = [
        "origin",
        "target_timestamp",
        "horizon",
        "observed_mw",
        "model",
        "feature_set",
        "split_id",
    ]
    averaged = (
        members.group_by(keys, maintain_order=True)
        .agg(
            pl.col("predicted_mw").mean().alias("predicted_mw"),
            pl.col("seed").n_unique().alias("member_count"),
        )
        .with_columns(pl.lit(ENSEMBLE_SEED, dtype=pl.Int64).alias("seed"))
    )
    if not averaged["member_count"].eq(len(expected_seeds)).all():
        raise DataContractError("neural member predictions are incomplete")
    return _validated_groups(
        averaged.select(PREDICTION_COLUMNS), description="neural ensemble"
    )


def _stream_records(
    models: tuple[str, ...], feature_sets: tuple[FeatureSet, ...], classical_seed: int
) -> list[dict[str, object]]:
    return [
        {
            "model": model,
            "feature_set": feature_set,
            "seeds": list(PAPER_SEEDS if model in _NEURAL_MODELS else (classical_seed,)),
        }
        for model in models
        for feature_set in feature_sets
    ]


def _manifest_identity(
    *,
    profile: Literal["paper", "smoke"],
    hashes: Mapping[str, str],
    models: tuple[str, ...],
    feature_sets: tuple[FeatureSet, ...],
    classical_seed: int,
    schemas: Mapping[str, Mapping[str, list[str]]],
    execution_overrides: Mapping[str, int],
    config: PaperBaselineConfig,
) -> dict[str, object]:
    return {
        "schema_version": 2,
        "profile": profile,
        "input_hashes": dict(hashes),
        "models": list(models),
        "feature_sets": list(feature_sets),
        "classical_seed": classical_seed,
        "neural_seeds": list(PAPER_SEEDS),
        "ensemble_seed": ENSEMBLE_SEED,
        "feature_schemas": dict(schemas),
        "preprocessing": config.preprocessing.to_manifest(),
        "execution_overrides": dict(execution_overrides),
    }


def _load_manifest(path: Path, identity: Mapping[str, object]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactMismatch("baseline manifest is missing or invalid") from error
    if not isinstance(raw, dict) or set(raw) != {*identity, "stages"}:
        raise ArtifactMismatch("baseline manifest schema differs")
    for name, expected in identity.items():
        if raw.get(name) != expected:
            if name == "input_hashes" and isinstance(raw.get(name), dict):
                actual_hashes = raw[name]
                for hash_name in PAPER_HASH_KEYS:
                    if actual_hashes.get(hash_name) != expected.get(hash_name):
                        raise ArtifactMismatch(
                            f"{hash_name} differs from the baseline manifest"
                        )
            label = "feature schema" if name == "feature_schemas" else name
            raise ArtifactMismatch(f"{label} differs from the baseline manifest")
    if not isinstance(raw.get("stages"), dict) or not set(raw["stages"]).issubset(
        {"oof", "final"}
    ):
        raise ArtifactMismatch("baseline manifest stages are invalid")
    return raw


def _artifact_paths(run_dir: Path, stage: Literal["oof", "final"]) -> tuple[Path, Path, Path]:
    predictions = run_dir / "predictions"
    if stage == "oof":
        members = predictions / "oof_members.parquet"
        point = predictions / "oof.parquet"
    else:
        members = predictions / "final_2024_members.parquet"
        point = predictions / "final_2024.parquet"
    return members, point, predictions / "baseline_manifest.json"


def _verify_artifact_hashes(
    run_dir: Path, stage: str, record: Mapping[str, Any]
) -> tuple[Path, Path]:
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"members", "point"}:
        raise ArtifactMismatch(f"{stage} artifact manifest is invalid")
    paths: dict[str, Path] = {}
    for name in ("members", "point"):
        entry = artifacts[name]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ArtifactMismatch(f"{stage} {name} artifact entry is invalid")
        relative = entry["path"]
        if not isinstance(relative, str):
            raise ArtifactMismatch(f"{stage} {name} path is invalid")
        path = run_dir / relative
        if not path.is_file():
            raise ArtifactMismatch(f"partial {stage} baseline publication")
        if file_sha256(path) != entry["sha256"]:
            raise ArtifactMismatch(f"{stage} {name} SHA-256 differs from the manifest")
        paths[name] = path
    return paths["members"], paths["point"]


def _expected_combinations(
    *,
    models: tuple[str, ...],
    feature_sets: tuple[FeatureSet, ...],
    split_ids: tuple[str, ...],
    classical_seed: int,
    members: bool,
) -> set[tuple[str, str, int, str]]:
    combinations: set[tuple[str, str, int, str]] = set()
    for model in models:
        if members and model not in _NEURAL_MODELS:
            continue
        seeds = PAPER_SEEDS if members else (
            (ENSEMBLE_SEED,) if model in _NEURAL_MODELS else (classical_seed,)
        )
        for feature_set in feature_sets:
            for seed in seeds:
                for split_id in split_ids:
                    combinations.add((model, feature_set, seed, split_id))
    return combinations


def _coverage_record(frame: pl.DataFrame) -> dict[str, object]:
    if tuple(frame.columns) != _COVERAGE_COLUMNS or frame.is_empty():
        raise DataContractError("expected prediction coverage must be non-empty and exact")
    normalized = frame.select(
        pl.col("origin").cast(pl.Datetime("ns"), strict=True),
        pl.col("target_timestamp").cast(pl.Datetime("ns"), strict=True),
        pl.col("horizon").cast(pl.Int64, strict=True),
        pl.col("split_id").cast(pl.String, strict=True),
        pl.col("observed_mw").cast(pl.Float64, strict=True),
    )
    if any(normalized[column].null_count() for column in _COVERAGE_COLUMNS):
        raise DataContractError("expected prediction coverage must not contain nulls")
    if not normalized["observed_mw"].is_finite().all():
        raise DataContractError("expected prediction coverage values must be finite")
    canonical = (
        normalized.with_columns(
            pl.col("origin").cast(pl.Int64),
            pl.col("target_timestamp").cast(pl.Int64),
        )
        .sort("split_id", "origin", "target_timestamp", "horizon")
    )
    digest = hashlib.sha256()
    for origin, target, horizon, split_id, observed in canonical.iter_rows():
        encoded_split = split_id.encode("utf-8")
        digest.update(
            struct.pack(">qqqI", origin, target, horizon, len(encoded_split))
        )
        digest.update(encoded_split)
        digest.update(struct.pack(">d", observed))
    return {"count": canonical.height, "sha256": digest.hexdigest()}


def _expected_stage_coverage(
    matrix: ForecastMatrix, folds: tuple[AnnualFold, ...]
) -> dict[str, object]:
    frames: list[pl.DataFrame] = []
    for fold in folds:
        _, evaluation_indices = select_fold_samples(matrix, fold)
        if evaluation_indices.size == 0:
            raise DataContractError(
                f"{fold.split_id} requires non-empty expected prediction coverage"
            )
        evaluation = matrix.take(evaluation_indices)
        frames.append(
            predictions_to_frame(
                evaluation,
                evaluation.target,
                model="expected-coverage",
                feature_set="expected-coverage",
                seed=0,
                fold=fold,
            ).select(_COVERAGE_COLUMNS)
        )
    return _coverage_record(pl.concat(frames, how="vertical"))


def _timestamp_population(times: np.ndarray, *, unit: str) -> dict[str, object]:
    values = np.asarray(times).reshape(-1).astype("datetime64[ns]")
    if not values.size or np.isnat(values).any():
        raise DataContractError("preprocessing population timestamps must be complete")
    unique = np.unique(values)
    return {
        "count": int(unique.size),
        "start": np.datetime_as_string(unique[0], unit="s"),
        "end": np.datetime_as_string(unique[-1], unit="s"),
        "unit": unit,
    }


def _preprocessing_populations(
    matrix: ForecastMatrix,
    folds: tuple[AnnualFold, ...],
    *,
    models: tuple[str, ...],
    feature_sets: tuple[FeatureSet, ...],
    validation_days: int,
) -> list[dict[str, object]]:
    """Derive the exact timestamp populations every fitted scaler must use."""

    records: list[dict[str, object]] = []
    for model in models:
        for feature_set in feature_sets:
            for fold in folds:
                train_indices, _ = select_fold_samples(matrix, fold)
                outer_train = matrix.take(train_indices)
                fit_train = (
                    outer_train
                    if model == "svr"
                    else chronological_validation_tail(
                        outer_train, days=validation_days
                    )[0]
                )
                fit_daily = _timestamp_population(
                    fit_train.origins, unit="daily-sample"
                )
                target = _timestamp_population(
                    fit_train.target_times, unit="unique-hour"
                )
                if target["count"] != len(fit_train.origins) * 24:
                    raise DataContractError(
                        "estimator-fit targets must be non-overlapping complete days"
                    )
                if model in _NEURAL_MODELS:
                    history_offsets = np.arange(-168, 0).astype("timedelta64[h]")
                    history_times = (
                        fit_train.origins[:, None].astype("datetime64[ns]")
                        + history_offsets[None, :]
                    )
                    scalers = {
                        "target": target,
                        "weather": _timestamp_population(
                            history_times, unit="unique-hour"
                        ),
                        "calendar": target,
                    }
                else:
                    scalers = {"x": fit_daily, "target": target}
                records.append(
                    {
                        "model": model,
                        "feature_set": feature_set,
                        "split_id": fold.split_id,
                        "scalers": scalers,
                    }
                )
    return records


def _require_expected_coverage(value: object) -> dict[str, object]:
    if (
        not isinstance(value, dict)
        or set(value) != {"count", "sha256"}
        or type(value.get("count")) is not int
        or value["count"] <= 0
        or not isinstance(value.get("sha256"), str)
        or _SHA256.fullmatch(value["sha256"]) is None
    ):
        raise ArtifactMismatch("expected coverage manifest is invalid")
    return value


def _validate_published_frames(
    members: pl.DataFrame,
    point: pl.DataFrame,
    *,
    models: tuple[str, ...],
    feature_sets: tuple[FeatureSet, ...],
    split_ids: tuple[str, ...],
    eval_years: tuple[int, ...],
    classical_seed: int,
    expected_coverage: Mapping[str, object],
) -> None:
    members = _validated_groups(members, description="member")
    point = _validated_groups(point, description="point")
    if point.is_empty():
        raise ArtifactMismatch("point prediction publication must not be empty")
    normalized_coverage = _require_expected_coverage(dict(expected_coverage))

    def combinations(frame: pl.DataFrame) -> set[tuple[str, str, int, str]]:
        if frame.is_empty():
            return set()
        return set(
            frame.select("model", "feature_set", "seed", "split_id")
            .unique()
            .iter_rows()
        )

    if combinations(point) != _expected_combinations(
        models=models,
        feature_sets=feature_sets,
        split_ids=split_ids,
        classical_seed=classical_seed,
        members=False,
    ):
        raise ArtifactMismatch("point model, feature, seed, or split identities differ")
    if combinations(members) != _expected_combinations(
        models=models,
        feature_sets=feature_sets,
        split_ids=split_ids,
        classical_seed=classical_seed,
        members=True,
    ):
        raise ArtifactMismatch("member model, feature, seed, or split identities differ")

    def validate_stream_coverage(frame: pl.DataFrame, *, description: str) -> None:
        if frame.is_empty():
            return
        for stream in frame.partition_by(
            ["model", "feature_set", "seed"], maintain_order=True
        ):
            try:
                actual = _coverage_record(stream.select(_COVERAGE_COLUMNS))
            except (DataContractError, TypeError, ValueError) as error:
                raise ArtifactMismatch(
                    f"{description} prediction coverage is invalid: {error}"
                ) from error
            if actual != normalized_coverage:
                raise ArtifactMismatch(
                    f"{description} prediction coverage differs from expected coverage"
                )

    validate_stream_coverage(point, description="point")
    validate_stream_coverage(members, description="member")
    if set(point["target_timestamp"].dt.year()) != set(eval_years):
        raise ArtifactMismatch("prediction evaluation years differ from the stage contract")
    if not members.is_empty() and set(members["target_timestamp"].dt.year()) != set(
        eval_years
    ):
        raise ArtifactMismatch("member evaluation years differ from the stage contract")

    point_stream_keys: set[tuple[str, str]] = set()
    reference_targets: set[tuple[str, Any]] | None = None
    for stream in point.partition_by(["model", "feature_set"], maintain_order=True):
        identity = (str(stream["model"].item(0)), str(stream["feature_set"].item(0)))
        point_stream_keys.add(identity)
        targets = set(stream.select("split_id", "target_timestamp").iter_rows())
        if reference_targets is None:
            reference_targets = targets
        elif targets != reference_targets:
            raise ArtifactMismatch("point streams do not cover identical timestamps")
    if point_stream_keys != {(model, feature) for model in models for feature in feature_sets}:
        raise ArtifactMismatch("point model/feature streams differ")

    neural_models = tuple(model for model in models if model in _NEURAL_MODELS)
    if neural_models:
        keys = [
            "origin",
            "target_timestamp",
            "horizon",
            "observed_mw",
            "model",
            "feature_set",
            "split_id",
        ]
        expected_mean = members.group_by(keys).agg(
            pl.col("predicted_mw").mean().alias("expected_prediction")
        )
        actual = point.filter(pl.col("model").is_in(neural_models)).join(
            expected_mean, on=keys, how="inner"
        )
        if actual.height != expected_mean.height or not np.array_equal(
            actual["predicted_mw"].to_numpy(), actual["expected_prediction"].to_numpy()
        ):
            raise ArtifactMismatch("neural ensemble is not the exact five-seed mean")
    elif not members.is_empty():
        raise ArtifactMismatch("classical-only publication must have no member rows")


def _read_parquet(path: Path, *, description: str) -> pl.DataFrame:
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ArtifactMismatch(f"unable to read {description} predictions") from error


def _validate_completed_stage(
    run_dir: Path,
    stage: Literal["oof", "final"],
    manifest: Mapping[str, Any],
    *,
    models: tuple[str, ...],
    feature_sets: tuple[FeatureSet, ...],
    split_ids: tuple[str, ...],
    eval_years: tuple[int, ...],
    classical_seed: int,
    expected_coverage: Mapping[str, object],
    expected_preprocessing_populations: list[dict[str, object]],
) -> tuple[Path, Path]:
    record = manifest["stages"].get(stage)
    if not isinstance(record, dict):
        raise ArtifactMismatch(f"partial {stage} baseline publication")
    required = {
        "artifacts",
        "eval_years",
        "expected_coverage",
        "preprocessing_populations",
        "split_ids",
        "streams",
    }
    if set(record) != required:
        raise ArtifactMismatch(f"{stage} stage manifest is invalid")
    if record["eval_years"] != list(eval_years) or record["split_ids"] != list(split_ids):
        raise ArtifactMismatch(f"{stage} fold identities differ from the manifest")
    if record["streams"] != _stream_records(models, feature_sets, classical_seed):
        raise ArtifactMismatch(f"{stage} model streams differ from the manifest")
    recorded_coverage = _require_expected_coverage(record["expected_coverage"])
    if recorded_coverage != dict(expected_coverage):
        raise ArtifactMismatch(f"{stage} expected coverage differs from the supplied matrix")
    if record["preprocessing_populations"] != expected_preprocessing_populations:
        raise ArtifactMismatch(
            f"{stage} preprocessing scaler populations differ from the supplied matrix"
        )
    members_path, point_path = _verify_artifact_hashes(run_dir, stage, record)
    _validate_published_frames(
        _read_parquet(members_path, description=f"{stage} member"),
        _read_parquet(point_path, description=f"{stage} point"),
        models=models,
        feature_sets=feature_sets,
        split_ids=split_ids,
        eval_years=eval_years,
        classical_seed=classical_seed,
        expected_coverage=expected_coverage,
    )
    return members_path, point_path


def _expected_fit_units(
    models: tuple[str, ...], feature_sets: tuple[FeatureSet, ...], folds: int
) -> int:
    return sum(
        (len(PAPER_SEEDS) if model in _NEURAL_MODELS else 1)
        * len(feature_sets)
        * folds
        for model in models
    )


@contextmanager
def _publication_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".baseline-publication.lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(prefix=".baseline.", suffix=suffix, dir=directory)
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


class _RecoveryImpossible(Exception):
    """Signal that a valid interrupted transaction must be rolled back."""


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_atomic_canonical_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = _temporary_path(path.parent, ".atomic.json")
    try:
        _write_fsynced(temporary, _canonical_json_bytes(value))
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _safe_journal_child(directory: Path, name: object) -> Path:
    if (
        not isinstance(name, str)
        or not name
        or Path(name).is_absolute()
        or Path(name).parts != (name,)
        or name in {".", ".."}
    ):
        raise ArtifactMismatch("unsafe publication journal path")
    path = directory / name
    if path.parent.resolve() != directory.resolve() or path.is_symlink():
        raise ArtifactMismatch("unsafe publication journal path")
    return path


def _require_transaction_manifest(value: object, *, description: str) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or set(value) != {*_MANIFEST_IDENTITY_KEYS, "stages"}
        or not isinstance(value.get("stages"), dict)
        or not set(value["stages"]).issubset({"oof", "final"})
    ):
        raise ArtifactMismatch(f"publication journal {description} manifest is invalid")
    return value


def _load_transaction(
    directory: Path,
) -> tuple[dict[str, Any], dict[str, Path]] | None:
    journal_path = directory / _TRANSACTION_NAME
    if journal_path.is_symlink():
        raise ArtifactMismatch("unsafe publication journal path")
    if not journal_path.exists():
        return None
    if not journal_path.is_file():
        raise ArtifactMismatch("publication journal is invalid")
    try:
        payload = journal_path.read_bytes()
        raw = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactMismatch("publication journal is invalid") from error
    required = {
        "schema_version",
        "stage",
        "members",
        "point",
        "manifest",
        "intended_manifest",
        "prior_manifest",
    }
    if (
        not isinstance(raw, dict)
        or set(raw) != required
        or type(raw.get("schema_version")) is not int
        or raw["schema_version"] != 1
        or raw.get("stage") not in {"oof", "final"}
    ):
        raise ArtifactMismatch("publication journal schema is invalid")
    try:
        if payload != _canonical_json_bytes(raw):
            raise ArtifactMismatch("publication journal is not canonical")
    except (TypeError, ValueError) as error:
        raise ArtifactMismatch("publication journal is invalid") from error

    stage: Literal["oof", "final"] = raw["stage"]
    expected_members, expected_point, expected_manifest = _artifact_paths(
        directory.parent, stage
    )
    expected_finals = {
        "members": expected_members.name,
        "point": expected_point.name,
        "manifest": expected_manifest.name,
    }
    expected_suffixes = {
        "members": ".members.parquet",
        "point": ".point.parquet",
        "manifest": ".manifest.json",
    }
    paths: dict[str, Path] = {"journal": journal_path}
    temporary_names: set[str] = set()
    for component in ("members", "point", "manifest"):
        entry = raw[component]
        if not isinstance(entry, dict) or set(entry) != {"temp", "final", "sha256"}:
            raise ArtifactMismatch(f"publication journal {component} entry is invalid")
        temporary = entry["temp"]
        final = entry["final"]
        digest = entry["sha256"]
        temporary_path = _safe_journal_child(directory, temporary)
        final_path = _safe_journal_child(directory, final)
        if (
            not temporary.startswith(".baseline.")
            or not temporary.endswith(expected_suffixes[component])
            or temporary in temporary_names
            or final != expected_finals[component]
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
        ):
            raise ArtifactMismatch(f"publication journal {component} entry is invalid")
        temporary_names.add(temporary)
        paths[f"{component}_temp"] = temporary_path
        paths[f"{component}_final"] = final_path

    intended = _require_transaction_manifest(
        raw["intended_manifest"], description="intended"
    )
    prior_value = raw["prior_manifest"]
    prior = (
        None
        if prior_value is None
        else _require_transaction_manifest(prior_value, description="prior")
    )
    intended_stage = intended["stages"].get(stage)
    if not isinstance(intended_stage, dict):
        raise ArtifactMismatch("publication journal intended stage is invalid")
    prior_stages = {} if prior is None else prior["stages"]
    if stage in prior_stages:
        raise ArtifactMismatch("publication journal prior stage is invalid")
    for name in _MANIFEST_IDENTITY_KEYS:
        if prior is not None and prior[name] != intended[name]:
            raise ArtifactMismatch("publication journal manifest identity differs")
    if {
        name: record for name, record in intended["stages"].items() if name != stage
    } != prior_stages:
        raise ArtifactMismatch("publication journal does not preserve prior stages")
    artifacts = intended_stage.get("artifacts")
    expected_artifacts = {
        "members": {
            "path": f"predictions/{expected_members.name}",
            "sha256": raw["members"]["sha256"],
        },
        "point": {
            "path": f"predictions/{expected_point.name}",
            "sha256": raw["point"]["sha256"],
        },
    }
    if artifacts != expected_artifacts:
        raise ArtifactMismatch("publication journal artifact binding is invalid")
    if hashlib.sha256(_canonical_json_bytes(intended)).hexdigest() != raw["manifest"][
        "sha256"
    ]:
        raise ArtifactMismatch("publication journal manifest hash differs")
    return raw, paths


def _matches_digest(path: Path, digest: str) -> bool:
    return path.is_file() and not path.is_symlink() and file_sha256(path) == digest


def _recover_component(
    *, temporary: Path, final: Path, digest: str, description: str
) -> None:
    if _matches_digest(final, digest):
        if temporary.exists():
            if not temporary.is_file() or temporary.is_symlink():
                raise _RecoveryImpossible(description)
            temporary.unlink()
        return
    if not _matches_digest(temporary, digest):
        raise _RecoveryImpossible(description)
    if final.exists() and (not final.is_file() or final.is_symlink()):
        raise _RecoveryImpossible(description)
    os.replace(temporary, final)


def _unlink_transaction_file(path: Path) -> None:
    if path.exists():
        if not path.is_file() or path.is_symlink():
            raise ArtifactMismatch("publication transaction path is invalid")
        path.unlink()


def _rollback_transaction(
    directory: Path, transaction: Mapping[str, Any], paths: Mapping[str, Path]
) -> None:
    for component in ("members", "point"):
        _unlink_transaction_file(paths[f"{component}_temp"])
        _unlink_transaction_file(paths[f"{component}_final"])
    _unlink_transaction_file(paths["manifest_temp"])
    manifest_path = paths["manifest_final"]
    prior = transaction["prior_manifest"]
    if prior is None:
        _unlink_transaction_file(manifest_path)
    else:
        _write_atomic_canonical_json(manifest_path, prior)
    _fsync_directory(directory)
    _unlink_transaction_file(paths["journal"])
    _fsync_directory(directory)


def _recover_publication(
    directory: Path, *, validate_completed: Callable[[], Any]
) -> Any | None:
    loaded = _load_transaction(directory)
    if loaded is None:
        return None
    transaction, paths = loaded
    try:
        for component in ("members", "point", "manifest"):
            _recover_component(
                temporary=paths[f"{component}_temp"],
                final=paths[f"{component}_final"],
                digest=transaction[component]["sha256"],
                description=component,
            )
        _fsync_directory(directory)
        validated = validate_completed()
        _unlink_transaction_file(paths["journal"])
        _fsync_directory(directory)
        return validated
    except _RecoveryImpossible:
        _rollback_transaction(directory, transaction, paths)
        return None
    except ArtifactMismatch:
        _rollback_transaction(directory, transaction, paths)
        raise


def _publish_stage(
    *,
    run_dir: Path,
    stage: Literal["oof", "final"],
    members: pl.DataFrame,
    point: pl.DataFrame,
    manifest: dict[str, Any],
    stage_record: dict[str, Any],
) -> tuple[Path, Path, Path]:
    members_path, point_path, manifest_path = _artifact_paths(run_dir, stage)
    directory = members_path.parent
    members_temp = _temporary_path(directory, ".members.parquet")
    point_temp = _temporary_path(directory, ".point.parquet")
    manifest_temp = _temporary_path(directory, ".manifest.json")
    journal_path = directory / _TRANSACTION_NAME
    prior_manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.is_file()
        else None
    )
    transaction: dict[str, Any] | None = None
    paths: dict[str, Path] | None = None
    try:
        members.write_parquet(members_temp)
        point.write_parquet(point_temp)
        _fsync_file(members_temp)
        _fsync_file(point_temp)
        intended_record = json.loads(_canonical_json_bytes(stage_record))
        intended_record["artifacts"] = {
            "members": {
                "path": members_path.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(members_temp),
            },
            "point": {
                "path": point_path.relative_to(run_dir).as_posix(),
                "sha256": file_sha256(point_temp),
            },
        }
        intended_manifest = json.loads(_canonical_json_bytes(manifest))
        intended_manifest["stages"][stage] = intended_record
        manifest_payload = _canonical_json_bytes(intended_manifest)
        _write_fsynced(manifest_temp, manifest_payload)
        transaction = {
            "schema_version": 1,
            "stage": stage,
            "members": {
                "temp": members_temp.name,
                "final": members_path.name,
                "sha256": intended_record["artifacts"]["members"]["sha256"],
            },
            "point": {
                "temp": point_temp.name,
                "final": point_path.name,
                "sha256": intended_record["artifacts"]["point"]["sha256"],
            },
            "manifest": {
                "temp": manifest_temp.name,
                "final": manifest_path.name,
                "sha256": hashlib.sha256(manifest_payload).hexdigest(),
            },
            "intended_manifest": intended_manifest,
            "prior_manifest": prior_manifest,
        }
        _write_atomic_canonical_json(journal_path, transaction)
        loaded = _load_transaction(directory)
        if loaded is None:  # pragma: no cover - atomic writer guarantees existence
            raise ArtifactMismatch("publication journal disappeared")
        _, paths = loaded
        os.replace(members_temp, members_path)
        _publication_boundary(f"{stage}-members-published")
        os.replace(point_temp, point_path)
        _publication_boundary(f"{stage}-point-published")
        os.replace(manifest_temp, manifest_path)
        _publication_boundary(f"{stage}-manifest-published")
        _fsync_directory(directory)
        journal_path.unlink()
        _fsync_directory(directory)
    except Exception:
        if transaction is not None and paths is not None:
            _rollback_transaction(directory, transaction, paths)
        else:
            members_temp.unlink(missing_ok=True)
            point_temp.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)
            journal_path.unlink(missing_ok=True)
            _fsync_directory(directory)
        raise
    return members_path, point_path, manifest_path


def _recorded_stage_folds(
    *,
    recorded_stage: Literal["oof", "final"],
    record: object,
    requested_stage: Literal["oof", "final"],
    requested_folds: tuple[AnnualFold, ...],
    profile: Literal["paper", "smoke"],
) -> tuple[AnnualFold, ...]:
    if recorded_stage == requested_stage:
        return requested_folds
    if recorded_stage == "final":
        return (final_fold(),)
    if profile == "paper":
        return expanding_oof_folds()
    if not isinstance(record, dict):
        raise ArtifactMismatch("oof stage manifest is invalid")
    split_ids = record.get("split_ids")
    if (
        not isinstance(split_ids, list)
        or not split_ids
        or any(not isinstance(split_id, str) for split_id in split_ids)
    ):
        raise ArtifactMismatch("smoke OOF split identities are invalid")
    allowed = expanding_oof_folds()
    selected = tuple(fold for fold in allowed if fold.split_id in split_ids)
    if tuple(fold.split_id for fold in selected) != tuple(split_ids):
        raise ArtifactMismatch("smoke OOF split identities are invalid")
    return selected


def _stage_preflight(
    *,
    run_dir: Path,
    stage: Literal["oof", "final"],
    identity: Mapping[str, object],
    models: tuple[str, ...],
    feature_sets: tuple[FeatureSet, ...],
    matrix: ForecastMatrix,
    folds: tuple[AnnualFold, ...],
    classical_seed: int,
    expected_coverage: Mapping[str, object],
    expected_preprocessing_populations: list[dict[str, object]],
    profile: Literal["paper", "smoke"],
    validation_days: int,
) -> PaperStageResult | tuple[dict[str, Any], Path, Path, Path]:
    members_path, point_path, manifest_path = _artifact_paths(run_dir, stage)
    manifest = _load_manifest(manifest_path, identity)
    stage_products = {
        name: _artifact_paths(run_dir, name)[:2] for name in ("oof", "final")
    }
    all_products = [path for products in stage_products.values() for path in products]
    validated_stages: dict[str, tuple[Path, Path]] = {}
    if manifest is None:
        if any(path.exists() or path.is_symlink() for path in all_products):
            raise ArtifactMismatch("partial baseline publication without a manifest")
        manifest = {**identity, "stages": {}}
    else:
        for recorded_stage, products in stage_products.items():
            present = [path.is_file() and not path.is_symlink() for path in products]
            exists = [path.exists() or path.is_symlink() for path in products]
            if recorded_stage not in manifest["stages"] and any(exists):
                raise ArtifactMismatch(
                    f"orphan {recorded_stage} baseline publication"
                )
            if recorded_stage in manifest["stages"] and not all(present):
                raise ArtifactMismatch(
                    f"partial {recorded_stage} baseline publication"
                )
        for recorded_stage, record in manifest["stages"].items():
            recorded_folds = _recorded_stage_folds(
                recorded_stage=recorded_stage,
                record=record,
                requested_stage=stage,
                requested_folds=folds,
                profile=profile,
            )
            recorded_split_ids = tuple(fold.split_id for fold in recorded_folds)
            recorded_eval_years = tuple(fold.eval_year for fold in recorded_folds)
            try:
                recorded_coverage = (
                    dict(expected_coverage)
                    if recorded_stage == stage
                    else _expected_stage_coverage(matrix, recorded_folds)
                )
                recorded_populations = (
                    expected_preprocessing_populations
                    if recorded_stage == stage
                    else _preprocessing_populations(
                        matrix,
                        recorded_folds,
                        models=models,
                        feature_sets=feature_sets,
                        validation_days=validation_days,
                    )
                )
            except DataContractError as error:
                raise ArtifactMismatch(
                    f"{recorded_stage} expected coverage cannot be derived"
                ) from error
            validated_stages[recorded_stage] = _validate_completed_stage(
                run_dir,
                recorded_stage,
                manifest,
                models=models,
                feature_sets=feature_sets,
                split_ids=recorded_split_ids,
                eval_years=recorded_eval_years,
                classical_seed=classical_seed,
                expected_coverage=recorded_coverage,
                expected_preprocessing_populations=recorded_populations,
            )
    stage_exists = members_path.exists() or point_path.exists() or stage in manifest["stages"]
    if not stage_exists:
        return manifest, members_path, point_path, manifest_path
    if not (members_path.is_file() and point_path.is_file() and stage in manifest["stages"]):
        raise ArtifactMismatch(f"partial {stage} baseline publication")
    validated_members, validated_point = validated_stages[stage]
    units = _expected_fit_units(models, feature_sets, len(folds))
    return PaperStageResult(
        stage=stage,
        members_path=validated_members,
        point_path=validated_point,
        manifest_path=manifest_path,
        fit_count=0,
        cache_hit_count=units,
        profile=profile,
    )


def _stage_folds(
    stage: Literal["oof", "final"], oof_years: tuple[int, ...] | None
) -> tuple[AnnualFold, ...]:
    if stage == "final":
        if oof_years is not None:
            raise DataContractError("oof_years is not valid for the final stage")
        return (final_fold(),)
    folds = expanding_oof_folds()
    if oof_years is None:
        return folds
    selected = tuple(fold for fold in folds if fold.eval_year in oof_years)
    if tuple(fold.eval_year for fold in selected) != oof_years:
        raise DataContractError("oof_years must be an ordered immutable OOF-year subset")
    return selected


def _run_paper_stage(
    *,
    stage: Literal["oof", "final"],
    matrices: Mapping[str, ForecastMatrix],
    config: PaperBaselineConfig,
    run_dir: Path,
    cache_dir: Path,
    artifact_hashes: Mapping[str, str],
    classical_seed: int,
    factory_builder: _FactoryBuilder,
    models: tuple[str, ...] | None,
    feature_sets: tuple[FeatureSet, ...] | None,
    profile: Literal["paper", "smoke"],
    oof_years: tuple[int, ...] | None,
    smoke_boosting_rounds: int | None,
) -> PaperStageResult:
    if isinstance(classical_seed, bool) or not isinstance(classical_seed, int):
        raise DataContractError("classical_seed must be an integer")
    selected_models, selected_features = _require_stage_selection(
        config=config,
        models=models,
        feature_sets=feature_sets,
        profile=profile,
    )
    if smoke_boosting_rounds is not None and (
        profile != "smoke"
        or isinstance(smoke_boosting_rounds, bool)
        or not isinstance(smoke_boosting_rounds, int)
        or smoke_boosting_rounds <= 0
    ):
        raise DataContractError(
            "smoke_boosting_rounds is a positive-integer non-paper override"
        )
    execution_overrides = (
        {}
        if smoke_boosting_rounds is None
        else {"boosting_rounds": smoke_boosting_rounds}
    )
    try:
        normalized_matrices, schemas = _matrix_contracts(matrices, selected_features)
    except DataContractError as error:
        if _artifact_paths(Path(run_dir), stage)[2].exists():
            raise ArtifactMismatch(f"feature schema differs: {error}") from error
        raise
    hashes = _require_artifact_hashes(artifact_hashes, config)
    folds = _stage_folds(stage, oof_years)
    if profile == "paper" and stage == "oof" and folds != expanding_oof_folds():
        raise DataContractError("paper OOF publication requires all four immutable folds")
    split_ids = tuple(fold.split_id for fold in folds)
    eval_years = tuple(fold.eval_year for fold in folds)
    expected_coverage = _expected_stage_coverage(
        normalized_matrices[selected_features[0]], folds
    )
    expected_preprocessing_populations = _preprocessing_populations(
        normalized_matrices[selected_features[0]],
        folds,
        models=selected_models,
        feature_sets=selected_features,
        validation_days=config.validation_days,
    )
    identity = _manifest_identity(
        profile=profile,
        hashes=hashes,
        models=selected_models,
        feature_sets=selected_features,
        classical_seed=classical_seed,
        schemas=schemas,
        execution_overrides=execution_overrides,
        config=config,
    )
    run = Path(run_dir)
    cache = PredictionCache(Path(cache_dir))
    predictions_dir = run / "predictions"
    with _publication_lock(predictions_dir):
        def preflight_publication() -> (
            PaperStageResult | tuple[dict[str, Any], Path, Path, Path]
        ):
            return _stage_preflight(
                run_dir=run,
                stage=stage,
                identity=identity,
                models=selected_models,
                feature_sets=selected_features,
                matrix=normalized_matrices[selected_features[0]],
                folds=folds,
                classical_seed=classical_seed,
                expected_coverage=expected_coverage,
                expected_preprocessing_populations=expected_preprocessing_populations,
                profile=profile,
                validation_days=config.validation_days,
            )

        recovered = _recover_publication(
            predictions_dir, validate_completed=preflight_publication
        )
        preflight = preflight_publication() if recovered is None else recovered
        if isinstance(preflight, PaperStageResult):
            return preflight
        manifest, _, _, _ = preflight
        point_frames: list[pl.DataFrame] = []
        member_frames: list[pl.DataFrame] = []
        fit_count = 0
        cache_hit_count = 0
        for model in selected_models:
            if smoke_boosting_rounds is not None and model in {"xgboost", "lightgbm"}:
                factory = factory_builder(
                    config,
                    model,
                    profile="smoke",
                    smoke_boosting_rounds=smoke_boosting_rounds,
                )
            else:
                factory = factory_builder(config, model)
            if getattr(factory, "name", None) != model:
                raise DataContractError("paper factory returned the wrong model name")
            seeds = PAPER_SEEDS if model in _NEURAL_MODELS else (classical_seed,)
            for feature_set in selected_features:
                stream_members: list[pl.DataFrame] = []
                stream_hashes = _cache_hashes(
                    hashes, schemas[feature_set], execution_overrides
                )
                for seed in seeds:
                    if stage == "oof":
                        result = generate_oof_stream(
                            normalized_matrices[feature_set],
                            factory,
                            feature_set,
                            cache,
                            stream_hashes,
                            seed,
                            folds=folds,
                            validation_days=config.validation_days,
                        )
                    else:
                        result = generate_cached_final(
                            normalized_matrices[feature_set],
                            factory,
                            feature_set,
                            cache,
                            stream_hashes,
                            seed,
                            validation_days=config.validation_days,
                        )
                    fit_count += result.fit_count
                    cache_hit_count += result.cache_hit_count
                    frame = result.combined_frame if stage == "oof" else result.frame
                    if model in _NEURAL_MODELS:
                        stream_members.append(frame)
                        member_frames.append(frame)
                    else:
                        point_frames.append(frame)
                if model in _NEURAL_MODELS:
                    point_frames.append(
                        _ensemble_frame(
                            stream_members,
                            model=model,
                            feature_set=feature_set,
                            expected_seeds=PAPER_SEEDS,
                        )
                    )
        members = (
            pl.concat(member_frames, how="vertical")
            if member_frames
            else _empty_prediction_frame()
        )
        point = pl.concat(point_frames, how="vertical")
        _validate_published_frames(
            members,
            point,
            models=selected_models,
            feature_sets=selected_features,
            split_ids=split_ids,
            eval_years=eval_years,
            classical_seed=classical_seed,
            expected_coverage=expected_coverage,
        )
        stage_record: dict[str, Any] = {
            "eval_years": list(eval_years),
            "expected_coverage": expected_coverage,
            "preprocessing_populations": expected_preprocessing_populations,
            "split_ids": list(split_ids),
            "streams": _stream_records(
                selected_models, selected_features, classical_seed
            ),
        }
        members_path, point_path, manifest_path = _publish_stage(
            run_dir=run,
            stage=stage,
            members=members,
            point=point,
            manifest=manifest,
            stage_record=stage_record,
        )
        return PaperStageResult(
            stage=stage,
            members_path=members_path,
            point_path=point_path,
            manifest_path=manifest_path,
            fit_count=fit_count,
            cache_hit_count=cache_hit_count,
            profile=profile,
        )


def run_paper_oof_stage(
    *,
    matrices: Mapping[str, ForecastMatrix],
    config: PaperBaselineConfig,
    run_dir: Path,
    cache_dir: Path,
    artifact_hashes: Mapping[str, str],
    classical_seed: int,
    factory_builder: _FactoryBuilder = make_paper_factory,
    models: tuple[str, ...] | None = None,
    feature_sets: tuple[FeatureSet, ...] | None = None,
    profile: Literal["paper", "smoke"] = "paper",
    oof_years: tuple[int, ...] | None = None,
    smoke_boosting_rounds: int | None = None,
) -> PaperStageResult:
    """Run/resume all requested OOF streams and publish one validated group."""

    return _run_paper_stage(
        stage="oof",
        matrices=matrices,
        config=config,
        run_dir=run_dir,
        cache_dir=cache_dir,
        artifact_hashes=artifact_hashes,
        classical_seed=classical_seed,
        factory_builder=factory_builder,
        models=models,
        feature_sets=feature_sets,
        profile=profile,
        oof_years=oof_years,
        smoke_boosting_rounds=smoke_boosting_rounds,
    )


def run_paper_final_stage(
    *,
    matrices: Mapping[str, ForecastMatrix],
    config: PaperBaselineConfig,
    run_dir: Path,
    cache_dir: Path,
    artifact_hashes: Mapping[str, str],
    classical_seed: int,
    factory_builder: _FactoryBuilder = make_paper_factory,
    models: tuple[str, ...] | None = None,
    feature_sets: tuple[FeatureSet, ...] | None = None,
    profile: Literal["paper", "smoke"] = "paper",
    smoke_boosting_rounds: int | None = None,
) -> PaperStageResult:
    """Run/resume all requested final streams and publish the 2024 group."""

    return _run_paper_stage(
        stage="final",
        matrices=matrices,
        config=config,
        run_dir=run_dir,
        cache_dir=cache_dir,
        artifact_hashes=artifact_hashes,
        classical_seed=classical_seed,
        factory_builder=factory_builder,
        models=models,
        feature_sets=feature_sets,
        profile=profile,
        oof_years=None,
        smoke_boosting_rounds=smoke_boosting_rounds,
    )


__all__ = [
    "ClassicalBaseline",
    "ENSEMBLE_SEED",
    "PaperStageResult",
    "make_paper_factory",
    "run_paper_final_stage",
    "run_paper_oof_stage",
]
