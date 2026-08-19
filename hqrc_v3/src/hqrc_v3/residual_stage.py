"""Production preparation of fold-standardized OOF event residuals."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any, Literal

import polars as pl

from hqrc_v3.baselines.config import MODEL_NAMES, PAPER_SEEDS, load_paper_baselines
from hqrc_v3.baselines.paper import (
    ENSEMBLE_SEED,
    PAPER_HASH_KEYS,
    validate_oof_publication_against_source_matrices,
)
from hqrc_v3.baselines.preprocessing import validate_population_contract
from hqrc_v3.config import load_config
from hqrc_v3.contracts import PREDICTION_COLUMNS, DataContractError, validate_prediction_frame
from hqrc_v3.data import (
    FIXED_END,
    FIXED_PUBLIC_HOLIDAY_DATES,
    FIXED_ROWS,
    FIXED_START,
    FIXED_SUBSTITUTE_OR_TEMPORARY_DATES,
    audit_hourly_data,
    load_temporary_holiday_availability,
    read_hourly_data,
)
from hqrc_v3.events import EventOccurrence, load_event_registry, load_holiday_calendar
from hqrc_v3.features import (
    attach_calendar_features,
    build_daily_forecast_matrix,
    feature_columns,
    history_columns,
)
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
from hqrc_v3.publication_fs import _fsync_directory as _portable_fsync_directory
from hqrc_v3.publication_fs import _fsync_file as _portable_fsync_file
from hqrc_v3.publication_fs import exclusive_lock
from hqrc_v3.residuals import compute_fold_scale, standardize_event_residuals
from hqrc_v3.splits import expanding_oof_folds, fold_for_split_id, is_oof_split_id

STANDARDIZED_RESIDUAL_COLUMNS = (
    "origin",
    "target_timestamp",
    "horizon",
    "observed_mw",
    "predicted_mw",
    "residual_mw",
    "standardized_residual",
    "sigma_n_mw",
    "model",
    "feature_set",
    "seed",
    "split_id",
    "occurrence_id",
    "holiday_type",
    "tau_days",
    "hour",
    "restriction",
)

_SCHEMA_VERSION = 2
_SCHEMA_KIND = "hqrc-v3.standardized-residuals.v2"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_BASELINE_KEYS = {
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
    "stages",
}
_BASELINE_OOF_KEYS = {
    "eval_years",
    "expected_coverage",
    "preprocessing_populations",
    "split_ids",
    "streams",
    "artifacts",
}
_MANIFEST_KEYS = {
    "schema_version",
    "schema_kind",
    "profile",
    "inputs",
    "outputs",
    "contexts",
    "manifest_sha256",
}
_INPUT_KEYS = {
    "baseline_oof_semantics",
    "data",
    "event_registry",
    "experiment_config",
    "holiday_calendar",
    "model_config",
    "oof_members",
    "oof_predictions",
    "temporary_holiday_availability",
}
_NEURAL_MODELS = frozenset(("seq2seq_lstm", "transformer"))
_FEATURE_SETS = ("B0", "B1")


@dataclass(frozen=True)
class FoldScaleSummary:
    """One scale computed exclusively from a single fold's non-event residuals."""

    model: str
    feature_set: str
    seed: int
    split_id: str
    sigma_n_mw: float
    non_event_rows: int


@dataclass(frozen=True)
class StandardizedResidualBuild:
    """Pure residual transformation result before artifact publication."""

    frame: pl.DataFrame
    scales: tuple[FoldScaleSummary, ...]


@dataclass(frozen=True)
class StandardizedResidualArtifact:
    """Published canonical residual artifact and its completion manifest."""

    residual_path: Path
    manifest_path: Path
    frame: pl.DataFrame
    reused: bool


def _validate_combined_predictions(frame: pl.DataFrame) -> pl.DataFrame:
    if not isinstance(frame, pl.DataFrame) or tuple(frame.columns) != PREDICTION_COLUMNS:
        raise DataContractError("combined OOF predictions must have the exact prediction schema")
    if frame.is_empty():
        raise DataContractError("combined OOF predictions must not be empty")
    groups: list[pl.DataFrame] = []
    try:
        for group in frame.partition_by(
            ["model", "feature_set", "seed", "split_id"], maintain_order=True
        ):
            validated = validate_prediction_frame(group)
            if not is_oof_split_id(str(validated["split_id"].item(0))):
                raise DataContractError("residual preparation accepts OOF predictions only")
            groups.append(validated)
    except (TypeError, ValueError) as error:
        if isinstance(error, DataContractError):
            raise
        raise DataContractError(f"combined OOF predictions are invalid: {error}") from error
    return pl.concat(groups, how="vertical")


def _event_lookup(events: Sequence[EventOccurrence]) -> pl.DataFrame:
    if not events:
        raise DataContractError("event registry must not be empty")
    ids: set[str] = set()
    occupied: set[datetime] = set()
    rows: list[dict[str, object]] = []
    for event in sorted(events, key=lambda item: item.window_start):
        if not isinstance(event, EventOccurrence):
            raise TypeError("events must contain EventOccurrence values")
        if event.occurrence_id in ids:
            raise DataContractError("event registry contains duplicate occurrence ids")
        ids.add(event.occurrence_id)
        central = datetime.combine(event.central_date, time.min)
        timestamp = datetime.combine(event.window_start, time.min)
        end = datetime.combine(event.window_end + timedelta(days=1), time.min)
        while timestamp < end:
            if timestamp in occupied:
                raise DataContractError("event correction windows overlap")
            occupied.add(timestamp)
            rows.append(
                {
                    "target_timestamp": timestamp,
                    "split_id": f"oof-{event.central_date.year}",
                    "occurrence_id": event.occurrence_id,
                    "holiday_type": event.holiday_type,
                    "tau_days": (timestamp - central).total_seconds() / (24.0 * 3600.0),
                    "hour": timestamp.hour,
                    "restriction": event.restriction,
                }
            )
            timestamp += timedelta(hours=1)
    return pl.DataFrame(rows).with_columns(
        pl.col("target_timestamp").cast(pl.Datetime("ns")),
        pl.col("split_id").cast(pl.String),
        pl.col("occurrence_id").cast(pl.String),
        pl.col("holiday_type").cast(pl.String),
        pl.col("tau_days").cast(pl.Float64),
        pl.col("hour").cast(pl.Int64),
        pl.col("restriction").cast(pl.Int64),
    )


def _expected_events_by_split(
    lookup: pl.DataFrame, split_ids: set[str]
) -> dict[str, dict[str, int]]:
    expected: dict[str, dict[str, int]] = {}
    for split_id, occurrence_id, count in (
        lookup.filter(pl.col("split_id").is_in(sorted(split_ids)))
        .group_by("split_id", "occurrence_id")
        .len()
        .iter_rows()
    ):
        expected.setdefault(str(split_id), {})[str(occurrence_id)] = int(count)
    return expected


def build_standardized_residuals(
    predictions: pl.DataFrame,
    events: Sequence[EventOccurrence],
) -> StandardizedResidualBuild:
    """Standardize only event rows using each point stream's own fold-local RMS."""

    validated = _validate_combined_predictions(predictions)
    lookup = _event_lookup(events)
    split_ids = set(str(value) for value in validated["split_id"].unique().to_list())
    expected_events = _expected_events_by_split(lookup, split_ids)
    if set(expected_events) != split_ids:
        raise DataContractError("every OOF split must contain registered correction events")

    enriched = (
        validated.join(lookup, on=["target_timestamp", "split_id"], how="left")
        .with_columns(
            (pl.col("observed_mw") - pl.col("predicted_mw")).alias("residual_mw"),
            pl.col("occurrence_id").is_not_null().alias("is_event"),
        )
        .with_columns(pl.col("residual_mw").cast(pl.Float64))
    )
    output: list[pl.DataFrame] = []
    scales: list[FoldScaleSummary] = []
    group_keys = ["model", "feature_set", "seed", "split_id"]
    for group in enriched.partition_by(group_keys, maintain_order=True):
        model = str(group["model"].item(0))
        feature_set = str(group["feature_set"].item(0))
        seed = int(group["seed"].item(0))
        split_id = str(group["split_id"].item(0))
        scale = compute_fold_scale(group)
        event_rows = group.filter(pl.col("is_event"))
        actual_counts = {
            str(occurrence_id): int(count)
            for occurrence_id, count in event_rows.group_by("occurrence_id").len().iter_rows()
        }
        if actual_counts != expected_events[split_id]:
            raise DataContractError(
                f"{model}/{feature_set}/seed-{seed}/{split_id} event windows are incomplete"
            )
        standardized = standardize_event_residuals(event_rows, scale).with_columns(
            pl.lit(float(scale), dtype=pl.Float64).alias("sigma_n_mw")
        )
        output.append(standardized.select(STANDARDIZED_RESIDUAL_COLUMNS))
        scales.append(
            FoldScaleSummary(
                model=model,
                feature_set=feature_set,
                seed=seed,
                split_id=split_id,
                sigma_n_mw=float(scale),
                non_event_rows=group.filter(~pl.col("is_event")).height,
            )
        )
    if not output:
        raise DataContractError("OOF predictions contain no standardizable event rows")
    frame = pl.concat(output, how="vertical").sort(
        "model", "feature_set", "seed", "target_timestamp"
    )
    duplicate_hours = frame.select(
        pl.struct("model", "feature_set", "seed", "target_timestamp")
        .is_duplicated()
        .any()
    ).item()
    if duplicate_hours:
        raise DataContractError("standardized residual artifact contains duplicate context hours")
    ordered_scales = tuple(
        sorted(
            scales,
            key=lambda item: (item.model, item.feature_set, item.seed, item.split_id),
        )
    )
    _validate_standardized_frame(frame)
    return StandardizedResidualBuild(frame=frame, scales=ordered_scales)


def _validate_standardized_frame(frame: pl.DataFrame) -> None:
    if not isinstance(frame, pl.DataFrame) or tuple(frame.columns) != STANDARDIZED_RESIDUAL_COLUMNS:
        raise ArtifactMismatch("standardized residual output schema differs")
    if frame.is_empty() or any(frame[column].null_count() for column in frame.columns):
        raise ArtifactMismatch("standardized residual output must be nonempty and complete")
    float_columns = (
        "observed_mw",
        "predicted_mw",
        "residual_mw",
        "standardized_residual",
        "sigma_n_mw",
        "tau_days",
    )
    if any(frame.schema[column] not in {pl.Float32, pl.Float64} for column in float_columns):
        raise ArtifactMismatch("standardized residual numeric schema differs")
    all_finite = pl.all_horizontal(
        *(pl.col(name).is_finite() for name in float_columns)
    ).all()
    if not frame.select(all_finite).item():
        raise ArtifactMismatch("standardized residual output contains nonfinite values")
    if not frame.select((pl.col("sigma_n_mw") > 0).all()).item():
        raise ArtifactMismatch("standardized residual scales must be positive")
    integer_columns = ("horizon", "seed", "hour", "restriction")
    integer_types = {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }
    if any(frame.schema[column] not in integer_types for column in integer_columns):
        raise ArtifactMismatch("standardized residual integer schema differs")
    string_types = {pl.String, pl.Categorical, pl.Enum}
    string_columns = ("model", "feature_set", "split_id", "occurrence_id", "holiday_type")
    if any(frame.schema[column] not in string_types for column in string_columns):
        raise ArtifactMismatch("standardized residual identifier schema differs")
    if frame.schema["origin"].base_type() != pl.Datetime or frame.schema[
        "target_timestamp"
    ].base_type() != pl.Datetime:
        raise ArtifactMismatch("standardized residual timestamps must be datetimes")
    if frame.schema["origin"].time_zone is not None or frame.schema[
        "target_timestamp"
    ].time_zone is not None:
        raise ArtifactMismatch("standardized residual timestamps must be timezone-naive")
    if not frame.filter(
        ~pl.col("model").is_in(MODEL_NAMES)
        | ~pl.col("feature_set").is_in(_FEATURE_SETS)
        | ~pl.col("holiday_type").is_in(("seollal", "chuseok"))
        | ~pl.col("restriction").is_in((0, 1))
        | (pl.col("hour") < 0)
        | (pl.col("hour") > 23)
    ).is_empty():
        raise ArtifactMismatch("standardized residual categorical values are invalid")
    if any(not is_oof_split_id(str(split_id)) for split_id in frame["split_id"].unique()):
        raise ArtifactMismatch("standardized residual split ids must be OOF folds")
    residual_error = (
        pl.col("residual_mw") - (pl.col("observed_mw") - pl.col("predicted_mw"))
    ).abs()
    standardized_error = (
        pl.col("standardized_residual") * pl.col("sigma_n_mw")
        - pl.col("residual_mw")
    ).abs()
    if not frame.select(
        (residual_error <= 1e-9).all() & (standardized_error <= 1e-8).all()
    ).item():
        raise ArtifactMismatch("standardized residual arithmetic differs")
    tau_hours = pl.col("tau_days") * 24.0
    if not frame.select(
        ((tau_hours - tau_hours.round(0)).abs() <= 1e-8).all()
        & (tau_hours.round(0).cast(pl.Int64).mod(24) == pl.col("hour")).all()
        & (pl.col("hour") == pl.col("target_timestamp").dt.hour()).all()
    ).item():
        raise ArtifactMismatch("standardized residual tau/hour alignment differs")
    ordered = frame.sort("model", "feature_set", "seed", "target_timestamp")
    if not frame.equals(ordered):
        raise ArtifactMismatch("standardized residual rows are not canonically ordered")
    key = ("model", "feature_set", "seed", "target_timestamp")
    if frame.select(pl.struct(*key).is_duplicated().any()).item():
        raise ArtifactMismatch("standardized residual context hours are duplicated")
    for event in frame.partition_by(
        ["model", "feature_set", "seed", "occurrence_id"], maintain_order=True
    ):
        if any(
            event[column].n_unique() != 1
            for column in ("holiday_type", "restriction", "split_id")
        ):
            raise ArtifactMismatch("event residual metadata varies within an occurrence")
        if not event.select(
            pl.col("target_timestamp").diff().drop_nulls().eq(pl.duration(hours=1)).all()
            & pl.col("tau_days")
            .diff()
            .drop_nulls()
            .sub(1.0 / 24.0)
            .abs()
            .le(1e-8)
            .all()
        ).item():
            raise ArtifactMismatch("event residual hours are not contiguous")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ArtifactMismatch("residual manifest must be finite canonical JSON") from error


def _read_canonical_json(path: Path, description: str) -> dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise ArtifactMismatch(f"{description} is missing or unsafe")
    try:
        payload = path.read_bytes()
        value = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactMismatch(f"{description} is unreadable") from error
    if not isinstance(value, dict) or payload != _canonical_json(value):
        raise ArtifactMismatch(f"{description} is not canonical JSON")
    return value


def _require_sha(value: object, description: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ArtifactMismatch(f"{description} SHA-256 is invalid")
    return value


def _source_identity(path: Path, description: str) -> dict[str, str]:
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ArtifactMismatch(f"{description} source is missing or unsafe")
    return {"path": source.resolve().as_posix(), "sha256": file_sha256(source)}


def _baseline_oof_projection(manifest: Mapping[str, Any]) -> dict[str, object]:
    """Return only immutable baseline identity plus the completed OOF stage."""

    if (
        set(manifest) != _BASELINE_KEYS
        or type(manifest.get("schema_version")) is not int
        or manifest.get("schema_version") != 2
    ):
        raise ArtifactMismatch("baseline manifest schema differs")
    stages = manifest.get("stages")
    if (
        not isinstance(stages, dict)
        or "oof" not in stages
        or not set(stages).issubset({"oof", "final"})
        or not isinstance(stages["oof"], dict)
        or set(stages["oof"]) != _BASELINE_OOF_KEYS
    ):
        raise ArtifactMismatch("baseline manifest has no OOF stage")
    return {
        key: manifest[key]
        for key in (
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
        )
    } | {"oof": stages["oof"]}


def _baseline_oof_digest(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical_json(_baseline_oof_projection(manifest))).hexdigest()


def _validate_preprocessing_populations(
    value: object,
    *,
    models: tuple[str, ...],
    feature_sets: tuple[str, ...],
    split_ids: tuple[str, ...],
) -> None:
    if not isinstance(value, list):
        raise ArtifactMismatch("baseline preprocessing populations are invalid")
    expected = {
        (model, feature_set, split_id)
        for model in models
        for feature_set in feature_sets
        for split_id in split_ids
    }
    actual: set[tuple[str, str, str]] = set()
    for record in value:
        if not isinstance(record, dict) or set(record) != {
            "model",
            "feature_set",
            "split_id",
            "scalers",
        }:
            raise ArtifactMismatch("baseline preprocessing population schema differs")
        if (
            not isinstance(record["model"], str)
            or not isinstance(record["feature_set"], str)
            or not isinstance(record["split_id"], str)
        ):
            raise ArtifactMismatch("baseline preprocessing population identity is invalid")
        key = (record["model"], record["feature_set"], record["split_id"])
        if key in actual:
            raise ArtifactMismatch("baseline preprocessing populations contain duplicates")
        actual.add(key)
        try:
            scalers = validate_population_contract(record["scalers"])
        except (DataContractError, TypeError, ValueError) as error:
            raise ArtifactMismatch(
                f"baseline preprocessing population is invalid: {error}"
            ) from error
        expected_scalers = (
            {"target", "weather", "calendar"}
            if record["model"] in _NEURAL_MODELS
            else {"x", "target"}
        )
        if set(scalers) != expected_scalers:
            raise ArtifactMismatch("baseline preprocessing scaler identities differ")
    if actual != expected:
        raise ArtifactMismatch("baseline preprocessing population coverage differs")


def _read_prediction_artifact(path: Path, description: str) -> pl.DataFrame:
    if not path.is_file() or path.is_symlink():
        raise ArtifactMismatch(f"{description} artifact is missing or unsafe")
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ArtifactMismatch(f"{description} artifact is unreadable") from error


def _load_baseline_source(
    *,
    run_dir: Path,
    data_path: Path,
    config_path: Path,
    model_config_path: Path,
    event_registry_path: Path,
    holiday_calendar_path: Path,
    temporary_holiday_availability_path: Path,
    profile: Literal["paper", "smoke"],
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, dict[str, str]]]:
    manifest_path = run_dir / "predictions/baseline_manifest.json"
    point_path = run_dir / "predictions/oof.parquet"
    members_path = run_dir / "predictions/oof_members.parquet"
    manifest = _read_canonical_json(manifest_path, "baseline manifest")
    if (
        set(manifest) != _BASELINE_KEYS
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 2
    ):
        raise ArtifactMismatch("baseline manifest schema differs")
    if manifest.get("profile") != profile:
        raise ArtifactMismatch("baseline profile differs from residual preparation profile")
    input_hashes = manifest.get("input_hashes")
    if not isinstance(input_hashes, dict) or set(input_hashes) != set(PAPER_HASH_KEYS):
        raise ArtifactMismatch("baseline input hash schema differs")
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in input_hashes.values()
    ):
        raise ArtifactMismatch("baseline input hashes are invalid")

    source_paths = {
        "data_sha256": (Path(data_path), "data"),
        "experiment_sha256": (Path(config_path), "experiment config"),
        "model_config_sha256": (Path(model_config_path), "model config"),
        "event_registry_sha256": (Path(event_registry_path), "event registry"),
        "holiday_calendar_sha256": (Path(holiday_calendar_path), "holiday calendar"),
        "temporary_holiday_availability_sha256": (
            Path(temporary_holiday_availability_path),
            "temporary holiday availability",
        ),
    }
    source_inputs: dict[str, dict[str, str]] = {}
    source_names = {
        "data_sha256": "data",
        "experiment_sha256": "experiment_config",
        "model_config_sha256": "model_config",
        "event_registry_sha256": "event_registry",
        "holiday_calendar_sha256": "holiday_calendar",
        "temporary_holiday_availability_sha256": "temporary_holiday_availability",
    }
    for hash_name, (source_path, description) in source_paths.items():
        identity = _source_identity(source_path, description)
        if input_hashes.get(hash_name) != identity["sha256"]:
            raise ArtifactMismatch(f"{description} differs from the baseline manifest")
        source_inputs[source_names[hash_name]] = identity

    load_config(Path(config_path))
    baseline_config = load_paper_baselines(Path(model_config_path))
    load_event_registry(Path(event_registry_path))
    calendar = load_holiday_calendar(Path(holiday_calendar_path))
    temporary_availability = load_temporary_holiday_availability(
        Path(temporary_holiday_availability_path)
    )
    models, feature_sets = manifest.get("models"), manifest.get("feature_sets")
    if (
        not isinstance(models, list)
        or not models
        or len(models) != len(set(models))
        or any(model not in MODEL_NAMES for model in models)
        or not isinstance(feature_sets, list)
        or not feature_sets
        or len(feature_sets) != len(set(feature_sets))
        or any(feature_set not in _FEATURE_SETS for feature_set in feature_sets)
    ):
        raise ArtifactMismatch("baseline model/feature-set identity is invalid")
    if models != [model for model in MODEL_NAMES if model in models] or feature_sets != [
        feature_set for feature_set in _FEATURE_SETS if feature_set in feature_sets
    ]:
        raise ArtifactMismatch("baseline model/feature-set order is not canonical")
    if profile == "paper" and (
        tuple(models) != MODEL_NAMES or tuple(feature_sets) != _FEATURE_SETS
    ):
        raise ArtifactMismatch("paper residual preparation requires all five models and B0/B1")

    if manifest.get("neural_seeds") != list(PAPER_SEEDS):
        raise ArtifactMismatch("baseline neural seed contract differs")
    if manifest.get("ensemble_seed") != ENSEMBLE_SEED:
        raise ArtifactMismatch("baseline ensemble seed differs")
    expected_schemas = {
        feature_set: {
            "history": list(history_columns(feature_set)),
            "future": list(feature_columns(feature_set)),
        }
        for feature_set in feature_sets
    }
    if manifest.get("feature_schemas") != expected_schemas:
        raise ArtifactMismatch("baseline feature schemas differ")
    if manifest.get("preprocessing") != baseline_config.preprocessing.to_manifest():
        raise ArtifactMismatch("baseline preprocessing contract differs")
    execution_overrides = manifest.get("execution_overrides")
    if profile == "paper":
        if execution_overrides != {}:
            raise ArtifactMismatch("paper baseline execution overrides differ")
    elif execution_overrides != {} and (
        not isinstance(execution_overrides, dict)
        or set(execution_overrides) != {"boosting_rounds"}
        or isinstance(execution_overrides["boosting_rounds"], bool)
        or not isinstance(execution_overrides["boosting_rounds"], int)
        or execution_overrides["boosting_rounds"] <= 0
    ):
        raise ArtifactMismatch("smoke baseline execution overrides are invalid")
    stages = manifest.get("stages")
    if not isinstance(stages, dict) or "oof" not in stages:
        raise ArtifactMismatch("baseline manifest has no OOF stage")
    stage = stages["oof"]
    if not isinstance(stage, dict) or set(stage) != _BASELINE_OOF_KEYS:
        raise ArtifactMismatch("baseline OOF stage schema differs")
    split_ids = stage.get("split_ids")
    eval_years = stage.get("eval_years")
    if (
        not isinstance(split_ids, list)
        or not split_ids
        or any(
            not isinstance(split_id, str) or not is_oof_split_id(split_id)
            for split_id in split_ids
        )
        or split_ids != sorted(set(split_ids))
        or not isinstance(eval_years, list)
        or eval_years != [fold_for_split_id(split_id).eval_year for split_id in split_ids]
    ):
        raise ArtifactMismatch("baseline OOF split identities are invalid")
    paper_splits = [fold.split_id for fold in expanding_oof_folds()]
    if profile == "paper" and split_ids != paper_splits:
        raise ArtifactMismatch("paper residual preparation requires all four OOF folds")
    models_tuple = tuple(models)
    feature_sets_tuple = tuple(feature_sets)
    split_ids_tuple = tuple(split_ids)
    expected_streams = [
        {
            "model": model,
            "feature_set": feature_set,
            "seeds": list(
                PAPER_SEEDS
                if model in _NEURAL_MODELS
                else (manifest["classical_seed"],)
            ),
        }
        for model in models_tuple
        for feature_set in feature_sets_tuple
    ]
    if stage.get("streams") != expected_streams:
        raise ArtifactMismatch("baseline OOF stream contract differs")
    _validate_preprocessing_populations(
        stage.get("preprocessing_populations"),
        models=models_tuple,
        feature_sets=feature_sets_tuple,
        split_ids=split_ids_tuple,
    )
    artifacts = stage.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"members", "point"}:
        raise ArtifactMismatch("baseline OOF artifact schema differs")
    expected_artifacts = {
        "members": (members_path, "predictions/oof_members.parquet"),
        "point": (point_path, "predictions/oof.parquet"),
    }
    artifact_inputs: dict[str, dict[str, str]] = {}
    loaded: dict[str, pl.DataFrame] = {}
    for name, (path, relative) in expected_artifacts.items():
        entry = artifacts[name]
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or entry.get("path") != relative
        ):
            raise ArtifactMismatch(f"baseline OOF {name} artifact identity differs")
        recorded_sha = _require_sha(entry.get("sha256"), f"OOF {name}")
        loaded[name] = _read_prediction_artifact(path, f"OOF {name}")
        actual_sha = file_sha256(path)
        if actual_sha != recorded_sha:
            label = "point prediction" if name == "point" else "member prediction"
            raise ArtifactMismatch(f"OOF {label} SHA-256 differs from the baseline manifest")
        artifact_inputs[f"oof_{'predictions' if name == 'point' else 'members'}"] = {
            "path": relative,
            "sha256": actual_sha,
        }
    try:
        predictions = _validate_combined_predictions(loaded["point"])
    except (DataContractError, TypeError, ValueError) as error:
        raise ArtifactMismatch(f"OOF prediction artifact is invalid: {error}") from error
    classical_seed = manifest.get("classical_seed")
    if (
        isinstance(classical_seed, bool)
        or not isinstance(classical_seed, int)
    ):
        raise ArtifactMismatch("baseline point-stream seeds are invalid")
    expected = {
        (
            model,
            feature_set,
            ENSEMBLE_SEED if model in _NEURAL_MODELS else classical_seed,
            split_id,
        )
        for model in models
        for feature_set in feature_sets
        for split_id in split_ids
    }
    actual = set(
        predictions.select("model", "feature_set", "seed", "split_id")
        .unique()
        .iter_rows()
    )
    if actual != expected:
        raise ArtifactMismatch("OOF point prediction contexts differ from the baseline manifest")
    _validate_exact_coverage(predictions, expected)
    try:
        paper_bounds = (
            {
                "expected_start": FIXED_START,
                "expected_end": FIXED_END,
                "expected_rows": FIXED_ROWS,
                "expected_public_holiday_dates": FIXED_PUBLIC_HOLIDAY_DATES,
                "expected_substitute_or_temporary_dates": (
                    FIXED_SUBSTITUTE_OR_TEMPORARY_DATES
                ),
                "temporary_holiday_availability": temporary_availability,
            }
            if profile == "paper"
            else {
                "expected_start": None,
                "expected_end": None,
                "expected_rows": None,
                "temporary_holiday_availability": temporary_availability,
            }
        )
        audited = audit_hourly_data(read_hourly_data(Path(data_path)), **paper_bounds)
        featured = attach_calendar_features(audited, calendar)
        matrices = {
            feature_set: build_daily_forecast_matrix(
                featured, feature_set=feature_set
            )
            for feature_set in feature_sets_tuple
        }
        validate_oof_publication_against_source_matrices(
            loaded["members"],
            predictions,
            stage_record=stage,
            matrices=matrices,
            config=baseline_config,
            models=models_tuple,
            feature_sets=feature_sets_tuple,
            split_ids=split_ids_tuple,
            eval_years=tuple(eval_years),
            classical_seed=classical_seed,
        )
    except (ArtifactMismatch, DataContractError, TypeError, ValueError) as error:
        if isinstance(error, ArtifactMismatch):
            raise
        raise ArtifactMismatch(f"baseline OOF publication is invalid: {error}") from error
    inputs = {
        **source_inputs,
        **artifact_inputs,
        "baseline_oof_semantics": {
            "path": "predictions/baseline_manifest.json",
            "sha256": _baseline_oof_digest(manifest),
        },
    }
    return predictions, manifest, inputs


def _validate_exact_coverage(
    predictions: pl.DataFrame,
    expected_contexts: set[tuple[object, object, object, object]],
) -> None:
    reference_observed: pl.DataFrame | None = None
    for model, feature_set, seed, split_id in sorted(expected_contexts):
        group = predictions.filter(
            (pl.col("model") == model)
            & (pl.col("feature_set") == feature_set)
            & (pl.col("seed") == seed)
            & (pl.col("split_id") == split_id)
        ).sort("target_timestamp")
        fold = fold_for_split_id(str(split_id))
        start = datetime.combine(fold.eval_start, time.min)
        end = datetime.combine(fold.eval_end, time(hour=23))
        expected_hours = int((end - start).total_seconds() // 3600) + 1
        if (
            group.height != expected_hours
            or group["target_timestamp"].item(0) != start
            or group["target_timestamp"].item(-1) != end
            or not group.select(
                pl.col("target_timestamp").diff().drop_nulls().eq(pl.duration(hours=1)).all()
            ).item()
        ):
            raise ArtifactMismatch("OOF point prediction hourly coverage is incomplete")
        observed = group.select("split_id", "target_timestamp", "observed_mw")
        if reference_observed is None:
            reference_observed = observed
        else:
            reference_for_split = reference_observed.filter(pl.col("split_id") == split_id)
            if reference_for_split.is_empty():
                reference_observed = pl.concat([reference_observed, observed])
            elif not observed.equals(reference_for_split):
                raise ArtifactMismatch("OOF point streams disagree on observed load")


def _context_records(build: StandardizedResidualBuild) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for context in build.frame.partition_by(
        ["model", "feature_set", "seed"], maintain_order=True
    ):
        model = str(context["model"].item(0))
        feature_set = str(context["feature_set"].item(0))
        seed = int(context["seed"].item(0))
        selected = [
            scale
            for scale in build.scales
            if (scale.model, scale.feature_set, scale.seed) == (model, feature_set, seed)
        ]
        selected.sort(key=lambda item: item.split_id)
        latest = max(selected, key=lambda item: fold_for_split_id(item.split_id).eval_year)
        records.append(
            {
                "model": model,
                "feature_set": feature_set,
                "seed": seed,
                "rows": context.height,
                "split_ids": [item.split_id for item in selected],
                "occurrence_ids": sorted(context["occurrence_id"].unique().to_list()),
                "fold_scales": [
                    {
                        "split_id": item.split_id,
                        "sigma_n_mw": item.sigma_n_mw,
                        "non_event_rows": item.non_event_rows,
                    }
                    for item in selected
                ],
                "latest_complete_oof_scale": {
                    "split_id": latest.split_id,
                    "sigma_n_mw": latest.sigma_n_mw,
                },
            }
        )
    return sorted(records, key=lambda item: (str(item["model"]), str(item["feature_set"])))


def _manifest_payload(
    *,
    profile: Literal["paper", "smoke"],
    inputs: Mapping[str, Mapping[str, str]],
    build: StandardizedResidualBuild,
    residual_sha256: str,
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "schema_kind": _SCHEMA_KIND,
        "profile": profile,
        "inputs": {key: dict(inputs[key]) for key in sorted(inputs)},
        "outputs": {
            "standardized_residuals": {
                "path": "inputs/standardized_residuals.parquet",
                "sha256": residual_sha256,
                "rows": build.frame.height,
            }
        },
        "contexts": _context_records(build),
    }
    return {**unsigned, "manifest_sha256": hashlib.sha256(_canonical_json(unsigned)).hexdigest()}


def _safe_bound_path(run_dir: Path, entry: object, *, expected: str | None = None) -> Path:
    schemas = ({"path", "sha256"}, {"path", "sha256", "rows"})
    if not isinstance(entry, dict) or set(entry) not in schemas:
        raise ArtifactMismatch("residual manifest artifact entry is invalid")
    relative = entry.get("path")
    _require_sha(entry.get("sha256"), "residual manifest artifact")
    if (
        not isinstance(relative, str)
        or Path(relative).is_absolute()
        or ".." in Path(relative).parts
    ):
        raise ArtifactMismatch("residual manifest artifact path is unsafe")
    if expected is not None and relative != expected:
        raise ArtifactMismatch("residual manifest artifact path differs")
    path = run_dir / relative
    allowed_parents = {
        run_dir.resolve(),
        (run_dir / "inputs").resolve(),
        (run_dir / "predictions").resolve(),
    }
    if path.resolve().parent not in allowed_parents:
        raise ArtifactMismatch("residual manifest artifact escapes the run directory")
    return path


def _validate_context_manifest(value: object, frame: pl.DataFrame) -> None:
    if not isinstance(value, list) or not value:
        raise ArtifactMismatch("residual manifest contexts must be a nonempty list")
    actual_keys = set(
        frame.select("model", "feature_set", "seed").unique().iter_rows()
    )
    recorded_keys: set[tuple[str, str, int]] = set()
    for record in value:
        required = {
            "model",
            "feature_set",
            "seed",
            "rows",
            "split_ids",
            "occurrence_ids",
            "fold_scales",
            "latest_complete_oof_scale",
        }
        if not isinstance(record, dict) or set(record) != required:
            raise ArtifactMismatch("residual context manifest schema differs")
        model, feature_set, seed = record["model"], record["feature_set"], record["seed"]
        if (
            not isinstance(model, str)
            or model not in MODEL_NAMES
            or feature_set not in _FEATURE_SETS
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise ArtifactMismatch("residual context identity is invalid")
        key = (model, feature_set, seed)
        if key in recorded_keys:
            raise ArtifactMismatch("residual context manifest contains duplicates")
        recorded_keys.add(key)
        context = frame.filter(
            (pl.col("model") == model)
            & (pl.col("feature_set") == feature_set)
            & (pl.col("seed") == seed)
        )
        if type(record["rows"]) is not int or record["rows"] != context.height:
            raise ArtifactMismatch("residual context row count differs")
        split_ids = sorted(context["split_id"].unique().to_list())
        occurrence_ids = sorted(context["occurrence_id"].unique().to_list())
        if record["split_ids"] != split_ids or record["occurrence_ids"] != occurrence_ids:
            raise ArtifactMismatch("residual context coverage differs")
        scales = record["fold_scales"]
        recorded_splits = [
            item.get("split_id") for item in scales if isinstance(item, dict)
        ] if isinstance(scales, list) else []
        if not isinstance(scales, list) or recorded_splits != split_ids:
            raise ArtifactMismatch("residual context fold scales differ")
        for scale in scales:
            if (
                not isinstance(scale, dict)
                or set(scale) != {"split_id", "sigma_n_mw", "non_event_rows"}
                or not isinstance(scale["sigma_n_mw"], (int, float))
                or isinstance(scale["sigma_n_mw"], bool)
                or not math.isfinite(scale["sigma_n_mw"])
                or scale["sigma_n_mw"] <= 0
                or type(scale["non_event_rows"]) is not int
                or scale["non_event_rows"] <= 0
            ):
                raise ArtifactMismatch("residual context fold scale is invalid")
            artifact_scale = context.filter(pl.col("split_id") == scale["split_id"])[
                "sigma_n_mw"
            ].unique()
            if artifact_scale.len() != 1 or not math.isclose(
                float(artifact_scale.item()),
                float(scale["sigma_n_mw"]),
                rel_tol=0.0,
                abs_tol=1e-12,
            ):
                raise ArtifactMismatch("residual context scale differs from the artifact")
        latest = record["latest_complete_oof_scale"]
        expected_latest = max(
            scales, key=lambda item: fold_for_split_id(item["split_id"]).eval_year
        )
        expected_latest_identity = {
            "split_id": expected_latest["split_id"],
            "sigma_n_mw": expected_latest["sigma_n_mw"],
        }
        if latest != expected_latest_identity:
            raise ArtifactMismatch("latest complete OOF scale differs")
    if recorded_keys != actual_keys:
        raise ArtifactMismatch("residual manifest context coverage differs from the artifact")


def load_standardized_residual_manifest(
    manifest_path: Path,
    *,
    run_dir: Path,
    config_sha256: str,
    event_sha256: str,
) -> dict[str, Any]:
    """Strictly reload a completed residual publication and all hash bindings."""

    manifest = _read_canonical_json(Path(manifest_path), "standardized residual manifest")
    if (
        set(manifest) != _MANIFEST_KEYS
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != _SCHEMA_VERSION
        or manifest.get("schema_kind") != _SCHEMA_KIND
        or manifest.get("profile") not in {"paper", "smoke"}
    ):
        raise ArtifactMismatch("standardized residual manifest schema differs")
    unsigned = {key: manifest[key] for key in sorted(_MANIFEST_KEYS - {"manifest_sha256"})}
    expected_digest = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    if manifest.get("manifest_sha256") != expected_digest:
        raise ArtifactMismatch("standardized residual manifest digest differs")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict) or set(inputs) != _INPUT_KEYS:
        raise ArtifactMismatch("standardized residual manifest inputs differ")
    for name in _INPUT_KEYS:
        entry = inputs[name]
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise ArtifactMismatch("standardized residual input identity is invalid")
        if not isinstance(entry.get("path"), str) or not entry["path"]:
            raise ArtifactMismatch("standardized residual input path is invalid")
        _require_sha(entry.get("sha256"), f"{name} input")
    if inputs["experiment_config"]["sha256"] != config_sha256:
        raise ArtifactMismatch("experiment config differs from the residual manifest")
    if inputs["event_registry"]["sha256"] != event_sha256:
        raise ArtifactMismatch("event registry differs from the residual manifest")
    for name, expected_path in (
        ("oof_predictions", "predictions/oof.parquet"),
        ("oof_members", "predictions/oof_members.parquet"),
    ):
        path = _safe_bound_path(Path(run_dir), inputs[name], expected=expected_path)
        if not path.is_file() or path.is_symlink() or file_sha256(path) != inputs[name]["sha256"]:
            raise ArtifactMismatch(f"{name} input differs from the residual manifest")
    baseline_manifest_path = _safe_bound_path(
        Path(run_dir),
        inputs["baseline_oof_semantics"],
        expected="predictions/baseline_manifest.json",
    )
    baseline_manifest = _read_canonical_json(baseline_manifest_path, "baseline manifest")
    if _baseline_oof_digest(baseline_manifest) != inputs["baseline_oof_semantics"]["sha256"]:
        raise ArtifactMismatch("baseline OOF semantics differ from the residual manifest")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != {"standardized_residuals"}:
        raise ArtifactMismatch("standardized residual output manifest differs")
    output = outputs["standardized_residuals"]
    residual_path = _safe_bound_path(
        Path(run_dir), output, expected="inputs/standardized_residuals.parquet"
    )
    if (
        not residual_path.is_file()
        or residual_path.is_symlink()
        or file_sha256(residual_path) != output["sha256"]
    ):
        raise ArtifactMismatch("standardized residual output SHA-256 differs")
    try:
        frame = pl.read_parquet(residual_path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ArtifactMismatch("standardized residual output is unreadable") from error
    if type(output.get("rows")) is not int or output["rows"] != frame.height:
        raise ArtifactMismatch("standardized residual output schema or row count differs")
    _validate_standardized_frame(frame)
    _validate_context_manifest(manifest.get("contexts"), frame)
    return manifest


@contextmanager
def _publication_lock(directory: Path) -> Iterator[None]:
    directory.mkdir(parents=True, exist_ok=True)
    lock_path = directory / ".standardized-residuals.lock"
    if lock_path.is_symlink():
        raise ArtifactMismatch("standardized residual lock path is unsafe")
    with exclusive_lock(lock_path):
        yield


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".standardized-residuals.", suffix=suffix, dir=directory
    )
    os.close(descriptor)
    return Path(name)


def _fsync_file(path: Path) -> None:
    _portable_fsync_file(path)


def _fsync_directory(path: Path) -> None:
    _portable_fsync_directory(path)


def prepare_standardized_residual_artifact(
    *,
    run_dir: Path,
    data_path: Path,
    config_path: Path,
    model_config_path: Path,
    event_registry_path: Path,
    holiday_calendar_path: Path,
    temporary_holiday_availability_path: Path,
    profile: Literal["paper", "smoke"] = "paper",
) -> StandardizedResidualArtifact:
    """Build or strictly reuse the canonical all-context OOF residual artifact."""

    run = Path(run_dir)
    config = Path(config_path)
    event_path = Path(event_registry_path)
    events = load_event_registry(event_path)
    inputs_dir = run / "inputs"
    residual_path = inputs_dir / "standardized_residuals.parquet"
    manifest_path = inputs_dir / "standardized_residuals_manifest.json"
    with _publication_lock(inputs_dir):
        predictions, _, inputs = _load_baseline_source(
            run_dir=run,
            data_path=Path(data_path),
            config_path=config,
            model_config_path=Path(model_config_path),
            event_registry_path=event_path,
            holiday_calendar_path=Path(holiday_calendar_path),
            temporary_holiday_availability_path=Path(
                temporary_holiday_availability_path
            ),
            profile=profile,
        )
        config_sha = inputs["experiment_config"]["sha256"]
        event_sha = inputs["event_registry"]["sha256"]
        if manifest_path.exists() or manifest_path.is_symlink():
            manifest = load_standardized_residual_manifest(
                manifest_path,
                run_dir=run,
                config_sha256=config_sha,
                event_sha256=event_sha,
            )
            if manifest["profile"] != profile:
                raise ArtifactMismatch("residual artifact profile differs")
            return StandardizedResidualArtifact(
                residual_path=residual_path,
                manifest_path=manifest_path,
                frame=pl.read_parquet(residual_path),
                reused=True,
            )
        if residual_path.is_symlink():
            raise ArtifactMismatch("standardized residual output path is unsafe")
        build = build_standardized_residuals(predictions, events)
        residual_temp = _temporary_path(inputs_dir, ".parquet")
        manifest_temp = _temporary_path(inputs_dir, ".json")
        try:
            build.frame.write_parquet(residual_temp)
            _fsync_file(residual_temp)
            residual_sha = file_sha256(residual_temp)
            manifest = _manifest_payload(
                profile=profile,
                inputs=inputs,
                build=build,
                residual_sha256=residual_sha,
            )
            with manifest_temp.open("wb") as stream:
                stream.write(_canonical_json(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(residual_temp, residual_path)
            os.replace(manifest_temp, manifest_path)
            _fsync_directory(inputs_dir)
        finally:
            residual_temp.unlink(missing_ok=True)
            manifest_temp.unlink(missing_ok=True)
        load_standardized_residual_manifest(
            manifest_path,
            run_dir=run,
            config_sha256=config_sha,
            event_sha256=event_sha,
        )
        return StandardizedResidualArtifact(
            residual_path=residual_path,
            manifest_path=manifest_path,
            frame=build.frame,
            reused=False,
        )


def select_diagnostic_residual_context(
    frame: pl.DataFrame,
    manifest: Mapping[str, Any],
    *,
    events: Sequence[EventOccurrence],
    model: str,
    feature_set: str,
    seed: int | None,
    through: int,
) -> pl.DataFrame:
    """Select one manifest-declared context and prove its registered event coverage."""

    _validate_standardized_frame(frame)
    if model not in MODEL_NAMES or feature_set not in _FEATURE_SETS:
        raise ArtifactMismatch("diagnostic context selector is invalid")
    if isinstance(through, bool) or not isinstance(through, int):
        raise ArtifactMismatch("diagnostic through year must be an integer")
    profile = manifest.get("profile")
    if profile not in {"paper", "smoke"}:
        raise ArtifactMismatch("diagnostic residual profile is invalid")
    if profile == "paper" and through != 2023:
        raise ArtifactMismatch("paper AR diagnostics must stop at OOF year 2023")
    contexts = manifest.get("contexts")
    if not isinstance(contexts, list):
        raise ArtifactMismatch("diagnostic residual contexts are invalid")
    matches = [
        record
        for record in contexts
        if isinstance(record, dict)
        and record.get("model") == model
        and record.get("feature_set") == feature_set
        and (seed is None or record.get("seed") == seed)
    ]
    if len(matches) != 1:
        raise ArtifactMismatch("diagnostic selectors must match one manifest context")
    record = matches[0]
    selected_seed = record.get("seed")
    if isinstance(selected_seed, bool) or not isinstance(selected_seed, int):
        raise ArtifactMismatch("diagnostic context seed is invalid")
    selected = frame.filter(
        (pl.col("model") == model)
        & (pl.col("feature_set") == feature_set)
        & (pl.col("seed") == selected_seed)
    ).sort("model", "feature_set", "seed", "target_timestamp")
    if selected.is_empty():
        raise ArtifactMismatch("manifest-declared diagnostic context is absent")

    split_ids = record.get("split_ids")
    if (
        not isinstance(split_ids, list)
        or not split_ids
        or any(not isinstance(value, str) or not is_oof_split_id(value) for value in split_ids)
    ):
        raise ArtifactMismatch("diagnostic split identities are invalid")
    split_years = [fold_for_split_id(value).eval_year for value in split_ids]
    if split_ids != [f"oof-{year}" for year in split_years] or split_years != sorted(
        set(split_years)
    ):
        raise ArtifactMismatch("diagnostic split identities are not canonical")
    if max(split_years) != through or any(year > through for year in split_years):
        raise ArtifactMismatch("diagnostic residuals do not match the numeric through year")
    if profile == "paper" and split_ids != [f"oof-{year}" for year in range(2020, 2024)]:
        raise ArtifactMismatch("paper diagnostics require exact OOF 2020-2023 splits")
    if sorted(selected["split_id"].unique().to_list()) != split_ids:
        raise ArtifactMismatch("diagnostic artifact splits differ from its manifest context")

    lookup = _event_lookup(events).filter(pl.col("split_id").is_in(split_ids))
    expected_occurrences = sorted(lookup["occurrence_id"].unique().to_list())
    if record.get("occurrence_ids") != expected_occurrences:
        raise ArtifactMismatch("diagnostic occurrence identities differ from the registry")
    if sorted(selected["occurrence_id"].unique().to_list()) != expected_occurrences:
        raise ArtifactMismatch("diagnostic occurrence coverage differs from the registry")
    if profile == "paper" and (
        len(expected_occurrences) != 8 or lookup.height != 1_032 or selected.height != 1_032
    ):
        raise ArtifactMismatch("paper diagnostic event-window coverage differs")

    metadata = (
        "target_timestamp",
        "split_id",
        "occurrence_id",
        "holiday_type",
        "tau_days",
        "hour",
        "restriction",
    )
    expected_metadata = lookup.select(metadata).sort("target_timestamp")
    actual_metadata = selected.select(metadata).sort("target_timestamp")
    if not actual_metadata.equals(expected_metadata):
        raise ArtifactMismatch("diagnostic event timestamps or metadata differ from the registry")
    return selected


__all__ = [
    "FoldScaleSummary",
    "STANDARDIZED_RESIDUAL_COLUMNS",
    "StandardizedResidualArtifact",
    "StandardizedResidualBuild",
    "build_standardized_residuals",
    "load_standardized_residual_manifest",
    "prepare_standardized_residual_artifact",
    "select_diagnostic_residual_context",
]
