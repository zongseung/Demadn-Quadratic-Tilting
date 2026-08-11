"""Production preparation of fold-standardized OOF event residuals."""

from __future__ import annotations

import fcntl
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

from hqrc_v3.baselines.config import MODEL_NAMES
from hqrc_v3.baselines.paper import ENSEMBLE_SEED, PAPER_HASH_KEYS
from hqrc_v3.contracts import PREDICTION_COLUMNS, DataContractError, validate_prediction_frame
from hqrc_v3.events import EventOccurrence, load_event_registry
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
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

_SCHEMA_VERSION = 1
_SCHEMA_KIND = "hqrc-v3.standardized-residuals.v1"
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
_INPUT_KEYS = {"oof_predictions", "baseline_manifest", "experiment_config", "event_registry"}
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


def _load_baseline_source(
    *,
    run_dir: Path,
    config_path: Path,
    event_registry_path: Path,
    profile: Literal["paper", "smoke"],
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, dict[str, str]]]:
    manifest_path = run_dir / "predictions/baseline_manifest.json"
    point_path = run_dir / "predictions/oof.parquet"
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
    config_sha = file_sha256(config_path)
    event_sha = file_sha256(event_registry_path)
    if input_hashes.get("experiment_sha256") != config_sha:
        raise ArtifactMismatch("experiment config differs from the baseline manifest")
    if input_hashes.get("event_registry_sha256") != event_sha:
        raise ArtifactMismatch("event registry differs from the baseline manifest")
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
    artifacts = stage.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {"members", "point"}:
        raise ArtifactMismatch("baseline OOF artifact schema differs")
    point = artifacts["point"]
    if (
        not isinstance(point, dict)
        or set(point) != {"path", "sha256"}
        or point.get("path") != "predictions/oof.parquet"
    ):
        raise ArtifactMismatch("baseline OOF point artifact identity differs")
    recorded_prediction_sha = _require_sha(point.get("sha256"), "OOF prediction")
    if not point_path.is_file() or point_path.is_symlink():
        raise ArtifactMismatch("OOF prediction artifact is missing or unsafe")
    prediction_sha = file_sha256(point_path)
    if prediction_sha != recorded_prediction_sha:
        raise ArtifactMismatch("OOF prediction SHA-256 differs from the baseline manifest")
    try:
        predictions = _validate_combined_predictions(pl.read_parquet(point_path))
    except (OSError, pl.exceptions.PolarsError, DataContractError) as error:
        raise ArtifactMismatch(f"OOF prediction artifact is invalid: {error}") from error
    classical_seed = manifest.get("classical_seed")
    ensemble_seed = manifest.get("ensemble_seed")
    if (
        isinstance(classical_seed, bool)
        or not isinstance(classical_seed, int)
        or isinstance(ensemble_seed, bool)
        or not isinstance(ensemble_seed, int)
        or ensemble_seed != ENSEMBLE_SEED
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
    inputs = {
        "oof_predictions": {
            "path": "predictions/oof.parquet",
            "sha256": prediction_sha,
        },
        "baseline_manifest": {
            "path": "predictions/baseline_manifest.json",
            "sha256": file_sha256(manifest_path),
        },
        "experiment_config": {"path": config_path.as_posix(), "sha256": config_sha},
        "event_registry": {"path": event_registry_path.as_posix(), "sha256": event_sha},
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
        _require_sha(entry.get("sha256"), f"{name} input")
    if inputs["experiment_config"]["sha256"] != config_sha256:
        raise ArtifactMismatch("experiment config differs from the residual manifest")
    if inputs["event_registry"]["sha256"] != event_sha256:
        raise ArtifactMismatch("event registry differs from the residual manifest")
    for name, expected_path in (
        ("oof_predictions", "predictions/oof.parquet"),
        ("baseline_manifest", "predictions/baseline_manifest.json"),
    ):
        path = _safe_bound_path(Path(run_dir), inputs[name], expected=expected_path)
        if not path.is_file() or path.is_symlink() or file_sha256(path) != inputs[name]["sha256"]:
            label = "prediction" if name == "oof_predictions" else "baseline manifest"
            raise ArtifactMismatch(f"{label} input differs from the residual manifest")
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
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _temporary_path(directory: Path, suffix: str) -> Path:
    descriptor, name = tempfile.mkstemp(
        prefix=".standardized-residuals.", suffix=suffix, dir=directory
    )
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


def prepare_standardized_residual_artifact(
    *,
    run_dir: Path,
    config_path: Path,
    event_registry_path: Path,
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
            config_path=config,
            event_registry_path=event_path,
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


__all__ = [
    "FoldScaleSummary",
    "STANDARDIZED_RESIDUAL_COLUMNS",
    "StandardizedResidualArtifact",
    "StandardizedResidualBuild",
    "build_standardized_residuals",
    "load_standardized_residual_manifest",
    "prepare_standardized_residual_artifact",
]
