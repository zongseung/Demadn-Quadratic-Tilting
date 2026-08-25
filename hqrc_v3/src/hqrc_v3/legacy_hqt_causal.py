"""Causal 2024 evaluation for the accepted AR-free conference HQT model."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import arviz as az
import polars as pl

from hqrc_v3._loeo_contract import derive_loeo_seed
from hqrc_v3.baselines.config import MODEL_NAMES
from hqrc_v3.bayes.legacy_hqt import (
    LEGACY_HQT_MODEL_SPEC,
    new_event_correction_draws,
    sample_legacy_hqt,
)
from hqrc_v3.bayes.samplers import PYMC_INITIALIZATION_CHOICES, validate_inference_data
from hqrc_v3.correction_source import ValidatedCorrectionSource, validate_correction_source
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.evaluation.metrics import point_metric_frame
from hqrc_v3.events import EventOccurrence
from hqrc_v3.features import B1W_WINDOW_VERSION, feature_columns, history_columns
from hqrc_v3.legacy_hqt_loeo import build_legacy_hqt_data_from_frame, cosine_boundary_taper
from hqrc_v3.provenance import file_sha256

_HOLIDAY_INDEX = {"seollal": 0, "chuseok": 1}
_CORRECTIONS = ("H0", "H1", "H2", "H2-taper")
_FORECAST_COLUMNS = {
    "H0": "H0_mw",
    "H1": "H1_mw",
    "H2": "H2_mw",
    "H2-taper": "H2_taper_mw",
}
_TRAINING_IDS = tuple(
    f"{holiday_type}-{year}"
    for holiday_type in ("chuseok", "seollal")
    for year in range(2020, 2024)
)
_EVALUATION_IDS = ("seollal-2024", "chuseok-2024")
_ARTIFACT_FILES = (
    "input_identity.json",
    "posterior.nc",
    "hourly_predictions.parquet",
    "event_metrics.parquet",
    "pooled_metrics.parquet",
)
_PARTIAL_ARTIFACT_FILES = ("input_identity.json", "posterior.nc")
_COMPLETE_NAMESPACE = frozenset((*_ARTIFACT_FILES, "manifest.json", "COMPLETE"))


class LegacyHQTCausalError(ValueError):
    """Raised when the causal conference-HQT contract is violated."""


@dataclass(frozen=True, slots=True)
class LegacyHQTCausalContextResult:
    context: EventResidualContext
    output_dir: Path
    sampler_fit_count: int
    reused: bool


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise LegacyHQTCausalError("causal HQT metadata is not canonical JSON") from error


def _json_sha256(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_parquet_atomic(frame: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".parquet", dir=path.parent
    )
    os.close(descriptor)
    try:
        frame.write_parquet(temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_netcdf_atomic(idata: az.InferenceData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".nc", dir=path.parent)
    os.close(descriptor)
    try:
        az.to_netcdf(idata, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _require_real_file(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise LegacyHQTCausalError(f"{description} is missing or unsafe") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise LegacyHQTCausalError(f"{description} is missing or unsafe")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    _require_real_file(path, description)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyHQTCausalError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise LegacyHQTCausalError(f"{description} is not canonical")
    return value


def _event_hours(event: EventOccurrence) -> tuple[datetime, ...]:
    timestamp = datetime.combine(event.window_start, time.min)
    end = datetime.combine(event.window_end + timedelta(days=1), time.min)
    values = []
    while timestamp < end:
        values.append(timestamp)
        timestamp += timedelta(hours=1)
    return tuple(values)


def _event_registry(
    source: ValidatedCorrectionSource,
) -> tuple[dict[str, EventOccurrence], tuple[EventOccurrence, ...]]:
    by_id = {event.occurrence_id: event for event in source.events}
    if len(by_id) != len(source.events):
        raise LegacyHQTCausalError("causal HQT event registry contains duplicate ids")
    if (set(_TRAINING_IDS) | set(_EVALUATION_IDS)) - set(by_id):
        raise LegacyHQTCausalError("causal HQT event registry lacks the 2020-2024 universe")
    evaluation = tuple(by_id[event_id] for event_id in _EVALUATION_IDS)
    if tuple(event.holiday_type for event in evaluation) != ("seollal", "chuseok"):
        raise LegacyHQTCausalError("causal HQT 2024 event types differ")
    return by_id, evaluation


def build_causal_training_frame(
    source: ValidatedCorrectionSource, context: EventResidualContext
) -> tuple[pl.DataFrame, tuple[str, ...], float]:
    """Load exactly eight physical OOF events and the unique OOF-2023 scale."""

    if not isinstance(context, EventResidualContext) or context.feature_set != "B1W":
        raise LegacyHQTCausalError("causal HQT requires a B1W residual context")
    frame = source.load_standardized_context(context, through=2023)
    required = {
        "feature_set",
        "holiday_type",
        "hour",
        "model",
        "occurrence_id",
        "residual_mw",
        "restriction",
        "seed",
        "sigma_n_mw",
        "split_id",
        "standardized_residual",
        "target_timestamp",
        "tau_days",
    }
    if (
        not isinstance(frame, pl.DataFrame)
        or frame.is_empty()
        or not required.issubset(frame.columns)
    ):
        raise LegacyHQTCausalError("causal HQT training frame schema differs")
    if (
        frame["model"].unique().to_list() != [context.model]
        or frame["feature_set"].unique().to_list() != ["B1W"]
        or frame["seed"].unique().to_list() != [context.seed]
    ):
        raise LegacyHQTCausalError("causal HQT training context differs")
    frame = frame.sort("occurrence_id", "target_timestamp")
    ids = tuple(frame["occurrence_id"].unique(maintain_order=True).to_list())
    if ids != _TRAINING_IDS or any(event_id.endswith("-2024") for event_id in ids):
        raise LegacyHQTCausalError("causal HQT requires exactly eight pre-2024 events")
    expected_splits = {f"oof-{year}" for year in range(2020, 2024)}
    if set(frame["split_id"].unique().to_list()) != expected_splits:
        raise LegacyHQTCausalError("causal HQT requires exactly OOF 2020-2023 splits")
    for event_id in ids:
        selected = frame.filter(pl.col("occurrence_id") == event_id)
        year = event_id.rsplit("-", 1)[1]
        holiday_type = event_id.rsplit("-", 1)[0]
        if selected["split_id"].unique().to_list() != [f"oof-{year}"] or selected[
            "holiday_type"
        ].unique().to_list() != [holiday_type]:
            raise LegacyHQTCausalError("causal HQT physical event identity differs")
    scales = frame.filter(pl.col("split_id") == "oof-2023")["sigma_n_mw"].unique()
    if scales.len() != 1:
        raise LegacyHQTCausalError("pre-2024 evaluation scale is not unique")
    sigma_fit = float(scales.item())
    if not math.isfinite(sigma_fit) or sigma_fit <= 0:
        raise LegacyHQTCausalError("pre-2024 evaluation scale must be positive")
    return frame, ids, sigma_fit


def _evaluation_frames(
    source: ValidatedCorrectionSource, context: EventResidualContext
) -> tuple[tuple[str, EventOccurrence, pl.DataFrame], ...]:
    _, events = _event_registry(source)
    final = source.load_final_point_context(context)
    required = {
        "feature_set",
        "model",
        "observed_mw",
        "predicted_mw",
        "seed",
        "split_id",
        "target_timestamp",
    }
    if (
        not isinstance(final, pl.DataFrame)
        or final.is_empty()
        or not required.issubset(final.columns)
    ):
        raise LegacyHQTCausalError("causal HQT final-2024 frame schema differs")
    if (
        final["model"].unique().to_list() != [context.model]
        or final["feature_set"].unique().to_list() != ["B1W"]
        or final["seed"].unique().to_list() != [context.seed]
        or final["split_id"].unique().to_list() != ["final-2024"]
    ):
        raise LegacyHQTCausalError("causal HQT final-2024 context differs")
    evaluations = []
    for event in events:
        expected = _event_hours(event)
        selected = final.filter(pl.col("target_timestamp").is_in(expected)).sort("target_timestamp")
        observed_timestamps = tuple(selected["target_timestamp"].to_list())
        if selected.height != len(expected) or observed_timestamps != expected:
            raise LegacyHQTCausalError(
                f"causal HQT final event window is incomplete: {event.occurrence_id}"
            )
        center = datetime.combine(event.central_date, time.min)
        selected = selected.with_columns(
            pl.lit(event.occurrence_id).alias("occurrence_id"),
            pl.lit(event.holiday_type).alias("holiday_type"),
            pl.Series(
                "tau_days",
                [
                    (timestamp - center).total_seconds() / 86_400.0
                    for timestamp in observed_timestamps
                ],
                dtype=pl.Float64,
            ),
        )
        evaluations.append((event.occurrence_id, event, selected))
    return tuple(evaluations)


def _b1w_schema(source: ValidatedCorrectionSource) -> dict[str, object]:
    expected: dict[str, object] = {
        "history": list(history_columns("B1W")),
        "future": list(feature_columns("B1W")),
        "window_definition": [B1W_WINDOW_VERSION],
    }
    manifest = source.baseline_manifest
    schemas = manifest.get("feature_schemas") if isinstance(manifest, dict) else None
    schema = schemas.get("B1W") if isinstance(schemas, dict) else None
    if schema != expected:
        raise LegacyHQTCausalError("causal HQT source B1W feature schema differs")
    return expected


def _select_contexts(
    source: ValidatedCorrectionSource, *, models: Iterable[str], feature_set: str
) -> tuple[EventResidualContext, ...]:
    requested = tuple(models)
    if (
        not requested
        or len(requested) != len(set(requested))
        or any(model not in MODEL_NAMES for model in requested)
    ):
        raise LegacyHQTCausalError("causal HQT model scope differs")
    if feature_set != "B1W":
        raise LegacyHQTCausalError("causal HQT feature set must be B1W")
    available = {
        (context.model, context.feature_set): context for context in source.available_contexts
    }
    try:
        return tuple(available[(model, "B1W")] for model in MODEL_NAMES if model in requested)
    except KeyError as error:
        raise LegacyHQTCausalError("causal HQT source lacks a requested B1W context") from error


def _sampler_contract(
    *,
    profile: str,
    root_seed: int,
    context: EventResidualContext,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    cores: int | None,
    init: str | None,
    target_accept: float | None,
) -> dict[str, object]:
    if isinstance(root_seed, bool) or not isinstance(root_seed, int):
        raise LegacyHQTCausalError("causal HQT root seed must be an integer")
    if profile == "paper":
        resolved_draws = 1_000 if draws is None else draws
        resolved_tune = 1_000 if tune is None else tune
        resolved_chains = 4 if chains is None else chains
        resolved_target = 0.99 if target_accept is None else target_accept
        if (
            resolved_draws < 1_000
            or resolved_tune < 1_000
            or resolved_chains < 4
            or resolved_target != 0.99
        ):
            raise LegacyHQTCausalError(
                "paper causal HQT requires at least 4 chains, at least 1000 tune/draws, "
                "and target_accept=0.99"
            )
    elif profile == "smoke":
        if draws is None or tune is None or chains is None:
            raise LegacyHQTCausalError("smoke causal HQT requires explicit draws, tune, and chains")
        resolved_draws, resolved_tune, resolved_chains = draws, tune, chains
        resolved_target = 0.9 if target_accept is None else target_accept
    else:
        raise LegacyHQTCausalError("profile must be paper or smoke")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (resolved_draws, resolved_tune, resolved_chains)
    ):
        raise LegacyHQTCausalError("draws, tune, and chains must be positive integers")
    resolved_cores = min(resolved_chains, os.cpu_count() or 1) if cores is None else cores
    if (
        isinstance(resolved_cores, bool)
        or not isinstance(resolved_cores, int)
        or resolved_cores <= 0
        or resolved_cores > resolved_chains
    ):
        raise LegacyHQTCausalError("cores must be a positive integer no greater than chains")
    resolved_init = "adapt_diag" if init is None else init
    if resolved_init not in PYMC_INITIALIZATION_CHOICES:
        raise LegacyHQTCausalError("causal HQT sampler initialization is unsupported")
    if (
        isinstance(resolved_target, bool)
        or not isinstance(resolved_target, (int, float))
        or not math.isfinite(float(resolved_target))
        or not 0 < float(resolved_target) < 1
    ):
        raise LegacyHQTCausalError("causal HQT target_accept is invalid")
    seed_label = f"legacy-hqt-causal-2024:{context.model}:{context.feature_set}:seed-{context.seed}"
    return {
        "chains": resolved_chains,
        "cores": resolved_cores,
        "draws": resolved_draws,
        "init": resolved_init,
        "profile": profile,
        "root_seed": root_seed,
        "seed": derive_loeo_seed(root_seed, seed_label),
        "target_accept": float(resolved_target),
        "tune": resolved_tune,
    }


def _context_identity(
    source: ValidatedCorrectionSource,
    context: EventResidualContext,
    *,
    training_ids: tuple[str, ...],
    evaluation_ids: tuple[str, ...],
    sigma_fit: float,
    feature_schema: Mapping[str, object],
    sampler: Mapping[str, object],
) -> dict[str, object]:
    return {
        "context": {
            "feature_set": context.feature_set,
            "model": context.model,
            "seed": context.seed,
            "split_ids": list(context.split_ids),
        },
        "evaluation": "causal-2024",
        "evaluation_occurrence_ids": list(evaluation_ids),
        "feature_schema": dict(feature_schema),
        "model_spec": LEGACY_HQT_MODEL_SPEC,
        "sampler": dict(sampler),
        "schema_version": 1,
        "sigma_fit_oof_2023_mw": sigma_fit,
        "source": {
            "baseline_manifest_sha256": file_sha256(source.baseline_manifest_path),
            "final_members_sha256": source.final_members_sha256,
            "final_point_sha256": source.final_point_sha256,
            "residual_manifest_sha256": file_sha256(source.residual_manifest_path),
            "residual_sha256": source.residual_sha256,
        },
        "training_occurrence_ids": list(training_ids),
    }


def _same_type_shift(frame: pl.DataFrame, holiday_type: str) -> float:
    selected = frame.filter(pl.col("holiday_type") == holiday_type)["residual_mw"]
    if selected.is_empty():
        raise LegacyHQTCausalError("causal H1 has no same-type training events")
    shift = float(selected.mean())
    if not math.isfinite(shift):
        raise LegacyHQTCausalError("causal H1 shift is non-finite")
    return shift


def _build_products(
    training: pl.DataFrame,
    evaluations: tuple[tuple[str, EventOccurrence, pl.DataFrame], ...],
    idata: az.InferenceData,
    *,
    context: EventResidualContext,
    sigma_fit: float,
    root_seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, int]]:
    hourly_frames = []
    metric_frames = []
    prediction_seeds: dict[str, int] = {}
    for occurrence_id, event, frame in evaluations:
        observed = frame["observed_mw"].to_numpy()
        baseline = frame["predicted_mw"].to_numpy()
        h1_shift = _same_type_shift(training, event.holiday_type)
        prediction_seed = derive_loeo_seed(
            root_seed, f"legacy-hqt-causal-predictive:{occurrence_id}"
        )
        prediction_seeds[occurrence_id] = prediction_seed
        draws = new_event_correction_draws(
            idata,
            holiday_type_index=_HOLIDAY_INDEX[event.holiday_type],
            tau_days=frame["tau_days"].to_numpy(),
            seed=prediction_seed,
        )
        if draws.ndim != 2 or draws.shape[1] != frame.height or draws.shape[0] == 0:
            raise LegacyHQTCausalError("causal HQT new-event correction shape differs")
        standardized = draws.mean(axis=0)
        taper = cosine_boundary_taper(frame.height)
        h1 = baseline + h1_shift
        h2 = baseline + sigma_fit * standardized
        h2_taper = baseline + sigma_fit * standardized * taper
        hourly_frames.append(
            pl.DataFrame(
                {
                    "model": [context.model] * frame.height,
                    "feature_set": [context.feature_set] * frame.height,
                    "seed": [context.seed] * frame.height,
                    "occurrence_id": [occurrence_id] * frame.height,
                    "holiday_type": [event.holiday_type] * frame.height,
                    "target_timestamp": frame["target_timestamp"],
                    "observed_mw": observed,
                    "H0_mw": baseline,
                    "H1_mw": h1,
                    "H2_mw": h2,
                    "H2_taper_mw": h2_taper,
                    "h1_shift_mw": [h1_shift] * frame.height,
                    "h2_standardized_correction": standardized,
                    "taper_weight": taper,
                    "sigma_n_mw": [sigma_fit] * frame.height,
                }
            )
        )
        for correction, forecast in zip(_CORRECTIONS, (baseline, h1, h2, h2_taper)):
            metric_frames.append(
                point_metric_frame(occurrence_id, observed, forecast).with_columns(
                    pl.lit(context.model).alias("model"),
                    pl.lit(context.feature_set).alias("feature_set"),
                    pl.lit(context.seed).alias("seed"),
                    pl.lit(event.holiday_type).alias("holiday_type"),
                    pl.lit(correction).alias("correction"),
                )
            )
    hourly = pl.concat(hourly_frames, how="vertical")
    event_metrics = pl.concat(metric_frames, how="vertical").select(
        "model",
        "feature_set",
        "seed",
        "event_id",
        "holiday_type",
        "correction",
        "n_timestamps",
        "rmse",
        "mae",
        "mape",
        "smape",
        "r2",
    )
    pooled_rows = []
    for scope, selected in (
        ("all", hourly),
        ("seollal", hourly.filter(pl.col("holiday_type") == "seollal")),
        ("chuseok", hourly.filter(pl.col("holiday_type") == "chuseok")),
    ):
        for correction, column in _FORECAST_COLUMNS.items():
            pooled_rows.append(
                point_metric_frame(
                    scope,
                    selected["observed_mw"].to_numpy(),
                    selected[column].to_numpy(),
                ).with_columns(
                    pl.lit(context.model).alias("model"),
                    pl.lit(context.feature_set).alias("feature_set"),
                    pl.lit(context.seed).alias("seed"),
                    pl.lit(scope).alias("scope"),
                    pl.lit(correction).alias("correction"),
                )
            )
    pooled = pl.concat(pooled_rows, how="vertical").select(
        "model",
        "feature_set",
        "seed",
        "scope",
        "correction",
        "n_timestamps",
        "rmse",
        "mae",
        "mape",
        "smape",
        "r2",
    )
    return hourly, event_metrics, pooled, prediction_seeds


def _validate_posterior(
    idata: az.InferenceData,
    *,
    identity_sha256: str,
    paper_profile: bool,
) -> None:
    try:
        model_spec = json.loads(idata.attrs.get("legacy_hqt_model_json", "null"))
    except (TypeError, json.JSONDecodeError) as error:
        raise LegacyHQTCausalError("resumable causal HQT posterior model is unreadable") from error
    if model_spec != LEGACY_HQT_MODEL_SPEC:
        raise LegacyHQTCausalError("resumable causal HQT posterior model differs")
    if idata.attrs.get("legacy_hqt_causal_identity_sha256") != identity_sha256:
        raise LegacyHQTCausalError("resumable causal HQT posterior identity differs")
    validate_inference_data(idata, paper_profile=paper_profile)


def _load_complete_context(output_dir: Path, identity: Mapping[str, object]) -> bool:
    complete_path = output_dir / "COMPLETE"
    if not complete_path.exists():
        return False
    try:
        entries = {entry.name for entry in output_dir.iterdir()}
        if entries != _COMPLETE_NAMESPACE:
            raise LegacyHQTCausalError("completed causal HQT namespace differs")
        stored_identity = _read_json(
            output_dir / "input_identity.json", "completed causal HQT identity"
        )
        if stored_identity != dict(identity):
            raise LegacyHQTCausalError("completed causal HQT identity differs")
        manifest_path = output_dir / "manifest.json"
        manifest = _read_json(manifest_path, "completed causal HQT manifest")
        complete = _read_json(complete_path, "completed causal HQT completion marker")
        files = manifest.get("files")
        if (
            manifest.get("schema_version") != 1
            or manifest.get("identity_sha256") != _json_sha256(identity)
            or not isinstance(files, dict)
            or set(files) != set(_ARTIFACT_FILES)
            or complete != {"manifest_sha256": file_sha256(manifest_path), "schema_version": 1}
        ):
            raise LegacyHQTCausalError("completed causal HQT manifest differs")
        for name in _ARTIFACT_FILES:
            path = output_dir / name
            _require_real_file(path, f"completed causal HQT artifact {name}")
            if files.get(name) != file_sha256(path):
                raise LegacyHQTCausalError(f"completed causal HQT artifact digest differs: {name}")
        pl.read_parquet(output_dir / "hourly_predictions.parquet")
        pl.read_parquet(output_dir / "event_metrics.parquet")
        pl.read_parquet(output_dir / "pooled_metrics.parquet")
        idata = az.from_netcdf(output_dir / "posterior.nc")
        _validate_posterior(
            idata,
            identity_sha256=_json_sha256(identity),
            paper_profile=identity["sampler"]["profile"] == "paper",  # type: ignore[index]
        )
    except LegacyHQTCausalError:
        raise
    except (OSError, KeyError, TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise LegacyHQTCausalError("completed causal HQT checkpoint is unreadable") from error
    return True


def _load_resumable_posterior(
    output_dir: Path,
    *,
    identity: Mapping[str, object],
    paper_profile: bool,
) -> az.InferenceData | None:
    posterior_path = output_dir / "posterior.nc"
    manifest_path = output_dir / "manifest.json"
    posterior_exists = os.path.lexists(posterior_path)
    manifest_exists = os.path.lexists(manifest_path)
    if not posterior_exists:
        if manifest_exists:
            raise LegacyHQTCausalError("incomplete causal HQT manifest exists without a posterior")
        entries = {entry.name for entry in output_dir.iterdir()}
        if entries != {"input_identity.json"}:
            raise LegacyHQTCausalError("incomplete causal HQT checkpoint state differs")
        return None
    _require_real_file(posterior_path, "incomplete causal HQT posterior")
    identity_path = output_dir / "input_identity.json"
    if _read_json(identity_path, "incomplete causal HQT identity") != dict(identity):
        raise LegacyHQTCausalError("incomplete causal HQT identity differs")
    if not manifest_exists:
        raise LegacyHQTCausalError("incomplete causal HQT posterior manifest is missing")
    manifest = _read_json(manifest_path, "incomplete causal HQT manifest")
    files = manifest.get("files")
    manifest_files = frozenset(files) if isinstance(files, dict) else frozenset()
    if (
        manifest.get("schema_version") != 1
        or manifest.get("identity_sha256") != _json_sha256(identity)
        or not isinstance(files, dict)
        or manifest_files not in {frozenset(_PARTIAL_ARTIFACT_FILES), frozenset(_ARTIFACT_FILES)}
    ):
        raise LegacyHQTCausalError("incomplete causal HQT posterior manifest differs")
    for name in manifest_files:
        path = output_dir / name
        _require_real_file(path, f"incomplete causal HQT artifact {name}")
        if files.get(name) != file_sha256(path):
            raise LegacyHQTCausalError(f"incomplete causal HQT artifact digest differs: {name}")
    try:
        idata = az.from_netcdf(posterior_path)
    except (OSError, ValueError) as error:
        raise LegacyHQTCausalError("incomplete causal HQT posterior is unreadable") from error
    _validate_posterior(
        idata,
        identity_sha256=_json_sha256(identity),
        paper_profile=paper_profile,
    )
    return idata


def _run_context(
    source: ValidatedCorrectionSource,
    context: EventResidualContext,
    *,
    output_dir: Path,
    sampler: Mapping[str, object],
) -> LegacyHQTCausalContextResult:
    training, training_ids, sigma_fit = build_causal_training_frame(source, context)
    evaluations = _evaluation_frames(source, context)
    evaluation_ids = tuple(item[0] for item in evaluations)
    if evaluation_ids != _EVALUATION_IDS:
        raise LegacyHQTCausalError("causal HQT evaluation event ordering differs")
    identity = _context_identity(
        source,
        context,
        training_ids=training_ids,
        evaluation_ids=evaluation_ids,
        sigma_fit=sigma_fit,
        feature_schema=_b1w_schema(source),
        sampler=sampler,
    )
    if output_dir.exists() and _load_complete_context(output_dir, identity):
        return LegacyHQTCausalContextResult(context, output_dir, 0, True)
    entries: set[str] = set()
    if output_dir.exists():
        try:
            mode = output_dir.lstat().st_mode
            entries = {entry.name for entry in output_dir.iterdir()}
        except OSError as error:
            raise LegacyHQTCausalError("incomplete causal HQT namespace is unsafe") from error
        allowed = _COMPLETE_NAMESPACE - {"COMPLETE"}
        if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode) or not entries <= allowed:
            raise LegacyHQTCausalError("incomplete causal HQT namespace differs")
        for entry in output_dir.iterdir():
            _require_real_file(entry, f"incomplete causal HQT checkpoint {entry.name}")

    output_dir.mkdir(parents=True, exist_ok=True)
    identity_path = output_dir / "input_identity.json"
    if entries and "input_identity.json" not in entries:
        raise LegacyHQTCausalError(
            "incomplete causal HQT identity is missing beside checkpoint state"
        )
    if "input_identity.json" in entries:
        if _read_json(identity_path, "incomplete causal HQT identity") != identity:
            raise LegacyHQTCausalError("incomplete causal HQT identity differs")
    else:
        _write_atomic(identity_path, _canonical_json(identity))
    identity_digest = _json_sha256(identity)
    idata = _load_resumable_posterior(
        output_dir,
        identity=identity,
        paper_profile=sampler["profile"] == "paper",
    )
    fitted = idata is None
    if fitted:
        data = build_legacy_hqt_data_from_frame(training, occurrence_ids=training_ids)
        idata = sample_legacy_hqt(
            data,
            draws=int(sampler["draws"]),
            tune=int(sampler["tune"]),
            chains=int(sampler["chains"]),
            cores=int(sampler["cores"]),
            seed=int(sampler["seed"]),
            init=str(sampler["init"]),
            target_accept=float(sampler["target_accept"]),
            paper_profile=sampler["profile"] == "paper",
        )
        idata.attrs["legacy_hqt_causal_identity_sha256"] = identity_digest
        _write_netcdf_atomic(idata, output_dir / "posterior.nc")
        partial_manifest = {
            "files": {name: file_sha256(output_dir / name) for name in _PARTIAL_ARTIFACT_FILES},
            "identity_sha256": identity_digest,
            "schema_version": 1,
        }
        _write_atomic(output_dir / "manifest.json", _canonical_json(partial_manifest))
    assert idata is not None
    hourly, event_metrics, pooled_metrics, prediction_seeds = _build_products(
        training,
        evaluations,
        idata,
        context=context,
        sigma_fit=sigma_fit,
        root_seed=int(sampler["root_seed"]),
    )
    _write_parquet_atomic(hourly, output_dir / "hourly_predictions.parquet")
    _write_parquet_atomic(event_metrics, output_dir / "event_metrics.parquet")
    _write_parquet_atomic(pooled_metrics, output_dir / "pooled_metrics.parquet")
    manifest = {
        "files": {name: file_sha256(output_dir / name) for name in _ARTIFACT_FILES},
        "identity_sha256": identity_digest,
        "prediction_seeds": prediction_seeds,
        "schema_version": 1,
    }
    manifest_path = output_dir / "manifest.json"
    _write_atomic(manifest_path, _canonical_json(manifest))
    _write_atomic(
        output_dir / "COMPLETE",
        _canonical_json({"manifest_sha256": file_sha256(manifest_path), "schema_version": 1}),
    )
    return LegacyHQTCausalContextResult(context, output_dir, int(fitted), False)


def run_legacy_hqt_causal_2024(
    *,
    source_run_dir: Path,
    config_path: Path,
    output_root: Path,
    models: Iterable[str],
    feature_set: str,
    root_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    init: str | None = None,
    target_accept: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[LegacyHQTCausalContextResult, ...]:
    """Fit/reuse one causal HQT posterior per selected B1W model context."""

    source = validate_correction_source(
        run_dir=Path(source_run_dir).expanduser().resolve(),
        config_path=Path(config_path).expanduser().resolve(),
        profile=profile,
    )
    contexts = _select_contexts(source, models=models, feature_set=feature_set)
    results = []
    for index, context in enumerate(contexts, start=1):
        sampler = _sampler_contract(
            profile=profile,
            root_seed=root_seed,
            context=context,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            init=init,
            target_accept=target_accept,
        )
        label = f"{context.model}/{context.feature_set}/seed-{context.seed}"
        if progress:
            progress(f"[{index}/{len(contexts)} {label}] causal-2024")
        result = _run_context(
            source,
            context,
            output_dir=Path(output_root).expanduser().resolve()
            / context.model
            / context.feature_set,
            sampler=sampler,
        )
        results.append(result)
        if progress:
            state = "reused" if result.reused else "complete"
            progress(f"[{index}/{len(contexts)} {label}] {state}")
    return tuple(results)


__all__ = [
    "LegacyHQTCausalContextResult",
    "LegacyHQTCausalError",
    "build_causal_training_frame",
    "run_legacy_hqt_causal_2024",
]
