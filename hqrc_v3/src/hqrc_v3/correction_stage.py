"""Strict production orchestration for the single causal-2024 HQRC comparison."""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import polars as pl

from hqrc_v3.bayes.artifacts import load_hqrc_data, write_hqrc_data
from hqrc_v3.bayes.model import CYCLIC_HOUR_PARAMETERIZATION, HQRCData, HQRCModelOptions
from hqrc_v3.bayes.predictive import (
    corrected_predictive_draws,
    draw_new_event_correction,
    posterior_values,
    select_posterior_indices,
    simulate_stationary_ar1,
)
from hqrc_v3.bayes.samplers import (
    PYMC_INITIALIZATION,
    SAMPLER_GEOMETRY,
    sample_hqrc,
    validate_inference_data,
)
from hqrc_v3.contracts import DataContractError, validate_prediction_frame
from hqrc_v3.correction_source import CorrectionSourceError, validate_correction_source
from hqrc_v3.diagnostics.ar import (
    ApprovedARCalibration,
    EventResidualContext,
    load_approved_calibration,
    require_approved_calibration,
    validate_event_residual_context,
)
from hqrc_v3.evaluation.metrics import point_metric_frame, probabilistic_metric_frame
from hqrc_v3.events import EventOccurrence
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
from hqrc_v3.publication_fs import _fsync_directory as _portable_fsync_directory
from hqrc_v3.publication_fs import _fsync_file as _portable_fsync_file
from hqrc_v3.publication_fs import exclusive_lock

_CAUSAL_SPLITS = tuple(f"oof-{year}" for year in range(2020, 2024))
_HOLIDAY_INDEX = {"seollal": 0, "chuseok": 1}


class CausalCorrectionError(ValueError):
    """Raised when an input cannot belong to the frozen causal-2024 estimand."""


@dataclass(frozen=True)
class CausalPredictionData:
    """Validated Jan--Oct final stream and its two event-reset evaluation segments."""

    full_frame: pl.DataFrame
    event_frame: pl.DataFrame
    occurrence_ids: tuple[str, ...]
    segments: tuple[slice, ...]


@dataclass(frozen=True)
class CausalCorrectionInputs:
    """Every source-revalidated input needed by one context-specific H3 fit."""

    run_dir: Path
    source_profile: str
    approved: ApprovedARCalibration
    hqrc_data: HQRCData
    training_frame: pl.DataFrame
    prediction: CausalPredictionData
    sigma_n_mw: float
    residual_manifest: Mapping[str, Any]
    baseline_manifest: Mapping[str, Any]
    source_paths: Mapping[str, Path]
    source_hashes: Mapping[str, str]
    residual_path: Path
    final_members_path: Path
    final_point_path: Path


@dataclass(frozen=True)
class CausalCorrectionProducts:
    """Aligned deterministic and probabilistic products before publication."""

    event_predictions: pl.DataFrame
    full_period_point_predictions: pl.DataFrame
    event_metrics: pl.DataFrame
    full_period_point_metrics: pl.DataFrame


@dataclass(frozen=True)
class CausalCorrectionResult:
    """One completed or strictly reused context-specific causal publication."""

    output_dir: Path
    manifest_path: Path
    posterior_path: Path
    event_predictions_path: Path
    full_period_point_predictions_path: Path
    event_metrics_path: Path
    full_period_point_metrics_path: Path
    sampler_fit_count: int
    reused: bool


def _trusted(value: ApprovedARCalibration) -> ApprovedARCalibration:
    try:
        return require_approved_calibration(value)
    except (ArtifactMismatch, TypeError, ValueError) as error:
        raise CausalCorrectionError("approved AR calibration is not trusted") from error


def _causal_training_events(events: Sequence[EventOccurrence]) -> tuple[EventOccurrence, ...]:
    selected = tuple(
        sorted(
            (event for event in events if 2020 <= event.central_date.year <= 2023),
            key=lambda event: event.window_start,
        )
    )
    identities = {(event.holiday_type, event.central_date.year) for event in selected}
    expected = {(holiday, year) for year in range(2020, 2024) for holiday in ("seollal", "chuseok")}
    if len(selected) != 8 or identities != expected:
        raise CausalCorrectionError("causal training requires exactly eight 2020-2023 events")
    return selected


def _expected_event_metadata(events: Sequence[EventOccurrence]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for event in events:
        start = datetime.combine(event.window_start, time.min)
        end = datetime.combine(event.window_end + timedelta(days=1), time.min)
        center = datetime.combine(event.central_date, time.min)
        timestamp = start
        while timestamp < end:
            rows.append(
                {
                    "target_timestamp": timestamp,
                    "split_id": f"oof-{event.central_date.year}",
                    "occurrence_id": event.occurrence_id,
                    "holiday_type": event.holiday_type,
                    "tau_days": (timestamp - center).total_seconds() / 86_400.0,
                    "hour": timestamp.hour,
                    "restriction": event.restriction,
                }
            )
            timestamp += timedelta(hours=1)
    return pl.DataFrame(rows).sort("target_timestamp")


def build_causal_hqrc_data(
    frame: pl.DataFrame,
    approved: ApprovedARCalibration,
    *,
    events: Sequence[EventOccurrence],
) -> HQRCData:
    """Bind complete 2020--2023 OOF event segments to one approved context."""

    trusted = _trusted(approved)
    if trusted.context.split_ids != _CAUSAL_SPLITS:
        raise CausalCorrectionError("approved context must contain exact OOF 2020-2023 splits")
    training_events = _causal_training_events(events)
    occurrence_ids = tuple(event.occurrence_id for event in training_events)
    if set(trusted.calibration.event_ids) != set(occurrence_ids):
        raise CausalCorrectionError("approved event ids differ from the eight training events")
    required = {
        "target_timestamp",
        "standardized_residual",
        "model",
        "feature_set",
        "seed",
        "split_id",
        "occurrence_id",
        "holiday_type",
        "tau_days",
        "hour",
        "restriction",
    }
    if not isinstance(frame, pl.DataFrame) or not required.issubset(frame.columns):
        raise CausalCorrectionError("training residual frame schema differs")
    selected = frame.select(sorted(required)).sort("target_timestamp")
    try:
        actual_context = validate_event_residual_context(selected)
    except (TypeError, ValueError) as error:
        raise CausalCorrectionError("training residual context is invalid") from error
    if actual_context != trusted.context:
        raise CausalCorrectionError("training residual context differs from the approval")
    if set(selected["occurrence_id"].unique().to_list()) != set(occurrence_ids):
        raise CausalCorrectionError("training occurrence population differs")
    expected = _expected_event_metadata(training_events)
    metadata = tuple(expected.columns)
    actual = selected.select(metadata).sort("target_timestamp")
    if not actual.equals(expected):
        raise CausalCorrectionError("training event timestamps or metadata differ from registry")

    event_index = {event_id: index for index, event_id in enumerate(occurrence_ids)}
    ordered = selected.sort("target_timestamp")
    return HQRCData(
        observations=ordered["standardized_residual"].to_numpy(),
        occurrence_index=np.asarray(
            [event_index[value] for value in ordered["occurrence_id"].to_list()],
            dtype=np.int64,
        ),
        holiday_type_index=np.asarray(
            [_HOLIDAY_INDEX[value] for value in ordered["holiday_type"].to_list()],
            dtype=np.int64,
        ),
        tau_days=ordered["tau_days"].to_numpy(),
        hour=ordered["hour"].to_numpy(),
        restriction=ordered["restriction"].to_numpy(),
        occurrence_ids=occurrence_ids,
    )


def select_latest_oof_scale(manifest: Mapping[str, Any], approved: ApprovedARCalibration) -> float:
    """Return only the manifest-declared fold-local ``oof-2023`` MW scale."""

    trusted = _trusted(approved)
    contexts = manifest.get("contexts") if isinstance(manifest, Mapping) else None
    if not isinstance(contexts, list):
        raise CausalCorrectionError("residual manifest contexts are invalid")
    matches = [
        item
        for item in contexts
        if isinstance(item, dict)
        and item.get("model") == trusted.context.model
        and item.get("feature_set") == trusted.context.feature_set
        and item.get("seed") == trusted.context.seed
    ]
    if len(matches) != 1:
        raise CausalCorrectionError("approved residual context is not unique in manifest")
    latest = matches[0].get("latest_complete_oof_scale")
    if not isinstance(latest, dict) or set(latest) != {"split_id", "sigma_n_mw"}:
        raise CausalCorrectionError("latest complete OOF scale is invalid")
    scale = latest.get("sigma_n_mw")
    if latest.get("split_id") != "oof-2023":
        raise CausalCorrectionError("latest correction scale must be oof-2023")
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise CausalCorrectionError("latest OOF scale must be finite and positive")
    return float(scale)


def _evaluation_events(events: Sequence[EventOccurrence]) -> tuple[EventOccurrence, ...]:
    selected = tuple(
        sorted(
            (event for event in events if event.central_date.year == 2024),
            key=lambda event: event.window_start,
        )
    )
    if tuple(event.occurrence_id for event in selected) != (
        "seollal-2024",
        "chuseok-2024",
    ):
        raise CausalCorrectionError("evaluation registry must contain only the two 2024 events")
    return selected


def build_causal_prediction_data(
    frame: pl.DataFrame,
    *,
    context: EventResidualContext,
    events: Sequence[EventOccurrence],
) -> CausalPredictionData:
    """Validate one final point stream and extract exact registered event hours."""

    try:
        validated = validate_prediction_frame(frame).sort("target_timestamp")
    except (DataContractError, TypeError, ValueError) as error:
        raise CausalCorrectionError("final baseline stream is invalid") from error
    identity = (
        str(validated["model"].item(0)),
        str(validated["feature_set"].item(0)),
        int(validated["seed"].item(0)),
        str(validated["split_id"].item(0)),
    )
    if identity != (context.model, context.feature_set, context.seed, "final-2024"):
        raise CausalCorrectionError("final baseline stream differs from approved context")
    expected_start, expected_end = datetime(2024, 1, 1), datetime(2024, 10, 31, 23)
    if (
        validated.height != 7_320
        or validated["target_timestamp"].item(0) != expected_start
        or validated["target_timestamp"].item(-1) != expected_end
    ):
        raise CausalCorrectionError("final baseline must cover exactly Jan-Oct 2024")

    frames: list[pl.DataFrame] = []
    segments: list[slice] = []
    offset = 0
    evaluation_events = _evaluation_events(events)
    for event in evaluation_events:
        start = datetime.combine(event.window_start, time.min)
        end = datetime.combine(event.window_end + timedelta(days=1), time.min)
        center = datetime.combine(event.central_date, time.min)
        selected = validated.filter(
            (pl.col("target_timestamp") >= start) & (pl.col("target_timestamp") < end)
        ).with_columns(
            pl.lit(event.occurrence_id).alias("occurrence_id"),
            pl.lit(event.holiday_type).alias("holiday_type"),
            ((pl.col("target_timestamp") - pl.lit(center)).dt.total_seconds() / 86_400.0).alias(
                "tau_days"
            ),
            pl.lit(event.restriction).cast(pl.Int64).alias("restriction"),
        )
        expected_hours = int((end - start).total_seconds() / 3_600)
        if selected.height != expected_hours:
            raise CausalCorrectionError("2024 event-window coverage differs")
        frames.append(selected)
        segments.append(slice(offset, offset + selected.height))
        offset += selected.height
    event_frame = pl.concat(frames, how="vertical")
    if (
        event_frame.height != 264
        or datetime(2024, 10, 1) in event_frame["target_timestamp"].to_list()
    ):
        raise CausalCorrectionError("causal evaluation must contain exactly 264 event hours")
    return CausalPredictionData(
        full_frame=validated,
        event_frame=event_frame,
        occurrence_ids=tuple(event.occurrence_id for event in evaluation_events),
        segments=tuple(segments),
    )


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise CausalCorrectionError("causal correction metadata is not canonical JSON") from error


def _read_canonical_json(path: Path, description: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise CausalCorrectionError(f"{description} is missing or unsafe")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise CausalCorrectionError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise CausalCorrectionError(f"{description} is not canonical JSON")
    return value


def prepare_causal_correction_inputs(
    *,
    run_dir: Path,
    config_path: Path,
    approved_ar_path: Path,
    profile: str,
) -> CausalCorrectionInputs:
    """Rebuild source truth and reuse, never fit, the existing final baseline."""

    if profile not in {"paper", "smoke"}:
        raise CausalCorrectionError("profile must be paper or smoke")
    approved_candidate = Path(approved_ar_path)
    if not approved_candidate.is_file() or approved_candidate.is_symlink():
        raise CausalCorrectionError("approved AR artifact is missing or unsafe")
    run = Path(run_dir)
    try:
        source = validate_correction_source(
            run_dir=run,
            config_path=Path(config_path),
            profile=profile,
        )
    except CorrectionSourceError as error:
        raise CausalCorrectionError(str(error)) from error
    residual_manifest = source.residual_manifest
    source_profile = source.source_profile
    residual_entry = residual_manifest["outputs"]["standardized_residuals"]
    approved = _trusted(
        load_approved_calibration(
            approved_candidate,
            current_residual_sha256=residual_entry["sha256"],
            current_config_sha256=source.source_hashes["experiment_config"],
            current_event_sha256=source.source_hashes["event_registry"],
        )
    )

    try:
        training_frame = source.load_standardized_context(approved.context, through=2023)
        final_frame = source.load_final_point_context(approved.context)
    except CorrectionSourceError as error:
        raise CausalCorrectionError(str(error)) from error
    hqrc_data = build_causal_hqrc_data(training_frame, approved, events=source.events)
    sigma_n_mw = select_latest_oof_scale(residual_manifest, approved)
    prediction = build_causal_prediction_data(
        final_frame, context=approved.context, events=source.events
    )
    return CausalCorrectionInputs(
        run_dir=run,
        source_profile=source_profile,
        approved=approved,
        hqrc_data=hqrc_data,
        training_frame=training_frame,
        prediction=prediction,
        sigma_n_mw=sigma_n_mw,
        residual_manifest=residual_manifest,
        baseline_manifest=source.baseline_manifest,
        source_paths=source.source_paths,
        source_hashes=source.source_hashes,
        residual_path=source.residual_path,
        final_members_path=source.final_members_path,
        final_point_path=source.final_point_path,
    )


def _posterior_mapping(idata: object) -> Mapping[str, object]:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or not hasattr(posterior, "data_vars"):
        raise CausalCorrectionError("posterior InferenceData is missing")
    return posterior


def generate_causal_products(
    inputs: CausalCorrectionInputs,
    idata: object,
    *,
    seed: int,
    predictive_draws: int | None = None,
) -> CausalCorrectionProducts:
    """Generate H3 trajectories with one independent stationary AR reset per event."""

    if not isinstance(inputs, CausalCorrectionInputs):
        raise TypeError("inputs must be CausalCorrectionInputs")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise CausalCorrectionError("predictive seed must be an integer")
    posterior = _posterior_mapping(idata)
    required = {"mu", "between_cholesky", "gamma", "phi", "sigma_r"}
    if not required.issubset(posterior):
        raise CausalCorrectionError("H3 posterior variables are incomplete")
    sample_count = int(posterior.sizes.get("chain", 0)) * int(posterior.sizes.get("draw", 0))
    draws = sample_count if predictive_draws is None else predictive_draws
    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise CausalCorrectionError("predictive_draws must be a positive integer")
    specifications = {
        "mu": 2,
        "between_cholesky": 3,
        "gamma": 2,
        "phi": 0,
        "sigma_r": 0,
    }
    try:
        indices = select_posterior_indices(
            posterior, specifications=specifications, draws=draws, seed=seed
        )
        gamma = posterior_values(posterior, "gamma", trailing=2, indices=indices)
        phi = posterior_values(posterior, "phi", trailing=0, indices=indices)
        sigma_r = posterior_values(posterior, "sigma_r", trailing=0, indices=indices)
    except (TypeError, ValueError) as error:
        raise CausalCorrectionError("H3 posterior shape differs") from error

    event_outputs: list[pl.DataFrame] = []
    point_metrics: list[pl.DataFrame] = []
    probabilistic_metrics: list[pl.DataFrame] = []
    event_point_values: list[np.ndarray] = []
    for event_index, (occurrence_id, segment) in enumerate(
        zip(inputs.prediction.occurrence_ids, inputs.prediction.segments, strict=True)
    ):
        event = inputs.prediction.event_frame[segment]
        holiday = 0 if str(event["holiday_type"].item(0)) == "seollal" else 1
        restriction = int(event["restriction"].item(0))
        try:
            beta = draw_new_event_correction(
                posterior,
                holiday_type=holiday,
                pooling="partial",
                draws=draws,
                seed=seed + 101 * (event_index + 1),
                restriction=restriction,
                sample_indices=indices,
            )
        except (TypeError, ValueError) as error:
            raise CausalCorrectionError("new-event coefficient draw failed") from error
        tau = event["tau_days"].to_numpy()
        hour = event["target_timestamp"].dt.hour().to_numpy()
        q = (
            beta[:, 0, None]
            + beta[:, 1, None] * tau[None, :]
            + beta[:, 2, None] * tau[None, :] ** 2
            + gamma[:, holiday, hour]
        )
        e = simulate_stationary_ar1(
            phi=phi,
            sigma=sigma_r,
            horizon=event.height,
            seed=seed + 10_000 + event_index,
        )
        baseline = event["predicted_mw"].to_numpy()
        predictive = corrected_predictive_draws(baseline, inputs.sigma_n_mw, q, e)
        point = baseline + inputs.sigma_n_mw * q.mean(axis=0)
        observed = event["observed_mw"].to_numpy()
        output = event.select(
            "target_timestamp", "occurrence_id", "holiday_type", "tau_days"
        ).with_columns(
            pl.Series("observed_mw", observed),
            pl.Series("baseline_mw", baseline),
            pl.Series("q_mean_standardized", q.mean(axis=0)),
            pl.Series("corrected_point_mw", point),
            pl.Series("predictive_draws_mw", predictive.T.tolist()),
        )
        event_outputs.append(output)
        point_metrics.append(point_metric_frame(occurrence_id, observed, point))
        probabilistic_metrics.append(
            probabilistic_metric_frame(occurrence_id, observed, predictive)
        )
        event_point_values.append(point)

    event_predictions = pl.concat(event_outputs, how="vertical")
    event_metrics = pl.concat(point_metrics, how="vertical").join(
        pl.concat(probabilistic_metrics, how="vertical"), on="event_id", how="inner"
    )
    full = inputs.prediction.full_frame
    baseline_full = full["predicted_mw"].to_numpy()
    corrected_full = baseline_full.copy()
    positions = {
        timestamp: index for index, timestamp in enumerate(full["target_timestamp"].to_list())
    }
    event_mask = np.zeros(full.height, dtype=bool)
    for timestamps, points in zip(
        (
            inputs.prediction.event_frame[segment]["target_timestamp"].to_list()
            for segment in inputs.prediction.segments
        ),
        event_point_values,
        strict=True,
    ):
        indices_in_full = np.asarray([positions[timestamp] for timestamp in timestamps])
        corrected_full[indices_in_full] = points
        event_mask[indices_in_full] = True
    if not np.array_equal(corrected_full[~event_mask], baseline_full[~event_mask]):
        raise CausalCorrectionError("HQRC changed a point outside registered event windows")
    full_predictions = full.select("target_timestamp", "observed_mw").with_columns(
        pl.Series("baseline_mw", baseline_full),
        pl.Series("corrected_point_mw", corrected_full),
        pl.Series("is_hqrc_event", event_mask),
    )
    full_metrics = point_metric_frame("full-2024", full["observed_mw"].to_numpy(), corrected_full)
    return CausalCorrectionProducts(
        event_predictions=event_predictions,
        full_period_point_predictions=full_predictions,
        event_metrics=event_metrics,
        full_period_point_metrics=full_metrics,
    )


_MODEL_OPTIONS = HQRCModelOptions(
    covariance="full",
    include_restriction=True,
    innovation="normal_ar1",
)
_PRODUCT_FILENAMES = {
    "event_predictions": "event_predictions.parquet",
    "full_period_point_predictions": "full_period_point_predictions.parquet",
    "event_metrics": "event_metrics.parquet",
    "full_period_point_metrics": "full_period_point_metrics.parquet",
}
_CHECKPOINT_ENTRIES = {
    "hqrc_data.current.json",
    ".hqrc_data.generations",
    "posterior.nc",
    "posterior.checkpoint.json",
}
_DOWNSTREAM_FILENAMES = (*_PRODUCT_FILENAMES.values(), "manifest.json")


def _publication_boundary(_: str) -> None:
    """Failure-injection hook used to prove crash and resume behavior."""


def _fsync_file(path: Path) -> None:
    _portable_fsync_file(path)


def _fsync_directory(path: Path) -> None:
    _portable_fsync_directory(path)


def _atomic_bytes(path: Path, payload: bytes) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _atomic_json(path: Path, value: Mapping[str, Any]) -> Path:
    return _atomic_bytes(path, _canonical_json(dict(value)) + b"\n")


def _read_publication_json(path: Path, description: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise CausalCorrectionError(f"{description} is missing or unsafe")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise CausalCorrectionError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value) + b"\n":
        raise CausalCorrectionError(f"{description} is not canonical JSON")
    return value


@contextmanager
def _context_lock(directory: Path):
    candidate = Path(directory)
    if candidate.exists() and candidate.is_symlink():
        raise CausalCorrectionError("correction context directory is unsafe")
    candidate.mkdir(parents=True, exist_ok=True)
    lock_path = candidate / ".correction.lock"
    if lock_path.is_symlink():
        raise CausalCorrectionError("correction context lock is unsafe")
    with exclusive_lock(lock_path):
        yield


def _sampler_contract(
    profile: str,
    *,
    seed: int,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    cores: int | None = None,
) -> dict[str, object]:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CausalCorrectionError("sampler seed must be a non-negative integer")
    if profile == "paper":
        resolved = (
            1_000 if draws is None else draws,
            1_000 if tune is None else tune,
            4 if chains is None else chains,
        )
        if resolved[0] < 1_000 or resolved[1] < 1_000 or resolved[2] != 4:
            raise CausalCorrectionError(
                "paper profile requires 4 chains and at least 1000 tune/draws"
            )
    elif profile == "smoke":
        if draws is None or tune is None or chains is None:
            raise CausalCorrectionError("smoke profile requires explicit draws, tune, and chains")
        resolved = (draws, tune, chains)
    else:
        raise CausalCorrectionError("profile must be paper or smoke")
    resolved_cores = 1 if cores is None else cores
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolved
    ):
        raise CausalCorrectionError("draws, tune, and chains must be positive integers")
    if (
        isinstance(resolved_cores, bool)
        or not isinstance(resolved_cores, int)
        or resolved_cores <= 0
    ):
        raise CausalCorrectionError("cores must be a positive integer")
    if resolved_cores > resolved[2]:
        raise CausalCorrectionError("cores must not exceed chains")
    return {
        "backend": "pymc",
        "seed": seed,
        "draws": resolved[0],
        "tune": resolved[1],
        "chains": resolved[2],
        "cores": resolved_cores,
        "target_accept": 0.99 if profile == "paper" else 0.9,
        "profile": profile,
        "init": PYMC_INITIALIZATION,
        "geometry": SAMPLER_GEOMETRY,
    }


def _approved_payload(approved: ApprovedARCalibration) -> dict[str, Any]:
    try:
        return json.loads(approved.artifact_path.read_bytes())
    except (OSError, json.JSONDecodeError) as error:
        raise CausalCorrectionError("approved AR artifact became unreadable") from error


def _context_mapping(context: EventResidualContext) -> dict[str, object]:
    return {
        "model": context.model,
        "feature_set": context.feature_set,
        "seed": context.seed,
        "split_ids": list(context.split_ids),
    }


def _input_identity(
    inputs: CausalCorrectionInputs, sampler: Mapping[str, object]
) -> dict[str, Any]:
    approved = _trusted(inputs.approved)
    approved_payload = _approved_payload(approved)
    residual_manifest_path = inputs.run_dir / "inputs/standardized_residuals_manifest.json"
    baseline_manifest_path = inputs.run_dir / "predictions/baseline_manifest.json"
    identity: dict[str, Any] = {
        "schema_version": 1,
        "evaluation": "causal-2024",
        "source_profile": inputs.source_profile,
        "sampler_profile": sampler["profile"],
        "source_hashes": dict(sorted(inputs.source_hashes.items())),
        "source_paths": {
            name: path.resolve().as_posix() for name, path in sorted(inputs.source_paths.items())
        },
        "residuals": {
            "manifest_sha256": file_sha256(residual_manifest_path),
            "artifact_sha256": file_sha256(inputs.residual_path),
        },
        "final_baseline": {
            "manifest_sha256": file_sha256(baseline_manifest_path),
            "members_sha256": file_sha256(inputs.final_members_path),
            "point_sha256": file_sha256(inputs.final_point_path),
        },
        "approved_ar": {
            "path": approved.artifact_path.resolve().as_posix(),
            "file_sha256": file_sha256(approved.artifact_path),
            "artifact_digest": approved.artifact_digest,
            "proposal_digest": approved_payload.get("proposal_digest"),
            "context": _context_mapping(approved.context),
            "event_ids": list(approved.calibration.event_ids),
        },
        "training_occurrence_ids": list(inputs.hqrc_data.occurrence_ids),
        "evaluation_occurrence_ids": list(inputs.prediction.occurrence_ids),
        "latest_complete_oof_scale": {
            "split_id": "oof-2023",
            "sigma_n_mw": inputs.sigma_n_mw,
        },
        "model": {
            "variant": "H3",
            "pooling": "partial",
            "options": {
                "covariance": "full",
                "include_restriction": True,
                "lkj_eta": 2.0,
                "between_scale_prior": 1.0,
                "innovation": "normal_ar1",
            },
            "cyclic_hour_parameterization": CYCLIC_HOUR_PARAMETERIZATION,
        },
        "sampler": dict(sampler),
        "coverage": {
            "training_rows": int(inputs.hqrc_data.observations.size),
            "final_rows": inputs.prediction.full_frame.height,
            "event_rows": inputs.prediction.event_frame.height,
            "training_timestamp_sha256": hashlib.sha256(
                inputs.training_frame["target_timestamp"].to_numpy().tobytes()
            ).hexdigest(),
            "evaluation_timestamp_sha256": hashlib.sha256(
                inputs.prediction.event_frame["target_timestamp"].to_numpy().tobytes()
            ).hexdigest(),
        },
    }
    return identity


def _posterior_metadata_matches(
    idata: object,
    inputs: CausalCorrectionInputs,
    sampler: Mapping[str, object],
) -> None:
    try:
        calibration = json.loads(idata.attrs["hqrc_calibration_json"])
        model = json.loads(idata.attrs["hqrc_model_json"])
        recorded_sampler = json.loads(idata.attrs["hqrc_sampler_json"])
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise CausalCorrectionError("posterior provenance metadata is missing") from error
    expected_calibration = {
        "artifact_digest": inputs.approved.artifact_digest,
        "residual_sha256": inputs.approved.residual_sha256,
        "config_sha256": inputs.approved.config_sha256,
        "event_sha256": inputs.approved.event_sha256,
        "context": _context_mapping(inputs.approved.context),
    }
    if any(calibration.get(key) != value for key, value in expected_calibration.items()):
        raise CausalCorrectionError("posterior approved context differs")
    if model != {
        "variant": "H3",
        "pooling": "partial",
        "options": {
            "covariance": "full",
            "include_restriction": True,
            "lkj_eta": 2.0,
            "between_scale_prior": 1.0,
            "innovation": "normal_ar1",
        },
        "cyclic_hour_parameterization": CYCLIC_HOUR_PARAMETERIZATION,
    }:
        raise CausalCorrectionError("posterior H3 model contract differs")
    if idata.attrs.get("hqrc_backend") != sampler["backend"]:
        raise CausalCorrectionError("posterior sampler backend differs")
    expected_sampler = {
        "seed": sampler["seed"],
        "draws": sampler["draws"],
        "tune": sampler["tune"],
        "chains": sampler["chains"],
        "cores": sampler["cores"],
        "target_accept": sampler["target_accept"],
        "paper_profile": sampler["profile"] == "paper",
        "init": sampler["init"],
        "geometry": sampler["geometry"],
    }
    if recorded_sampler != expected_sampler:
        raise CausalCorrectionError("posterior sampler contract differs")


def _write_posterior_checkpoint(
    directory: Path,
    idata: object,
    *,
    inputs: CausalCorrectionInputs,
    sampler: Mapping[str, object],
    identity_sha256: str,
) -> Path:
    _posterior_metadata_matches(idata, inputs, sampler)
    diagnostics = validate_inference_data(idata, paper_profile=sampler["profile"] == "paper")
    posterior_path = directory / "posterior.nc"
    descriptor, name = tempfile.mkstemp(prefix=".posterior.", suffix=".nc", dir=directory)
    os.close(descriptor)
    temporary = Path(name)
    try:
        az.to_netcdf(idata, temporary)
        _fsync_file(temporary)
        os.replace(temporary, posterior_path)
        _fsync_directory(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    checkpoint = {
        "schema_version": 1,
        "state": "POSTERIOR_COMPLETE",
        "identity_sha256": identity_sha256,
        "posterior_sha256": file_sha256(posterior_path),
        "diagnostics": {
            "max_rhat": (diagnostics.max_rhat if math.isfinite(diagnostics.max_rhat) else None),
            "min_bulk_ess": (
                diagnostics.min_bulk_ess if math.isfinite(diagnostics.min_bulk_ess) else None
            ),
            "min_tail_ess": (
                diagnostics.min_tail_ess if math.isfinite(diagnostics.min_tail_ess) else None
            ),
            "divergences": diagnostics.divergences,
        },
    }
    _atomic_json(directory / "posterior.checkpoint.json", checkpoint)
    return posterior_path


def _load_posterior_checkpoint(
    directory: Path,
    *,
    inputs: CausalCorrectionInputs,
    sampler: Mapping[str, object],
    identity_sha256: str,
):
    checkpoint = _read_publication_json(
        directory / "posterior.checkpoint.json", "posterior checkpoint"
    )
    posterior_path = directory / "posterior.nc"
    if (
        set(checkpoint)
        != {
            "schema_version",
            "state",
            "identity_sha256",
            "posterior_sha256",
            "diagnostics",
        }
        or checkpoint["schema_version"] != 1
        or checkpoint["state"] != "POSTERIOR_COMPLETE"
        or checkpoint["identity_sha256"] != identity_sha256
        or not posterior_path.is_file()
        or posterior_path.is_symlink()
        or file_sha256(posterior_path) != checkpoint["posterior_sha256"]
    ):
        raise CausalCorrectionError("posterior checkpoint differs")
    try:
        idata = az.from_netcdf(posterior_path)
    except (OSError, ValueError) as error:
        raise CausalCorrectionError("posterior checkpoint is unreadable") from error
    _posterior_metadata_matches(idata, inputs, sampler)
    diagnostics = validate_inference_data(idata, paper_profile=sampler["profile"] == "paper")
    if checkpoint["diagnostics"] != {
        "max_rhat": diagnostics.max_rhat if math.isfinite(diagnostics.max_rhat) else None,
        "min_bulk_ess": (
            diagnostics.min_bulk_ess if math.isfinite(diagnostics.min_bulk_ess) else None
        ),
        "min_tail_ess": (
            diagnostics.min_tail_ess if math.isfinite(diagnostics.min_tail_ess) else None
        ),
        "divergences": diagnostics.divergences,
    }:
        raise CausalCorrectionError("posterior checkpoint diagnostics differ")
    return idata


def _same_hqrc_data(left: HQRCData, right: HQRCData) -> bool:
    return left.occurrence_ids == right.occurrence_ids and all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in (
            "observations",
            "occurrence_index",
            "holiday_type_index",
            "tau_days",
            "hour",
            "restriction",
        )
    )


def _product_frames(products: CausalCorrectionProducts) -> dict[str, pl.DataFrame]:
    return {
        "event_predictions": products.event_predictions,
        "full_period_point_predictions": products.full_period_point_predictions,
        "event_metrics": products.event_metrics,
        "full_period_point_metrics": products.full_period_point_metrics,
    }


def _frames_exact(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    return (
        left.columns == right.columns
        and left.schema == right.schema
        and left.equals(right, null_equal=True)
    )


def _write_products(directory: Path, products: CausalCorrectionProducts) -> None:
    frames = _product_frames(products)
    temporaries: dict[str, Path] = {}
    try:
        for name, frame in frames.items():
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{name}.", suffix=".parquet", dir=directory
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            frame.write_parquet(temporary)
            _fsync_file(temporary)
            temporaries[name] = temporary
        for name in frames:
            os.replace(temporaries.pop(name), directory / _PRODUCT_FILENAMES[name])
            _publication_boundary(f"{name}-published")
        _fsync_directory(directory)
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def _require_real_file(path: Path, description: str) -> None:
    try:
        identity = path.lstat()
    except OSError as error:
        raise CausalCorrectionError(f"{description} is missing or unsafe") from error
    if not stat.S_ISREG(identity.st_mode):
        raise CausalCorrectionError(f"{description} is missing or unsafe")


def _current_hqrc_generation_paths(directory: Path) -> tuple[Path, Path, Path]:
    pointer_path = directory / "hqrc_data.current.json"
    pointer = _read_publication_json(pointer_path, "HQRCData pointer")
    generation_directory = directory / ".hqrc_data.generations"
    try:
        namespace_identity = generation_directory.lstat()
    except OSError as error:
        raise CausalCorrectionError("HQRCData generation namespace is unsafe") from error
    if not stat.S_ISDIR(namespace_identity.st_mode):
        raise CausalCorrectionError("HQRCData generation namespace is unsafe")

    targets: list[Path] = []
    for key in ("npz", "metadata"):
        relative = pointer.get(key)
        if (
            not isinstance(relative, str)
            or not relative
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise CausalCorrectionError("HQRCData pointer generation path is unsafe")
        target = directory / relative
        if target.parent != generation_directory:
            raise CausalCorrectionError("HQRCData pointer generation path is unsafe")
        _require_real_file(target, f"HQRCData current {key}")
        targets.append(target)
    if targets[0] == targets[1]:
        raise CausalCorrectionError("HQRCData current generation paths collide")
    try:
        generation_entries = tuple(generation_directory.iterdir())
    except OSError as error:
        raise CausalCorrectionError("HQRCData generation namespace is unreadable") from error
    if {path.name for path in generation_entries} != {path.name for path in targets}:
        raise CausalCorrectionError("HQRCData generation namespace contains unknown entries")
    for path in generation_entries:
        _require_real_file(path, "HQRCData generation entry")
    return pointer_path, targets[0], targets[1]


def _canonical_output_records(directory: Path) -> dict[str, dict[str, str]]:
    pointer_path, generation_npz, generation_metadata = _current_hqrc_generation_paths(directory)
    paths = {
        "hqrc_data_pointer": pointer_path,
        "hqrc_data_npz": generation_npz,
        "hqrc_data_metadata": generation_metadata,
        "posterior": directory / "posterior.nc",
        "posterior_checkpoint": directory / "posterior.checkpoint.json",
        **{name: directory / filename for name, filename in _PRODUCT_FILENAMES.items()},
    }
    records: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        _require_real_file(path, f"correction output {name}")
        records[name] = {
            "path": path.relative_to(directory).as_posix(),
            "sha256": file_sha256(path),
        }
    return records


def _manifest_payload(
    directory: Path,
    *,
    identity: Mapping[str, Any],
    products: CausalCorrectionProducts,
) -> dict[str, Any]:
    outputs = _canonical_output_records(directory)
    rows = {
        "event_predictions": products.event_predictions.height,
        "full_period_point_predictions": products.full_period_point_predictions.height,
        "event_metrics": products.event_metrics.height,
        "full_period_point_metrics": products.full_period_point_metrics.height,
    }
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "state": "COMPLETE",
        "identity": dict(identity),
        "outputs": outputs,
        "rows": rows,
    }
    return {
        **unsigned,
        "manifest_digest": hashlib.sha256(_canonical_json(unsigned)).hexdigest(),
    }


def _result(directory: Path, *, reused: bool, sampler_fit_count: int) -> CausalCorrectionResult:
    return CausalCorrectionResult(
        output_dir=directory,
        manifest_path=directory / "manifest.json",
        posterior_path=directory / "posterior.nc",
        event_predictions_path=directory / _PRODUCT_FILENAMES["event_predictions"],
        full_period_point_predictions_path=directory
        / _PRODUCT_FILENAMES["full_period_point_predictions"],
        event_metrics_path=directory / _PRODUCT_FILENAMES["event_metrics"],
        full_period_point_metrics_path=directory / _PRODUCT_FILENAMES["full_period_point_metrics"],
        sampler_fit_count=sampler_fit_count,
        reused=reused,
    )


def _validate_complete(
    directory: Path,
    *,
    identity: Mapping[str, Any],
    inputs: CausalCorrectionInputs,
    sampler: Mapping[str, object],
) -> CausalCorrectionResult:
    allowed_entries = {
        ".correction.lock",
        ".hqrc_data.generations",
        "hqrc_data.current.json",
        "posterior.nc",
        "posterior.checkpoint.json",
        *_PRODUCT_FILENAMES.values(),
        "manifest.json",
        "COMPLETE",
    }
    if {path.name for path in directory.iterdir()} != allowed_entries:
        raise CausalCorrectionError("completed correction directory contains unknown entries")
    manifest = _read_publication_json(directory / "manifest.json", "correction manifest")
    complete = _read_publication_json(directory / "COMPLETE", "correction completion marker")
    unsigned = {
        key: manifest[key]
        for key in ("schema_version", "state", "identity", "outputs", "rows")
        if key in manifest
    }
    if (
        set(manifest)
        != {"schema_version", "state", "identity", "outputs", "rows", "manifest_digest"}
        or manifest.get("schema_version") != 1
        or manifest.get("state") != "COMPLETE"
        or manifest.get("identity") != dict(identity)
        or manifest.get("manifest_digest") != hashlib.sha256(_canonical_json(unsigned)).hexdigest()
        or complete
        != {
            "manifest_sha256": file_sha256(directory / "manifest.json"),
            "state": "COMPLETE",
        }
    ):
        raise CausalCorrectionError("completed correction manifest differs")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or outputs != _canonical_output_records(directory):
        raise CausalCorrectionError("correction output manifest differs")
    data, settings = load_hqrc_data(directory / "hqrc_data.npz")
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    if not _same_hqrc_data(data, inputs.hqrc_data) or settings != {
        "identity_sha256": identity_sha256,
        "model": identity["model"],
    }:
        raise CausalCorrectionError("published HQRCData differs")
    idata = _load_posterior_checkpoint(
        directory,
        inputs=inputs,
        sampler=sampler,
        identity_sha256=identity_sha256,
    )
    recorded_sampler = manifest["identity"].get("sampler")
    if not isinstance(recorded_sampler, dict) or recorded_sampler != dict(sampler):
        raise CausalCorrectionError("published sampler identity differs")
    expected = generate_causal_products(
        inputs,
        idata,
        seed=int(recorded_sampler["seed"]),
        predictive_draws=int(recorded_sampler["draws"]) * int(recorded_sampler["chains"]),
    )
    expected_frames = _product_frames(expected)
    if manifest.get("rows") != {name: frame.height for name, frame in expected_frames.items()}:
        raise CausalCorrectionError("published correction semantics differ")
    for name, expected_frame in expected_frames.items():
        try:
            published_frame = pl.read_parquet(directory / _PRODUCT_FILENAMES[name])
        except (OSError, pl.exceptions.PolarsError) as error:
            raise CausalCorrectionError("published correction semantics differ") from error
        if not _frames_exact(published_frame, expected_frame):
            raise CausalCorrectionError("published correction semantics differ")
    return _result(directory, reused=True, sampler_fit_count=0)


def _load_resumable_checkpoint(
    directory: Path,
    *,
    identity: Mapping[str, Any],
    identity_sha256: str,
    inputs: CausalCorrectionInputs,
    sampler: Mapping[str, object],
):
    entries = {path.name: path for path in directory.iterdir() if path.name != ".correction.lock"}
    allowed = _CHECKPOINT_ENTRIES | set(_DOWNSTREAM_FILENAMES)
    if set(entries) - allowed or not _CHECKPOINT_ENTRIES.issubset(entries):
        raise CausalCorrectionError("unsafe partial correction publication")
    for name, path in entries.items():
        if path.is_symlink():
            raise CausalCorrectionError("unsafe partial correction publication")
        if name == ".hqrc_data.generations":
            if not path.is_dir():
                raise CausalCorrectionError("unsafe partial correction publication")
        elif not path.is_file():
            raise CausalCorrectionError("unsafe partial correction publication")
    _current_hqrc_generation_paths(directory)
    try:
        data, settings = load_hqrc_data(directory / "hqrc_data.npz")
    except (OSError, ValueError) as error:
        raise CausalCorrectionError("checkpoint HQRCData differs") from error
    if not _same_hqrc_data(data, inputs.hqrc_data) or settings != {
        "identity_sha256": identity_sha256,
        "model": identity["model"],
    }:
        raise CausalCorrectionError("checkpoint HQRCData differs")
    idata = _load_posterior_checkpoint(
        directory,
        inputs=inputs,
        sampler=sampler,
        identity_sha256=identity_sha256,
    )
    for filename in _DOWNSTREAM_FILENAMES:
        path = entries.get(filename)
        if path is not None:
            path.unlink()
    _fsync_directory(directory)
    return idata


def fit_causal_2024_correction(
    *,
    run_dir: Path,
    config_path: Path,
    approved_ar_path: Path,
    sampler_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    output_root: Path | None = None,
) -> CausalCorrectionResult:
    """Fit or strictly reuse the one frozen H3 partial-pooling causal comparison."""

    sampler = _sampler_contract(
        profile, seed=sampler_seed, draws=draws, tune=tune, chains=chains, cores=cores
    )
    inputs = prepare_causal_correction_inputs(
        run_dir=Path(run_dir),
        config_path=Path(config_path),
        approved_ar_path=Path(approved_ar_path),
        profile=profile,
    )
    if inputs.source_profile not in {"paper", "smoke"} or (
        profile == "paper" and inputs.source_profile != "paper"
    ):
        raise CausalCorrectionError("source and sampler profiles are incompatible")
    context = inputs.approved.context
    root = Path(run_dir) if output_root is None else Path(output_root)
    directory = (
        root
        / "corrections/causal-2024"
        / context.model
        / context.feature_set
        / f"seed-{context.seed}"
        / profile
        / f"sampler-seed-{sampler_seed}"
        / f"init-{sampler['init']}-geometry-{sampler['geometry']}"
        / (
            f"draws-{sampler['draws']}-tune-{sampler['tune']}-chains-{sampler['chains']}-cores-{sampler['cores']}"
        )
    )
    identity = _input_identity(inputs, sampler)
    identity_sha256 = hashlib.sha256(_canonical_json(identity)).hexdigest()
    with _context_lock(directory):
        complete_path = directory / "COMPLETE"
        manifest_path = directory / "manifest.json"
        complete_exists = complete_path.exists() or complete_path.is_symlink()
        manifest_exists = manifest_path.exists() or manifest_path.is_symlink()
        if complete_exists:
            if not manifest_exists:
                raise CausalCorrectionError("partial correction publication")
            return _validate_complete(directory, identity=identity, inputs=inputs, sampler=sampler)

        existing = {path.name for path in directory.iterdir() if path.name != ".correction.lock"}
        if existing:
            idata = _load_resumable_checkpoint(
                directory,
                identity=identity,
                identity_sha256=identity_sha256,
                inputs=inputs,
                sampler=sampler,
            )
            sampler_fit_count = 0
        else:
            write_hqrc_data(
                directory / "hqrc_data.npz",
                inputs.hqrc_data,
                settings={"identity_sha256": identity_sha256, "model": identity["model"]},
            )
            _publication_boundary("hqrc-data-published")
            idata = sample_hqrc(
                inputs.hqrc_data,
                inputs.approved,
                variant="H3",
                pooling="partial",
                options=_MODEL_OPTIONS,
                draws=int(sampler["draws"]),
                tune=int(sampler["tune"]),
                chains=int(sampler["chains"]),
                cores=int(sampler["cores"]),
                seed=int(sampler["seed"]),
                backend="pymc",
                paper_profile=profile == "paper",
            )
            _write_posterior_checkpoint(
                directory,
                idata,
                inputs=inputs,
                sampler=sampler,
                identity_sha256=identity_sha256,
            )
            sampler_fit_count = 1
        _publication_boundary("posterior-checkpointed")
        products = generate_causal_products(
            inputs,
            idata,
            seed=sampler_seed,
            predictive_draws=int(sampler["draws"]) * int(sampler["chains"]),
        )
        _write_products(directory, products)
        manifest = _manifest_payload(directory, identity=identity, products=products)
        _atomic_json(directory / "manifest.json", manifest)
        _publication_boundary("manifest-published")
        _atomic_json(
            directory / "COMPLETE",
            {
                "manifest_sha256": file_sha256(directory / "manifest.json"),
                "state": "COMPLETE",
            },
        )
        _publication_boundary("complete-published")
        return _result(directory, reused=False, sampler_fit_count=sampler_fit_count)


__all__ = [
    "CausalCorrectionError",
    "CausalCorrectionInputs",
    "CausalCorrectionProducts",
    "CausalCorrectionResult",
    "CausalPredictionData",
    "build_causal_hqrc_data",
    "build_causal_prediction_data",
    "generate_causal_products",
    "fit_causal_2024_correction",
    "prepare_causal_correction_inputs",
    "select_latest_oof_scale",
]
