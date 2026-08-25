"""Reviewer-scoped H0/H1/H2 LOEO evaluation for the accepted HQT model."""

from __future__ import annotations

import json
import math
import os
import stat
import tempfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import polars as pl

from hqrc_v3._loeo_contract import derive_loeo_seed
from hqrc_v3.baselines.config import MODEL_NAMES
from hqrc_v3.bayes.legacy_hqt import (
    LEGACY_HQT_MODEL_SPEC,
    new_event_correction_draws,
    sample_legacy_hqt,
)
from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.bayes.samplers import PYMC_INITIALIZATION_CHOICES, validate_inference_data
from hqrc_v3.correction_source import ValidatedCorrectionSource, validate_correction_source
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.diagnostics.loeo import (
    LOEOFold,
    LOEOHeldOut,
    LOEOPublication,
    load_loeo_event,
    load_loeo_fold,
    load_loeo_universe,
    publish_loeo_universe,
)
from hqrc_v3.evaluation.metrics import point_metric_frame
from hqrc_v3.provenance import file_sha256

_HOLIDAY_INDEX = {"seollal": 0, "chuseok": 1}
_CORRECTIONS = ("H0", "H1", "H2", "H2-taper")
_ARTIFACT_FILES = (
    "input_identity.json",
    "posterior.nc",
    "hourly_predictions.parquet",
    "metrics.parquet",
)
_PARTIAL_ARTIFACT_FILES = ("input_identity.json", "posterior.nc")
_COMPLETE_NAMESPACE = frozenset((*_ARTIFACT_FILES, "manifest.json", "COMPLETE"))


class LegacyHQTError(ValueError):
    """Raised when an original-paper HQT run violates its frozen contract."""


@dataclass(frozen=True, slots=True)
class LegacyHQTContextResult:
    context: EventResidualContext
    output_dir: Path
    occurrence_ids: tuple[str, ...]
    sampler_fit_count: int
    reused_fold_count: int


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise LegacyHQTError("legacy HQT metadata is not canonical JSON") from error


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
        raise LegacyHQTError(f"{description} is missing or unsafe") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise LegacyHQTError(f"{description} is missing or unsafe")


def _read_identity(path: Path, description: str = "legacy HQT identity") -> dict[str, Any]:
    _require_real_file(path, description)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise LegacyHQTError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise LegacyHQTError(f"{description} is not canonical")
    return value


def build_legacy_hqt_data_from_frame(
    frame: pl.DataFrame, *, occurrence_ids: tuple[str, ...]
) -> HQRCData:
    """Convert an ordered residual frame for either physical or causal HQT training."""

    required = {
        "holiday_type",
        "hour",
        "occurrence_id",
        "restriction",
        "standardized_residual",
        "tau_days",
    }
    if not required.issubset(frame.columns) or not occurrence_ids:
        raise LegacyHQTError("legacy HQT training frame schema differs")
    observed_order = tuple(frame["occurrence_id"].unique(maintain_order=True).to_list())
    if observed_order != occurrence_ids:
        raise LegacyHQTError("legacy HQT occurrence ordering differs")
    event_index = {event_id: index for index, event_id in enumerate(occurrence_ids)}
    try:
        return HQRCData(
            observations=frame["standardized_residual"].to_numpy(),
            occurrence_index=np.asarray(
                [event_index[value] for value in frame["occurrence_id"].to_list()],
                dtype=np.int64,
            ),
            holiday_type_index=np.asarray(
                [_HOLIDAY_INDEX[value] for value in frame["holiday_type"].to_list()],
                dtype=np.int64,
            ),
            tau_days=frame["tau_days"].to_numpy(),
            hour=frame["hour"].to_numpy(),
            restriction=frame["restriction"].to_numpy(),
            occurrence_ids=occurrence_ids,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LegacyHQTError("legacy HQT training frame cannot form model data") from error


def build_legacy_hqt_data(fold: LOEOFold) -> HQRCData:
    """Convert one physical nine-event LOEO fold without any AR calibration."""

    if not isinstance(fold, LOEOFold) or fold.causal is not False:
        raise LegacyHQTError("legacy HQT requires a physical causal=false LOEO fold")
    if len(fold.occurrence_ids) != 9 or fold.held_out_occurrence_id in fold.occurrence_ids:
        raise LegacyHQTError("legacy HQT LOEO training must contain exactly nine other events")
    required = {
        "causal",
        "feature_set",
        "holiday_type",
        "hour",
        "model",
        "occurrence_id",
        "restriction",
        "seed",
        "standardized_residual",
        "tau_days",
    }
    if not required.issubset(fold.frame.columns):
        raise LegacyHQTError("legacy HQT physical fold schema differs")
    if fold.frame["causal"].unique().to_list() != [False]:
        raise LegacyHQTError("legacy HQT training rows must be causal=false")
    if tuple(fold.frame["occurrence_id"].unique(maintain_order=True).to_list()) != (
        fold.occurrence_ids
    ):
        raise LegacyHQTError("legacy HQT occurrence ordering differs")
    for column, expected in (
        ("model", fold.context.model),
        ("feature_set", fold.context.feature_set),
        ("seed", fold.context.seed),
    ):
        if fold.frame[column].unique().to_list() != [expected]:
            raise LegacyHQTError(f"legacy HQT fold {column} context differs")

    return build_legacy_hqt_data_from_frame(fold.frame, occurrence_ids=fold.occurrence_ids)


def event_type_mean_shift_mw(fold: LOEOFold, holiday_type: str) -> float:
    """Return the accepted paper's H1 raw-MW event-type residual mean."""

    if holiday_type not in _HOLIDAY_INDEX:
        raise LegacyHQTError("holiday type must be seollal or chuseok")
    if "residual_mw" not in fold.frame.columns:
        raise LegacyHQTError("legacy H1 requires raw-MW training residuals")
    selected = fold.frame.filter(pl.col("holiday_type") == holiday_type)["residual_mw"]
    if selected.len() == 0:
        raise LegacyHQTError("legacy H1 has no same-type training events")
    shift = float(selected.mean())
    if not math.isfinite(shift):
        raise LegacyHQTError("legacy H1 shift is non-finite")
    return shift


def cosine_boundary_taper(length: int, *, edge_hours: int = 24) -> np.ndarray:
    """Taper only the one-day buffers added around the official holiday sequence."""

    if isinstance(length, bool) or not isinstance(length, int) or length <= 2:
        raise LegacyHQTError("taper length must be an integer greater than two")
    if (
        isinstance(edge_hours, bool)
        or not isinstance(edge_hours, int)
        or edge_hours <= 0
        or 2 * edge_hours >= length
    ):
        raise LegacyHQTError("taper edge must leave a non-empty untapered center")
    phase = np.arange(edge_hours, dtype=float) / edge_hours
    ramp = 0.5 * (1.0 - np.cos(np.pi * phase))
    weight = np.ones(length, dtype=float)
    weight[:edge_hours] = ramp
    weight[-edge_hours:] = ramp[::-1]
    return weight


def _evaluation_scale(held_out: LOEOHeldOut) -> float:
    required = {"causal", "occurrence_id", "sigma_n_mw"}
    if not required.issubset(held_out.frame.columns) or held_out.frame.is_empty():
        raise LegacyHQTError("legacy HQT held-out event schema differs")
    if held_out.frame["causal"].unique().to_list() != [False]:
        raise LegacyHQTError("legacy HQT held-out event must be causal=false")
    scales = held_out.frame["sigma_n_mw"].unique()
    if scales.len() != 1:
        raise LegacyHQTError("legacy HQT held-out scale is not unique")
    scale = float(scales.item())
    if not math.isfinite(scale) or scale <= 0:
        raise LegacyHQTError("legacy HQT held-out scale must be positive")
    return scale


def _sampler_contract(
    *,
    profile: str,
    root_seed: int,
    held_out_occurrence_id: str,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    cores: int | None,
    init: str | None,
    target_accept: float | None,
) -> dict[str, object]:
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
            raise LegacyHQTError(
                "paper HQT requires at least 4 chains, at least 1000 tune/draws, "
                "and target_accept=0.99"
            )
    elif profile == "smoke":
        if draws is None or tune is None or chains is None:
            raise LegacyHQTError("smoke HQT requires explicit draws, tune, and chains")
        resolved_draws, resolved_tune, resolved_chains = draws, tune, chains
        resolved_target = 0.9 if target_accept is None else target_accept
    else:
        raise LegacyHQTError("profile must be paper or smoke")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (resolved_draws, resolved_tune, resolved_chains)
    ):
        raise LegacyHQTError("draws, tune, and chains must be positive integers")
    resolved_cores = min(resolved_chains, os.cpu_count() or 1) if cores is None else cores
    if (
        isinstance(resolved_cores, bool)
        or not isinstance(resolved_cores, int)
        or resolved_cores <= 0
        or resolved_cores > resolved_chains
    ):
        raise LegacyHQTError("cores must be a positive integer no greater than chains")
    resolved_init = "adapt_diag" if init is None else init
    if resolved_init not in PYMC_INITIALIZATION_CHOICES:
        raise LegacyHQTError("legacy HQT sampler initialization is unsupported")
    if (
        isinstance(resolved_target, bool)
        or not isinstance(resolved_target, (int, float))
        or not math.isfinite(float(resolved_target))
        or not 0 < float(resolved_target) < 1
    ):
        raise LegacyHQTError("legacy HQT target_accept is invalid")
    return {
        "chains": resolved_chains,
        "cores": resolved_cores,
        "draws": resolved_draws,
        "init": resolved_init,
        "profile": profile,
        "root_seed": root_seed,
        "seed": derive_loeo_seed(root_seed, f"legacy-hqt:{held_out_occurrence_id}"),
        "target_accept": float(resolved_target),
        "tune": resolved_tune,
    }


def _fold_identity(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    fold: LOEOFold,
    sampler: Mapping[str, object],
) -> dict[str, object]:
    return {
        "context": {
            "feature_set": publication.context.feature_set,
            "model": publication.context.model,
            "seed": publication.context.seed,
            "split_ids": list(publication.context.split_ids),
        },
        "evaluation": "retrospective-10-event-loeo",
        "fold_sha256": fold.residual_sha256,
        "held_out_occurrence_id": fold.held_out_occurrence_id,
        "loeo_universe_sha256": publication.universe_sha256,
        "model_spec": LEGACY_HQT_MODEL_SPEC,
        "sampler": dict(sampler),
        "schema_version": 1,
        "source": {
            "baseline_manifest_sha256": file_sha256(source.baseline_manifest_path),
            "residual_sha256": source.residual_sha256,
        },
    }


def _fold_products(
    fold: LOEOFold,
    held_out: LOEOHeldOut,
    idata: az.InferenceData,
    *,
    prediction_seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    frame = held_out.frame.sort("target_timestamp")
    required = {
        "holiday_type",
        "observed_mw",
        "predicted_mw",
        "target_timestamp",
        "tau_days",
    }
    if not required.issubset(frame.columns):
        raise LegacyHQTError("legacy HQT held-out product schema differs")
    holidays = frame["holiday_type"].unique().to_list()
    if len(holidays) != 1 or holidays[0] not in _HOLIDAY_INDEX:
        raise LegacyHQTError("legacy HQT held-out holiday type differs")
    holiday_type = str(holidays[0])
    observed = frame["observed_mw"].to_numpy()
    baseline = frame["predicted_mw"].to_numpy()
    sigma_eval = _evaluation_scale(held_out)
    h1_shift = event_type_mean_shift_mw(fold, holiday_type)
    correction_draws = new_event_correction_draws(
        idata,
        holiday_type_index=_HOLIDAY_INDEX[holiday_type],
        tau_days=frame["tau_days"].to_numpy(),
        seed=prediction_seed,
    )
    standardized_correction = correction_draws.mean(axis=0)
    h1 = baseline + h1_shift
    h2 = baseline + sigma_eval * standardized_correction
    taper = cosine_boundary_taper(frame.height)
    h2_taper = baseline + sigma_eval * standardized_correction * taper
    hourly = pl.DataFrame(
        {
            "occurrence_id": [held_out.occurrence_id] * frame.height,
            "holiday_type": [holiday_type] * frame.height,
            "target_timestamp": frame["target_timestamp"],
            "observed_mw": observed,
            "H0_mw": baseline,
            "H1_mw": h1,
            "H2_mw": h2,
            "H2_taper_mw": h2_taper,
            "h1_shift_mw": [h1_shift] * frame.height,
            "h2_standardized_correction": standardized_correction,
            "taper_weight": taper,
            "sigma_n_mw": [sigma_eval] * frame.height,
        }
    )
    metrics = []
    for correction, forecast in zip(_CORRECTIONS, (baseline, h1, h2, h2_taper)):
        metrics.append(
            point_metric_frame(held_out.occurrence_id, observed, forecast).with_columns(
                pl.lit(holiday_type).alias("holiday_type"),
                pl.lit(correction).alias("correction"),
            )
        )
    return hourly, pl.concat(metrics, how="vertical")


def _validate_posterior(
    idata: az.InferenceData,
    *,
    identity_sha256: str,
    paper_profile: bool,
) -> None:
    try:
        model_spec = json.loads(idata.attrs.get("legacy_hqt_model_json", "null"))
    except (TypeError, json.JSONDecodeError) as error:
        raise LegacyHQTError("resumable legacy HQT posterior model is unreadable") from error
    if model_spec != LEGACY_HQT_MODEL_SPEC:
        raise LegacyHQTError("resumable legacy HQT posterior model differs")
    if idata.attrs.get("legacy_hqt_loeo_identity_sha256") != identity_sha256:
        raise LegacyHQTError("resumable legacy HQT posterior identity differs")
    validate_inference_data(idata, paper_profile=paper_profile)


def _manifest_files(
    manifest: Mapping[str, object],
    identity: Mapping[str, object],
    *,
    completed: bool,
    expected_prediction_seed: int,
) -> frozenset[str]:
    files = manifest.get("files")
    file_names = frozenset(files) if isinstance(files, dict) else frozenset()
    partial_names = frozenset(_PARTIAL_ARTIFACT_FILES)
    full_names = frozenset(_ARTIFACT_FILES)
    has_full_manifest = file_names == full_names
    expected_keys = {"files", "identity_sha256", "schema_version"}
    if has_full_manifest:
        expected_keys.add("prediction_seed")
    if (
        set(manifest) != expected_keys
        or manifest.get("schema_version") != 1
        or manifest.get("identity_sha256") != _json_sha256(identity)
        or not isinstance(files, dict)
        or (completed and not has_full_manifest)
        or (not completed and file_names not in {partial_names, full_names})
        or (has_full_manifest and manifest.get("prediction_seed") != expected_prediction_seed)
    ):
        state = "completed" if completed else "incomplete"
        raise LegacyHQTError(f"{state} legacy HQT fold manifest differs")
    return file_names


def _checkpoint_entries(fold_dir: Path) -> set[str]:
    try:
        mode = fold_dir.lstat().st_mode
        entries = tuple(fold_dir.iterdir())
    except OSError as error:
        raise LegacyHQTError("legacy HQT fold namespace is unsafe") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise LegacyHQTError("legacy HQT fold namespace is unsafe")
    for entry in entries:
        _require_real_file(entry, f"legacy HQT fold checkpoint {entry.name}")
    return {entry.name for entry in entries}


def _load_complete_fold(
    fold_dir: Path, identity: Mapping[str, object], *, prediction_seed: int
) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    complete_path = fold_dir / "COMPLETE"
    if not os.path.lexists(complete_path):
        return None
    try:
        entries = _checkpoint_entries(fold_dir)
        if entries != _COMPLETE_NAMESPACE:
            raise LegacyHQTError("completed legacy HQT fold namespace differs")
        stored_identity = _read_identity(
            fold_dir / "input_identity.json", "completed legacy HQT fold identity"
        )
        if stored_identity != dict(identity):
            raise LegacyHQTError("completed legacy HQT fold identity differs")
        manifest_path = fold_dir / "manifest.json"
        manifest = _read_identity(manifest_path, "completed legacy HQT fold manifest")
        complete = _read_identity(complete_path, "completed legacy HQT completion marker")
        files = manifest["files"]
        file_names = _manifest_files(
            manifest,
            identity,
            completed=True,
            expected_prediction_seed=prediction_seed,
        )
        if complete != {"manifest_sha256": file_sha256(manifest_path), "schema_version": 1}:
            raise LegacyHQTError("completed legacy HQT fold completion marker differs")
        for name in file_names:
            path = fold_dir / name
            _require_real_file(path, f"completed legacy HQT fold artifact {name}")
            if files[name] != file_sha256(path):  # type: ignore[index]
                raise LegacyHQTError(f"completed legacy HQT fold artifact digest differs: {name}")
        idata = az.from_netcdf(fold_dir / "posterior.nc")
        _validate_posterior(
            idata,
            identity_sha256=_json_sha256(identity),
            paper_profile=identity["sampler"]["profile"] == "paper",  # type: ignore[index]
        )
        hourly = pl.read_parquet(fold_dir / "hourly_predictions.parquet")
        metrics = pl.read_parquet(fold_dir / "metrics.parquet")
    except LegacyHQTError:
        raise
    except (OSError, KeyError, TypeError, ValueError, pl.exceptions.PolarsError) as error:
        raise LegacyHQTError("completed legacy HQT fold checkpoint is unreadable") from error
    return hourly, metrics


def _load_resumable_posterior(
    fold_dir: Path,
    *,
    identity: Mapping[str, object],
    paper_profile: bool,
    prediction_seed: int,
) -> az.InferenceData | None:
    posterior_path = fold_dir / "posterior.nc"
    manifest_path = fold_dir / "manifest.json"
    posterior_exists = os.path.lexists(posterior_path)
    manifest_exists = os.path.lexists(manifest_path)
    if not posterior_exists:
        if manifest_exists:
            raise LegacyHQTError("incomplete legacy HQT manifest exists without a posterior")
        if _checkpoint_entries(fold_dir) != {"input_identity.json"}:
            raise LegacyHQTError("incomplete legacy HQT checkpoint state differs")
        return None
    _require_real_file(posterior_path, "incomplete legacy HQT posterior")
    if not manifest_exists:
        raise LegacyHQTError("incomplete legacy HQT posterior manifest is missing")
    manifest = _read_identity(manifest_path, "incomplete legacy HQT manifest")
    file_names = _manifest_files(
        manifest,
        identity,
        completed=False,
        expected_prediction_seed=prediction_seed,
    )
    files = manifest["files"]
    for name in file_names:
        path = fold_dir / name
        _require_real_file(path, f"incomplete legacy HQT artifact {name}")
        if files[name] != file_sha256(path):  # type: ignore[index]
            raise LegacyHQTError(f"incomplete legacy HQT artifact digest differs: {name}")
    try:
        idata = az.from_netcdf(posterior_path)
    except (OSError, ValueError) as error:
        raise LegacyHQTError("incomplete legacy HQT posterior is unreadable") from error
    _validate_posterior(
        idata,
        identity_sha256=_json_sha256(identity),
        paper_profile=paper_profile,
    )
    return idata


def _fit_fold(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    *,
    held_out_occurrence_id: str,
    sampler: Mapping[str, object],
    fold_dir: Path,
) -> tuple[pl.DataFrame, pl.DataFrame, bool]:
    trusted = load_loeo_universe(source, publication.context, output_dir=publication.output_dir)
    if trusted != publication:
        raise LegacyHQTError("legacy HQT LOEO publication changed")
    fold = load_loeo_fold(
        source,
        publication.context,
        output_dir=publication.output_dir,
        held_out_occurrence_id=held_out_occurrence_id,
    )
    held_out = load_loeo_event(
        source,
        publication.context,
        output_dir=publication.output_dir,
        occurrence_id=held_out_occurrence_id,
    )
    identity = _fold_identity(source, publication, fold, sampler)
    prediction_seed = derive_loeo_seed(
        int(sampler["root_seed"]), f"legacy-hqt-predictive:{held_out_occurrence_id}"
    )
    entries: set[str] = set()
    if os.path.lexists(fold_dir):
        entries = _checkpoint_entries(fold_dir)
        completed = _load_complete_fold(
            fold_dir,
            identity,
            prediction_seed=prediction_seed,
        )
        if completed is not None:
            return completed[0], completed[1], True
        if not entries <= _COMPLETE_NAMESPACE - {"COMPLETE"}:
            raise LegacyHQTError("incomplete legacy HQT fold namespace differs")
    else:
        fold_dir.mkdir(parents=True, exist_ok=False)
    identity_path = fold_dir / "input_identity.json"
    if entries and "input_identity.json" not in entries:
        raise LegacyHQTError("incomplete legacy HQT identity is missing beside checkpoint state")
    if "input_identity.json" in entries:
        if _read_identity(identity_path, "incomplete legacy HQT identity") != identity:
            raise LegacyHQTError("incomplete legacy HQT fold identity differs")
    else:
        _write_atomic(identity_path, _canonical_json(identity))
    identity_digest = _json_sha256(identity)
    idata = _load_resumable_posterior(
        fold_dir,
        identity=identity,
        paper_profile=sampler["profile"] == "paper",
        prediction_seed=prediction_seed,
    )
    fitted = idata is None
    if fitted:
        data = build_legacy_hqt_data(fold)
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
        idata.attrs["legacy_hqt_loeo_identity_sha256"] = identity_digest
        posterior_path = fold_dir / "posterior.nc"
        _write_netcdf_atomic(idata, posterior_path)
        partial_manifest = {
            "files": {name: file_sha256(fold_dir / name) for name in _PARTIAL_ARTIFACT_FILES},
            "identity_sha256": identity_digest,
            "schema_version": 1,
        }
        _write_atomic(fold_dir / "manifest.json", _canonical_json(partial_manifest))
    assert idata is not None
    posterior_path = fold_dir / "posterior.nc"
    hourly, metrics = _fold_products(fold, held_out, idata, prediction_seed=prediction_seed)
    hourly_path = fold_dir / "hourly_predictions.parquet"
    metrics_path = fold_dir / "metrics.parquet"
    _write_parquet_atomic(hourly, hourly_path)
    _write_parquet_atomic(metrics, metrics_path)
    manifest = {
        "files": {name: file_sha256(fold_dir / name) for name in _ARTIFACT_FILES},
        "identity_sha256": identity_digest,
        "prediction_seed": prediction_seed,
        "schema_version": 1,
    }
    manifest_path = fold_dir / "manifest.json"
    _write_atomic(manifest_path, _canonical_json(manifest))
    _write_atomic(
        fold_dir / "COMPLETE",
        _canonical_json({"manifest_sha256": file_sha256(manifest_path), "schema_version": 1}),
    )
    return hourly, metrics, not fitted


def _scale_stability(
    source: ValidatedCorrectionSource, context: EventResidualContext
) -> pl.DataFrame:
    residuals = source.load_standardized_context(context, through=2023)
    fit_scales = residuals.filter(pl.col("split_id") == "oof-2023")["sigma_n_mw"].unique()
    if fit_scales.len() != 1:
        raise LegacyHQTError("OOF-2023 fitting scale is not unique")
    sigma_fit = float(fit_scales.item())
    final = source.load_final_point_context(context)
    event_dates = {
        event.window_start + timedelta(days=offset)
        for event in source.events
        if event.central_date.year == 2024
        for offset in range((event.window_end - event.window_start).days + 1)
    }
    non_event = final.filter(~pl.col("target_timestamp").dt.date().is_in(sorted(event_dates)))
    error = non_event["observed_mw"].to_numpy() - non_event["predicted_mw"].to_numpy()
    sigma_2024 = float(np.sqrt(np.mean(error**2)))
    if not math.isfinite(sigma_2024) or sigma_2024 <= 0:
        raise LegacyHQTError("2024 non-event residual scale is invalid")
    return pl.DataFrame(
        {
            "model": [context.model],
            "feature_set": [context.feature_set],
            "seed": [context.seed],
            "sigma_fit_oof_2023_mw": [sigma_fit],
            "sigma_2024_non_event_mw": [sigma_2024],
            "scale_ratio": [sigma_fit / sigma_2024],
            "n_2024_non_event_hours": [non_event.height],
        }
    )


def _aggregate_products(
    hourly_frames: list[pl.DataFrame], metrics_frames: list[pl.DataFrame]
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    hourly = pl.concat(hourly_frames, how="vertical")
    event_metrics = pl.concat(metrics_frames, how="vertical").select(
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
    forecast_columns = {
        "H0": "H0_mw",
        "H1": "H1_mw",
        "H2": "H2_mw",
        "H2-taper": "H2_taper_mw",
    }
    for scope, selected in (
        ("all", hourly),
        ("seollal", hourly.filter(pl.col("holiday_type") == "seollal")),
        ("chuseok", hourly.filter(pl.col("holiday_type") == "chuseok")),
    ):
        if selected.is_empty():
            continue
        for correction, column in forecast_columns.items():
            pooled_rows.append(
                point_metric_frame(
                    scope,
                    selected["observed_mw"].to_numpy(),
                    selected[column].to_numpy(),
                ).with_columns(pl.lit(scope).alias("scope"), pl.lit(correction).alias("correction"))
            )
    pooled = pl.concat(pooled_rows, how="vertical")

    by_key = {(row["event_id"], row["correction"]): row for row in event_metrics.to_dicts()}
    improvement_rows = []
    for event_id in event_metrics["event_id"].unique(maintain_order=True).to_list():
        base = by_key[(event_id, "H0")]
        for correction in ("H1", "H2", "H2-taper"):
            corrected = by_key[(event_id, correction)]
            improvement_rows.append(
                {
                    "event_id": event_id,
                    "holiday_type": base["holiday_type"],
                    "correction": correction,
                    "baseline_rmse": base["rmse"],
                    "corrected_rmse": corrected["rmse"],
                    "rmse_improvement_pct": 100.0 * (1.0 - corrected["rmse"] / base["rmse"]),
                }
            )
    improvements = pl.DataFrame(improvement_rows)
    summary_rows = []
    for correction in ("H1", "H2", "H2-taper"):
        for scope in ("all", "seollal", "chuseok"):
            selected = improvements.filter(pl.col("correction") == correction)
            if scope != "all":
                selected = selected.filter(pl.col("holiday_type") == scope)
            if selected.is_empty():
                continue
            values = selected["rmse_improvement_pct"].to_numpy()
            summary_rows.append(
                {
                    "scope": scope,
                    "correction": correction,
                    "events": values.size,
                    "improved_events": int((values > 0).sum()),
                    "mean_improvement_pct": float(np.mean(values)),
                    "median_improvement_pct": float(np.median(values)),
                    "q25_improvement_pct": float(np.quantile(values, 0.25)),
                    "q75_improvement_pct": float(np.quantile(values, 0.75)),
                    "min_improvement_pct": float(np.min(values)),
                    "max_improvement_pct": float(np.max(values)),
                }
            )
    return event_metrics, pooled, improvements, pl.DataFrame(summary_rows)


def _select_contexts(
    source: ValidatedCorrectionSource,
    *,
    models: Iterable[str],
    feature_sets: Iterable[str],
) -> tuple[EventResidualContext, ...]:
    requested_models = tuple(models)
    requested_features = tuple(feature_sets)
    if any(model not in MODEL_NAMES for model in requested_models):
        raise LegacyHQTError("legacy HQT model scope differs")
    if any(feature not in {"B0", "B1", "B1W"} for feature in requested_features):
        raise LegacyHQTError("legacy HQT feature scope differs")
    available = {
        (context.model, context.feature_set): context for context in source.available_contexts
    }
    try:
        return tuple(
            available[(model, feature)]
            for model in MODEL_NAMES
            if model in requested_models
            for feature in ("B0", "B1", "B1W")
            if feature in requested_features
        )
    except KeyError as error:
        raise LegacyHQTError("legacy HQT source lacks a requested context") from error


def run_legacy_hqt_loeo(
    *,
    source_run_dir: Path,
    config_path: Path,
    output_root: Path,
    models: Iterable[str],
    feature_sets: Iterable[str],
    held_out_occurrence_ids: Iterable[str] | None,
    root_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    init: str | None = None,
    target_accept: float | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[LegacyHQTContextResult, ...]:
    """Run/reuse reviewer-scoped HQT folds in fixed manuscript model order."""

    source = validate_correction_source(
        run_dir=Path(source_run_dir).expanduser().resolve(),
        config_path=Path(config_path).expanduser().resolve(),
        profile=profile,
    )
    contexts = _select_contexts(source, models=models, feature_sets=feature_sets)
    results = []
    for context_index, context in enumerate(contexts, start=1):
        label = f"{context.model}/{context.feature_set}/seed-{context.seed}"
        context_root = (
            Path(output_root).expanduser().resolve()
            / context.model
            / context.feature_set
            / f"seed-{context.seed}"
            / "legacy-hqt"
        )
        if progress:
            progress(f"[{context_index}/{len(contexts)} {label}] publishing LOEO universe")
        publication = publish_loeo_universe(source, context, output_dir=context_root / "loeo")
        selected = (
            publication.occurrence_ids
            if held_out_occurrence_ids is None
            else tuple(held_out_occurrence_ids)
        )
        if (
            not selected
            or len(set(selected)) != len(selected)
            or not set(selected) <= set(publication.occurrence_ids)
        ):
            raise LegacyHQTError("legacy HQT held-out scope differs from LOEO publication")
        hourly_frames: list[pl.DataFrame] = []
        metrics_frames: list[pl.DataFrame] = []
        fit_count = 0
        reused_count = 0
        for fold_index, held_out in enumerate(selected, start=1):
            sampler = _sampler_contract(
                profile=profile,
                root_seed=root_seed,
                held_out_occurrence_id=held_out,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                init=init,
                target_accept=target_accept,
            )
            if progress:
                progress(
                    f"[{context_index}/{len(contexts)} {label}] "
                    f"fold {fold_index}/{len(selected)} {held_out}"
                )
            hourly, metrics, reused = _fit_fold(
                source,
                publication,
                held_out_occurrence_id=held_out,
                sampler=sampler,
                fold_dir=context_root / "folds" / held_out,
            )
            hourly_frames.append(hourly)
            metrics_frames.append(metrics)
            fit_count += int(not reused)
            reused_count += int(reused)
        event_metrics, pooled, improvements, summary = _aggregate_products(
            hourly_frames, metrics_frames
        )
        _write_parquet_atomic(event_metrics, context_root / "event_metrics.parquet")
        _write_parquet_atomic(pooled, context_root / "pooled_metrics.parquet")
        _write_parquet_atomic(improvements, context_root / "event_improvements.parquet")
        _write_parquet_atomic(summary, context_root / "improvement_summary.parquet")
        _write_parquet_atomic(
            _scale_stability(source, context), context_root / "scale_stability.parquet"
        )
        results.append(
            LegacyHQTContextResult(
                context=context,
                output_dir=context_root,
                occurrence_ids=selected,
                sampler_fit_count=fit_count,
                reused_fold_count=reused_count,
            )
        )
        if progress:
            progress(f"[{context_index}/{len(contexts)} {label}] complete")
    return tuple(results)


__all__ = [
    "LegacyHQTContextResult",
    "LegacyHQTError",
    "build_legacy_hqt_data",
    "build_legacy_hqt_data_from_frame",
    "cosine_boundary_taper",
    "event_type_mean_shift_mw",
    "run_legacy_hqt_loeo",
]
