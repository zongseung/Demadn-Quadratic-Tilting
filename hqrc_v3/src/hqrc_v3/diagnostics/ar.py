"""Event-reset AR(1) diagnostics and immutable prior-calibration artifacts."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import timedelta
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf, pacf

from hqrc_v3.provenance import ArtifactMismatch

_SCHEMA_VERSION = "hqrc-v3.ar-calibration.v1"
_FORMULA_VERSION = "robust-mad-beta-v1"
_LAGS = 48
_HARMONICS = 3
_MIN_EVENT_ROWS = 2 * _LAGS + 4
_REQUIRED_COLUMNS = frozenset(
    {
        "occurrence_id",
        "tau_days",
        "hour",
        "standardized_residual",
        "model",
        "feature_set",
        "seed",
    }
)


class ARCalibrationError(ValueError):
    """Raised when residual diagnostics or a suggested AR prior are invalid."""


@dataclass(frozen=True)
class EventResidualContext:
    """One immutable model/feature/seed residual context used for diagnostics."""

    model: str
    feature_set: str
    seed: int
    split_id: str | None = None


@dataclass(frozen=True)
class EventARDiagnostic:
    """Serializable occurrence-local AR diagnostics, all arrays covering lags 1--48."""

    occurrence_id: str
    phi: float
    raw_acf: tuple[float, ...]
    raw_pacf: tuple[float, ...]
    detrended_acf: tuple[float, ...]
    detrended_pacf: tuple[float, ...]
    bartlett_95: float
    innovation_acf: tuple[float, ...]
    innovation_pacf: tuple[float, ...]
    ljung_box_lag: int
    ljung_box_statistic: float
    ljung_box_pvalue: float
    diagnostic_warning: bool


@dataclass(frozen=True)
class ARCalibration:
    """A data-derived *suggestion* for the transformed AR(1) Beta prior."""

    a: float
    b: float
    phi_center: float
    event_count: int
    event_ids: tuple[str, ...]
    phi_estimates: tuple[float, ...]
    formula_version: str = _FORMULA_VERSION


def _timestamp_column(frame: pl.DataFrame) -> str:
    found = [name for name in ("timestamp", "target_timestamp") if name in frame.columns]
    if len(found) != 1:
        raise ARCalibrationError("residual context requires exactly one hourly timestamp column")
    return found[0]


def _is_datetime(dtype: pl.DataType) -> bool:
    return dtype.base_type() == pl.Datetime and dtype.time_zone is None


def _context_from_frame(frame: pl.DataFrame) -> tuple[EventResidualContext, str]:
    if not isinstance(frame, pl.DataFrame):
        raise ARCalibrationError("event residual context must be a Polars DataFrame")
    missing = _REQUIRED_COLUMNS - set(frame.columns)
    if missing:
        raise ARCalibrationError(f"event residual context is missing columns: {sorted(missing)}")
    if frame.is_empty():
        raise ARCalibrationError("event residual context must not be empty")
    timestamp_column = _timestamp_column(frame)
    if not _is_datetime(frame.schema[timestamp_column]):
        raise ARCalibrationError("event residual timestamp must be a naive Datetime column")
    if any(frame[column].null_count() for column in _REQUIRED_COLUMNS | {timestamp_column}):
        raise ARCalibrationError("event residual context must not contain null values")
    if frame["standardized_residual"].dtype not in {pl.Float32, pl.Float64}:
        raise ARCalibrationError("standardized_residual must be floating point")
    if not frame.select(pl.col("standardized_residual").is_finite().all()).item():
        raise ARCalibrationError("standardized_residual must be finite")
    if frame["tau_days"].dtype not in {pl.Float32, pl.Float64}:
        raise ARCalibrationError("tau_days must be floating point")
    if not frame.select(pl.col("tau_days").is_finite().all()).item():
        raise ARCalibrationError("tau_days must be finite")
    if frame["hour"].dtype not in {
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
    }:
        raise ARCalibrationError("hour must be an integer")
    if not frame.filter((pl.col("hour") < 0) | (pl.col("hour") > 23)).is_empty():
        raise ARCalibrationError("hour must be between 0 and 23")
    if not frame.filter(pl.col("hour") != pl.col(timestamp_column).dt.hour()).is_empty():
        raise ARCalibrationError("hour must match the timestamp hour")
    identifiers = ("model", "feature_set", "seed")
    if any(frame[column].n_unique() != 1 for column in identifiers):
        raise ARCalibrationError("event residual context must have one model/feature/seed")
    if any(not str(frame[column].item(0)).strip() for column in ("model", "feature_set")):
        raise ARCalibrationError("event residual context identifiers must not be blank")
    split_id = None
    if "split_id" in frame.columns:
        if frame["split_id"].null_count() or frame["split_id"].n_unique() != 1:
            raise ARCalibrationError("event residual context must have one known split_id")
        split_id = str(frame["split_id"].item(0))
        if not split_id.strip():
            raise ARCalibrationError("event residual context split_id must not be blank")
    context = EventResidualContext(
        model=str(frame["model"].item(0)),
        feature_set=str(frame["feature_set"].item(0)),
        seed=int(frame["seed"].item(0)),
        split_id=split_id,
    )
    return context, timestamp_column


def _validate_event_rows(frame: pl.DataFrame, timestamp_column: str) -> None:
    if frame["occurrence_id"].dtype not in {pl.String, pl.Categorical, pl.Enum}:
        raise ARCalibrationError("occurrence_id must be a string")
    if frame.filter(pl.col("occurrence_id").str.strip_chars() == "").height:
        raise ARCalibrationError("occurrence_id must not be blank")
    if frame.select(pl.struct("occurrence_id", timestamp_column).is_duplicated().any()).item():
        raise ARCalibrationError("duplicate occurrence timestamp")
    for occurrence_id in sorted(frame["occurrence_id"].unique().to_list()):
        event = frame.filter(pl.col("occurrence_id") == occurrence_id)
        timestamps = event[timestamp_column].to_list()
        if len(timestamps) < _MIN_EVENT_ROWS:
            raise ARCalibrationError(f"event {occurrence_id!r} is too short for 48-lag diagnostics")
        if timestamps != sorted(timestamps):
            raise ARCalibrationError(f"event {occurrence_id!r} timestamps must be ordered")
        if any(
            current - previous != timedelta(hours=1)
            for previous, current in zip(timestamps, timestamps[1:])
        ):
            raise ARCalibrationError(f"event {occurrence_id!r} has an hourly timestamp gap")
    ranges = sorted(
        (
            event[timestamp_column][0],
            event[timestamp_column][-1],
            event_id,
        )
        for event_id in frame["occurrence_id"].unique().to_list()
        for event in (frame.filter(pl.col("occurrence_id") == event_id),)
    )
    for previous, current in zip(ranges, ranges[1:]):
        if current[0] <= previous[1] + timedelta(hours=1):
            raise ARCalibrationError("event residuals must not concatenate event boundaries")


def validate_event_residual_context(frame: pl.DataFrame) -> EventResidualContext:
    """Validate standardized hourly residuals without concatenating event boundaries."""

    context, timestamp_column = _context_from_frame(frame)
    _validate_event_rows(frame, timestamp_column)
    return context


def _validated(frame: pl.DataFrame) -> tuple[EventResidualContext, str]:
    context = validate_event_residual_context(frame)
    return context, _timestamp_column(frame)


def detrend_event_residuals(frame: pl.DataFrame) -> pl.DataFrame:
    """Remove the quadratic-plus-three-harmonic diagnostic trend per occurrence."""

    _, _ = _validated(frame)
    detrended = np.empty(frame.height, dtype=float)
    for occurrence_id in frame["occurrence_id"].unique().to_list():
        indices = np.flatnonzero(frame["occurrence_id"].to_numpy() == occurrence_id)
        event = frame[indices.tolist()]
        tau = event["tau_days"].to_numpy().astype(float)
        hour = event["hour"].to_numpy().astype(float)
        columns = [np.ones(len(event)), tau, tau**2]
        for harmonic in range(1, _HARMONICS + 1):
            angle = 2.0 * np.pi * harmonic * hour / 24.0
            columns.extend((np.sin(angle), np.cos(angle)))
        design = np.column_stack(columns)
        values = event["standardized_residual"].to_numpy().astype(float)
        coefficients, _, _, _ = np.linalg.lstsq(design, values, rcond=None)
        detrended[indices] = values - design @ coefficients
    if not np.isfinite(detrended).all():
        raise ARCalibrationError("detrending produced nonfinite residuals")
    return frame.with_columns(pl.Series("detrended_residual", detrended))


def estimate_event_phi(residual: np.ndarray) -> float:
    """Estimate a conditional AR(1) coefficient inside exactly one occurrence."""

    values = np.asarray(residual, dtype=float)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ARCalibrationError("event residual must be a finite one-dimensional series")
    denominator = float(np.dot(values[:-1], values[:-1]))
    if denominator <= np.finfo(float).eps:
        raise ARCalibrationError("event residual has zero lag variance")
    estimate = float(np.dot(values[:-1], values[1:]) / denominator)
    return float(np.clip(estimate, -0.98, 0.98))


def estimate_event_phis(residuals: Mapping[str, np.ndarray]) -> dict[str, float]:
    """Estimate each occurrence separately; no lag pair can cross an event boundary."""

    if not residuals or any(
        not isinstance(event_id, str) or not event_id for event_id in residuals
    ):
        raise ARCalibrationError("residuals must contain named occurrences")
    return {event_id: estimate_event_phi(residuals[event_id]) for event_id in sorted(residuals)}


def _lag_values(values: np.ndarray) -> tuple[tuple[float, ...], tuple[float, ...]]:
    return (
        tuple(float(value) for value in acf(values, nlags=_LAGS, fft=True, adjusted=False)[1:]),
        tuple(float(value) for value in pacf(values, nlags=_LAGS, method="ywmle")[1:]),
    )


def _warning(
    pacf_values: tuple[float, ...], innovation_acf: tuple[float, ...], bound: float
) -> bool:
    lag_one_is_dominant = all(abs(value) < abs(pacf_values[0]) for value in pacf_values[1:6])
    innovation_reduces_acf = abs(innovation_acf[0]) < abs(pacf_values[0])
    return not (abs(pacf_values[0]) > bound and lag_one_is_dominant and innovation_reduces_acf)


def diagnose_event_residuals(frame: pl.DataFrame) -> tuple[EventARDiagnostic, ...]:
    """Compute serial-correlation diagnostics for each event after independent detrending."""

    _, _ = _validated(frame)
    detrended = detrend_event_residuals(frame)
    diagnostics: list[EventARDiagnostic] = []
    for occurrence_id in sorted(detrended["occurrence_id"].unique().to_list()):
        event = detrended.filter(pl.col("occurrence_id") == occurrence_id)
        raw = event["standardized_residual"].to_numpy().astype(float)
        values = event["detrended_residual"].to_numpy().astype(float)
        phi = estimate_event_phi(values)
        innovation = values[1:] - phi * values[:-1]
        raw_acf, raw_pacf = _lag_values(raw)
        detrended_acf, detrended_pacf = _lag_values(values)
        innovation_acf, innovation_pacf = _lag_values(innovation)
        lag = min(24, len(innovation) - 1)
        result = acorr_ljungbox(innovation, lags=[lag], return_df=True)
        statistic = float(result["lb_stat"].iloc[0])
        pvalue = float(result["lb_pvalue"].iloc[0])
        bound = float(1.96 / math.sqrt(len(values)))
        diagnostics.append(
            EventARDiagnostic(
                occurrence_id=occurrence_id,
                phi=phi,
                raw_acf=raw_acf,
                raw_pacf=raw_pacf,
                detrended_acf=detrended_acf,
                detrended_pacf=detrended_pacf,
                bartlett_95=bound,
                innovation_acf=innovation_acf,
                innovation_pacf=innovation_pacf,
                ljung_box_lag=lag,
                ljung_box_statistic=statistic,
                ljung_box_pvalue=pvalue,
                diagnostic_warning=_warning(detrended_pacf, innovation_acf, bound),
            )
        )
    return tuple(diagnostics)


def calibrate_beta_prior(
    phi: np.ndarray, *, event_ids: tuple[str, ...] | None = None
) -> ARCalibration:
    """Apply the approved median/MAD Beta-moment rule to occurrence-local estimates."""

    values = np.asarray(phi, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ARCalibrationError("at least two occurrence-specific phi estimates are required")
    if not np.isfinite(values).all():
        raise ARCalibrationError("phi estimates must be finite")
    if np.any(values < -0.98) or np.any(values > 0.98):
        raise ARCalibrationError("phi estimates must be in the allowed [-0.98, 0.98] range")
    ids = event_ids or tuple(f"event-{index}" for index in range(len(values)))
    if (
        len(ids) != len(values)
        or len(set(ids)) != len(ids)
        or any(not event_id for event_id in ids)
    ):
        raise ARCalibrationError("phi estimates require unique event ids")
    estimates = dict(zip(ids, values, strict=True))
    ordered_ids = tuple(sorted(estimates))
    ordered_phi = np.asarray([estimates[event_id] for event_id in ordered_ids], dtype=float)
    transformed = (ordered_phi + 1.0) / 2.0
    center = float(np.clip(np.median(transformed), 0.10, 0.975))
    robust_sd = max(1.4826 * float(np.median(np.abs(transformed - np.median(transformed)))), 0.025)
    kappa = float(np.clip(center * (1.0 - center) / robust_sd**2 - 1.0, 8.0, 40.0))
    a = max(1.05, center * kappa)
    b = max(1.05, (1.0 - center) * kappa)
    if a + b > 40.0:
        scale = 40.0 / (a + b)
        a, b = a * scale, b * scale
    return ARCalibration(
        a=float(a),
        b=float(b),
        phi_center=float(2.0 * center - 1.0),
        event_count=len(ordered_phi),
        event_ids=ordered_ids,
        phi_estimates=tuple(float(value) for value in ordered_phi),
    )


def _canonical(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
        "utf-8"
    )


def _digest(payload: dict[str, Any]) -> str:
    return sha256(_canonical(payload)).hexdigest()


def _hashes(residual_sha256: str, config_sha256: str, event_sha256: str) -> dict[str, str]:
    hashes = {
        "residual_sha256": residual_sha256,
        "config_sha256": config_sha256,
        "event_sha256": event_sha256,
    }
    if any(not isinstance(value, str) or not value for value in hashes.values()):
        raise ARCalibrationError("residual, config, and event hashes must be nonempty strings")
    return hashes


def _context_mapping(context: EventResidualContext | dict[str, Any]) -> dict[str, Any]:
    value = asdict(context) if isinstance(context, EventResidualContext) else dict(context)
    required = {"model", "feature_set", "seed"}
    if set(value) - (required | {"split_id"}) or required - set(value):
        raise ARCalibrationError(
            "diagnostic artifact context must contain model, feature_set, and seed"
        )
    if not isinstance(value["model"], str) or not value["model"].strip():
        raise ARCalibrationError("diagnostic artifact model context must be nonblank")
    if not isinstance(value["feature_set"], str) or not value["feature_set"].strip():
        raise ARCalibrationError("diagnostic artifact feature context must be nonblank")
    if isinstance(value["seed"], bool) or not isinstance(value["seed"], int):
        raise ARCalibrationError("diagnostic artifact seed must be an integer")
    if value.get("split_id") is None:
        value.pop("split_id", None)
    elif not isinstance(value["split_id"], str) or not value["split_id"].strip():
        raise ARCalibrationError("diagnostic artifact split context must be nonblank")
    return dict(sorted(value.items()))


def _calibration_mapping(calibration: ARCalibration) -> dict[str, Any]:
    if not isinstance(calibration, ARCalibration):
        raise ARCalibrationError("calibration must be an ARCalibration")
    if calibration.event_count != len(calibration.event_ids) or calibration.event_count != len(
        calibration.phi_estimates
    ):
        raise ARCalibrationError("calibration event counts do not match estimates")
    numeric = (calibration.a, calibration.b, calibration.phi_center, *calibration.phi_estimates)
    if (
        not all(math.isfinite(value) for value in numeric)
        or calibration.a <= 1
        or calibration.b <= 1
    ):
        raise ARCalibrationError("calibration values must be finite with a,b > 1")
    return asdict(calibration)


def _diagnostic_mappings(diagnostics: tuple[EventARDiagnostic, ...]) -> list[dict[str, Any]]:
    items = []
    for diagnostic in diagnostics:
        if not isinstance(diagnostic, EventARDiagnostic):
            raise ARCalibrationError("diagnostics must contain EventARDiagnostic values")
        item = asdict(diagnostic)
        numbers = (
            diagnostic.phi,
            diagnostic.bartlett_95,
            diagnostic.ljung_box_statistic,
            diagnostic.ljung_box_pvalue,
            *diagnostic.raw_acf,
            *diagnostic.raw_pacf,
            *diagnostic.detrended_acf,
            *diagnostic.detrended_pacf,
            *diagnostic.innovation_acf,
            *diagnostic.innovation_pacf,
        )
        if not all(math.isfinite(float(value)) for value in numbers):
            raise ARCalibrationError("diagnostics must contain finite numbers")
        items.append(item)
    return sorted(items, key=lambda item: str(item["occurrence_id"]))


def _proposal_payload(
    diagnostics: tuple[EventARDiagnostic, ...],
    calibration: ARCalibration,
    *,
    residual_sha256: str,
    config_sha256: str,
    event_sha256: str,
    context: EventResidualContext | dict[str, Any],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "approved": False,
        "context": _context_mapping(context),
        "hashes": _hashes(residual_sha256, config_sha256, event_sha256),
        "diagnostics": _diagnostic_mappings(diagnostics),
        "calibration": _calibration_mapping(calibration),
    }
    payload["proposal_digest"] = _digest(payload)
    payload["artifact_digest"] = _digest(payload)
    return payload


def _write_atomic(path: Path, payload: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = _canonical(payload) + b"\n"
    if path.exists():
        existing = _read_artifact(path)
        if _canonical(existing) != _canonical(payload):
            raise ArtifactMismatch(f"refusing incompatible overwrite of AR artifact {path}")
        return path
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _proposal_digest_content(payload: dict[str, Any]) -> dict[str, Any]:
    content = dict(payload)
    content.pop("artifact_digest", None)
    content.pop("proposal_digest", None)
    if content.get("approved") is True:
        content["approved"] = False
        content.pop("approved_proposal_digest", None)
        content.pop("approved_schema_version", None)
    return content


def _read_artifact(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ArtifactMismatch(f"invalid AR calibration artifact {path}: {error}") from error
    if not isinstance(payload, dict) or payload.get("schema_version") != _SCHEMA_VERSION:
        raise ArtifactMismatch("unsupported AR calibration artifact schema")
    artifact_digest = payload.get("artifact_digest")
    content = dict(payload)
    content.pop("artifact_digest", None)
    if not isinstance(artifact_digest, str) or artifact_digest != _digest(content):
        raise ArtifactMismatch("AR calibration artifact digest differs")
    proposal_digest = payload.get("proposal_digest")
    if not isinstance(proposal_digest, str) or proposal_digest != _digest(
        _proposal_digest_content(payload)
    ):
        raise ArtifactMismatch("AR calibration proposal digest differs")
    if not isinstance(payload.get("hashes"), dict):
        raise ArtifactMismatch("AR calibration artifact hashes are invalid")
    _hashes(**payload["hashes"])
    return payload


def write_ar_diagnostics(
    path: Path,
    diagnostics: tuple[EventARDiagnostic, ...],
    calibration: ARCalibration,
    *,
    residual_sha256: str,
    config_sha256: str,
    event_sha256: str,
    context: EventResidualContext | dict[str, Any],
) -> Path:
    """Atomically write a deterministic, unapproved calibration proposal."""

    return _write_atomic(
        path,
        _proposal_payload(
            diagnostics,
            calibration,
            residual_sha256=residual_sha256,
            config_sha256=config_sha256,
            event_sha256=event_sha256,
            context=context,
        ),
    )


def write_calibration(path: Path, calibration: ARCalibration, **kwargs: Any) -> Path:
    """Compatibility wrapper for callers which only have the proposed calibration."""

    return write_ar_diagnostics(path, (), calibration, **kwargs)


def _assert_current_hashes(
    payload: dict[str, Any],
    *,
    current_residual_sha256: str,
    current_config_sha256: str | None = None,
    current_event_sha256: str | None = None,
) -> None:
    current = {"residual_sha256": current_residual_sha256}
    if current_config_sha256 is not None:
        current["config_sha256"] = current_config_sha256
    if current_event_sha256 is not None:
        current["event_sha256"] = current_event_sha256
    for name, value in current.items():
        if not isinstance(value, str) or not value:
            raise ARCalibrationError(f"current {name} must be a nonempty string")
        if payload["hashes"].get(name) != value:
            raise ArtifactMismatch(f"{name} differs from the AR calibration artifact")


def approve_calibration(
    proposal_path: Path,
    approved_path: Path,
    *,
    current_residual_sha256: str,
    current_config_sha256: str | None = None,
    current_event_sha256: str | None = None,
) -> Path:
    """Freeze a reviewed proposal without recalculating or accepting manual values."""

    proposal = _read_artifact(proposal_path)
    if proposal.get("approved") is not False:
        raise ARCalibrationError("only an unapproved AR proposal may be approved")
    _assert_current_hashes(
        proposal,
        current_residual_sha256=current_residual_sha256,
        current_config_sha256=current_config_sha256,
        current_event_sha256=current_event_sha256,
    )
    approved = dict(proposal)
    approved["approved"] = True
    approved["approved_proposal_digest"] = proposal["proposal_digest"]
    approved["approved_schema_version"] = proposal["schema_version"]
    approved.pop("artifact_digest")
    approved["artifact_digest"] = _digest(approved)
    return _write_atomic(approved_path, approved)


def _calibration_from_mapping(value: Any) -> ARCalibration:
    if not isinstance(value, dict):
        raise ArtifactMismatch("AR calibration payload is invalid")
    try:
        calibration = ARCalibration(
            a=float(value["a"]),
            b=float(value["b"]),
            phi_center=float(value["phi_center"]),
            event_count=int(value["event_count"]),
            event_ids=tuple(str(event_id) for event_id in value["event_ids"]),
            phi_estimates=tuple(float(phi) for phi in value["phi_estimates"]),
            formula_version=str(value["formula_version"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ArtifactMismatch("AR calibration payload is invalid") from error
    _calibration_mapping(calibration)
    return calibration


def load_approved_calibration(
    path: Path,
    *,
    current_residual_sha256: str,
    current_config_sha256: str | None = None,
    current_event_sha256: str | None = None,
) -> ARCalibration:
    """Load only an approved, digest-valid calibration compatible with current inputs."""

    payload = _read_artifact(path)
    if payload.get("approved") is not True:
        raise ARCalibrationError("AR calibration artifact is not approved")
    if payload.get("approved_proposal_digest") != payload.get("proposal_digest"):
        raise ArtifactMismatch("approved AR artifact does not preserve the proposal digest")
    if payload.get("approved_schema_version") != _SCHEMA_VERSION:
        raise ArtifactMismatch("approved AR artifact schema version differs")
    _assert_current_hashes(
        payload,
        current_residual_sha256=current_residual_sha256,
        current_config_sha256=current_config_sha256,
        current_event_sha256=current_event_sha256,
    )
    return _calibration_from_mapping(payload.get("calibration"))
