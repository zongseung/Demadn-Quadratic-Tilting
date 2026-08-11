"""Strict production orchestration for the single causal-2024 HQRC comparison."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any

import numpy as np
import polars as pl

from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.contracts import DataContractError, validate_prediction_frame
from hqrc_v3.diagnostics.ar import (
    ApprovedARCalibration,
    EventResidualContext,
    require_approved_calibration,
    validate_event_residual_context,
)
from hqrc_v3.events import EventOccurrence
from hqrc_v3.provenance import ArtifactMismatch

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


__all__ = [
    "CausalCorrectionError",
    "CausalPredictionData",
    "build_causal_hqrc_data",
    "build_causal_prediction_data",
    "select_latest_oof_scale",
]
