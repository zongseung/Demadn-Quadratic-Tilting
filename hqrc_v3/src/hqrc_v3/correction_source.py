"""Immutable, source-derived preflight shared by correction evaluation stages."""

from __future__ import annotations

import json
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import polars as pl

from hqrc_v3.baselines.config import MODEL_NAMES, load_paper_baselines
from hqrc_v3.baselines.paper import (
    PAPER_FEATURE_SUITES,
    PAPER_HASH_KEYS,
    run_paper_final_stage,
)
from hqrc_v3.config import load_config
from hqrc_v3.contracts import DataContractError, validate_prediction_frame
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
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.events import (
    EventOccurrence,
    load_event_registry,
    load_holiday_calendar,
    validate_feature_event_alignment,
)
from hqrc_v3.features import attach_calendar_features, build_daily_forecast_matrix
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
from hqrc_v3.residual_stage import (
    build_standardized_residuals,
    load_standardized_residual_manifest,
    select_residual_context,
)
from hqrc_v3.splits import expanding_oof_folds, final_fold, select_fold_samples

_SOURCE_KEYS = (
    "data",
    "experiment_config",
    "model_config",
    "event_registry",
    "holiday_calendar",
    "temporary_holiday_availability",
)
_SOURCE_TO_BASELINE_HASH = {
    "data": "data_sha256",
    "experiment_config": "experiment_sha256",
    "model_config": "model_config_sha256",
    "event_registry": "event_registry_sha256",
    "holiday_calendar": "holiday_calendar_sha256",
    "temporary_holiday_availability": "temporary_holiday_availability_sha256",
}
_INPUT_NAMESPACE = frozenset(
    {
        ".standardized-residuals.lock",
        "standardized_residuals.parquet",
        "standardized_residuals_manifest.json",
    }
)
_PREDICTION_NAMESPACE = frozenset(
    {
        ".baseline-publication.lock",
        "baseline_manifest.json",
        "oof.parquet",
        "oof_members.parquet",
        "final_2024.parquet",
        "final_2024_members.parquet",
    }
)
_FEATURE_SETS = frozenset(("B0", "B1", "B1W"))


class CorrectionSourceError(ValueError):
    """Raised when a publication cannot serve as immutable correction input."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise CorrectionSourceError("correction source metadata is not canonical JSON") from error


def _read_canonical_json(path: Path, description: str) -> dict[str, Any]:
    candidate = Path(path)
    if not candidate.is_file() or candidate.is_symlink():
        raise CorrectionSourceError(f"{description} is missing or unsafe")
    try:
        raw = candidate.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise CorrectionSourceError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise CorrectionSourceError(f"{description} is not canonical JSON")
    return value


def _require_real_file(path: Path, description: str) -> None:
    try:
        identity = Path(path).lstat()
    except OSError as error:
        raise CorrectionSourceError(f"{description} is missing or unsafe") from error
    if not stat.S_ISREG(identity.st_mode):
        raise CorrectionSourceError(f"{description} is missing or unsafe")


def _require_exact_namespace(directory: Path, allowed: frozenset[str], description: str) -> None:
    candidate = Path(directory)
    try:
        identity = candidate.lstat()
        entries = tuple(candidate.iterdir())
    except OSError as error:
        raise CorrectionSourceError(f"{description} namespace is missing or unsafe") from error
    if not stat.S_ISDIR(identity.st_mode):
        raise CorrectionSourceError(f"{description} namespace is missing or unsafe")
    names = {entry.name for entry in entries}
    if names - allowed:
        raise CorrectionSourceError(f"{description} namespace contains unknown entries")
    for entry in entries:
        _require_real_file(entry, f"{description} publication entry")


def _resolve_source_paths(
    run_dir: Path, config_path: Path
) -> tuple[dict[str, Path], dict[str, str], dict[str, Any]]:
    """Resolve all six source identities before trusting downstream paths."""

    manifest_path = Path(run_dir) / "inputs/standardized_residuals_manifest.json"
    raw_manifest = _read_canonical_json(manifest_path, "standardized residual manifest")
    inputs = raw_manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise CorrectionSourceError("residual manifest inputs are invalid")
    paths: dict[str, Path] = {}
    hashes: dict[str, str] = {}
    for name in _SOURCE_KEYS:
        entry = inputs.get(name)
        if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
            raise CorrectionSourceError(f"residual manifest {name} source is invalid")
        raw_path, digest = entry.get("path"), entry.get("sha256")
        if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
            raise CorrectionSourceError(f"residual manifest {name} path must be absolute")
        path = Path(raw_path)
        _require_real_file(path, f"residual manifest {name} source")
        if not isinstance(digest, str) or file_sha256(path) != digest:
            raise CorrectionSourceError(f"residual manifest {name} source hash differs")
        paths[name], hashes[name] = path, digest
    requested_config = Path(config_path)
    _require_real_file(requested_config, "requested experiment config")
    if paths["experiment_config"].resolve() != requested_config.resolve():
        raise CorrectionSourceError(
            "requested experiment config differs from residual source"
        )
    return paths, hashes, raw_manifest


def _paper_bounds(profile: str, availability: object) -> dict[str, object]:
    if profile == "paper":
        return {
            "expected_start": FIXED_START,
            "expected_end": FIXED_END,
            "expected_rows": FIXED_ROWS,
            "expected_public_holiday_dates": FIXED_PUBLIC_HOLIDAY_DATES,
            "expected_substitute_or_temporary_dates": (
                FIXED_SUBSTITUTE_OR_TEMPORARY_DATES
            ),
            "temporary_holiday_availability": availability,
        }
    return {
        "expected_start": None,
        "expected_end": None,
        "expected_rows": None,
        "temporary_holiday_availability": availability,
    }


def _read_bound_parquet(path: Path, digest: str, description: str) -> pl.DataFrame:
    _require_real_file(path, description)
    if file_sha256(path) != digest:
        raise CorrectionSourceError(f"{description} hash differs")
    try:
        return pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise CorrectionSourceError(f"{description} is unreadable") from error


def _context_key(context: EventResidualContext) -> tuple[str, str, int]:
    if not isinstance(context, EventResidualContext):
        raise TypeError("context must be an EventResidualContext")
    if (
        not isinstance(context.model, str)
        or not isinstance(context.feature_set, str)
        or isinstance(context.seed, bool)
        or not isinstance(context.seed, int)
    ):
        raise CorrectionSourceError("requested context is invalid")
    return context.model, context.feature_set, context.seed


def _require_canonical_context(
    context: EventResidualContext, *, description: str
) -> EventResidualContext:
    _context_key(context)
    split_ids = context.split_ids
    if not isinstance(split_ids, tuple) or not split_ids:
        raise CorrectionSourceError(f"{description} split identity is not canonical")
    allowed = tuple(fold.split_id for fold in expanding_oof_folds())
    canonical = tuple(split_id for split_id in allowed if split_id in split_ids)
    if split_ids != canonical:
        raise CorrectionSourceError(f"{description} split identity is not canonical")
    return context


def _select_point_context(
    frame: pl.DataFrame,
    *,
    context: EventResidualContext,
    expected_split_ids: tuple[str, ...],
    description: str,
) -> pl.DataFrame:
    key = _context_key(context)
    selected = frame.filter(
        (pl.col("model") == key[0])
        & (pl.col("feature_set") == key[1])
        & (pl.col("seed") == key[2])
    )
    if selected.is_empty():
        raise CorrectionSourceError(f"requested context is absent from {description}")
    groups: list[pl.DataFrame] = []
    try:
        for split_id in expected_split_ids:
            split = selected.filter(pl.col("split_id") == split_id)
            if split.is_empty():
                raise CorrectionSourceError(
                    f"requested context split is absent from {description}"
                )
            groups.append(validate_prediction_frame(split).sort("target_timestamp"))
    except (DataContractError, TypeError, ValueError) as error:
        if isinstance(error, CorrectionSourceError):
            raise
        raise CorrectionSourceError(f"requested {description} stream is invalid") from error
    unexpected = set(selected["split_id"].unique().to_list()) - set(expected_split_ids)
    if unexpected:
        raise CorrectionSourceError(f"requested {description} stream has unknown splits")
    return pl.concat(groups, how="vertical")


@dataclass(frozen=True, slots=True)
class ValidatedCorrectionSource:
    """Hash-bound source publication with no mutable frame held in memory."""

    run_dir: Path
    source_profile: str
    events: tuple[EventOccurrence, ...]
    source_paths: Mapping[str, Path]
    source_hashes: Mapping[str, str]
    available_contexts: tuple[EventResidualContext, ...]
    residual_manifest_path: Path
    residual_path: Path
    baseline_manifest_path: Path
    oof_members_path: Path
    oof_point_path: Path
    final_members_path: Path
    final_point_path: Path
    residual_sha256: str
    oof_members_sha256: str
    oof_point_sha256: str
    final_members_sha256: str
    final_point_sha256: str
    _residual_manifest_json: bytes
    _baseline_manifest_json: bytes

    def __post_init__(self) -> None:
        object.__setattr__(self, "run_dir", Path(self.run_dir))
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(
            self, "source_paths", MappingProxyType(dict(self.source_paths))
        )
        object.__setattr__(
            self, "source_hashes", MappingProxyType(dict(self.source_hashes))
        )
        contexts = tuple(self.available_contexts)
        for context in contexts:
            _require_canonical_context(context, description="available context")
        if len(contexts) != len(set(contexts)):
            raise CorrectionSourceError("available contexts are duplicated")
        object.__setattr__(self, "available_contexts", contexts)

    @property
    def residual_manifest(self) -> dict[str, Any]:
        """Return a detached JSON copy without exposing mutable source state."""

        return json.loads(self._residual_manifest_json)

    @property
    def baseline_manifest(self) -> dict[str, Any]:
        """Return a detached JSON copy without exposing mutable source state."""

        return json.loads(self._baseline_manifest_json)

    def _require_context(self, context: EventResidualContext) -> tuple[str, str, int]:
        _require_exact_namespace(self.run_dir / "inputs", _INPUT_NAMESPACE, "residual")
        _require_exact_namespace(
            self.run_dir / "predictions", _PREDICTION_NAMESPACE, "baseline"
        )
        _require_canonical_context(context, description="requested context")
        if context not in self.available_contexts:
            raise CorrectionSourceError("requested context is not present in validated sources")
        return _context_key(context)

    def load_standardized_context(
        self, context: EventResidualContext, *, through: int
    ) -> pl.DataFrame:
        """Load one manifest-declared standardized OOF context defensively."""

        self._require_context(context)
        frame = _read_bound_parquet(
            self.residual_path,
            self.residual_sha256,
            "standardized residual artifact",
        )
        try:
            return select_residual_context(
                frame,
                self.residual_manifest,
                events=self.events,
                model=context.model,
                feature_set=context.feature_set,
                seed=context.seed,
                through=through,
            )
        except (ArtifactMismatch, DataContractError, TypeError, ValueError) as error:
            raise CorrectionSourceError(
                "requested standardized residual context is invalid"
            ) from error

    def load_oof_point_context(self, context: EventResidualContext) -> pl.DataFrame:
        """Load one exact canonical OOF point stream from the validated publication."""

        self._require_context(context)
        frame = _read_bound_parquet(
            self.oof_point_path, self.oof_point_sha256, "OOF point publication"
        )
        return _select_point_context(
            frame,
            context=context,
            expected_split_ids=context.split_ids,
            description="OOF point publication",
        )

    def load_final_point_context(self, context: EventResidualContext) -> pl.DataFrame:
        """Load one exact canonical final-2024 point stream defensively."""

        self._require_context(context)
        frame = _read_bound_parquet(
            self.final_point_path,
            self.final_point_sha256,
            "final point publication",
        )
        return _select_point_context(
            frame,
            context=context,
            expected_split_ids=("final-2024",),
            description="final point publication",
        )


def _manifest_artifact(
    manifest: Mapping[str, Any], stage: str, artifact: str, expected: str
) -> tuple[Path, str]:
    try:
        entry = manifest["stages"][stage]["artifacts"][artifact]
    except (KeyError, TypeError) as error:
        raise CorrectionSourceError(
            f"baseline {stage} {artifact} artifact binding is invalid"
        ) from error
    if (
        not isinstance(entry, dict)
        or set(entry) != {"path", "sha256"}
        or entry.get("path") != expected
        or not isinstance(entry.get("sha256"), str)
    ):
        raise CorrectionSourceError(
            f"baseline {stage} {artifact} artifact binding is invalid"
        )
    return Path(expected), str(entry["sha256"])


def _preflight_baseline_publication(
    run: Path, manifest: Mapping[str, Any]
) -> tuple[dict[str, tuple[Path, str]], dict[str, pl.DataFrame]]:
    """Validate every canonical baseline file without invoking a baseline stage."""

    _require_exact_namespace(run / "inputs", _INPUT_NAMESPACE, "residual")
    _require_exact_namespace(run / "predictions", _PREDICTION_NAMESPACE, "baseline")
    expected_paths = {
        "oof_members": run / "predictions/oof_members.parquet",
        "oof_point": run / "predictions/oof.parquet",
        "final_members": run / "predictions/final_2024_members.parquet",
        "final_point": run / "predictions/final_2024.parquet",
    }
    artifacts: dict[str, tuple[Path, str]] = {}
    frames: dict[str, pl.DataFrame] = {}
    for key, stage, artifact, relative in (
        ("oof_members", "oof", "members", "predictions/oof_members.parquet"),
        ("oof_point", "oof", "point", "predictions/oof.parquet"),
        (
            "final_members",
            "final",
            "members",
            "predictions/final_2024_members.parquet",
        ),
        ("final_point", "final", "point", "predictions/final_2024.parquet"),
    ):
        bound_relative, digest = _manifest_artifact(manifest, stage, artifact, relative)
        path = run / bound_relative
        frame = _read_bound_parquet(path, digest, f"baseline {key} artifact")
        if path.resolve() != expected_paths[key].resolve():
            raise CorrectionSourceError(f"baseline {key} path is substituted")
        artifacts[key] = path, digest
        frames[key] = frame
    return artifacts, frames


def _manifest_contexts(manifest: Mapping[str, Any]) -> tuple[EventResidualContext, ...]:
    records = manifest.get("contexts")
    if not isinstance(records, list) or not records:
        raise CorrectionSourceError("residual publication has no contexts")
    contexts: list[EventResidualContext] = []
    for record in records:
        if not isinstance(record, dict):
            raise CorrectionSourceError("residual publication context is invalid")
        model, feature_set, seed = (
            record.get("model"),
            record.get("feature_set"),
            record.get("seed"),
        )
        if (
            not isinstance(model, str)
            or not isinstance(feature_set, str)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise CorrectionSourceError("residual publication context is invalid")
        split_ids = record.get("split_ids")
        if not isinstance(split_ids, list) or not all(
            isinstance(split_id, str) for split_id in split_ids
        ):
            raise CorrectionSourceError("residual publication context is invalid")
        context = EventResidualContext(model, feature_set, seed, tuple(split_ids))
        contexts.append(
            _require_canonical_context(context, description="residual publication context")
        )
    keys = [_context_key(context) for context in contexts]
    if len(contexts) != len(set(contexts)) or len(keys) != len(set(keys)):
        raise CorrectionSourceError("residual publication contexts are duplicated")
    return tuple(contexts)


def _frame_contexts(frame: pl.DataFrame) -> set[tuple[str, str, int]]:
    required = {"model", "feature_set", "seed"}
    if not required.issubset(frame.columns):
        raise CorrectionSourceError("point publication context columns are missing")
    return {
        (str(model), str(feature_set), int(seed))
        for model, feature_set, seed in frame.select(
            "model", "feature_set", "seed"
        ).unique().iter_rows()
    }


def _validate_final_source_truth(final: pl.DataFrame, matrix: object) -> None:
    _, evaluation_indices = select_fold_samples(matrix, final_fold())
    evaluation = matrix.take(evaluation_indices)
    expected_origin = np.repeat(evaluation.origins, 24).astype("datetime64[us]")
    expected_timestamp = evaluation.target_times.reshape(-1).astype("datetime64[us]")
    expected_horizon = np.tile(np.arange(1, 25, dtype=np.int64), len(evaluation_indices))
    expected_observed = evaluation.target.reshape(-1).astype(float)
    for stream in final.partition_by(["model", "feature_set", "seed"], maintain_order=True):
        ordered = stream.sort("target_timestamp")
        if (
            ordered.height != expected_observed.size
            or not np.array_equal(
                ordered["origin"].to_numpy().astype("datetime64[us]"), expected_origin
            )
            or not np.array_equal(
                ordered["target_timestamp"].to_numpy().astype("datetime64[us]"),
                expected_timestamp,
            )
            or not np.array_equal(ordered["horizon"].to_numpy(), expected_horizon)
            or not np.array_equal(ordered["observed_mw"].to_numpy(), expected_observed)
        ):
            raise CorrectionSourceError(
                "final baseline observed targets differ from source-derived truth"
            )


def validate_correction_source(
    *, run_dir: Path, config_path: Path, profile: str
) -> ValidatedCorrectionSource:
    """Rebuild source truth and validate canonical OOF/final publications without fitting."""

    if profile not in {"paper", "smoke"}:
        raise CorrectionSourceError("profile must be paper or smoke")
    run = Path(run_dir)
    sources, source_hashes, _ = _resolve_source_paths(run, Path(config_path))
    load_config(sources["experiment_config"])
    events = load_event_registry(sources["event_registry"])
    calendar = load_holiday_calendar(sources["holiday_calendar"])
    validate_feature_event_alignment(calendar, events)
    availability = load_temporary_holiday_availability(
        sources["temporary_holiday_availability"]
    )
    residual_manifest_path = run / "inputs/standardized_residuals_manifest.json"
    residual_manifest = load_standardized_residual_manifest(
        residual_manifest_path,
        run_dir=run,
        config_sha256=source_hashes["experiment_config"],
        event_sha256=source_hashes["event_registry"],
    )
    source_profile = residual_manifest.get("profile")
    if source_profile not in {"paper", "smoke"} or (
        profile == "paper" and source_profile != "paper"
    ):
        raise CorrectionSourceError("residual publication profile cannot serve this fit")

    baseline_manifest_path = run / "predictions/baseline_manifest.json"
    baseline_manifest = _read_canonical_json(baseline_manifest_path, "baseline manifest")
    if baseline_manifest.get("profile") != source_profile:
        raise CorrectionSourceError("baseline profile differs")
    stages = baseline_manifest.get("stages")
    if not isinstance(stages, dict) or set(stages) != {"oof", "final"}:
        raise CorrectionSourceError("both canonical baseline stages are required")
    models = baseline_manifest.get("models")
    feature_sets = baseline_manifest.get("feature_sets")
    if (
        not isinstance(models, list)
        or not isinstance(feature_sets, list)
        or not models
        or not feature_sets
        or len(models) != len(set(models))
        or len(feature_sets) != len(set(feature_sets))
        or any(model not in MODEL_NAMES for model in models)
        or any(feature not in _FEATURE_SETS for feature in feature_sets)
    ):
        raise CorrectionSourceError("baseline manifest model coverage is invalid")
    if source_profile == "paper" and (
        tuple(models) != MODEL_NAMES or tuple(feature_sets) not in PAPER_FEATURE_SUITES
    ):
        raise CorrectionSourceError(
            "paper correction requires all models and an exact paper feature suite"
        )

    artifacts, _ = _preflight_baseline_publication(run, baseline_manifest)

    baseline_config = load_paper_baselines(sources["model_config"])
    audited = audit_hourly_data(
        read_hourly_data(sources["data"]),
        **_paper_bounds(str(source_profile), availability),
    )
    featured = attach_calendar_features(audited, calendar)
    matrices = {
        feature_set: build_daily_forecast_matrix(featured, feature_set=feature_set)
        for feature_set in feature_sets
    }
    artifact_hashes = {
        _SOURCE_TO_BASELINE_HASH[name]: source_hashes[name] for name in _SOURCE_KEYS
    }
    if set(artifact_hashes) != set(PAPER_HASH_KEYS):
        raise CorrectionSourceError("baseline source hash schema differs")
    execution_overrides = baseline_manifest.get("execution_overrides")
    smoke_rounds = (
        execution_overrides.get("boosting_rounds")
        if isinstance(execution_overrides, dict)
        else None
    )
    classical_seed = baseline_manifest.get("classical_seed")
    if isinstance(classical_seed, bool) or not isinstance(classical_seed, int):
        raise CorrectionSourceError("baseline classical seed is invalid")
    final_result = run_paper_final_stage(
        matrices=matrices,
        config=baseline_config,
        run_dir=run,
        cache_dir=run / "prediction-stream-cache",
        artifact_hashes=artifact_hashes,
        classical_seed=classical_seed,
        models=tuple(models),
        feature_sets=tuple(feature_sets),
        profile=source_profile,
        smoke_boosting_rounds=smoke_rounds,
    )
    if final_result.fit_count != 0:
        raise CorrectionSourceError("final baseline validation unexpectedly fitted a model")

    if (
        Path(final_result.members_path).resolve()
        != artifacts["final_members"][0].resolve()
        or Path(final_result.point_path).resolve()
        != artifacts["final_point"][0].resolve()
        or Path(final_result.manifest_path).resolve() != baseline_manifest_path.resolve()
    ):
        raise CorrectionSourceError("final baseline reuse returned substituted paths")
    if _read_canonical_json(baseline_manifest_path, "baseline manifest") != baseline_manifest:
        raise CorrectionSourceError("baseline manifest changed during source validation")
    artifacts, frames = _preflight_baseline_publication(run, baseline_manifest)

    residual_entry = residual_manifest.get("outputs", {}).get("standardized_residuals")
    if (
        not isinstance(residual_entry, dict)
        or residual_entry.get("path") != "inputs/standardized_residuals.parquet"
        or not isinstance(residual_entry.get("sha256"), str)
    ):
        raise CorrectionSourceError("standardized residual artifact binding is invalid")
    residual_path = run / "inputs/standardized_residuals.parquet"
    residual_frame = _read_bound_parquet(
        residual_path,
        str(residual_entry["sha256"]),
        "standardized residual artifact",
    )
    oof_point = frames["oof_point"]
    rebuilt_residual = build_standardized_residuals(oof_point, events).frame
    if (
        residual_frame.columns != rebuilt_residual.columns
        or residual_frame.schema != rebuilt_residual.schema
        or not residual_frame.equals(rebuilt_residual, null_equal=True)
    ):
        raise CorrectionSourceError(
            "standardized residual semantics differ from canonical OOF predictions"
        )
    final_point = frames["final_point"]
    _validate_final_source_truth(final_point, matrices[feature_sets[0]])

    available_contexts = _manifest_contexts(residual_manifest)
    expected_contexts = {_context_key(context) for context in available_contexts}
    if _frame_contexts(oof_point) != expected_contexts or _frame_contexts(
        final_point
    ) != expected_contexts:
        raise CorrectionSourceError(
            "OOF/final context coverage differs from the residual publication"
        )
    return ValidatedCorrectionSource(
        run_dir=run,
        source_profile=str(source_profile),
        events=tuple(events),
        source_paths=sources,
        source_hashes=source_hashes,
        available_contexts=available_contexts,
        residual_manifest_path=residual_manifest_path,
        residual_path=residual_path,
        baseline_manifest_path=baseline_manifest_path,
        oof_members_path=artifacts["oof_members"][0],
        oof_point_path=artifacts["oof_point"][0],
        final_members_path=artifacts["final_members"][0],
        final_point_path=artifacts["final_point"][0],
        residual_sha256=str(residual_entry["sha256"]),
        oof_members_sha256=artifacts["oof_members"][1],
        oof_point_sha256=artifacts["oof_point"][1],
        final_members_sha256=artifacts["final_members"][1],
        final_point_sha256=artifacts["final_point"][1],
        _residual_manifest_json=_canonical_json(residual_manifest),
        _baseline_manifest_json=_canonical_json(baseline_manifest),
    )


__all__ = [
    "CorrectionSourceError",
    "ValidatedCorrectionSource",
    "validate_correction_source",
]
