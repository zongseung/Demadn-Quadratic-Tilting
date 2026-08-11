"""Immutable retrospective LOEO residual universes and physical training folds."""

from __future__ import annotations

import fcntl
import json
import math
import os
import stat
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from hashlib import sha256
from pathlib import Path
from types import MappingProxyType
from typing import Any

import polars as pl

from hqrc_v3.correction_source import CorrectionSourceError, ValidatedCorrectionSource
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.events import EventOccurrence
from hqrc_v3.provenance import file_sha256
from hqrc_v3.residual_stage import STANDARDIZED_RESIDUAL_COLUMNS

_SCHEMA_VERSION = "hqrc-v3.loeo-universe.v1"
_ROOT_NAMESPACE = frozenset({".loeo.lock", "generations", "current.json"})
_GENERATION_NAMESPACE = frozenset({"universe.parquet", "folds", "manifest.json", "COMPLETE"})
_COLUMNS = (*STANDARDIZED_RESIDUAL_COLUMNS, "causal")
_YEARS = tuple(range(2020, 2025))


class LOEOError(ValueError):
    """Raised when a LOEO publication cannot be trusted or constructed."""


@dataclass(frozen=True, slots=True)
class LOEOPublication:
    """Hash-bound immutable universe plus all physical fold identities."""

    output_dir: Path
    generation_dir: Path
    manifest_path: Path
    universe_path: Path
    universe_sha256: str
    occurrence_ids: tuple[str, ...]
    fold_paths: Mapping[str, Path]
    fold_sha256: Mapping[str, str]
    context: EventResidualContext
    causal: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "generation_dir", Path(self.generation_dir))
        object.__setattr__(self, "manifest_path", Path(self.manifest_path))
        object.__setattr__(self, "universe_path", Path(self.universe_path))
        object.__setattr__(
            self,
            "fold_paths",
            MappingProxyType({key: Path(value) for key, value in self.fold_paths.items()}),
        )
        object.__setattr__(self, "fold_sha256", MappingProxyType(dict(self.fold_sha256)))


@dataclass(frozen=True, slots=True)
class LOEOFold:
    """One re-read physical training fold, never a view supplied by a caller."""

    held_out_occurrence_id: str
    occurrence_ids: tuple[str, ...]
    path: Path
    residual_sha256: str
    frame: pl.DataFrame
    context: EventResidualContext
    causal: bool


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise LOEOError("LOEO metadata is not canonical JSON") from error


def _sha_json(value: object) -> str:
    return sha256(_canonical_json(value)).hexdigest()


def _require_real_file(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise LOEOError(f"{description} is missing or unsafe") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise LOEOError(f"{description} is missing or unsafe")


def _require_real_directory(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise LOEOError(f"{description} is missing or unsafe") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise LOEOError(f"{description} is missing or unsafe")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    _require_real_file(path, description)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise LOEOError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise LOEOError(f"{description} is not canonical JSON")
    return value


def _write_json(path: Path, value: object) -> None:
    payload = _canonical_json(value)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


@contextmanager
def _publication_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".loeo.lock"
    if lock_path.exists():
        _require_real_file(lock_path, "LOEO publication lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _context_mapping(context: EventResidualContext) -> dict[str, object]:
    if not isinstance(context, EventResidualContext):
        raise LOEOError("LOEO requires one EventResidualContext")
    return {
        "model": context.model,
        "feature_set": context.feature_set,
        "seed": context.seed,
        "split_ids": list(context.split_ids),
    }


def _ordered_events(events: tuple[EventOccurrence, ...]) -> tuple[EventOccurrence, ...]:
    if not isinstance(events, tuple) or len(events) != 10:
        raise LOEOError("LOEO requires the exact ten-event correction registry")
    ordered = tuple(sorted(events, key=lambda event: (event.central_date, event.occurrence_id)))
    expected = tuple(f"{holiday}-{year}" for year in _YEARS for holiday in ("seollal", "chuseok"))
    if tuple(event.occurrence_id for event in ordered) != expected:
        raise LOEOError("LOEO registry occurrence identities or ordering differ")
    if any(event.holiday_type != event.occurrence_id.split("-", 1)[0] for event in ordered):
        raise LOEOError("LOEO registry holiday identities differ")
    return ordered


def _event_grid(events: tuple[EventOccurrence, ...]) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for order, event in enumerate(events):
        start = datetime.combine(event.window_start, time.min)
        end = datetime.combine(event.window_end + timedelta(days=1), time.min)
        timestamp = start
        while timestamp < end:
            rows.append(
                {
                    "event_order": order,
                    "target_timestamp": timestamp,
                    "split_id": f"oof-{event.central_date.year}"
                    if event.central_date.year < 2024
                    else "final-2024",
                    "occurrence_id": event.occurrence_id,
                    "holiday_type": event.holiday_type,
                    "tau_days": (
                        timestamp - datetime.combine(event.central_date, time.min)
                    ).total_seconds()
                    / 86400.0,
                    "hour": timestamp.hour,
                    "restriction": event.restriction,
                }
            )
            timestamp += timedelta(hours=1)
    grid = pl.DataFrame(rows).with_columns(
        pl.col("event_order").cast(pl.Int64),
        pl.col("target_timestamp").cast(pl.Datetime("ns")),
        pl.col("split_id").cast(pl.String),
        pl.col("occurrence_id").cast(pl.String),
        pl.col("holiday_type").cast(pl.String),
        pl.col("tau_days").cast(pl.Float64),
        pl.col("hour").cast(pl.Int64),
        pl.col("restriction").cast(pl.Int64),
    )
    if grid.height != 1_296:
        raise LOEOError("LOEO registry grids must contain exactly 1,296 hourly rows")
    return grid


def _require_frame_columns(frame: pl.DataFrame, columns: tuple[str, ...], description: str) -> None:
    if not isinstance(frame, pl.DataFrame) or tuple(frame.columns) != columns:
        raise LOEOError(f"{description} schema differs")


def _same_frame(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    return (
        left.columns == right.columns
        and left.schema == right.schema
        and left.equals(right, null_equal=True)
    )


def _validate_grid(frame: pl.DataFrame, grid: pl.DataFrame, description: str) -> None:
    metadata = (
        "target_timestamp",
        "split_id",
        "occurrence_id",
        "holiday_type",
        "tau_days",
        "hour",
        "restriction",
    )
    actual = frame.select(metadata).sort("target_timestamp")
    expected = grid.select(metadata).sort("target_timestamp")
    if not _same_frame(actual, expected):
        raise LOEOError(f"{description} event grids or metadata differ from the registry")


def _source_identity(
    source: ValidatedCorrectionSource,
    context: EventResidualContext,
    events: tuple[EventOccurrence, ...],
) -> dict[str, object]:
    return {
        "context": _context_mapping(context),
        "registry": [
            {
                "occurrence_id": event.occurrence_id,
                "holiday_type": event.holiday_type,
                "central_date": event.central_date.isoformat(),
                "official_start": event.official_start.isoformat(),
                "official_end": event.official_end.isoformat(),
                "restriction": event.restriction,
            }
            for event in events
        ],
        "source_hashes": dict(sorted(source.source_hashes.items())),
        "residual_sha256": source.residual_sha256,
        "oof_point_sha256": source.oof_point_sha256,
        "final_point_sha256": source.final_point_sha256,
    }


def _build_universe(
    source: ValidatedCorrectionSource, context: EventResidualContext
) -> tuple[pl.DataFrame, tuple[EventOccurrence, ...], dict[str, object]]:
    if not isinstance(source, ValidatedCorrectionSource):
        raise TypeError("LOEO requires a ValidatedCorrectionSource")
    if context not in source.available_contexts:
        raise LOEOError("LOEO context is not an exact validated source context")
    events = _ordered_events(source.events)
    grid = _event_grid(events)
    oof_grid = grid.filter(pl.col("split_id") != "final-2024")
    final_grid = grid.filter(pl.col("split_id") == "final-2024")
    try:
        oof = source.load_standardized_context(context, through=2023)
        final = source.load_final_point_context(context)
    except (CorrectionSourceError, ValueError, TypeError) as error:
        raise LOEOError("validated correction source could not reload LOEO inputs") from error
    _require_frame_columns(oof, STANDARDIZED_RESIDUAL_COLUMNS, "standardized OOF source")
    _validate_grid(oof, oof_grid, "standardized OOF source")
    if (
        oof["standardized_residual"].null_count()
        or not oof.select(pl.col("standardized_residual").is_finite().all()).item()
    ):
        raise LOEOError("standardized OOF source residuals are invalid")
    scale = oof.filter(pl.col("split_id") == "oof-2023")["sigma_n_mw"].unique()
    if scale.len() != 1 or not math.isfinite(float(scale.item())) or float(scale.item()) <= 0:
        raise LOEOError("LOEO 2024 scale must be the unique positive oof-2023 sigma_N")
    final_required = (
        "origin",
        "target_timestamp",
        "horizon",
        "observed_mw",
        "predicted_mw",
        "model",
        "feature_set",
        "seed",
        "split_id",
    )
    if not isinstance(final, pl.DataFrame) or not set(final_required).issubset(final.columns):
        raise LOEOError("final point source schema differs")
    if any(final[column].null_count() for column in final_required):
        raise LOEOError("final point source contains null LOEO inputs")
    final_values = final.select(final_required).join(
        final_grid, on=["target_timestamp", "split_id"], how="inner"
    )
    if (
        final_values.height != final_grid.height
        or final_values.select("target_timestamp").is_duplicated().any()
    ):
        raise LOEOError("final point source event metadata differs from the registry")
    final_standardized = final_values.with_columns(
        (pl.col("observed_mw") - pl.col("predicted_mw")).cast(pl.Float64).alias("residual_mw"),
        pl.lit(float(scale.item()), dtype=pl.Float64).alias("sigma_n_mw"),
    ).with_columns(
        (pl.col("residual_mw") / pl.col("sigma_n_mw")).alias("standardized_residual"),
    )
    oof_ordered = oof.join(
        oof_grid.select("target_timestamp", "event_order"), on="target_timestamp", how="inner"
    )
    final_ordered = final_standardized
    universe = (
        pl.concat(
            [
                oof_ordered.select(STANDARDIZED_RESIDUAL_COLUMNS + ("event_order",)),
                final_ordered.select(STANDARDIZED_RESIDUAL_COLUMNS + ("event_order",)),
            ],
            how="vertical",
        )
        .sort("event_order", "target_timestamp")
        .drop("event_order")
        .with_columns(pl.lit(False, dtype=pl.Boolean).alias("causal"))
        .select(_COLUMNS)
    )
    _validate_universe_frame(universe, events)
    return universe, events, _source_identity(source, context, events)


def _validate_universe_frame(frame: pl.DataFrame, events: tuple[EventOccurrence, ...]) -> None:
    _require_frame_columns(frame, _COLUMNS, "LOEO universe")
    grid = _event_grid(events)
    if (
        frame.height != 1_296
        or frame["causal"].dtype != pl.Boolean
        or frame["causal"].unique().to_list() != [False]
    ):
        raise LOEOError("LOEO universe causal metadata differs")
    _validate_grid(frame, grid, "LOEO universe")
    if frame.select(pl.col("occurrence_id").unique(maintain_order=True))[
        "occurrence_id"
    ].to_list() != [event.occurrence_id for event in events]:
        raise LOEOError("LOEO universe occurrence ordering differs")
    # The registry grid, rather than lexical IDs, defines canonical occurrence order.
    ordered = (
        frame.join(
            grid.select("target_timestamp", "event_order"), on="target_timestamp", how="left"
        )
        .sort("event_order", "target_timestamp")
        .drop("event_order")
    )
    if not _same_frame(frame, ordered):
        raise LOEOError("LOEO universe rows are not canonically ordered")
    if any(frame[column].null_count() for column in _COLUMNS):
        raise LOEOError("LOEO universe contains null values")
    if not frame.select(pl.col("standardized_residual").is_finite().all()).item():
        raise LOEOError("LOEO universe standardized residuals are invalid")


def _generation_identity(source_identity: Mapping[str, object]) -> str:
    return _sha_json(
        {"schema_version": _SCHEMA_VERSION, "source": source_identity, "causal": False}
    )


def _manifest_payload(
    *,
    identity: str,
    source: Mapping[str, object],
    universe_path: Path,
    universe_sha: str,
    universe: pl.DataFrame,
    events: tuple[EventOccurrence, ...],
    folds: Mapping[str, tuple[Path, str, pl.DataFrame]],
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "schema_version": _SCHEMA_VERSION,
        "generation_identity": identity,
        "causal": False,
        "source": dict(source),
        "occurrence_ids": [event.occurrence_id for event in events],
        "universe": {
            "path": universe_path.as_posix(),
            "sha256": universe_sha,
            "rows": universe.height,
        },
        "folds": {
            held_out: {
                "path": path.as_posix(),
                "sha256": digest,
                "rows": frame.height,
                "occurrence_ids": frame["occurrence_id"].unique(maintain_order=True).to_list(),
            }
            for held_out, (path, digest, frame) in folds.items()
        },
    }
    return {**unsigned, "manifest_sha256": _sha_json(unsigned)}


def _validate_root(root: Path, *, allow_empty: bool) -> None:
    if root.exists():
        _require_real_directory(root, "LOEO publication directory")
    elif allow_empty:
        root.mkdir(parents=True, mode=0o700)
    else:
        raise LOEOError("LOEO publication directory is missing")
    entries = {path.name: path for path in root.iterdir()}
    if not set(entries).issubset(_ROOT_NAMESPACE):
        raise LOEOError("LOEO publication directory contains unknown entries")
    for name, path in entries.items():
        if name == "generations":
            _require_real_directory(path, "LOEO generation namespace")
        else:
            _require_real_file(path, f"LOEO publication {name}")


def _publication_from_manifest(
    root: Path, generation: Path, manifest: Mapping[str, Any], context: EventResidualContext
) -> LOEOPublication:
    universe = manifest["universe"]
    folds = manifest["folds"]
    return LOEOPublication(
        output_dir=root,
        generation_dir=generation,
        manifest_path=generation / "manifest.json",
        universe_path=generation / str(universe["path"]),
        universe_sha256=str(universe["sha256"]),
        occurrence_ids=tuple(manifest["occurrence_ids"]),
        fold_paths={held: generation / str(entry["path"]) for held, entry in folds.items()},
        fold_sha256={held: str(entry["sha256"]) for held, entry in folds.items()},
        context=context,
        causal=False,
    )


def _load_publication(
    source: ValidatedCorrectionSource, context: EventResidualContext, output_dir: Path
) -> tuple[LOEOPublication, pl.DataFrame, tuple[EventOccurrence, ...]]:
    expected, events, source_identity = _build_universe(source, context)
    root = Path(output_dir)
    _validate_root(root, allow_empty=False)
    current_path = root / "current.json"
    current = _read_json(current_path, "LOEO current pointer")
    identity = _generation_identity(source_identity)
    if (
        set(current) != {"generation", "generation_identity", "manifest_sha256"}
        or current.get("generation") != identity
        or current.get("generation_identity") != identity
    ):
        raise LOEOError("LOEO current publication is incompatible with validated inputs")
    generations = root / "generations"
    _require_real_directory(generations, "LOEO generation namespace")
    entries = {path.name: path for path in generations.iterdir()}
    if set(entries) != {identity}:
        raise LOEOError("LOEO generation namespace contains partial or incompatible entries")
    generation = generations / identity
    _require_real_directory(generation, "LOEO generation")
    generation_entries = {path.name: path for path in generation.iterdir()}
    if set(generation_entries) != _GENERATION_NAMESPACE:
        raise LOEOError("LOEO completed generation contains unknown or partial entries")
    _require_real_directory(generation / "folds", "LOEO fold namespace")
    for name in ("universe.parquet", "manifest.json", "COMPLETE"):
        _require_real_file(generation / name, f"LOEO generation {name}")
    manifest = _read_json(generation / "manifest.json", "LOEO manifest")
    complete = _read_json(generation / "COMPLETE", "LOEO completion marker")
    if (
        set(complete) != {"manifest_sha256"}
        or complete["manifest_sha256"] != manifest.get("manifest_sha256")
        or current["manifest_sha256"] != manifest.get("manifest_sha256")
    ):
        raise LOEOError("LOEO completion hashes differ")
    required = {
        "schema_version",
        "generation_identity",
        "causal",
        "source",
        "occurrence_ids",
        "universe",
        "folds",
        "manifest_sha256",
    }
    if (
        set(manifest) != required
        or manifest.get("schema_version") != _SCHEMA_VERSION
        or manifest.get("generation_identity") != identity
        or manifest.get("causal") is not False
        or manifest.get("source") != source_identity
    ):
        raise LOEOError("LOEO manifest identity differs")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    if manifest["manifest_sha256"] != _sha_json(unsigned):
        raise LOEOError("LOEO manifest hash differs")
    expected_ids = tuple(event.occurrence_id for event in events)
    if manifest.get("occurrence_ids") != list(expected_ids):
        raise LOEOError("LOEO manifest registry differs")
    universe = manifest.get("universe")
    if (
        not isinstance(universe, dict)
        or universe.get("path") != "universe.parquet"
        or universe.get("rows") != 1_296
        or not isinstance(universe.get("sha256"), str)
    ):
        raise LOEOError("LOEO universe manifest entry differs")
    universe_path = generation / "universe.parquet"
    if file_sha256(universe_path) != universe["sha256"]:
        raise LOEOError("LOEO universe hash differs")
    try:
        actual_universe = pl.read_parquet(universe_path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise LOEOError("LOEO universe is unreadable") from error
    _validate_universe_frame(actual_universe, events)
    if not _same_frame(actual_universe, expected):
        raise LOEOError("LOEO universe semantics differ from validated sources")
    folds = manifest.get("folds")
    if not isinstance(folds, dict) or set(folds) != set(expected_ids):
        raise LOEOError("LOEO fold manifest identities differ")
    fold_files = {path.name: path for path in (generation / "folds").iterdir()}
    expected_files = {f"{identifier}.parquet" for identifier in expected_ids}
    if set(fold_files) != expected_files:
        raise LOEOError("LOEO fold namespace contains unknown or missing entries")
    for held_out in expected_ids:
        entry = folds[held_out]
        expected_occurrences = tuple(
            identifier for identifier in expected_ids if identifier != held_out
        )
        expected_fold = expected.filter(pl.col("occurrence_id") != held_out)
        if (
            not isinstance(entry, dict)
            or entry.get("path") != f"folds/{held_out}.parquet"
            or entry.get("rows") != expected_fold.height
            or entry.get("occurrence_ids") != list(expected_occurrences)
            or not isinstance(entry.get("sha256"), str)
        ):
            raise LOEOError("LOEO fold manifest entry differs")
        path = generation / str(entry["path"])
        _require_real_file(path, f"LOEO fold {held_out}")
        if file_sha256(path) != entry["sha256"]:
            raise LOEOError("LOEO fold hash differs")
        try:
            actual_fold = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError) as error:
            raise LOEOError("LOEO fold is unreadable") from error
        _require_frame_columns(actual_fold, _COLUMNS, "LOEO fold")
        if actual_fold["causal"].dtype != pl.Boolean or actual_fold[
            "causal"
        ].unique().to_list() != [False]:
            raise LOEOError("LOEO fold causal metadata differs")
        if held_out in actual_fold["occurrence_id"].unique().to_list() or not _same_frame(
            actual_fold, expected_fold
        ):
            raise LOEOError("LOEO fold contains held-out rows or differs semantically")
    return _publication_from_manifest(root, generation, manifest, context), actual_universe, events


def publish_loeo_universe(
    source: ValidatedCorrectionSource, context: EventResidualContext, *, output_dir: Path
) -> LOEOPublication:
    """Construct and atomically publish the one immutable ten-event LOEO universe."""

    universe, events, source_identity = _build_universe(source, context)
    root = Path(output_dir)
    _validate_root(root, allow_empty=True)
    with _publication_lock(root):
        _validate_root(root, allow_empty=False)
        current = root / "current.json"
        if current.exists():
            publication, _, _ = _load_publication(source, context, root)
            return publication
        entries = {path.name for path in root.iterdir() if path.name != ".loeo.lock"}
        if entries:
            raise LOEOError("LOEO publication directory contains a partial generation")
        identity = _generation_identity(source_identity)
        generations = root / "generations"
        generations.mkdir(mode=0o700)
        _require_real_directory(generations, "LOEO generation namespace")
        staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=generations))
        try:
            universe_path = staging / "universe.parquet"
            universe.write_parquet(universe_path)
            folds_dir = staging / "folds"
            folds_dir.mkdir(mode=0o700)
            folds: dict[str, tuple[Path, str, pl.DataFrame]] = {}
            for event in events:
                held_out = event.occurrence_id
                frame = universe.filter(pl.col("occurrence_id") != held_out)
                path = folds_dir / f"{held_out}.parquet"
                frame.write_parquet(path)
                folds[held_out] = (Path(f"folds/{held_out}.parquet"), file_sha256(path), frame)
            manifest = _manifest_payload(
                identity=identity,
                source=source_identity,
                universe_path=Path("universe.parquet"),
                universe_sha=file_sha256(universe_path),
                universe=universe,
                events=events,
                folds=folds,
            )
            _write_json(staging / "manifest.json", manifest)
            _write_json(staging / "COMPLETE", {"manifest_sha256": manifest["manifest_sha256"]})
            target = generations / identity
            if target.exists():
                raise LOEOError("LOEO generation already exists without a trusted pointer")
            os.replace(staging, target)
            pointer = root / ".current.tmp"
            _write_json(
                pointer,
                {
                    "generation": identity,
                    "generation_identity": identity,
                    "manifest_sha256": manifest["manifest_sha256"],
                },
            )
            os.replace(pointer, current)
        except Exception:
            if staging.exists():
                # A staging directory is never trusted; leave no reusable partial state behind.
                for path in sorted(staging.rglob("*"), reverse=True):
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                    elif path.is_dir():
                        path.rmdir()
                staging.rmdir()
            raise
    publication, _, _ = _load_publication(source, context, root)
    return publication


def load_loeo_universe(
    source: ValidatedCorrectionSource, context: EventResidualContext, *, output_dir: Path
) -> LOEOPublication:
    """Reload and fully revalidate a published universe against immutable source inputs."""

    publication, _, _ = _load_publication(source, context, Path(output_dir))
    return publication


def load_loeo_fold(
    source: ValidatedCorrectionSource,
    context: EventResidualContext,
    *,
    output_dir: Path,
    held_out_occurrence_id: str,
) -> LOEOFold:
    """Re-read one physical nine-event Parquet after complete publication validation."""

    publication, _, _ = _load_publication(source, context, Path(output_dir))
    if held_out_occurrence_id not in publication.occurrence_ids:
        raise LOEOError("LOEO held-out occurrence is not registered")
    path = publication.fold_paths[held_out_occurrence_id]
    try:
        frame = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise LOEOError("LOEO fold is unreadable") from error
    occurrence_ids = tuple(
        identifier
        for identifier in publication.occurrence_ids
        if identifier != held_out_occurrence_id
    )
    if tuple(frame["occurrence_id"].unique(maintain_order=True).to_list()) != occurrence_ids:
        raise LOEOError("LOEO fold occurrence ordering differs")
    return LOEOFold(
        held_out_occurrence_id,
        occurrence_ids,
        path,
        publication.fold_sha256[held_out_occurrence_id],
        frame,
        context,
        False,
    )


__all__ = [
    "LOEOError",
    "LOEOFold",
    "LOEOPublication",
    "load_loeo_fold",
    "load_loeo_universe",
    "publish_loeo_universe",
]
