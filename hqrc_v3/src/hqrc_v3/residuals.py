"""Fold-scoped residual standardization and immutable prediction artifacts."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hqrc_v3.contracts import PREDICTION_COLUMNS, DataContractError
from hqrc_v3.provenance import ArtifactMismatch, file_sha256
from hqrc_v3.splits import is_oof_split_id

_RESIDUAL_COLUMNS = ("is_event", "residual_mw")
_CACHE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class ResidualContext:
    """The model fit identity that owns an OOF residual scale."""

    model: str
    feature_set: str
    seed: int
    split_id: str


@dataclass(frozen=True)
class PredictionCacheEntry:
    """A prediction frame and the fitted preprocessing population that owns it."""

    frame: pl.DataFrame
    population_contract: dict[str, dict[str, object]] | None


class FoldScale(float):
    """A numeric residual scale carrying the immutable context that produced it."""

    context: ResidualContext

    def __new__(cls, value: float, context: ResidualContext) -> FoldScale:
        scale = super().__new__(cls, value)
        scale.context = context
        return scale


def _residual_context(frame: pl.DataFrame) -> ResidualContext:
    missing = (set(PREDICTION_COLUMNS) | set(_RESIDUAL_COLUMNS)) - set(frame.columns)
    if missing:
        raise DataContractError(f"residual frame is missing required columns: {sorted(missing)}")
    if frame.is_empty():
        raise DataContractError("residual frame must not be empty")
    if frame["is_event"].dtype != pl.Boolean or frame["is_event"].null_count():
        raise DataContractError("is_event must be a non-null boolean column")
    residual_dtype = frame["residual_mw"].dtype
    if frame["residual_mw"].null_count() or residual_dtype not in {pl.Float32, pl.Float64}:
        raise DataContractError("residual_mw must be a non-null floating point column")
    if not frame.select(pl.col("residual_mw").is_finite().all()).item():
        raise DataContractError("residual_mw must be finite")
    identifiers = ("model", "feature_set", "seed", "split_id")
    if any(column not in frame.columns or frame[column].null_count() for column in identifiers):
        raise DataContractError("residual frame must have complete identifiers")
    if any(frame[column].n_unique() != 1 for column in identifiers):
        raise DataContractError("residual scale requires one model/feature/seed/fold context")
    return ResidualContext(
        model=str(frame["model"].item(0)),
        feature_set=str(frame["feature_set"].item(0)),
        seed=int(frame["seed"].item(0)),
        split_id=str(frame["split_id"].item(0)),
    )


def compute_fold_scale(frame: pl.DataFrame) -> FoldScale:
    """Compute RMS scale from only non-event residuals in one OOF context."""

    context = _residual_context(frame)
    if not is_oof_split_id(context.split_id):
        raise DataContractError("fold scale must be computed from an OOF prediction context")
    non_event = frame.filter(~pl.col("is_event"))["residual_mw"]
    if non_event.is_empty():
        raise DataContractError("cannot compute a fold scale without non-event residuals")
    scale = math.sqrt(float((non_event * non_event).mean()))
    if not math.isfinite(scale) or scale <= 0:
        raise DataContractError("fold scale must be finite and positive")
    return FoldScale(scale, context)


def standardize_event_residuals(frame: pl.DataFrame, scale: FoldScale) -> pl.DataFrame:
    """Add event residual z-scores, refusing a scale from a different OOF fold."""

    context = _residual_context(frame)
    if not isinstance(scale, FoldScale) or scale.context != context:
        raise DataContractError("event residual scale context does not match the prediction frame")
    if not math.isfinite(scale) or scale <= 0:
        raise DataContractError("fold scale must be finite and positive")
    if not frame.select(pl.col("is_event").all()).item():
        raise DataContractError("standardize_event_residuals accepts event rows only")
    return frame.with_columns((pl.col("residual_mw") / float(scale)).alias("standardized_residual"))


class PredictionCache:
    """A hash-immutable Parquet cache with JSON metadata as its completion marker."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read(self, key: str, expected_hashes: Mapping[str, str]) -> pl.DataFrame | None:
        """Return None when absent; return a compatible frame; raise on hash mismatch."""

        self._paths(key)
        with self._lock(key, exclusive=False):
            return self._read_unlocked(key, expected_hashes)

    def read_entry(
        self, key: str, expected_hashes: Mapping[str, str]
    ) -> PredictionCacheEntry | None:
        """Read a frame together with integrity-bound fitted population provenance."""

        self._paths(key)
        with self._lock(key, exclusive=False):
            return self._read_entry_unlocked(key, expected_hashes)

    def _read_unlocked(self, key: str, expected_hashes: Mapping[str, str]) -> pl.DataFrame | None:
        entry = self._read_entry_unlocked(key, expected_hashes)
        return None if entry is None else entry.frame

    def _read_entry_unlocked(
        self, key: str, expected_hashes: Mapping[str, str]
    ) -> PredictionCacheEntry | None:
        parquet_path, metadata_path = self._paths(key)
        parquet_exists, metadata_exists = parquet_path.is_file(), metadata_path.is_file()
        if not parquet_exists and not metadata_exists:
            return None
        if parquet_exists != metadata_exists:
            raise ArtifactMismatch(f"partial prediction cache artifact for key {key!r}")
        metadata = self._read_metadata(metadata_path)
        self._assert_hashes(metadata, expected_hashes)
        required = {
            "schema_version",
            "hashes",
            "parquet_sha256",
            "population_contract",
            "entry_sha256",
        }
        if (
            set(metadata) != required
            or type(metadata.get("schema_version")) is not int
            or metadata["schema_version"] != 2
            or not isinstance(metadata.get("parquet_sha256"), str)
            or not isinstance(metadata.get("entry_sha256"), str)
        ):
            raise ArtifactMismatch(f"invalid cache metadata {metadata_path.name}")
        unsigned = {name: metadata[name] for name in sorted(required - {"entry_sha256"})}
        if metadata["entry_sha256"] != self._entry_digest(unsigned):
            raise ArtifactMismatch(
                f"prediction cache metadata entry digest differs for key {key!r}"
            )
        try:
            parquet_digest = file_sha256(parquet_path)
        except OSError as error:
            raise ArtifactMismatch(
                f"unable to hash cached prediction artifact {key!r}: {error}"
            ) from error
        if parquet_digest != metadata["parquet_sha256"]:
            raise ArtifactMismatch(f"prediction cache parquet digest differs for key {key!r}")
        population = self._normalize_population(metadata["population_contract"])
        try:
            frame = pl.read_parquet(parquet_path)
        except (OSError, pl.exceptions.PolarsError) as error:
            raise ArtifactMismatch(
                f"unable to read cached prediction artifact {key!r}: {error}"
            ) from error
        from hqrc_v3.contracts import validate_prediction_frame

        return PredictionCacheEntry(
            frame=validate_prediction_frame(frame),
            population_contract=population,
        )

    def write(self, key: str, frame: pl.DataFrame, hashes: Mapping[str, str]) -> pl.DataFrame:
        """Atomically persist a validated frame and return it without overwriting mismatches."""

        from hqrc_v3.contracts import validate_prediction_frame

        normalized = validate_prediction_frame(frame)
        hashes_dict = self._normalize_hashes(hashes)
        self._paths(key)
        with self._lock(key, exclusive=True):
            self._cleanup_stale_partial(key)
            existing = self._read_unlocked(key, expected_hashes=hashes_dict)
            if existing is not None:
                return existing
            return self._publish_unlocked(key, normalized, hashes_dict)

    def write_entry(
        self,
        key: str,
        frame: pl.DataFrame,
        hashes: Mapping[str, str],
        *,
        population_contract: Mapping[str, Mapping[str, object]],
    ) -> PredictionCacheEntry:
        """Atomically persist predictions and their actual fitted population."""

        from hqrc_v3.contracts import validate_prediction_frame

        normalized_frame = validate_prediction_frame(frame)
        normalized_hashes = self._normalize_hashes(hashes)
        normalized_population = self._normalize_population(population_contract)
        if normalized_population is None:  # pragma: no cover - mapping cannot normalize to None
            raise DataContractError("fitted preprocessing population must not be null")
        self._paths(key)
        with self._lock(key, exclusive=True):
            self._cleanup_stale_partial(key)
            existing = self._read_entry_unlocked(key, expected_hashes=normalized_hashes)
            if existing is not None:
                if existing.population_contract != normalized_population:
                    raise ArtifactMismatch(
                        "fitted preprocessing population differs from the cached artifact"
                    )
                return existing
            return self._publish_entry_unlocked(
                key,
                normalized_frame,
                normalized_hashes,
                normalized_population,
            )

    def _paths(self, key: str) -> tuple[Path, Path]:
        if not isinstance(key, str) or not _CACHE_KEY.fullmatch(key) or key in {".", ".."}:
            raise ValueError("prediction cache key must be a safe non-empty filename")
        return self.root / f"{key}.parquet", self.root / f"{key}.json"

    def _temporary_path(self, key: str, suffix: str) -> Path:
        descriptor, path = tempfile.mkstemp(prefix=f".{key}.", suffix=suffix, dir=self.root)
        os.close(descriptor)
        return Path(path)

    @contextmanager
    def _lock(self, key: str, *, exclusive: bool) -> Iterator[None]:
        """Hold a per-key advisory lock, released by the OS if the owner crashes."""

        self._paths(key)
        lock_path = self.root / f".{key}.lock"
        with lock_path.open("a+") as lock_file:
            mode = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_file.fileno(), mode)
            try:
                yield
            finally:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _cleanup_stale_partial(self, key: str) -> None:
        """Remove only an incomplete pair while holding the exclusive key lock."""

        parquet_path, metadata_path = self._paths(key)
        parquet_exists, metadata_exists = parquet_path.is_file(), metadata_path.is_file()
        if parquet_exists == metadata_exists:
            return
        parquet_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
        self._fsync_directory()

    def _publish_unlocked(
        self, key: str, frame: pl.DataFrame, hashes: Mapping[str, str]
    ) -> pl.DataFrame:
        """Publish a new pair while its exclusive key lock prevents observation gaps."""

        return self._publish_entry_unlocked(key, frame, hashes, None).frame

    def _publish_entry_unlocked(
        self,
        key: str,
        frame: pl.DataFrame,
        hashes: Mapping[str, str],
        population_contract: dict[str, dict[str, object]] | None,
    ) -> PredictionCacheEntry:
        """Publish one integrity-bound prediction/provenance pair."""

        parquet_path, metadata_path = self._paths(key)
        parquet_temp = self._temporary_path(key, ".parquet")
        metadata_temp = self._temporary_path(key, ".json")
        published_parquet = False
        try:
            frame.write_parquet(parquet_temp)
            self._fsync_file(parquet_temp)
            unsigned_metadata: dict[str, object] = {
                "schema_version": 2,
                "hashes": dict(hashes),
                "parquet_sha256": file_sha256(parquet_temp),
                "population_contract": population_contract,
            }
            metadata = {
                **unsigned_metadata,
                "entry_sha256": self._entry_digest(unsigned_metadata),
            }
            self._write_metadata(metadata_temp, metadata)
            os.replace(parquet_temp, parquet_path)
            published_parquet = True
            os.replace(metadata_temp, metadata_path)
            self._fsync_directory()
        except Exception:
            parquet_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
            if published_parquet:
                parquet_path.unlink(missing_ok=True)
                metadata_path.unlink(missing_ok=True)
            raise
        return PredictionCacheEntry(
            frame=frame,
            population_contract=population_contract,
        )

    @staticmethod
    def _normalize_hashes(hashes: Mapping[str, str]) -> dict[str, str]:
        valid_items = all(
            isinstance(key, str) and key and isinstance(value, str) and value
            for key, value in hashes.items()
        )
        if not hashes or not valid_items:
            raise ValueError("artifact hashes must be a non-empty mapping of names to strings")
        return dict(sorted(hashes.items()))

    def _read_metadata(self, path: Path) -> dict[str, object]:
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ArtifactMismatch(f"invalid cache metadata {path.name}: {error}") from error
        if not isinstance(metadata, dict):
            raise ArtifactMismatch(f"invalid cache metadata {path.name}")
        return metadata

    def _assert_hashes(
        self, metadata: Mapping[str, object], expected_hashes: Mapping[str, str]
    ) -> None:
        expected = self._normalize_hashes(expected_hashes)
        stored = metadata.get("hashes")
        if not isinstance(stored, dict):
            raise ArtifactMismatch("invalid cache metadata hashes")
        for name in sorted(set(stored) | set(expected)):
            if stored.get(name) != expected.get(name):
                raise ArtifactMismatch(f"{name} hash differs from the cached artifact")

    @staticmethod
    def _normalize_population(
        value: object,
    ) -> dict[str, dict[str, object]] | None:
        if value is None:
            return None
        from hqrc_v3.baselines.preprocessing import validate_population_contract

        try:
            return validate_population_contract(value)
        except DataContractError as error:
            raise ArtifactMismatch("invalid fitted preprocessing population metadata") from error

    @staticmethod
    def _entry_digest(metadata: Mapping[str, object]) -> str:
        payload = json.dumps(
            dict(metadata), sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as artifact:
            os.fsync(artifact.fileno())

    @staticmethod
    def _write_metadata(path: Path, metadata: Mapping[str, object]) -> None:
        with path.open("w", encoding="utf-8") as artifact:
            json.dump(metadata, artifact, sort_keys=True, separators=(",", ":"))
            artifact.flush()
            os.fsync(artifact.fileno())

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
