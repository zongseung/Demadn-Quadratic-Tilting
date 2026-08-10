"""Fold-scoped residual standardization and immutable prediction artifacts."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hqrc_v3.contracts import PREDICTION_COLUMNS, DataContractError
from hqrc_v3.provenance import ArtifactMismatch

_RESIDUAL_COLUMNS = ("is_event", "residual_mw")
_CACHE_KEY = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class ResidualContext:
    """The model fit identity that owns an OOF residual scale."""

    model: str
    feature_set: str
    seed: int
    split_id: str


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
    if not context.split_id.startswith("oof-"):
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

        parquet_path, metadata_path = self._paths(key)
        parquet_exists, metadata_exists = parquet_path.is_file(), metadata_path.is_file()
        if not parquet_exists and not metadata_exists:
            return None
        if parquet_exists != metadata_exists:
            raise ArtifactMismatch(f"partial prediction cache artifact for key {key!r}")
        metadata = self._read_metadata(metadata_path)
        self._assert_hashes(metadata, expected_hashes)
        try:
            frame = pl.read_parquet(parquet_path)
        except (OSError, pl.exceptions.PolarsError) as error:
            raise ArtifactMismatch(
                f"unable to read cached prediction artifact {key!r}: {error}"
            ) from error
        from hqrc_v3.contracts import validate_prediction_frame

        return validate_prediction_frame(frame)

    def write(self, key: str, frame: pl.DataFrame, hashes: Mapping[str, str]) -> pl.DataFrame:
        """Atomically persist a validated frame and return it without overwriting mismatches."""

        from hqrc_v3.contracts import validate_prediction_frame

        normalized = validate_prediction_frame(frame)
        hashes_dict = self._normalize_hashes(hashes)
        existing = self.read(key, expected_hashes=hashes_dict)
        if existing is not None:
            return existing

        parquet_path, metadata_path = self._paths(key)
        parquet_temp = self._temporary_path(key, ".parquet")
        metadata_temp = self._temporary_path(key, ".json")
        try:
            normalized.write_parquet(parquet_temp)
            self._fsync_file(parquet_temp)
            self._write_metadata(metadata_temp, hashes_dict)
            os.replace(parquet_temp, parquet_path)
            os.replace(metadata_temp, metadata_path)
            self._fsync_directory()
        except Exception:
            parquet_temp.unlink(missing_ok=True)
            metadata_temp.unlink(missing_ok=True)
            raise
        return normalized

    def _paths(self, key: str) -> tuple[Path, Path]:
        if not isinstance(key, str) or not _CACHE_KEY.fullmatch(key) or key in {".", ".."}:
            raise ValueError("prediction cache key must be a safe non-empty filename")
        return self.root / f"{key}.parquet", self.root / f"{key}.json"

    def _temporary_path(self, key: str, suffix: str) -> Path:
        descriptor, path = tempfile.mkstemp(prefix=f".{key}.", suffix=suffix, dir=self.root)
        os.close(descriptor)
        return Path(path)

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
        if not isinstance(metadata, dict) or not isinstance(metadata.get("hashes"), dict):
            raise ArtifactMismatch(f"invalid cache metadata {path.name}")
        return metadata

    def _assert_hashes(
        self, metadata: Mapping[str, object], expected_hashes: Mapping[str, str]
    ) -> None:
        expected = self._normalize_hashes(expected_hashes)
        stored = metadata["hashes"]
        if not isinstance(stored, dict):  # guarded by _read_metadata; retained for type narrowing
            raise ArtifactMismatch("invalid cache metadata hashes")
        for name in sorted(set(stored) | set(expected)):
            if stored.get(name) != expected.get(name):
                raise ArtifactMismatch(f"{name} hash differs from the cached artifact")

    @staticmethod
    def _fsync_file(path: Path) -> None:
        with path.open("rb") as artifact:
            os.fsync(artifact.fileno())

    @staticmethod
    def _write_metadata(path: Path, hashes: Mapping[str, str]) -> None:
        with path.open("w", encoding="utf-8") as artifact:
            json.dump({"hashes": hashes}, artifact, sort_keys=True, separators=(",", ":"))
            artifact.flush()
            os.fsync(artifact.fileno())

    def _fsync_directory(self) -> None:
        descriptor = os.open(self.root, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
