"""Versioned, hash-bound serialization for safe cross-process HQRC inputs."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.provenance import file_sha256

_VERSION = 1
_ARRAYS = frozenset(
    {
        "observations",
        "occurrence_index",
        "holiday_type_index",
        "tau_days",
        "hour",
        "restriction",
    }
)


class HQRCArtifactError(ValueError):
    """Raised for unsafe, incomplete, or tampered HQRC input artifacts."""


def _digest(value: dict[str, Any]) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe_json(value: object) -> object:
    if value is None or isinstance(value, (bool, int, float, str)):
        if isinstance(value, float) and not np.isfinite(value):
            raise HQRCArtifactError("settings must contain finite JSON values")
        if isinstance(value, str) and (
            not value or ".." in Path(value).parts or Path(value).is_absolute()
        ):
            raise HQRCArtifactError("settings contain an unsafe string")
        return value
    if isinstance(value, dict) and all(isinstance(key, str) and key for key in value):
        return {key: _safe_json(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    raise HQRCArtifactError("settings must be JSON-safe")


def _write_json_temporary(path: Path, value: dict[str, Any]) -> Path:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"))
            output.flush()
            os.fsync(output.fileno())
        return Path(temporary)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


@contextmanager
def _artifact_lock(destination: Path, *, exclusive: bool) -> Iterator[None]:
    lock_path = destination.with_name(f".{destination.stem}.lock")
    descriptor: int | None = None
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        fcntl.flock(descriptor, operation)
    except OSError as error:
        if descriptor is not None:
            os.close(descriptor)
        raise HQRCArtifactError("HQRC artifact pair lock is unavailable") from error
    try:
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def write_hqrc_data(
    path: Path, data: HQRCData, *, settings: dict[str, object]
) -> tuple[Path, Path]:
    """Atomically publish numeric arrays plus strict JSON identity metadata."""

    if not isinstance(data, HQRCData):
        raise TypeError("data must be HQRCData")
    destination = Path(path)
    if destination.suffix != ".npz":
        raise HQRCArtifactError("HQRC data artifact must use .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        "observations": np.asarray(data.observations, dtype=np.float64),
        "occurrence_index": np.asarray(data.occurrence_index, dtype=np.int64),
        "holiday_type_index": np.asarray(data.holiday_type_index, dtype=np.int64),
        "tau_days": np.asarray(data.tau_days, dtype=np.float64),
        "hour": np.asarray(data.hour, dtype=np.int64),
        "restriction": np.asarray(data.restriction, dtype=np.int64),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent
    )
    os.close(descriptor)
    temporary_npz = Path(temporary_name)
    temporary_metadata: Path | None = None
    try:
        np.savez_compressed(temporary_npz, **arrays)
        with temporary_npz.open("rb") as artifact:
            os.fsync(artifact.fileno())
        metadata_path = destination.with_suffix(".json")
        payload: dict[str, Any] = {
            "schema_version": _VERSION,
            "npz_sha256": file_sha256(temporary_npz),
            "arrays": {name: name for name in sorted(_ARRAYS)},
            "occurrence_ids": list(data.occurrence_ids),
            "settings": _safe_json(settings),
        }
        payload["artifact_sha256"] = _digest(payload)
        temporary_metadata = _write_json_temporary(metadata_path, payload)
        with _artifact_lock(destination, exclusive=True):
            os.replace(temporary_npz, destination)
            temporary_npz = None
            os.replace(temporary_metadata, metadata_path)
            temporary_metadata = None
    finally:
        if temporary_npz is not None:
            temporary_npz.unlink(missing_ok=True)
        if temporary_metadata is not None:
            temporary_metadata.unlink(missing_ok=True)
    return destination, metadata_path


def load_hqrc_data(
    path: Path, metadata_path: Path | None = None
) -> tuple[HQRCData, dict[str, object]]:
    """Load only an untampered allow-pickle-free data artifact through HQRCData checks."""

    source = Path(path)
    metadata_file = (
        Path(metadata_path) if metadata_path is not None else source.with_suffix(".json")
    )
    with _artifact_lock(source, exclusive=False):
        try:
            payload = json.loads(metadata_file.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise HQRCArtifactError("HQRC artifact metadata is unreadable") from error
        if (
            not isinstance(payload, dict)
            or set(payload)
            != {
                "schema_version",
                "npz_sha256",
                "arrays",
                "occurrence_ids",
                "settings",
                "artifact_sha256",
            }
            or payload["schema_version"] != _VERSION
        ):
            raise HQRCArtifactError("HQRC artifact metadata schema is invalid")
        digest = payload.pop("artifact_sha256")
        if not isinstance(digest, str) or digest != _digest(payload):
            raise HQRCArtifactError("HQRC artifact metadata digest differs")
        if not source.is_file() or payload["npz_sha256"] != file_sha256(source):
            raise HQRCArtifactError("HQRC artifact NPZ digest differs")
        if payload["arrays"] != {name: name for name in sorted(_ARRAYS)}:
            raise HQRCArtifactError("HQRC artifact arrays schema is invalid")
        if not isinstance(payload["occurrence_ids"], list) or not isinstance(
            payload["settings"], dict
        ):
            raise HQRCArtifactError("HQRC artifact metadata types are invalid")
        try:
            with np.load(source, allow_pickle=False) as archive:
                if set(archive.files) != _ARRAYS:
                    raise HQRCArtifactError("HQRC artifact arrays are missing or unknown")
                arrays = {name: archive[name] for name in _ARRAYS}
            data = HQRCData(
                **arrays,
                occurrence_ids=tuple(payload["occurrence_ids"]),
            )
        except (OSError, ValueError, TypeError) as error:
            raise HQRCArtifactError("HQRC artifact arrays fail validation") from error
        return data, dict(_safe_json(payload["settings"]))
