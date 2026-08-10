"""Versioned, hash-bound serialization for safe cross-process HQRC inputs."""

from __future__ import annotations

import json
import os
import re
import tempfile
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.provenance import file_sha256

_LEGACY_VERSION = 1
_GENERATION_VERSION = 2
_POINTER_VERSION = 1
_GENERATION_KIND = "hqrc-data-generation"
_POINTER_KIND = "hqrc-data-current-pointer"
_GENERATION_PATTERN = re.compile(r"[0-9a-f]{32}")
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
_LEGACY_METADATA_KEYS = {
    "schema_version",
    "npz_sha256",
    "arrays",
    "occurrence_ids",
    "settings",
    "artifact_sha256",
}
_GENERATION_METADATA_KEYS = _LEGACY_METADATA_KEYS | {"artifact_kind", "generation"}
_POINTER_KEYS = {
    "schema_version",
    "artifact_kind",
    "generation",
    "npz",
    "metadata",
    "npz_sha256",
    "metadata_sha256",
    "pointer_digest",
}


class HQRCArtifactError(ValueError):
    """Raised for unsafe, incomplete, or tampered HQRC input artifacts."""


def _canonical(value: object) -> bytes:
    try:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    except (TypeError, ValueError) as error:
        raise HQRCArtifactError("HQRC artifact metadata is not finite canonical JSON") from error


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _sha(value: object, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise HQRCArtifactError(f"{name} must be a SHA-256 digest")
    try:
        int(value, 16)
    except ValueError as error:
        raise HQRCArtifactError(f"{name} must be a SHA-256 digest") from error
    if value != value.lower():
        raise HQRCArtifactError(f"{name} must be a lowercase SHA-256 digest")
    return value


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
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        return Path(temporary)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor: int | None = None
    try:
        descriptor = os.open(path, flags)
        os.fsync(descriptor)
    except OSError as error:
        raise HQRCArtifactError(f"cannot synchronize artifact directory: {path}") from error
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _publication_boundary(name: str) -> None:
    """Deterministic no-op seam used to inject publication failures in tests."""


def _generation_directory(destination: Path) -> Path:
    return destination.with_name(f".{destination.stem}.generations")


def _pointer_path(destination: Path) -> Path:
    return destination.with_name(f"{destination.stem}.current.json")


def write_hqrc_data(
    path: Path, data: HQRCData, *, settings: dict[str, object]
) -> tuple[Path, Path]:
    """Publish an immutable generation and atomically advance its current pointer.

    The returned paths identify the immutable generation itself.  Passing the original
    logical ``.npz`` path to :func:`load_hqrc_data` resolves the current pointer.
    """

    if not isinstance(data, HQRCData):
        raise TypeError("data must be HQRCData")
    destination = Path(path)
    if destination.suffix != ".npz":
        raise HQRCArtifactError("HQRC data artifact must use .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    generation_directory = _generation_directory(destination)
    generation_directory.mkdir(exist_ok=True)
    generation = uuid.uuid4().hex
    generation_npz = generation_directory / f"{generation}.npz"
    generation_metadata = generation_directory / f"{generation}.json"
    arrays = {
        "observations": np.asarray(data.observations, dtype=np.float64),
        "occurrence_index": np.asarray(data.occurrence_index, dtype=np.int64),
        "holiday_type_index": np.asarray(data.holiday_type_index, dtype=np.int64),
        "tau_days": np.asarray(data.tau_days, dtype=np.float64),
        "hour": np.asarray(data.hour, dtype=np.int64),
        "restriction": np.asarray(data.restriction, dtype=np.int64),
    }
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{generation}.", suffix=".npz", dir=generation_directory
    )
    os.close(descriptor)
    temporary_npz: Path | None = Path(temporary_name)
    temporary_metadata: Path | None = None
    temporary_pointer: Path | None = None
    try:
        np.savez_compressed(temporary_npz, **arrays)
        with temporary_npz.open("r+b") as artifact:
            artifact.flush()
            os.fsync(artifact.fileno())
        payload: dict[str, Any] = {
            "schema_version": _GENERATION_VERSION,
            "artifact_kind": _GENERATION_KIND,
            "generation": generation,
            "npz_sha256": file_sha256(temporary_npz),
            "arrays": {name: name for name in sorted(_ARRAYS)},
            "occurrence_ids": list(data.occurrence_ids),
            "settings": _safe_json(settings),
        }
        payload["artifact_sha256"] = _digest(payload)
        temporary_metadata = _write_json_temporary(generation_metadata, payload)

        _publication_boundary("before-generation-npz-publish")
        os.replace(temporary_npz, generation_npz)
        temporary_npz = None
        _publication_boundary("generation-npz-published")
        os.replace(temporary_metadata, generation_metadata)
        temporary_metadata = None
        _publication_boundary("generation-metadata-published")
        _fsync_directory(generation_directory)
        _publication_boundary("generation-directory-synced")

        pointer_path = _pointer_path(destination)
        pointer: dict[str, Any] = {
            "schema_version": _POINTER_VERSION,
            "artifact_kind": _POINTER_KIND,
            "generation": generation,
            "npz": generation_npz.relative_to(destination.parent).as_posix(),
            "metadata": generation_metadata.relative_to(destination.parent).as_posix(),
            "npz_sha256": file_sha256(generation_npz),
            "metadata_sha256": file_sha256(generation_metadata),
        }
        pointer["pointer_digest"] = _digest(pointer)
        temporary_pointer = _write_json_temporary(pointer_path, pointer)
        _publication_boundary("before-pointer-swap")
        os.replace(temporary_pointer, pointer_path)
        temporary_pointer = None
        _publication_boundary("pointer-swapped")
        _fsync_directory(destination.parent)
        _publication_boundary("parent-directory-synced")
    finally:
        for temporary in (temporary_npz, temporary_metadata, temporary_pointer):
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return generation_npz, generation_metadata


def _read_json(path: Path, *, label: str, canonical: bool) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise HQRCArtifactError(f"{label} is unreadable") from error
    if not isinstance(payload, dict):
        raise HQRCArtifactError(f"{label} schema is invalid")
    if canonical and raw != _canonical(payload) + b"\n":
        raise HQRCArtifactError(f"{label} is not canonical JSON")
    return payload


def _safe_pointer_target(parent: Path, value: object, *, label: str) -> Path:
    if (
        not isinstance(value, str)
        or not value
        or Path(value).is_absolute()
        or ".." in Path(value).parts
    ):
        raise HQRCArtifactError(f"HQRC current pointer contains an unsafe {label} path")
    target = parent / value
    try:
        target.resolve().relative_to(parent.resolve())
    except ValueError as error:
        raise HQRCArtifactError(f"HQRC current pointer contains an unsafe {label} path") from error
    return target


def _resolve_pointer(source: Path) -> tuple[Path, Path]:
    pointer_path = _pointer_path(source)
    if pointer_path.is_symlink():
        raise HQRCArtifactError("HQRC current pointer is unsafe")
    pointer = _read_json(pointer_path, label="HQRC current pointer", canonical=True)
    if set(pointer) != _POINTER_KEYS:
        raise HQRCArtifactError("HQRC current pointer schema is invalid")
    digest = pointer.pop("pointer_digest")
    if not isinstance(digest, str) or digest != _digest(pointer):
        raise HQRCArtifactError("HQRC current pointer digest differs")
    pointer["pointer_digest"] = digest
    generation = pointer["generation"]
    if (
        type(pointer["schema_version"]) is not int
        or pointer["schema_version"] != _POINTER_VERSION
        or pointer["artifact_kind"] != _POINTER_KIND
        or not isinstance(generation, str)
        or _GENERATION_PATTERN.fullmatch(generation) is None
    ):
        raise HQRCArtifactError("HQRC current pointer identity is invalid")
    npz = _safe_pointer_target(source.parent, pointer["npz"], label="NPZ")
    metadata = _safe_pointer_target(source.parent, pointer["metadata"], label="metadata")
    generation_directory = _generation_directory(source)
    if generation_directory.is_symlink():
        raise HQRCArtifactError("HQRC generation directory is unsafe")
    if (
        npz != generation_directory / f"{generation}.npz"
        or metadata != generation_directory / f"{generation}.json"
        or npz.resolve().parent != generation_directory.resolve()
        or metadata.resolve().parent != generation_directory.resolve()
    ):
        raise HQRCArtifactError("HQRC current pointer generation paths differ")
    if not npz.is_file() or _sha(pointer["npz_sha256"], "pointer NPZ digest") != file_sha256(
        npz
    ):
        raise HQRCArtifactError("HQRC current pointer NPZ digest differs")
    if not metadata.is_file() or _sha(
        pointer["metadata_sha256"], "pointer metadata digest"
    ) != file_sha256(metadata):
        raise HQRCArtifactError("HQRC current pointer metadata digest differs")
    return npz, metadata


def _load_metadata(source: Path, metadata_file: Path) -> dict[str, Any]:
    payload = _read_json(metadata_file, label="HQRC artifact metadata", canonical=False)
    version = payload.get("schema_version")
    if type(version) is not int:
        raise HQRCArtifactError("HQRC artifact metadata version differs")
    if version == _GENERATION_VERSION:
        if set(payload) != _GENERATION_METADATA_KEYS:
            raise HQRCArtifactError("HQRC artifact metadata schema is invalid")
        if metadata_file.read_bytes() != _canonical(payload) + b"\n":
            raise HQRCArtifactError("HQRC artifact metadata schema is not canonical JSON")
        generation = payload["generation"]
        if (
            payload["artifact_kind"] != _GENERATION_KIND
            or not isinstance(generation, str)
            or _GENERATION_PATTERN.fullmatch(generation) is None
            or source.stem != generation
            or metadata_file.stem != generation
            or source.parent != metadata_file.parent
        ):
            raise HQRCArtifactError("HQRC artifact generation identity differs")
    elif version == _LEGACY_VERSION:
        if set(payload) != _LEGACY_METADATA_KEYS:
            raise HQRCArtifactError("HQRC artifact metadata schema is invalid")
    else:
        raise HQRCArtifactError("HQRC artifact metadata version differs")
    digest = payload.pop("artifact_sha256")
    if not isinstance(digest, str) or digest != _digest(payload):
        raise HQRCArtifactError("HQRC artifact metadata digest differs")
    payload["artifact_sha256"] = digest
    if not source.is_file() or _sha(payload["npz_sha256"], "artifact NPZ digest") != file_sha256(
        source
    ):
        raise HQRCArtifactError("HQRC artifact NPZ digest differs")
    if payload["arrays"] != {name: name for name in sorted(_ARRAYS)}:
        raise HQRCArtifactError("HQRC artifact arrays schema is invalid")
    if not isinstance(payload["occurrence_ids"], list) or not isinstance(
        payload["settings"], dict
    ):
        raise HQRCArtifactError("HQRC artifact metadata types are invalid")
    return payload


def load_hqrc_data(
    path: Path, metadata_path: Path | None = None
) -> tuple[HQRCData, dict[str, object]]:
    """Load an immutable generation or resolve and validate a logical current pointer."""

    source = Path(path)
    if source.suffix != ".npz":
        raise HQRCArtifactError("HQRC data artifact must use .npz")
    pointer_path = _pointer_path(source)
    if metadata_path is None and pointer_path.is_file():
        source, metadata_file = _resolve_pointer(source)
    elif metadata_path is None and _generation_directory(source).exists():
        raise HQRCArtifactError("HQRC current pointer is missing")
    else:
        metadata_file = (
            Path(metadata_path) if metadata_path is not None else source.with_suffix(".json")
        )
    payload = _load_metadata(source, metadata_file)
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
