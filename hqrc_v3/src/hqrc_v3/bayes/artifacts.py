"""Versioned, hash-bound serialization for safe cross-process HQRC inputs."""

from __future__ import annotations

import io
import json
import os
import re
import sys
import uuid
from hashlib import sha256
from pathlib import Path
from typing import Any

import numpy as np

from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.provenance import file_sha256
from hqrc_v3.publication_fs import (
    PublicationFSError,
    TrustedDirectory,
    atomic_write_bytes,
    entry_sha256,
    guard_trusted_directory,
    replace_entry,
    trusted_directory,
    unlink_entry,
)

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


def _write_json_temporary(
    directory: TrustedDirectory, path: Path, value: dict[str, Any]
) -> Path:
    name = f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        return atomic_write_bytes(directory, name, _canonical(value) + b"\n")
    except PublicationFSError as error:
        raise HQRCArtifactError("HQRC current pointer temporary cannot be written") from error


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
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


def _open_generation_namespace(destination: Path) -> TrustedDirectory:
    parent = destination.parent
    namespace = _generation_directory(destination)
    parent_directory: TrustedDirectory | None = None
    try:
        parent_directory = trusted_directory(parent, parent)
        try:
            if parent_directory.backend == "posix":
                os.mkdir(namespace.name, mode=0o700, dir_fd=parent_directory.descriptor)
            else:
                namespace.mkdir(mode=0o700)
        except FileExistsError:
            pass
        directory = trusted_directory(parent, namespace, backend=parent_directory.backend)
        guard_trusted_directory(directory)
        return directory
    except (OSError, PublicationFSError) as error:
        raise HQRCArtifactError("HQRC generation namespace cannot be opened safely") from error
    finally:
        if parent_directory is not None:
            parent_directory.close()


def _revalidate_generation_namespace(directory: TrustedDirectory) -> None:
    try:
        guard_trusted_directory(directory)
    except PublicationFSError as error:
        raise HQRCArtifactError("HQRC generation namespace identity changed") from error


def _write_relative_json(directory: TrustedDirectory, name: str, value: dict[str, Any]) -> None:
    try:
        atomic_write_bytes(directory, name, _canonical(value) + b"\n")
    except PublicationFSError as error:
        raise HQRCArtifactError("HQRC generation temporary file cannot be written") from error


def _relative_sha256(directory: TrustedDirectory, name: str) -> str:
    try:
        return entry_sha256(directory, name)
    except PublicationFSError as error:
        raise HQRCArtifactError("HQRC generation file cannot be opened safely") from error


def _replace_relative(directory: TrustedDirectory, source: str, destination: str) -> None:
    try:
        replace_entry(directory, source, destination)
    except PublicationFSError as error:
        raise HQRCArtifactError("HQRC generation file cannot be published") from error


def _cleanup_relative(directory: TrustedDirectory, names: set[str]) -> None:
    cleanup_error: PublicationFSError | None = None
    for name in sorted(names):
        try:
            unlink_entry(directory, name, missing_ok=True)
        except PublicationFSError as error:
            cleanup_error = cleanup_error or error
    if cleanup_error is not None:
        raise HQRCArtifactError("HQRC generation cleanup failed") from cleanup_error


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
    directory: TrustedDirectory | None = None
    parent_directory: TrustedDirectory | None = None
    temporary_names: set[str] = set()
    temporary_pointer: Path | None = None
    try:
        directory = _open_generation_namespace(destination)
        generation_directory = directory.path
        _publication_boundary("generation-namespace-opened")
        _revalidate_generation_namespace(directory)
        generation = uuid.uuid4().hex
        generation_npz_name = f"{generation}.npz"
        generation_metadata_name = f"{generation}.json"
        generation_npz = generation_directory / generation_npz_name
        generation_metadata = generation_directory / generation_metadata_name
        temporary_npz_name = f".{generation}.{uuid.uuid4().hex}.npz.tmp"
        temporary_metadata_name = f".{generation}.{uuid.uuid4().hex}.json.tmp"
        temporary_names.add(temporary_npz_name)
        arrays = {
            "observations": np.asarray(data.observations, dtype=np.float64),
            "occurrence_index": np.asarray(data.occurrence_index, dtype=np.int64),
            "holiday_type_index": np.asarray(data.holiday_type_index, dtype=np.int64),
            "tau_days": np.asarray(data.tau_days, dtype=np.float64),
            "hour": np.asarray(data.hour, dtype=np.int64),
            "restriction": np.asarray(data.restriction, dtype=np.int64),
        }
        try:
            artifact = io.BytesIO()
            np.savez_compressed(artifact, **arrays)
            atomic_write_bytes(directory, temporary_npz_name, artifact.getvalue())
        except PublicationFSError as error:
            raise HQRCArtifactError("HQRC generation temporary file cannot be written") from error
        payload: dict[str, Any] = {
            "schema_version": _GENERATION_VERSION,
            "artifact_kind": _GENERATION_KIND,
            "generation": generation,
            "npz_sha256": _relative_sha256(directory, temporary_npz_name),
            "arrays": {name: name for name in sorted(_ARRAYS)},
            "occurrence_ids": list(data.occurrence_ids),
            "settings": _safe_json(settings),
        }
        payload["artifact_sha256"] = _digest(payload)
        temporary_names.add(temporary_metadata_name)
        _write_relative_json(directory, temporary_metadata_name, payload)

        _publication_boundary("before-generation-npz-publish")
        _replace_relative(directory, temporary_npz_name, generation_npz_name)
        temporary_names.discard(temporary_npz_name)
        _publication_boundary("generation-npz-published")
        _replace_relative(
            directory, temporary_metadata_name, generation_metadata_name
        )
        temporary_names.discard(temporary_metadata_name)
        _publication_boundary("generation-metadata-published")
        _revalidate_generation_namespace(directory)
        _publication_boundary("generation-directory-synced")

        pointer_path = _pointer_path(destination)
        pointer: dict[str, Any] = {
            "schema_version": _POINTER_VERSION,
            "artifact_kind": _POINTER_KIND,
            "generation": generation,
            "npz": generation_npz.relative_to(destination.parent).as_posix(),
            "metadata": generation_metadata.relative_to(destination.parent).as_posix(),
            "npz_sha256": _relative_sha256(directory, generation_npz_name),
            "metadata_sha256": _relative_sha256(
                directory, generation_metadata_name
            ),
        }
        pointer["pointer_digest"] = _digest(pointer)
        _revalidate_generation_namespace(directory)
        parent_directory = trusted_directory(
            destination.parent, destination.parent, backend=directory.backend
        )
        temporary_pointer = _write_json_temporary(parent_directory, pointer_path, pointer)
        _publication_boundary("before-pointer-swap")
        _revalidate_generation_namespace(directory)
        try:
            replace_entry(parent_directory, temporary_pointer.name, pointer_path.name)
        except PublicationFSError as error:
            raise HQRCArtifactError("HQRC current pointer cannot be published") from error
        temporary_pointer = None
        _publication_boundary("pointer-swapped")
        _fsync_directory(destination.parent)
        _publication_boundary("parent-directory-synced")
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            if directory is not None:
                try:
                    if temporary_names:
                        try:
                            _cleanup_relative(directory, temporary_names)
                        except HQRCArtifactError:
                            if not active_error:
                                raise
                finally:
                    directory.close()
        finally:
            if temporary_pointer is not None:
                try:
                    if parent_directory is not None:
                        unlink_entry(parent_directory, temporary_pointer.name, missing_ok=True)
                    else:
                        temporary_pointer.unlink(missing_ok=True)
                except PublicationFSError:
                    pass
            if parent_directory is not None:
                parent_directory.close()
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
