"""Portable filesystem checks used at publication boundaries."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from filelock import FileLock


class PublicationFSError(ValueError):
    """Raised when a publication filesystem entry cannot be trusted."""


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    try:
        return path.is_symlink() or is_junction()
    except OSError as error:
        raise PublicationFSError(f"cannot inspect publication path: {path}") from error


def _reject_link_components(path: Path) -> None:
    path = Path(path)
    current = Path(path.anchor) if path.anchor else Path()
    for part in path.parts:
        if part == path.anchor:
            continue
        current /= part
        try:
            exists = current.exists()
        except OSError as error:
            raise PublicationFSError(f"cannot inspect publication path: {current}") from error
        if _is_link_or_junction(current):
            raise PublicationFSError(f"publication path contains a link or junction: {current}")
        if not exists:
            break


def require_local_entry(path: Path, *, kind: Literal["file", "directory"]) -> Path:
    """Return an existing local file or directory after rejecting links and junctions."""

    checked = Path(path)
    _reject_link_components(checked)
    if kind == "file" and not checked.is_file():
        raise PublicationFSError(f"publication entry is not a regular file: {checked}")
    if kind == "directory" and not checked.is_dir():
        raise PublicationFSError(f"publication entry is not a directory: {checked}")
    return checked


def require_within(root: Path, candidate: Path) -> Path:
    """Resolve a candidate and require that it remains below the trusted root."""

    _reject_link_components(Path(root))
    _reject_link_components(Path(candidate))
    resolved_root = Path(root).resolve(strict=False)
    resolved_candidate = Path(candidate).resolve(strict=False)
    if not resolved_candidate.is_relative_to(resolved_root):
        raise PublicationFSError(f"publication path is outside trusted root: {candidate}")
    return resolved_candidate


def _fsync_file(path: Path) -> None:
    checked = require_local_entry(path, kind="file")
    with checked.open("r+b" if os.name == "nt" else "rb") as artifact:
        os.fsync(artifact.fileno())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    checked = require_local_entry(path, kind="directory")
    descriptor = os.open(checked, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def exclusive_lock(path: Path, *, timeout: float = -1) -> Iterator[None]:
    """Hold a conservative cross-process exclusive publication lock."""

    checked = require_within(Path(path).parent, Path(path))
    if checked.exists():
        require_local_entry(checked, kind="file")
    with FileLock(str(checked), timeout=timeout):
        require_within(checked.parent, checked)
        yield
