"""Portable filesystem checks used at publication boundaries."""

from __future__ import annotations

import os
import stat
import tempfile
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

from filelock import FileLock


class PublicationFSError(ValueError):
    """Raised when a publication filesystem entry cannot be trusted."""


@dataclass
class TrustedDirectory:
    root: Path
    path: Path
    identity: tuple[int, int]
    backend: Literal["posix", "windows"]
    descriptor: int | None = None

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None


def _is_link_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", lambda: False)
    try:
        return path.is_symlink() or is_junction()
    except OSError as error:
        raise PublicationFSError(f"cannot inspect publication path: {path}") from error


def _reject_link_components(path: Path) -> None:
    path = Path(path)
    if ".." in path.parts:
        raise PublicationFSError(f"publication path contains parent traversal: {path}")
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


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _entry_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise PublicationFSError("publication entry name is unsafe")
    return name


def _directory_identity(path: Path) -> tuple[int, int]:
    checked = require_local_entry(path, kind="directory")
    try:
        identity = checked.stat(follow_symlinks=False)
    except OSError as error:
        raise PublicationFSError(f"cannot inspect publication directory: {checked}") from error
    if not stat.S_ISDIR(identity.st_mode):
        raise PublicationFSError(f"publication entry is not a directory: {checked}")
    return identity.st_dev, identity.st_ino


def trusted_directory(
    root: Path,
    path: Path,
    *,
    backend: Literal["posix", "windows"] | None = None,
) -> TrustedDirectory:
    """Hold a trusted publication directory using the native platform backend."""

    selected = ("windows" if os.name == "nt" else "posix") if backend is None else backend
    if selected not in {"posix", "windows"}:
        raise ValueError("backend must be posix or windows")
    resolved_root = require_local_entry(Path(root), kind="directory").resolve(strict=True)
    resolved_path = require_within(Path(root), Path(path))
    require_local_entry(resolved_path, kind="directory")
    if selected == "windows":
        return TrustedDirectory(
            resolved_root,
            resolved_path,
            _directory_identity(resolved_path),
            "windows",
        )

    descriptors: list[int] = []
    try:
        descriptors.append(os.open(resolved_root, _directory_flags()))
        relative = resolved_path.relative_to(resolved_root)
        for component in relative.parts:
            descriptors.append(os.open(component, _directory_flags(), dir_fd=descriptors[-1]))
        held = os.fstat(descriptors[-1])
        if not stat.S_ISDIR(held.st_mode):
            raise PublicationFSError(f"publication entry is not a directory: {resolved_path}")
        final = descriptors.pop()
        return TrustedDirectory(
            resolved_root,
            resolved_path,
            (held.st_dev, held.st_ino),
            "posix",
            final,
        )
    except (OSError, ValueError) as error:
        raise PublicationFSError(f"cannot open publication directory safely: {path}") from error
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def guard_trusted_directory(directory: TrustedDirectory) -> None:
    """Revalidate containment and the held directory identity."""

    if not isinstance(directory, TrustedDirectory):
        raise TypeError("directory must be a TrustedDirectory")
    if directory.backend == "windows":
        require_local_entry(directory.root, kind="directory")
        checked = require_within(directory.root, directory.path)
        if checked != directory.path or _directory_identity(checked) != directory.identity:
            raise PublicationFSError("publication directory identity changed")
        return
    if directory.descriptor is None:
        raise PublicationFSError("publication directory descriptor is unavailable")
    try:
        held = os.fstat(directory.descriptor)
    except OSError as error:
        raise PublicationFSError("publication directory descriptor is unavailable") from error
    if not stat.S_ISDIR(held.st_mode) or (held.st_dev, held.st_ino) != directory.identity:
        raise PublicationFSError("publication directory identity changed")
    current = trusted_directory(directory.root, directory.path, backend="posix")
    try:
        if current.identity != directory.identity:
            raise PublicationFSError("publication directory identity changed")
    finally:
        current.close()


def _relative_identity(directory: TrustedDirectory, name: str) -> os.stat_result:
    safe = _entry_name(name)
    try:
        if directory.backend == "posix":
            assert directory.descriptor is not None
            return os.stat(safe, dir_fd=directory.descriptor, follow_symlinks=False)
        return (directory.path / safe).stat(follow_symlinks=False)
    except OSError as error:
        raise PublicationFSError(f"cannot inspect publication entry: {safe}") from error


def entry_sha256(directory: TrustedDirectory, name: str) -> str:
    """Hash one regular entry while retaining its path and directory identity."""

    safe = _entry_name(name)
    descriptor = -1
    try:
        guard_trusted_directory(directory)
        if directory.backend == "posix":
            assert directory.descriptor is not None
            descriptor = os.open(
                safe,
                os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory.descriptor,
            )
        else:
            path = require_local_entry(directory.path / safe, kind="file")
            descriptor = os.open(path, os.O_RDONLY)
        identity = os.fstat(descriptor)
        current = _relative_identity(directory, safe)
        if not stat.S_ISREG(identity.st_mode) or not stat.S_ISREG(current.st_mode):
            raise PublicationFSError(f"publication entry is not a regular file: {safe}")
        if (identity.st_dev, identity.st_ino) != (current.st_dev, current.st_ino):
            raise PublicationFSError(f"publication entry identity changed: {safe}")
        digest = sha256()
        with os.fdopen(descriptor, "rb") as source:
            descriptor = -1
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        guard_trusted_directory(directory)
        current = _relative_identity(directory, safe)
        if (identity.st_dev, identity.st_ino) != (current.st_dev, current.st_ino):
            raise PublicationFSError(f"publication entry identity changed: {safe}")
        return digest.hexdigest()
    except PublicationFSError:
        raise
    except OSError as error:
        raise PublicationFSError(f"cannot hash publication entry: {safe}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def replace_entry(directory: TrustedDirectory, source_name: str, target_name: str) -> Path:
    """Atomically replace one entry while retaining the trusted directory binding."""

    source = _entry_name(source_name)
    target = _entry_name(target_name)
    guard_trusted_directory(directory)
    identity = _relative_identity(directory, source)
    if not stat.S_ISREG(identity.st_mode):
        raise PublicationFSError(f"publication entry is not a regular file: {source}")
    try:
        target_identity = _relative_identity(directory, target)
    except PublicationFSError as error:
        if not isinstance(error.__cause__, FileNotFoundError):
            raise
    else:
        if not stat.S_ISREG(target_identity.st_mode):
            raise PublicationFSError(f"publication target is not a regular file: {target}")
    try:
        if directory.backend == "posix":
            assert directory.descriptor is not None
            os.replace(
                source,
                target,
                src_dir_fd=directory.descriptor,
                dst_dir_fd=directory.descriptor,
            )
            os.fsync(directory.descriptor)
        else:
            os.replace(directory.path / source, directory.path / target)
    except OSError as error:
        raise PublicationFSError(f"cannot replace publication entry: {target}") from error
    guard_trusted_directory(directory)
    target_identity = _relative_identity(directory, target)
    if not stat.S_ISREG(target_identity.st_mode):
        raise PublicationFSError(f"publication entry is not a regular file: {target}")
    return directory.path / target


def unlink_entry(directory: TrustedDirectory, name: str, *, missing_ok: bool = False) -> None:
    """Remove one regular entry from a trusted directory."""

    safe = _entry_name(name)
    guard_trusted_directory(directory)
    try:
        identity = _relative_identity(directory, safe)
    except PublicationFSError as error:
        if missing_ok and isinstance(error.__cause__, FileNotFoundError):
            return
        raise
    if not stat.S_ISREG(identity.st_mode):
        raise PublicationFSError(f"publication entry is not a regular file: {safe}")
    try:
        if directory.backend == "posix":
            assert directory.descriptor is not None
            os.unlink(safe, dir_fd=directory.descriptor)
            os.fsync(directory.descriptor)
        else:
            (directory.path / safe).unlink()
    except FileNotFoundError:
        if missing_ok:
            return
        raise PublicationFSError(f"publication entry is missing: {safe}")
    except OSError as error:
        raise PublicationFSError(f"cannot remove publication entry: {safe}") from error
    guard_trusted_directory(directory)


def atomic_write_bytes(directory: TrustedDirectory, name: str, content: bytes) -> Path:
    """Fsync and atomically publish bytes in a trusted directory."""

    target = _entry_name(name)
    if not isinstance(content, bytes):
        raise TypeError("content must be bytes")
    guard_trusted_directory(directory)
    temporary_name: str
    descriptor: int
    if directory.backend == "windows":
        descriptor, temporary = tempfile.mkstemp(
            prefix=".publication-", suffix=".tmp", dir=directory.path
        )
        temporary_name = Path(temporary).name
    else:
        assert directory.descriptor is not None
        temporary_name = f".publication-{uuid.uuid4().hex}.tmp"
        try:
            descriptor = os.open(
                temporary_name,
                os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
                0o600,
                dir_fd=directory.descriptor,
            )
        except OSError as error:
            raise PublicationFSError("cannot create publication temporary file") from error
    identity = os.fstat(descriptor)
    published = False
    try:
        if not stat.S_ISREG(identity.st_mode):
            raise PublicationFSError("publication temporary entry is not a regular file")
        with os.fdopen(descriptor, "wb") as output:
            descriptor = -1
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        guard_trusted_directory(directory)
        result = replace_entry(directory, temporary_name, target)
        published = True
        return result
    except OSError as error:
        raise PublicationFSError(f"cannot write publication entry: {target}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
        if not published:
            try:
                current = _relative_identity(directory, temporary_name)
                if (current.st_dev, current.st_ino) == (identity.st_dev, identity.st_ino):
                    unlink_entry(directory, temporary_name, missing_ok=True)
            except PublicationFSError:
                pass


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
    if os.name == "nt":
        with checked.open("a+b"):
            require_local_entry(checked, kind="file")
            with FileLock(str(checked), timeout=timeout):
                require_within(checked.parent, checked)
                yield
        return
    with FileLock(str(checked), timeout=timeout):
        require_within(checked.parent, checked)
        yield
