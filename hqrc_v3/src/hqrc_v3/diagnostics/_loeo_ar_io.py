"""Private canonical-file primitives for the immutable LOEO AR publication."""

from __future__ import annotations

import json
import os
import stat
from hashlib import sha256
from pathlib import Path
from typing import Any

from hqrc_v3.publication_fs import _fsync_directory as _portable_fsync_directory


class LOEOARProposalError(ValueError):
    """Raised when a LOEO AR set is incomplete, incompatible, or untrusted."""


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise LOEOARProposalError("LOEO AR metadata is not canonical JSON") from error


def sha_json(value: object) -> str:
    return sha256(canonical_json(value)).hexdigest()


def require_file(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise LOEOARProposalError(f"{description} is missing or unsafe") from error
    if stat.S_ISLNK(mode) or not stat.S_ISREG(mode):
        raise LOEOARProposalError(f"{description} is missing or unsafe")


def require_directory(path: Path, description: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as error:
        raise LOEOARProposalError(f"{description} is missing or unsafe") from error
    if stat.S_ISLNK(mode) or not stat.S_ISDIR(mode):
        raise LOEOARProposalError(f"{description} is missing or unsafe")


def read_json(path: Path, description: str) -> dict[str, Any]:
    require_file(path, description)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise LOEOARProposalError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != canonical_json(value):
        raise LOEOARProposalError(f"{description} is not canonical JSON")
    return value


def write_json(path: Path, value: object) -> None:
    payload = canonical_json(value)
    with path.open("xb") as destination:
        destination.write(payload)
        destination.flush()
        os.fsync(destination.fileno())


def fsync_directory(path: Path) -> None:
    _portable_fsync_directory(path)
