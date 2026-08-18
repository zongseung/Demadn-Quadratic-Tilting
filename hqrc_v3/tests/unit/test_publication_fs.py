from __future__ import annotations

import multiprocessing
import os
import subprocess
from pathlib import Path

import pytest
from filelock import Timeout

import hqrc_v3.publication_fs as publication_fs
from hqrc_v3.publication_fs import (
    PublicationFSError,
    atomic_write_bytes,
    exclusive_lock,
    require_local_entry,
    require_within,
    trusted_directory,
)


def _lock_attempt(lock_path: str, timeout: float, result_queue) -> None:
    try:
        with exclusive_lock(Path(lock_path), timeout=timeout):
            result_queue.put("acquired")
    except Timeout:
        result_queue.put("blocked")


def _spawn_lock_attempt(lock_path: Path, *, timeout: float) -> str:
    context = multiprocessing.get_context("spawn")
    result_queue = context.Queue()
    process = context.Process(target=_lock_attempt, args=(str(lock_path), timeout, result_queue))
    process.start()
    process.join(timeout=10)
    if process.is_alive():
        process.terminate()
        process.join(timeout=2)
        pytest.fail("lock attempt did not finish")
    assert process.exitcode == 0
    return result_queue.get(timeout=2)


def _make_platform_link(link: Path, target: Path) -> Path:
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"cannot create a directory link on this platform: {error}")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode:
            pytest.skip(f"cannot create a directory link or junction: {error}")
    return link


def _trusted_windows_directory(tmp_path: Path):
    root = tmp_path / "root"
    target = root / "fold"
    target.mkdir(parents=True)
    return trusted_directory(root, target, backend="windows")


def _raise_permission_error(*_args, **_kwargs):
    raise PermissionError("injected replace failure")


def test_exclusive_lock_blocks_a_spawned_writer(tmp_path):
    lock_path = tmp_path / "publication.lock"
    with exclusive_lock(lock_path):
        assert _spawn_lock_attempt(lock_path, timeout=0.2) == "blocked"
    assert _spawn_lock_attempt(lock_path, timeout=2.0) == "acquired"


def test_require_local_entry_rejects_symlink_or_junction(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = _make_platform_link(tmp_path / "link", target)
    with pytest.raises(PublicationFSError, match="link or junction"):
        require_local_entry(link, kind="directory")


def test_require_within_rejects_escape(tmp_path):
    with pytest.raises(PublicationFSError, match="outside trusted root"):
        require_within(tmp_path / "root", tmp_path / "escape")


def test_require_within_rejects_parent_traversal_before_a_junction(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    target = root / "target"
    target.mkdir()
    junction = _make_platform_link(root / "junction", target)

    with pytest.raises(PublicationFSError, match="parent traversal"):
        require_within(root, root / "missing" / ".." / junction.name)


def test_windows_fsync_file_opens_a_regular_file_writable(tmp_path, monkeypatch):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"content")
    opened_modes: list[str] = []
    original_open = Path.open

    def record_open(path, mode="r", *args, **kwargs):
        opened_modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(publication_fs.os, "name", "nt")
    monkeypatch.setattr(Path, "open", record_open)

    publication_fs._fsync_file(artifact)

    assert opened_modes == ["r+b"]


def test_windows_fsync_directory_skips_unsupported_descriptor_open(tmp_path, monkeypatch):
    monkeypatch.setattr(publication_fs.os, "name", "nt")
    monkeypatch.setattr(
        publication_fs.os,
        "open",
        lambda *_args: pytest.fail("Windows directory fsync must not open a descriptor"),
    )

    publication_fs._fsync_directory(tmp_path)


def test_windows_backend_publishes_atomically_without_dir_fd(tmp_path):
    root = tmp_path / "root"
    target = root / "fold"
    target.mkdir(parents=True)
    directory = trusted_directory(root, target, backend="windows")
    atomic_write_bytes(directory, "COMPLETE", b"ok\n")
    assert (target / "COMPLETE").read_bytes() == b"ok\n"
    assert not list(target.glob(".publication-*.tmp"))


def test_windows_backend_preserves_existing_target_on_replace_failure(tmp_path, monkeypatch):
    directory = _trusted_windows_directory(tmp_path)
    (directory.path / "result.json").write_bytes(b"old")
    monkeypatch.setattr(os, "replace", _raise_permission_error)
    with pytest.raises(PublicationFSError):
        atomic_write_bytes(directory, "result.json", b"new")
    assert (directory.path / "result.json").read_bytes() == b"old"
