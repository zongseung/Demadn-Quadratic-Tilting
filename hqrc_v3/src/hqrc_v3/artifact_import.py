"""Hash-checked import of the archived paper correction source."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from zipfile import BadZipFile, ZipFile, ZipInfo

from hqrc_v3.correction_source import validate_correction_source
from hqrc_v3.provenance import file_sha256
from hqrc_v3.publication_fs import (
    PublicationFSError,
    exclusive_lock,
    require_local_entry,
    require_within,
)

_ARCHIVE_PREFIX = PurePosixPath("artifacts/hqrc-v3-paper-20260811")
_EXTRACTED_NAMESPACES = frozenset({"predictions", "inputs", "prediction-stream-cache"})
_IGNORED_RUN_NAMESPACES = frozenset(
    {"ar_diagnostics", "corrections", "diagnostics", "quarantine", "reports"}
)
_SOURCE_OUTPUTS = {"data": "power_demand_final.csv"}

DEMAND_DATA_SHA256 = "ccb27f1d5ab7c5e2315bcd4cfc82f3c64d439b1b8d746ba65439fc89062969f9"


class ArchiveImportError(ValueError):
    """Raised when archived correction-source evidence cannot be trusted."""


@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    revision: str
    repository_path: str
    output_name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ImportedCorrectionSource:
    run_dir: Path
    config_path: Path
    reused: bool


SOURCE_SPECS = (
    SourceSpec(
        "experiment_config",
        "07e771fd",
        "hqrc_v3/configs/experiment.toml",
        "experiment.toml",
        "526416f140e3e2fe15777ee72e3e0155c9dff7d629c13d9ff88f1e032cc29c9f",
    ),
    SourceSpec(
        "model_config",
        "03d9b68a",
        "hqrc_v3/configs/model_spaces.toml",
        "model_spaces.toml",
        "b06d00880cd4311c5a0ff73c427800bd330f238d757c11165627d70f6465a1e9",
    ),
    SourceSpec(
        "event_registry",
        "3ed23767",
        "hqrc_v3/configs/events.csv",
        "events.csv",
        "147a823ba37501240460cd4cd53c914ca529c353dd43cf3cc302a65026206e7b",
    ),
    SourceSpec(
        "holiday_calendar",
        "03d9b68a",
        "hqrc_v3/configs/holiday_calendar.csv",
        "holiday_calendar.csv",
        "a72d93b49c9eaedd469ff04fb14534fcc718c893ac509ebc2152f532609c398e",
    ),
    SourceSpec(
        "temporary_holiday_availability",
        "03d9b68a",
        "hqrc_v3/configs/temporary_holiday_availability.csv",
        "temporary_holiday_availability.csv",
        "4b695cbbd5e737c43733e842eefb92ce19a6c75d6b0886853696818525fb2b0d",
    ),
)


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ArchiveImportError("residual manifest is not canonical JSON") from error


def residual_manifest_digest(manifest: dict[str, object]) -> str:
    """Return the canonical digest used by standardized residual manifests."""

    unsigned = {key: manifest[key] for key in sorted(manifest) if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(unsigned)).hexdigest()


def _safe_member_path(info: ZipInfo) -> PurePosixPath:
    name = info.filename
    windows = PureWindowsPath(name)
    if (
        not name
        or "\x00" in name
        or "\\" in name
        or name.startswith("/")
        or windows.is_absolute()
        or windows.drive
    ):
        raise ArchiveImportError(f"unsafe archive member: {name!r}")
    parts = name.split("/")
    if any(part in {"", ".", ".."} for part in parts[:-1]) or (
        parts[-1] in {"", ".", ".."} and not info.is_dir()
    ):
        raise ArchiveImportError(f"unsafe archive member: {name!r}")
    path = PurePosixPath(name)
    mode = info.external_attr >> 16
    file_type = stat.S_IFMT(mode)
    if stat.S_ISLNK(mode):
        raise ArchiveImportError(f"archive member is a link: {name!r}")
    if file_type not in {0, stat.S_IFREG, stat.S_IFDIR}:
        raise ArchiveImportError(f"archive member is not a regular file: {name!r}")
    return path


@dataclass(frozen=True, slots=True)
class _ArchivePlan:
    files: dict[str, ZipInfo]
    directories: frozenset[str]
    hashes: dict[str, str]
    manifest: dict[str, Any]


def _archive_plan(archive: Path) -> _ArchivePlan:
    files: dict[str, ZipInfo] = {}
    directories: set[str] = set()
    hashes: dict[str, str] = {}
    try:
        with ZipFile(archive) as bundle:
            for info in bundle.infolist():
                member = _safe_member_path(info)
                try:
                    relative = member.relative_to(_ARCHIVE_PREFIX)
                except ValueError:
                    continue
                if relative == PurePosixPath("."):
                    continue
                namespace = relative.parts[0]
                if namespace in _IGNORED_RUN_NAMESPACES:
                    continue
                if namespace not in _EXTRACTED_NAMESPACES:
                    raise ArchiveImportError(f"unexpected archive namespace: {namespace}")
                relative_name = relative.as_posix().rstrip("/")
                if not relative_name:
                    continue
                if info.is_dir():
                    directories.add(relative_name)
                    continue
                if relative_name in files:
                    raise ArchiveImportError(f"duplicate archive member: {relative_name}")
                files[relative_name] = info

            for relative_name in (*files, *directories):
                path = PurePosixPath(relative_name)
                if relative_name in files and relative_name in directories:
                    raise ArchiveImportError(f"conflicting archive member: {relative_name}")
                if any(parent.as_posix() in files for parent in path.parents[:-1]):
                    raise ArchiveImportError(f"conflicting archive member: {relative_name}")

            manifest_name = "inputs/standardized_residuals_manifest.json"
            if manifest_name not in files:
                raise ArchiveImportError("archive residual manifest is missing")
            for relative_name, info in files.items():
                digest = hashlib.sha256()
                with bundle.open(info) as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        digest.update(block)
                hashes[relative_name] = digest.hexdigest()
            raw_manifest = bundle.read(files[manifest_name])
    except ArchiveImportError:
        raise
    except (BadZipFile, KeyError, OSError, RuntimeError) as error:
        raise ArchiveImportError("archive is unreadable") from error

    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise ArchiveImportError("archive residual manifest is unreadable") from error
    if not isinstance(manifest, dict) or raw_manifest != _canonical_json(manifest):
        raise ArchiveImportError("archive residual manifest is not canonical JSON")
    if manifest.get("manifest_sha256") != residual_manifest_digest(manifest):
        raise ArchiveImportError("archive residual manifest digest differs")
    return _ArchivePlan(files, frozenset(directories), hashes, manifest)


def _git_sources(repository: Path) -> dict[str, bytes]:
    restored: dict[str, bytes] = {}
    for spec in SOURCE_SPECS:
        try:
            result = subprocess.run(
                ["git", "show", f"{spec.revision}:{spec.repository_path}"],
                cwd=repository,
                check=True,
                capture_output=True,
            )
        except (OSError, subprocess.CalledProcessError) as error:
            raise ArchiveImportError(f"unable to restore Git source: {spec.name}") from error
        if hashlib.sha256(result.stdout).hexdigest() != spec.sha256:
            raise ArchiveImportError(f"Git source hash differs: {spec.name}")
        restored[spec.name] = result.stdout
    return restored


def _source_outputs() -> dict[str, str]:
    return {**_SOURCE_OUTPUTS, **{spec.name: spec.output_name for spec in SOURCE_SPECS}}


def _rebound_manifest(manifest: dict[str, Any], run: Path) -> dict[str, Any]:
    rebound = json.loads(_canonical_json(manifest))
    inputs = rebound.get("inputs")
    outputs = _source_outputs()
    expected_hashes = {
        "data": DEMAND_DATA_SHA256,
        **{spec.name: spec.sha256 for spec in SOURCE_SPECS},
    }
    if not isinstance(inputs, dict):
        raise ArchiveImportError("archive residual manifest inputs are invalid")
    for name, output_name in outputs.items():
        entry = inputs.get(name)
        if (
            not isinstance(entry, dict)
            or set(entry) != {"path", "sha256"}
            or entry.get("sha256") != expected_hashes[name]
        ):
            raise ArchiveImportError(f"archive residual manifest source differs: {name}")
        entry["path"] = str((run / "sources" / output_name).resolve())
    rebound["manifest_sha256"] = residual_manifest_digest(rebound)
    return rebound


def _expected_tree(
    plan: _ArchivePlan, final_manifest: dict[str, Any], git_sources: dict[str, bytes]
) -> tuple[dict[str, str], set[str]]:
    hashes = dict(plan.hashes)
    hashes["inputs/standardized_residuals_manifest.json"] = hashlib.sha256(
        _canonical_json(final_manifest)
    ).hexdigest()
    hashes["sources/power_demand_final.csv"] = DEMAND_DATA_SHA256
    for spec in SOURCE_SPECS:
        hashes[f"sources/{spec.output_name}"] = hashlib.sha256(git_sources[spec.name]).hexdigest()
    directories = set(plan.directories) | {"sources"}
    for relative in hashes:
        parent = PurePosixPath(relative).parent
        while parent != PurePosixPath("."):
            directories.add(parent.as_posix())
            parent = parent.parent
    return hashes, directories


def _validate_exact_tree(
    run: Path, expected_hashes: dict[str, str], expected_dirs: set[str]
) -> None:
    try:
        require_local_entry(run, kind="directory")
        actual_files: dict[str, str] = {}
        actual_dirs: set[str] = set()
        for current, names, files in os.walk(run, followlinks=False):
            directory = require_local_entry(Path(current), kind="directory")
            if directory != run:
                actual_dirs.add(directory.relative_to(run).as_posix())
            for name in names:
                child = require_local_entry(directory / name, kind="directory")
                actual_dirs.add(child.relative_to(run).as_posix())
            for name in files:
                child = require_local_entry(directory / name, kind="file")
                actual_files[child.relative_to(run).as_posix()] = file_sha256(child)
    except (OSError, PublicationFSError, ValueError) as error:
        raise ArchiveImportError("destination differs from archived correction source") from error
    if actual_files != expected_hashes or actual_dirs != expected_dirs:
        raise ArchiveImportError("destination differs from archived correction source")


def _extract(plan: _ArchivePlan, archive: Path, staging: Path) -> None:
    for directory in sorted(plan.directories):
        target = staging / Path(directory)
        require_within(staging, target)
        target.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(archive) as bundle:
            for relative_name, info in plan.files.items():
                target = require_within(staging, staging / Path(relative_name))
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(info) as source, target.open("xb") as output:
                    shutil.copyfileobj(source, output)
    except (BadZipFile, OSError, RuntimeError) as error:
        raise ArchiveImportError("archive extraction failed; staging evidence preserved") from error


def _write_sources(
    staging: Path, data: Path, git_sources: dict[str, bytes], manifest: dict[str, Any]
) -> Path:
    sources = staging / "sources"
    sources.mkdir()
    data_target = sources / "power_demand_final.csv"
    shutil.copyfile(data, data_target)
    if file_sha256(data_target) != DEMAND_DATA_SHA256:
        raise ArchiveImportError("copied demand data hash differs")
    for spec in SOURCE_SPECS:
        target = sources / spec.output_name
        target.write_bytes(git_sources[spec.name])
        if file_sha256(target) != spec.sha256:
            raise ArchiveImportError(f"restored source hash differs: {spec.name}")
    manifest_path = staging / "inputs/standardized_residuals_manifest.json"
    manifest_path.write_bytes(_canonical_json(manifest))
    return sources / "experiment.toml"


def _validate_source(run: Path, profile: str, description: str) -> None:
    try:
        validate_correction_source(
            run_dir=run,
            config_path=run / "sources/experiment.toml",
            profile=profile,
        )
    except Exception as error:
        raise ArchiveImportError(f"{description} validation failed; evidence preserved") from error


def _resolved_inputs(
    archive_path: Path, data_path: Path, repository_root: Path, run_dir: Path
) -> tuple[Path, Path, Path, Path]:
    try:
        archive = require_local_entry(Path(archive_path), kind="file").resolve(strict=True)
        data = require_local_entry(Path(data_path), kind="file").resolve(strict=True)
        repository = require_local_entry(Path(repository_root), kind="directory").resolve(
            strict=True
        )
        run = Path(run_dir).resolve(strict=False)
        if run == repository or repository.is_relative_to(run):
            raise ArchiveImportError("destination overlaps repository source")
        if archive.is_relative_to(run) or data.is_relative_to(run):
            raise ArchiveImportError("destination overlaps file source")
        require_within(run.parent, run)
    except ArchiveImportError:
        raise
    except (OSError, PublicationFSError) as error:
        raise ArchiveImportError("import path is missing or unsafe") from error
    return archive, data, repository, run


def import_archived_correction_source(
    archive_path: Path,
    data_path: Path,
    repository_root: Path,
    run_dir: Path,
    *,
    profile: str = "paper",
) -> ImportedCorrectionSource:
    """Restore, rebind, atomically publish, and validate one archived source."""

    try:
        checked_archive = require_local_entry(Path(archive_path), kind="file").resolve(strict=True)
    except (OSError, PublicationFSError) as error:
        raise ArchiveImportError("archive path is missing or unsafe") from error
    plan = _archive_plan(checked_archive)
    archive, data, repository, run = _resolved_inputs(
        archive_path, data_path, repository_root, run_dir
    )
    if file_sha256(data) != DEMAND_DATA_SHA256:
        raise ArchiveImportError("demand data hash differs")
    git_sources = _git_sources(repository)
    final_manifest = _rebound_manifest(plan.manifest, run)
    expected_hashes, expected_dirs = _expected_tree(plan, final_manifest, git_sources)

    run.parent.mkdir(parents=True, exist_ok=True)
    lock_path = run.parent / f".{run.name}.import.lock"
    staging = run.parent / f".{run.name}.import-staging"
    try:
        with exclusive_lock(lock_path):
            if staging.exists():
                raise ArchiveImportError("staging evidence already exists and is preserved")
            if run.exists():
                _validate_exact_tree(run, expected_hashes, expected_dirs)
                _validate_source(run, profile, "destination")
                return ImportedCorrectionSource(run, run / "sources/experiment.toml", True)
            staging.mkdir(mode=0o700)
            _extract(plan, archive, staging)
            staging_manifest = _rebound_manifest(plan.manifest, staging)
            _write_sources(staging, data, git_sources, staging_manifest)
            _validate_source(staging, profile, "staging")
            (staging / "inputs/standardized_residuals_manifest.json").write_bytes(
                _canonical_json(final_manifest)
            )
            os.replace(staging, run)
            _validate_exact_tree(run, expected_hashes, expected_dirs)
            _validate_source(run, profile, "final")
    except ArchiveImportError:
        raise
    except (OSError, PublicationFSError) as error:
        raise ArchiveImportError("archive import failed; evidence preserved") from error
    return ImportedCorrectionSource(run, run / "sources/experiment.toml", False)


__all__ = [
    "ArchiveImportError",
    "ImportedCorrectionSource",
    "SourceSpec",
    "import_archived_correction_source",
    "residual_manifest_digest",
]
