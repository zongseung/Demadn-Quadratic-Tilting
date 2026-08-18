"""Trust-boundary tests for archived correction-source import."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from hqrc_v3 import artifact_import
from hqrc_v3.artifact_import import (
    ArchiveImportError,
    SourceSpec,
    import_archived_correction_source,
    residual_manifest_digest,
)

PREFIX = "artifacts/hqrc-v3-paper-20260811/"
SOURCE_NAMES = (
    "data",
    "experiment_config",
    "model_config",
    "event_registry",
    "holiday_calendar",
    "temporary_holiday_availability",
)


def _sha(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _tree_hashes(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _sha(path.read_bytes())
        for path in root.rglob("*")
        if path.is_file()
    }


@dataclass(frozen=True)
class ImportFixture:
    archive_path: Path
    data_path: Path
    repository_root: Path
    run_dir: Path
    source_bytes: dict[str, bytes]
    parquet_bytes: dict[str, bytes]

    def arguments(self) -> dict[str, Path]:
        return {
            "archive_path": self.archive_path,
            "data_path": self.data_path,
            "repository_root": self.repository_root,
            "run_dir": self.run_dir,
        }


@pytest.fixture
def import_fixture(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ImportFixture:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "-q")
    _git(repository, "config", "user.email", "tests@example.invalid")
    _git(repository, "config", "user.name", "Tests")
    _git(repository, "config", "core.autocrlf", "false")

    source_bytes = {
        "data": b"synthetic demand data\n",
        "experiment_config": b"experiment = true\n",
        "model_config": b"models = []\n",
        "event_registry": b"event\n",
        "holiday_calendar": b"holiday\n",
        "temporary_holiday_availability": b"availability\n",
    }
    files = {
        "experiment_config": ("configs/experiment.toml", "experiment.toml"),
        "model_config": ("configs/model_spaces.toml", "model_spaces.toml"),
        "event_registry": ("configs/events.csv", "events.csv"),
        "holiday_calendar": ("configs/holiday_calendar.csv", "holiday_calendar.csv"),
        "temporary_holiday_availability": (
            "configs/temporary_holiday_availability.csv",
            "temporary_holiday_availability.csv",
        ),
    }
    for name, (repository_path, _) in files.items():
        path = repository / repository_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(source_bytes[name])
    _git(repository, "add", "configs")
    _git(repository, "commit", "-qm", "fixture sources")
    revision = _git(repository, "rev-parse", "HEAD")
    monkeypatch.setattr(
        artifact_import,
        "SOURCE_SPECS",
        tuple(
            SourceSpec(name, revision, repository_path, output_name, _sha(source_bytes[name]))
            for name, (repository_path, output_name) in files.items()
        ),
    )
    monkeypatch.setattr(artifact_import, "DEMAND_DATA_SHA256", _sha(source_bytes["data"]))

    data = tmp_path / "controlled-data.csv"
    data.write_bytes(source_bytes["data"])
    inputs = {
        name: {"path": f"/old/macos/{name}", "sha256": _sha(source_bytes[name])}
        for name in SOURCE_NAMES
    }
    manifest: dict[str, object] = {
        "contexts": [{"sentinel": "unchanged"}],
        "inputs": inputs,
        "nested": {"path": "/not/a/source/path"},
    }
    manifest["manifest_sha256"] = residual_manifest_digest(manifest)
    parquet_bytes = {
        "predictions/oof.parquet": b"oof parquet bytes",
        "predictions/final_2024.parquet": b"final parquet bytes",
        "inputs/standardized_residuals.parquet": b"residual parquet bytes",
        "prediction-stream-cache/cache.parquet": b"cache parquet bytes",
    }
    archive = tmp_path / "controlled-artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        for directory in (
            "__MACOSX/",
            "artifacts/",
            "artifacts/baseline_fit2023_eval2024/",
            "artifacts/hqrc-v3-causal-2024-report-20260813/",
            "artifacts/hqt_fit2023_eval2024/",
            PREFIX,
            PREFIX + "inputs/",
            PREFIX + "predictions/",
            PREFIX + "prediction-stream-cache/",
            PREFIX + "corrections/",
        ):
            bundle.writestr(directory, b"")
        for relative, content in parquet_bytes.items():
            bundle.writestr(PREFIX + relative, content)
        bundle.writestr(
            PREFIX + "inputs/standardized_residuals_manifest.json",
            json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
        )
        bundle.writestr(PREFIX + "corrections/causal-2024/result.nc", b"ignored")
        bundle.writestr("__MACOSX/._artifacts", b"ignored")
        bundle.writestr("artifacts/.DS_Store", b"ignored")
        bundle.writestr("artifacts/baseline_fit2023_eval2024/result.bin", b"ignored")
        bundle.writestr("artifacts/hqrc-v3-causal-2024-report-20260813/report.bin", b"ignored")
        bundle.writestr("artifacts/hqt_fit2023_eval2024/result.bin", b"ignored")

    return ImportFixture(
        archive,
        data,
        repository,
        tmp_path / "run",
        source_bytes,
        parquet_bytes,
    )


def _validating_spy(calls: list[tuple[Path, Path, str]]):
    def validate(*, run_dir: Path, config_path: Path, profile: str) -> object:
        manifest = json.loads(
            (run_dir / "inputs/standardized_residuals_manifest.json").read_bytes()
        )
        assert manifest["manifest_sha256"] == residual_manifest_digest(manifest)
        for name in SOURCE_NAMES:
            assert Path(manifest["inputs"][name]["path"]).parent == run_dir / "sources"
        assert config_path == run_dir / "sources/experiment.toml"
        calls.append((run_dir, config_path, profile))
        return object()

    return validate


def _symlink_or_skip(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")


def test_import_rejects_traversal_before_destination_creation(tmp_path: Path):
    archive = tmp_path / "artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(PREFIX + "../../escape", b"x")
    destination = tmp_path / "run"

    with pytest.raises(ArchiveImportError, match="unsafe archive member"):
        import_archived_correction_source(archive, tmp_path / "data.csv", tmp_path, destination)

    assert not destination.exists()
    assert not list(tmp_path.glob(".run.import-staging*"))


@pytest.mark.parametrize(
    "member",
    [
        PREFIX + "surprise/file.txt",
        PREFIX + "predictions/../escape.txt",
        "C:/absolute.txt",
        "/absolute.txt",
        PREFIX + "predictions\\..\\escape.txt",
    ],
)
def test_import_rejects_unsafe_or_unexpected_members_before_staging(tmp_path: Path, member: str):
    archive = tmp_path / "artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"x")

    with pytest.raises(ArchiveImportError):
        import_archived_correction_source(
            archive, tmp_path / "data.csv", tmp_path, tmp_path / "run"
        )

    assert not list(tmp_path.glob(".run.import-staging*"))


@pytest.mark.parametrize(
    "member",
    [
        "outside/file.txt",
        "artifacts/unrelated-run/file.txt",
        PREFIX + "diagnostics/file.txt",
        PREFIX + "reports/file.txt",
    ],
)
def test_import_rejects_every_unapproved_namespace_before_staging(tmp_path: Path, member: str):
    archive = tmp_path / "artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(member, b"x")

    with pytest.raises(ArchiveImportError, match="unexpected archive namespace"):
        import_archived_correction_source(
            archive, tmp_path / "data.csv", tmp_path, tmp_path / "run"
        )

    assert not list(tmp_path.glob(".run.import-staging*"))


def test_import_rejects_link_member_before_staging(tmp_path: Path):
    archive = tmp_path / "artifacts.zip"
    member = ZipInfo(PREFIX + "inputs/link")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(member, "target")

    with pytest.raises(ArchiveImportError, match="link"):
        import_archived_correction_source(
            archive, tmp_path / "data.csv", tmp_path, tmp_path / "run"
        )

    assert not list(tmp_path.glob(".run.import-staging*"))


def test_import_rejects_conflicting_member_paths_before_staging(tmp_path: Path):
    archive = tmp_path / "artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(PREFIX + "inputs/conflict", b"file")
        bundle.writestr(PREFIX + "inputs/conflict/child", b"child")

    with pytest.raises(ArchiveImportError, match="conflicting archive member"):
        import_archived_correction_source(
            archive, tmp_path / "data.csv", tmp_path, tmp_path / "run"
        )

    assert not list(tmp_path.glob(".run.import-staging*"))


def test_import_rejects_duplicate_directories_before_staging(tmp_path: Path):
    archive = tmp_path / "artifacts.zip"
    with pytest.warns(UserWarning, match="Duplicate name"):
        with ZipFile(archive, "w") as bundle:
            bundle.writestr("artifacts/", b"")
            bundle.writestr("artifacts/", b"")

    with pytest.raises(ArchiveImportError, match="duplicate archive member"):
        import_archived_correction_source(
            archive, tmp_path / "data.csv", tmp_path, tmp_path / "run"
        )

    assert not list(tmp_path.glob(".run.import-staging*"))


def test_import_rejects_ignored_namespace_ancestor_conflict_before_staging(
    tmp_path: Path,
):
    archive = tmp_path / "artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(PREFIX + "corrections/conflict", b"file")
        bundle.writestr(PREFIX + "corrections/conflict/child", b"child")

    with pytest.raises(ArchiveImportError, match="conflicting archive member"):
        import_archived_correction_source(
            archive, tmp_path / "data.csv", tmp_path, tmp_path / "run"
        )

    assert not list(tmp_path.glob(".run.import-staging*"))


def test_import_rejects_casefold_aliases_in_ignored_namespace_before_staging(
    tmp_path: Path,
):
    archive = tmp_path / "artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("artifacts/baseline_fit2023_eval2024/Result.bin", b"one")
        bundle.writestr("artifacts/baseline_fit2023_eval2024/result.bin", b"two")

    with pytest.raises(ArchiveImportError, match="case-equivalent archive member"):
        import_archived_correction_source(
            archive, tmp_path / "data.csv", tmp_path, tmp_path / "run"
        )

    assert not list(tmp_path.glob(".run.import-staging*"))


def test_import_rejects_source_hash_mismatch_before_staging(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    first, *rest = artifact_import.SOURCE_SPECS
    monkeypatch.setattr(
        artifact_import,
        "SOURCE_SPECS",
        (
            SourceSpec(
                first.name,
                first.revision,
                first.repository_path,
                first.output_name,
                "0" * 64,
            ),
            *rest,
        ),
    )

    with pytest.raises(ArchiveImportError, match="Git source hash differs"):
        import_archived_correction_source(**import_fixture.arguments())

    assert not import_fixture.run_dir.exists()
    assert not list(import_fixture.run_dir.parent.glob(".run.import-staging*"))


def test_import_rejects_and_preserves_symlink_destination(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    target = import_fixture.run_dir.parent / "redirect-target"
    target.mkdir()
    link = import_fixture.run_dir.parent / "redirect-run"
    _symlink_or_skip(link, target)
    monkeypatch.setattr(artifact_import, "validate_correction_source", lambda **_: object())

    with pytest.raises(ArchiveImportError, match="import path is missing or unsafe"):
        import_archived_correction_source(**{**import_fixture.arguments(), "run_dir": link})

    assert link.is_symlink()
    assert not any(target.iterdir())


def test_import_rejects_and_preserves_dangling_destination_link(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    target = import_fixture.run_dir.parent / "missing-target"
    link = import_fixture.run_dir.parent / "dangling-run"
    _symlink_or_skip(link, target)
    monkeypatch.setattr(artifact_import, "validate_correction_source", lambda **_: object())

    with pytest.raises(ArchiveImportError, match="import path is missing or unsafe"):
        import_archived_correction_source(**{**import_fixture.arguments(), "run_dir": link})

    assert link.is_symlink()
    assert not target.exists()


def test_import_rejects_linked_destination_parent(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    target = import_fixture.run_dir.parent / "actual-parent"
    target.mkdir()
    linked_parent = import_fixture.run_dir.parent / "linked-parent"
    _symlink_or_skip(linked_parent, target)
    monkeypatch.setattr(artifact_import, "validate_correction_source", lambda **_: object())

    with pytest.raises(ArchiveImportError, match="import path is missing or unsafe"):
        import_archived_correction_source(
            **{**import_fixture.arguments(), "run_dir": linked_parent / "run"}
        )

    assert linked_parent.is_symlink()
    assert not (target / "run").exists()


def test_import_rejects_raw_destination_parent_traversal(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    raw_run = import_fixture.run_dir.parent / "unused" / ".." / "escaped-run"
    monkeypatch.setattr(artifact_import, "validate_correction_source", lambda **_: object())

    with pytest.raises(ArchiveImportError, match="import path is missing or unsafe"):
        import_archived_correction_source(**{**import_fixture.arguments(), "run_dir": raw_run})

    assert not (import_fixture.run_dir.parent / "escaped-run").exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows junction fallback")
def test_import_rejects_junction_destination_parent_without_symlink_privilege(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    target = import_fixture.run_dir.parent / "junction-target"
    target.mkdir()
    junction = import_fixture.run_dir.parent / "junction-parent"
    subprocess.run(
        ["cmd.exe", "/c", "mklink", "/J", str(junction), str(target)],
        check=True,
        capture_output=True,
    )
    monkeypatch.setattr(artifact_import, "validate_correction_source", lambda **_: object())
    try:
        assert junction.is_junction()
        with pytest.raises(ArchiveImportError, match="import path is missing or unsafe"):
            import_archived_correction_source(
                **{**import_fixture.arguments(), "run_dir": junction / "run"}
            )
        assert junction.is_junction()
        assert not (target / "run").exists()
    finally:
        junction.rmdir()


def test_import_preserves_failed_staging_evidence(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    def reject(**_: object) -> object:
        raise ValueError("controlled validation failure")

    monkeypatch.setattr(artifact_import, "validate_correction_source", reject)

    with pytest.raises(ArchiveImportError, match="staging validation failed"):
        import_archived_correction_source(**import_fixture.arguments())

    assert not import_fixture.run_dir.exists()
    staging = list(import_fixture.run_dir.parent.glob(".run.import-staging*"))
    assert len(staging) == 1
    assert (staging[0] / "inputs/standardized_residuals_manifest.json").is_file()


def test_import_rebinds_only_six_sources_and_reuses_identical_destination(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    validated: list[tuple[Path, Path, str]] = []
    monkeypatch.setattr(artifact_import, "validate_correction_source", _validating_spy(validated))

    first = import_archived_correction_source(**import_fixture.arguments())
    hashes = _tree_hashes(first.run_dir)
    manifest = json.loads(
        (first.run_dir / "inputs/standardized_residuals_manifest.json").read_bytes()
    )
    second = import_archived_correction_source(**import_fixture.arguments())

    assert first.reused is False
    assert second.reused is True
    assert first.config_path == first.run_dir / "sources/experiment.toml"
    assert manifest["nested"]["path"] == "/not/a/source/path"
    assert set(manifest["inputs"]) == set(SOURCE_NAMES)
    assert _tree_hashes(second.run_dir) == hashes
    assert [call[0] for call in validated] == [
        import_fixture.run_dir.parent / ".run.import-staging",
        first.run_dir,
        first.run_dir,
    ]
    for relative, content in import_fixture.parquet_bytes.items():
        assert (first.run_dir / relative).read_bytes() == content
    for name in SOURCE_NAMES:
        restored = Path(manifest["inputs"][name]["path"])
        assert restored.read_bytes() == import_fixture.source_bytes[name]


def test_existing_mismatched_destination_is_preserved_and_rejected(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(artifact_import, "validate_correction_source", lambda **_: object())
    first = import_archived_correction_source(**import_fixture.arguments())
    tampered = first.run_dir / "predictions/oof.parquet"
    tampered.write_bytes(b"tampered")

    with pytest.raises(ArchiveImportError, match="destination differs"):
        import_archived_correction_source(**import_fixture.arguments())

    assert tampered.read_bytes() == b"tampered"


def test_existing_staging_evidence_blocks_destination_reuse(
    import_fixture: ImportFixture, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(artifact_import, "validate_correction_source", lambda **_: object())
    import_archived_correction_source(**import_fixture.arguments())
    staging = import_fixture.run_dir.parent / ".run.import-staging"
    staging.mkdir()
    (staging / "partial").write_bytes(b"preserve me")

    with pytest.raises(ArchiveImportError, match="staging evidence"):
        import_archived_correction_source(**import_fixture.arguments())

    assert (staging / "partial").read_bytes() == b"preserve me"
