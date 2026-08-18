"""Task 15C ten-fold AR proposal-set and explicit-approval contracts."""

from __future__ import annotations

import json
import os
import stat
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import hqrc_v3.diagnostics.loeo_ar as loeo_ar_module
import numpy as np
import polars as pl
import pytest
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.diagnostics.ar import (
    calibrate_beta_prior,
    diagnose_event_residuals,
    require_approved_calibration,
)
from hqrc_v3.diagnostics.loeo import load_loeo_fold, publish_loeo_universe
from hqrc_v3.diagnostics.loeo_ar import (
    LOEOARProposalError,
    approve_loeo_ar_proposal_set,
    load_approved_loeo_ar_set,
    prepare_loeo_ar_proposal_set,
)
from hqrc_v3.provenance import file_sha256
from test_loeo_diagnostics import CONTEXT
from test_loeo_diagnostics import source as source_fixture

base_source = source_fixture


@pytest.fixture(name="source")
def _ar_source(
    base_source: ValidatedCorrectionSource, monkeypatch: pytest.MonkeyPatch
) -> ValidatedCorrectionSource:
    """Make Task 15B's linear publication nondegenerate after required detrending."""

    standardized = base_source.load_standardized_context(CONTEXT, through=2023).with_columns(
        (
            pl.col("standardized_residual")
            + 0.05 * (pl.int_range(pl.len()) * 0.31).sin()
            + 0.02 * (pl.int_range(pl.len()) * 0.13).cos()
        ).alias("standardized_residual")
    )
    final = base_source.load_final_point_context(CONTEXT).with_columns(
        (
            pl.col("observed_mw")
            + 1.1 * (pl.int_range(pl.len()) * 0.29).sin()
            + 0.3 * (pl.int_range(pl.len()) * 0.11).cos()
        ).alias("observed_mw")
    )
    monkeypatch.setattr(
        ValidatedCorrectionSource,
        "load_standardized_context",
        lambda _self, context, *, through: (
            standardized.clone()
            if context == CONTEXT and through == 2023
            else (_ for _ in ()).throw(AssertionError("wrong context"))
        ),
    )
    monkeypatch.setattr(
        ValidatedCorrectionSource,
        "load_final_point_context",
        lambda _self, context: (
            final.clone()
            if context == CONTEXT
            else (_ for _ in ()).throw(AssertionError("wrong context"))
        ),
    )
    return base_source


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return sha256(_canonical(value)).hexdigest()


def _write_canonical(path: Path, value: object) -> None:
    path.write_bytes(_canonical(value))


def _rehash_set(proposal, payload: dict[str, object]) -> None:
    unsigned = {key: value for key, value in payload.items() if key != "proposal_set_sha256"}
    payload["proposal_set_sha256"] = _digest(unsigned)
    _write_canonical(proposal.proposal_set_path, payload)
    _write_canonical(
        proposal.generation_dir / "COMPLETE",
        {"proposal_set_sha256": payload["proposal_set_sha256"]},
    )
    _write_canonical(
        proposal.output_dir / "current.json",
        {
            "proposal_set_sha256": payload["proposal_set_sha256"],
            "schema_version": "hqrc-v3.loeo-ar-set.v1",
        },
    )


def _prepare(source: ValidatedCorrectionSource, tmp_path: Path):
    loeo_dir = tmp_path / "loeo"
    publication = publish_loeo_universe(source, CONTEXT, output_dir=loeo_dir)
    proposal = prepare_loeo_ar_proposal_set(source, publication, output_dir=tmp_path / "loeo-ar")
    return publication, proposal


def _tree_snapshot(root: Path) -> dict[str, tuple[object, ...]]:
    snapshot: dict[str, tuple[object, ...]] = {}

    def record(path: Path) -> None:
        relative = path.relative_to(root).as_posix() if path != root else "."
        identity = path.lstat()
        if stat.S_ISDIR(identity.st_mode):
            snapshot[relative] = ("directory", identity.st_ino)
            for child in sorted(path.iterdir(), key=lambda item: item.name):
                record(child)
        elif stat.S_ISREG(identity.st_mode):
            snapshot[relative] = ("regular", identity.st_ino, path.read_bytes())
        elif stat.S_ISLNK(identity.st_mode):
            snapshot[relative] = ("symlink", os.readlink(path))
        else:
            snapshot[relative] = ("special", stat.S_IFMT(identity.st_mode))

    record(root)
    return snapshot


def _single_stage(root: Path, prefix: str) -> Path:
    stages = [path for path in root.iterdir() if path.name.startswith(prefix)]
    assert len(stages) == 1
    return stages[0]


def _leave_proposal_stage(
    source: ValidatedCorrectionSource,
    publication,
    output: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete: bool,
) -> Path:
    output.mkdir()
    if complete:
        original_replace = loeo_ar_module.os.replace

        def interrupt_generation(source_path, destination_path):
            if Path(destination_path).name == "generation":
                raise OSError("leave complete proposal stage")
            return original_replace(source_path, destination_path)

        monkeypatch.setattr(loeo_ar_module.os, "replace", interrupt_generation)
    else:
        original_writer = loeo_ar_module.write_ar_diagnostics
        calls = 0

        def interrupt_second_proposal(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("leave incomplete proposal stage")
            return original_writer(*args, **kwargs)

        monkeypatch.setattr(loeo_ar_module, "write_ar_diagnostics", interrupt_second_proposal)
    with pytest.raises(OSError, match="leave .* proposal stage"):
        prepare_loeo_ar_proposal_set(source, publication, output_dir=output)
    if complete:
        monkeypatch.setattr(loeo_ar_module.os, "replace", original_replace)
    else:
        monkeypatch.setattr(loeo_ar_module, "write_ar_diagnostics", original_writer)
    return _single_stage(output, ".staging-")


def _leave_approval_stage(
    source: ValidatedCorrectionSource,
    publication,
    proposal,
    monkeypatch: pytest.MonkeyPatch,
    *,
    complete: bool,
) -> Path:
    if complete:
        original_replace = loeo_ar_module.os.replace

        def interrupt_approval(source_path, destination_path):
            if Path(destination_path).name == "approval":
                raise OSError("leave complete approval stage")
            return original_replace(source_path, destination_path)

        monkeypatch.setattr(loeo_ar_module.os, "replace", interrupt_approval)
    else:
        original_approve = loeo_ar_module.approve_calibration
        calls = 0

        def interrupt_second_approval(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("leave incomplete approval stage")
            return original_approve(*args, **kwargs)

        monkeypatch.setattr(loeo_ar_module, "approve_calibration", interrupt_second_approval)
    with pytest.raises(OSError, match="leave .* approval stage"):
        approve_loeo_ar_proposal_set(
            source,
            publication,
            output_dir=proposal.output_dir,
            confirm_proposal_set_sha256=proposal.proposal_set_sha256,
        )
    if complete:
        monkeypatch.setattr(loeo_ar_module.os, "replace", original_replace)
    else:
        monkeypatch.setattr(loeo_ar_module, "approve_calibration", original_approve)
    return _single_stage(proposal.generation_dir, ".approval-staging-")


def test_prepare_publishes_exactly_ten_training_only_unapproved_proposals_and_plots(
    source: ValidatedCorrectionSource, tmp_path: Path
):
    publication, proposal = _prepare(source, tmp_path)

    assert proposal.approved is False
    assert proposal.occurrence_ids == publication.occurrence_ids
    assert tuple(proposal.training_sha256) == publication.occurrence_ids
    assert tuple(proposal.proposal_paths) == publication.occurrence_ids
    assert tuple(proposal.plot_paths) == publication.occurrence_ids
    assert proposal.source_context.context == publication.context
    assert proposal.source_context.loeo_manifest_sha256 == file_sha256(publication.manifest_path)
    assert proposal.source_context.universe_sha256 == publication.universe_sha256
    for held_out in publication.occurrence_ids:
        fold = load_loeo_fold(
            source,
            CONTEXT,
            output_dir=publication.output_dir,
            held_out_occurrence_id=held_out,
        )
        artifact = json.loads(proposal.proposal_paths[held_out].read_text())
        plot = proposal.plot_paths[held_out]
        assert proposal.training_sha256[held_out] == fold.residual_sha256
        assert artifact["approved"] is False
        assert artifact["hashes"]["residual_sha256"] == fold.residual_sha256
        assert artifact["calibration"]["event_ids"] == sorted(fold.occurrence_ids)
        assert len(artifact["diagnostics"]) == 9
        assert all("diagnostic_warning" in item for item in artifact["diagnostics"])
        assert stat.S_ISREG(plot.lstat().st_mode) and not plot.is_symlink()
        assert "raw ACF/PACF" in plot.read_text()
        assert "detrended ACF/PACF" in plot.read_text()
        assert "innovation ACF/PACF" in plot.read_text()


def test_prepare_reuses_event_reset_diagnostics_and_robust_beta_rule(
    source: ValidatedCorrectionSource, tmp_path: Path
):
    publication, proposal = _prepare(source, tmp_path)
    held_out = "seollal-2024"
    fold = load_loeo_fold(
        source,
        CONTEXT,
        output_dir=publication.output_dir,
        held_out_occurrence_id=held_out,
    )
    diagnostics = diagnose_event_residuals(fold.frame, allow_final_split=True)
    expected = calibrate_beta_prior(
        np.asarray([item.phi for item in diagnostics]),
        event_ids=tuple(item.occurrence_id for item in diagnostics),
    )
    artifact = json.loads(proposal.proposal_paths[held_out].read_text())

    assert artifact["calibration"]["a"] == pytest.approx(expected.a)
    assert artifact["calibration"]["b"] == pytest.approx(expected.b)
    assert artifact["calibration"]["phi_center"] == pytest.approx(expected.phi_center)
    assert artifact["calibration"]["phi_estimates"] == pytest.approx(expected.phi_estimates)


def test_prepare_never_invokes_approval(
    source: ValidatedCorrectionSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    publication = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "loeo")
    monkeypatch.setattr(
        loeo_ar_module,
        "approve_calibration",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("approval called")),
    )

    proposal = prepare_loeo_ar_proposal_set(source, publication, output_dir=tmp_path / "loeo-ar")

    assert proposal.approved is False


def test_batch_approval_requires_exact_digest_and_does_not_recompute(
    source: ValidatedCorrectionSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    publication, proposal = _prepare(source, tmp_path)
    with pytest.raises(LOEOARProposalError, match="confirmation"):
        approve_loeo_ar_proposal_set(
            source,
            publication,
            output_dir=proposal.output_dir,
            confirm_proposal_set_sha256="0" * 64,
        )
    with pytest.raises(LOEOARProposalError, match="confirmation"):
        approve_loeo_ar_proposal_set(
            source,
            publication,
            output_dir=proposal.output_dir,
            confirm_proposal_set_sha256="not-a-digest",
        )
    monkeypatch.setattr(
        loeo_ar_module,
        "diagnose_event_residuals",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("diagnostics rerun")),
    )
    monkeypatch.setattr(
        loeo_ar_module,
        "calibrate_beta_prior",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("calibration rerun")),
    )

    approved_path = approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=proposal.proposal_set_sha256,
    )

    approved_payload = json.loads(approved_path.read_bytes())
    assert approved_payload["proposal_set_sha256"] == proposal.proposal_set_sha256
    assert approved_payload["occurrence_ids"] == list(publication.occurrence_ids)


def test_load_approved_set_returns_tokened_per_fold_calibrations(
    source: ValidatedCorrectionSource, tmp_path: Path
):
    publication, proposal = _prepare(source, tmp_path)
    approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=proposal.proposal_set_sha256,
    )

    loaded = load_approved_loeo_ar_set(source, publication, output_dir=proposal.output_dir)
    selected = loaded.calibration_for("chuseok-2024")

    assert tuple(loaded.calibrations) == publication.occurrence_ids
    assert tuple(loaded.proposal_sha256) == publication.occurrence_ids
    assert tuple(loaded.approved_paths) == publication.occurrence_ids
    assert loaded.source_context == proposal.source_context
    assert require_approved_calibration(selected) == selected
    assert selected.calibration.event_ids == tuple(
        sorted(set(publication.occurrence_ids) - {"chuseok-2024"})
    )


def test_held_out_outcome_mutation_cannot_change_its_proposal_or_plot(
    source: ValidatedCorrectionSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first_publication, first = _prepare(source, tmp_path / "first")
    held_out = "seollal-2024"
    target_times = (
        pl.read_parquet(first_publication.universe_path)
        .filter(pl.col("occurrence_id") == held_out)["target_timestamp"]
        .to_list()
    )
    original = source.load_final_point_context(CONTEXT)
    changed = original.with_columns(
        pl.when(pl.col("target_timestamp").is_in(target_times))
        .then(pl.col("observed_mw") + 19.0)
        .otherwise(pl.col("observed_mw"))
        .alias("observed_mw")
    )
    monkeypatch.setattr(
        ValidatedCorrectionSource,
        "load_final_point_context",
        lambda _self, _context: changed.clone(),
    )
    second_publication = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "second/loeo")
    second = prepare_loeo_ar_proposal_set(
        source, second_publication, output_dir=tmp_path / "second/loeo-ar"
    )

    assert first.training_sha256[held_out] == second.training_sha256[held_out]
    assert first.proposal_digests[held_out] == second.proposal_digests[held_out]
    assert first.plot_sha256[held_out] == second.plot_sha256[held_out]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "reordered", "substituted"])
def test_rehashed_registry_population_mutation_is_rejected(
    source: ValidatedCorrectionSource, tmp_path: Path, mutation: str
):
    publication, proposal = _prepare(source, tmp_path)
    payload = json.loads(proposal.proposal_set_path.read_bytes())
    if mutation == "duplicate":
        payload["occurrence_ids"][-1] = payload["occurrence_ids"][0]
    elif mutation == "missing":
        payload["occurrence_ids"].pop()
        payload["folds"].pop()
    elif mutation == "reordered":
        payload["occurrence_ids"].reverse()
        payload["folds"].reverse()
    else:
        payload["occurrence_ids"][-1] = "seollal-2025"
        payload["folds"][-1]["held_out_occurrence_id"] = "seollal-2025"
    _rehash_set(proposal, payload)

    with pytest.raises(LOEOARProposalError, match="registry|complete|order|identit"):
        prepare_loeo_ar_proposal_set(source, publication, output_dir=proposal.output_dir)


@pytest.mark.parametrize("mutation", ["proposal", "plot"])
def test_rehashed_semantic_proposal_or_plot_mutation_is_rejected_after_approval(
    source: ValidatedCorrectionSource, tmp_path: Path, mutation: str
):
    publication, proposal = _prepare(source, tmp_path)
    held_out = "seollal-2024"
    set_payload = json.loads(proposal.proposal_set_path.read_bytes())
    entry = next(
        item for item in set_payload["folds"] if item["held_out_occurrence_id"] == held_out
    )
    if mutation == "plot":
        proposal.plot_paths[held_out].write_bytes(
            proposal.plot_paths[held_out].read_bytes() + b"<!-- rehashed mutation -->\n"
        )
        entry["plot_sha256"] = file_sha256(proposal.plot_paths[held_out])
    else:
        artifact = json.loads(proposal.proposal_paths[held_out].read_text())
        artifact["diagnostics"][0]["raw_acf"][0] += 0.01
        artifact.pop("artifact_digest")
        artifact.pop("proposal_digest")
        artifact["proposal_digest"] = _digest(artifact)
        artifact["artifact_digest"] = _digest(artifact)
        proposal.proposal_paths[held_out].write_bytes(_canonical(artifact) + b"\n")
        entry["proposal_digest"] = artifact["proposal_digest"]
        entry["proposal_sha256"] = file_sha256(proposal.proposal_paths[held_out])
    _rehash_set(proposal, set_payload)
    approved_path = approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=set_payload["proposal_set_sha256"],
    )
    assert (
        json.loads(approved_path.read_bytes())["proposal_set_sha256"]
        == set_payload["proposal_set_sha256"]
    )

    with pytest.raises(LOEOARProposalError, match="semantic"):
        load_approved_loeo_ar_set(source, publication, output_dir=proposal.output_dir)


@pytest.mark.parametrize("mutation", ["unknown", "plot", "proposal", "fold-swap"])
def test_approved_loader_fails_closed_on_bound_artifact_mutation(
    source: ValidatedCorrectionSource, tmp_path: Path, mutation: str
):
    publication, proposal = _prepare(source, tmp_path)
    approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=proposal.proposal_set_sha256,
    )
    if mutation == "unknown":
        (proposal.generation_dir / "unexpected").write_text("bad")
    elif mutation == "plot":
        proposal.plot_paths["seollal-2024"].write_text("changed")
    elif mutation == "proposal":
        proposal.proposal_paths["seollal-2024"].write_text("{}")
    else:
        left = proposal.proposal_paths["seollal-2024"]
        right = proposal.proposal_paths["chuseok-2024"]
        content = left.read_bytes()
        left.write_bytes(right.read_bytes())
        right.write_bytes(content)

    with pytest.raises((LOEOARProposalError, ValueError)):
        load_approved_loeo_ar_set(source, publication, output_dir=proposal.output_dir)


def test_rejects_causal_eight_event_context_mutation_and_single_approval(
    source: ValidatedCorrectionSource, tmp_path: Path
):
    publication, proposal = _prepare(source, tmp_path)
    causal_eight = replace(publication, causal=True, occurrence_ids=publication.occurrence_ids[:8])
    with pytest.raises(LOEOARProposalError, match="causal=false ten-event"):
        prepare_loeo_ar_proposal_set(source, causal_eight, output_dir=tmp_path / "bad")
    changed_context = replace(
        publication,
        context=replace(publication.context, split_ids=publication.context.split_ids[:-1]),
    )
    with pytest.raises(ValueError):
        prepare_loeo_ar_proposal_set(source, changed_context, output_dir=tmp_path / "context")
    approved_path = approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=proposal.proposal_set_sha256,
    )
    one_artifact = approved_path.parent / f"{publication.occurrence_ids[0]}.json"
    with pytest.raises(LOEOARProposalError):
        load_approved_loeo_ar_set(source, publication, output_dir=one_artifact)


def test_loader_rejects_symlink_plot_and_changed_physical_fold(
    source: ValidatedCorrectionSource, tmp_path: Path
):
    publication, proposal = _prepare(source, tmp_path)
    approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=proposal.proposal_set_sha256,
    )
    held_out = "seollal-2024"
    target = tmp_path / "plot-copy.svg"
    target.write_bytes(proposal.plot_paths[held_out].read_bytes())
    proposal.plot_paths[held_out].unlink()
    proposal.plot_paths[held_out].symlink_to(target)
    with pytest.raises(LOEOARProposalError, match="unsafe"):
        load_approved_loeo_ar_set(source, publication, output_dir=proposal.output_dir)

    proposal.plot_paths[held_out].unlink()
    proposal.plot_paths[held_out].write_bytes(target.read_bytes())
    publication.fold_paths[held_out].write_bytes(b"changed physical fold")
    with pytest.raises(ValueError):
        load_approved_loeo_ar_set(source, publication, output_dir=proposal.output_dir)


@pytest.mark.parametrize("boundary", ["generation", "pointer"])
def test_prepare_interruption_retries_without_partial_reuse(
    source: ValidatedCorrectionSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
):
    publication = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "loeo")
    output = tmp_path / "loeo-ar"
    original_replace = os.replace

    def interrupted_replace(source_path, destination_path):
        destination = Path(destination_path)
        if destination.name == boundary or (
            boundary == "pointer" and destination.name == "current.json"
        ):
            raise OSError("injected Task 15C interruption")
        return original_replace(source_path, destination_path)

    monkeypatch.setattr(loeo_ar_module.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="Task 15C interruption"):
        prepare_loeo_ar_proposal_set(source, publication, output_dir=output)
    monkeypatch.setattr(loeo_ar_module.os, "replace", original_replace)

    recovered = prepare_loeo_ar_proposal_set(source, publication, output_dir=output)

    assert recovered.approved is False
    assert len(recovered.proposal_paths) == 10


def test_approval_interruption_retries_and_missing_confirmation_is_a_type_error(
    source: ValidatedCorrectionSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    publication, proposal = _prepare(source, tmp_path)
    with pytest.raises(TypeError):
        approve_loeo_ar_proposal_set(  # type: ignore[call-arg]
            source, publication, output_dir=proposal.output_dir
        )
    original_replace = os.replace

    def interrupted_replace(source_path, destination_path):
        if Path(destination_path).name == "approval":
            raise OSError("injected approval interruption")
        return original_replace(source_path, destination_path)

    monkeypatch.setattr(loeo_ar_module.os, "replace", interrupted_replace)
    with pytest.raises(OSError, match="approval interruption"):
        approve_loeo_ar_proposal_set(
            source,
            publication,
            output_dir=proposal.output_dir,
            confirm_proposal_set_sha256=proposal.proposal_set_sha256,
        )
    monkeypatch.setattr(loeo_ar_module.os, "replace", original_replace)

    approved_path = approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=proposal.proposal_set_sha256,
    )

    assert approved_path.is_file()
    approved = load_approved_loeo_ar_set(source, publication, output_dir=proposal.output_dir)
    assert len(approved.calibrations) == 10


@pytest.mark.parametrize("target_exists", [False, True])
def test_publication_lock_rejects_symlink_without_touching_target(
    tmp_path: Path, target_exists: bool
):
    root = tmp_path / "publication"
    root.mkdir()
    target = tmp_path / "outside-lock"
    if target_exists:
        target.write_bytes(b"outside evidence")
    (root / ".loeo-ar.lock").symlink_to(target)
    before = target.read_bytes() if target_exists else None

    with pytest.raises(LOEOARProposalError, match="lock|unsafe"):
        with loeo_ar_module._publication_lock(root):
            pytest.fail("symlink lock was acquired")

    assert target.exists() is target_exists
    if target_exists:
        assert target.read_bytes() == before


@pytest.mark.parametrize("target_exists", [False, True])
def test_approval_entry_symlink_is_preserved_and_target_untouched(
    source: ValidatedCorrectionSource,
    tmp_path: Path,
    target_exists: bool,
):
    publication, proposal = _prepare(source, tmp_path)
    target = tmp_path / "outside-approval"
    if target_exists:
        target.mkdir()
        (target / "evidence").write_bytes(b"outside evidence")
    approval = proposal.generation_dir / "approval"
    approval.symlink_to(target, target_is_directory=True)
    target_state = _tree_snapshot(target) if target_exists else None

    with pytest.raises(LOEOARProposalError, match="approval|unsafe"):
        approve_loeo_ar_proposal_set(
            source,
            publication,
            output_dir=proposal.output_dir,
            confirm_proposal_set_sha256=proposal.proposal_set_sha256,
        )

    assert approval.is_symlink()
    assert target.exists() is target_exists
    if target_exists:
        assert _tree_snapshot(target) == target_state


@pytest.mark.parametrize("complete", [False, True])
def test_fresh_prepare_resumes_matching_stage_without_deleting_verified_files(
    source: ValidatedCorrectionSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
):
    publication = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "loeo")
    output = tmp_path / "loeo-ar"
    stage = _leave_proposal_stage(source, publication, output, monkeypatch, complete=complete)
    identity_raw = (stage / "STAGE.json").read_bytes()
    assert identity_raw == _canonical(json.loads(identity_raw))
    preserved = next((stage / "proposals").glob("*.json"))
    preserved_inode = preserved.stat().st_ino
    preserved_bytes = preserved.read_bytes()

    recovered = prepare_loeo_ar_proposal_set(source, publication, output_dir=output)

    recovered_path = recovered.generation_dir / "proposals" / preserved.name
    assert recovered_path.stat().st_ino == preserved_inode
    assert recovered_path.read_bytes() == preserved_bytes
    assert not any(path.name.startswith(".staging-") for path in output.iterdir())


@pytest.mark.parametrize("mutation", ["foreign", "unknown", "symlink"])
def test_fresh_prepare_preserves_foreign_or_unsafe_stage(
    source: ValidatedCorrectionSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    publication = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "loeo")
    output = tmp_path / "loeo-ar"
    stage = _leave_proposal_stage(source, publication, output, monkeypatch, complete=False)
    if mutation == "foreign":
        identity = json.loads((stage / "STAGE.json").read_bytes())
        identity["stage_identity_sha256"] = "0" * 64
        _write_canonical(stage / "STAGE.json", identity)
    elif mutation == "unknown":
        (stage / "unexpected").write_bytes(b"foreign evidence")
    else:
        target = tmp_path / "outside-stage"
        target.write_bytes(b"outside evidence")
        (stage / "COMPLETE").symlink_to(target)
    before = _tree_snapshot(stage)

    with pytest.raises(LOEOARProposalError, match="stage|staging|identity|unsafe|preserved"):
        prepare_loeo_ar_proposal_set(source, publication, output_dir=output)

    assert _tree_snapshot(stage) == before


@pytest.mark.parametrize("complete", [False, True])
def test_fresh_approval_resumes_matching_stage_without_deleting_verified_files(
    source: ValidatedCorrectionSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    complete: bool,
):
    publication, proposal = _prepare(source, tmp_path)
    stage = _leave_approval_stage(source, publication, proposal, monkeypatch, complete=complete)
    identity_raw = (stage / "STAGE.json").read_bytes()
    assert identity_raw == _canonical(json.loads(identity_raw))
    preserved = next(stage.glob("*.json"))
    while preserved.name in {"STAGE.json", "approved-set.json"}:
        preserved = next(
            path
            for path in stage.glob("*.json")
            if path.name
            not in {
                "STAGE.json",
                "approved-set.json",
            }
        )
    preserved_inode = preserved.stat().st_ino
    preserved_bytes = preserved.read_bytes()

    approved_path = approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=proposal.proposal_set_sha256,
    )

    recovered_path = approved_path.parent / preserved.name
    assert recovered_path.stat().st_ino == preserved_inode
    assert recovered_path.read_bytes() == preserved_bytes


def test_wrong_confirmation_preserves_preexisting_approval_stage(
    source: ValidatedCorrectionSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    publication, proposal = _prepare(source, tmp_path)
    stage = _leave_approval_stage(source, publication, proposal, monkeypatch, complete=False)
    before = _tree_snapshot(stage)

    with pytest.raises(LOEOARProposalError, match="confirmation"):
        approve_loeo_ar_proposal_set(
            source,
            publication,
            output_dir=proposal.output_dir,
            confirm_proposal_set_sha256="0" * 64,
        )

    assert _tree_snapshot(stage) == before


@pytest.mark.parametrize("mutation", ["foreign", "unknown", "symlink"])
def test_fresh_approval_preserves_foreign_or_unsafe_stage(
    source: ValidatedCorrectionSource,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
):
    publication, proposal = _prepare(source, tmp_path)
    stage = _leave_approval_stage(source, publication, proposal, monkeypatch, complete=False)
    if mutation == "foreign":
        identity = json.loads((stage / "STAGE.json").read_bytes())
        identity["stage_identity_sha256"] = "0" * 64
        _write_canonical(stage / "STAGE.json", identity)
    elif mutation == "unknown":
        (stage / "unexpected").write_bytes(b"foreign evidence")
    else:
        target = tmp_path / "outside-approval-stage"
        target.write_bytes(b"outside evidence")
        (stage / "COMPLETE").symlink_to(target)
    before = _tree_snapshot(stage)

    with pytest.raises(LOEOARProposalError, match="stage|staging|identity|unsafe|preserved"):
        approve_loeo_ar_proposal_set(
            source,
            publication,
            output_dir=proposal.output_dir,
            confirm_proposal_set_sha256=proposal.proposal_set_sha256,
        )

    assert _tree_snapshot(stage) == before
