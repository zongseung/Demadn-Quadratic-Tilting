"""Immutable ten-fold LOEO AR proposals and explicit batch approval."""

from __future__ import annotations

import fcntl
import os
import re
import stat
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

import hqrc_v3.diagnostics.ar as ar_module
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.diagnostics._loeo_ar_io import (
    LOEOARProposalError,
)
from hqrc_v3.diagnostics._loeo_ar_io import (
    canonical_json as _canonical,
)
from hqrc_v3.diagnostics._loeo_ar_io import (
    fsync_directory as _fsync_directory,
)
from hqrc_v3.diagnostics._loeo_ar_io import (
    read_json as _read_json,
)
from hqrc_v3.diagnostics._loeo_ar_io import (
    require_directory as _require_directory,
)
from hqrc_v3.diagnostics._loeo_ar_io import (
    require_file as _require_file,
)
from hqrc_v3.diagnostics._loeo_ar_io import (
    sha_json as _sha_json,
)
from hqrc_v3.diagnostics._loeo_ar_io import (
    write_json as _write_json,
)
from hqrc_v3.diagnostics._loeo_ar_plot import render_loeo_ar_svg
from hqrc_v3.diagnostics.ar import (
    ApprovedARCalibration,
    ARCalibration,
    EventARDiagnostic,
    EventResidualContext,
    approve_calibration,
    calibrate_beta_prior,
    diagnose_event_residuals,
    load_approved_calibration,
    require_approved_calibration,
    validate_event_residual_context,
    write_ar_diagnostics,
)
from hqrc_v3.diagnostics.loeo import (
    LOEOFold,
    LOEOPublication,
    load_loeo_fold,
    load_loeo_universe,
)
from hqrc_v3.provenance import ArtifactMismatch, file_sha256

_SCHEMA_VERSION = "hqrc-v3.loeo-ar-set.v1"
_APPROVED_SCHEMA_VERSION = "hqrc-v3.loeo-ar-approved-set.v1"
_PROPOSAL_STAGE_SCHEMA_VERSION = "hqrc-v3.loeo-ar-proposal-stage.v1"
_APPROVAL_STAGE_SCHEMA_VERSION = "hqrc-v3.loeo-ar-approval-stage.v1"
_STAGE_FILENAME = "STAGE.json"
_ROOT_NAMESPACE = frozenset({".loeo-ar.lock", "generation", "current.json", ".current.tmp"})
_GENERATION_NAMESPACE = frozenset(
    {"proposals", "plots", "proposal-set.json", "COMPLETE", _STAGE_FILENAME}
)
_APPROVAL_NAMESPACE = frozenset({"approved-set.json", "COMPLETE", _STAGE_FILENAME})
_DIGEST_PATTERN = re.compile(r"[0-9a-f]{64}")
_SET_TOKEN = object()


@dataclass(frozen=True, slots=True)
class LOEOARSourceContext:
    """Exact current LOEO publication identity bound into a proposal set."""

    context: EventResidualContext
    diagnostic_context: EventResidualContext
    loeo_output_dir: Path
    loeo_generation_dir: Path
    loeo_manifest_path: Path
    loeo_manifest_sha256: str
    universe_sha256: str
    causal: bool = False

    def __post_init__(self) -> None:
        for name in ("loeo_output_dir", "loeo_generation_dir", "loeo_manifest_path"):
            object.__setattr__(self, name, Path(getattr(self, name)))


@dataclass(frozen=True, slots=True)
class LOEOARProposalSet:
    """One immutable complete unapproved set of ten fold proposals."""

    output_dir: Path
    generation_dir: Path
    proposal_set_path: Path
    proposal_set_sha256: str
    context: EventResidualContext
    diagnostic_context: EventResidualContext
    source_context: LOEOARSourceContext
    occurrence_ids: tuple[str, ...]
    training_sha256: Mapping[str, str]
    proposal_paths: Mapping[str, Path]
    proposal_sha256: Mapping[str, str]
    proposal_digests: Mapping[str, str]
    plot_paths: Mapping[str, Path]
    plot_sha256: Mapping[str, str]
    calibrations: Mapping[str, ARCalibration]
    approved: bool = False

    def __post_init__(self) -> None:
        for name in ("output_dir", "generation_dir", "proposal_set_path"):
            object.__setattr__(self, name, Path(getattr(self, name)))
        for name in (
            "training_sha256",
            "proposal_sha256",
            "proposal_digests",
            "plot_sha256",
            "calibrations",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        for name in ("proposal_paths", "plot_paths"):
            object.__setattr__(
                self,
                name,
                MappingProxyType({key: Path(value) for key, value in getattr(self, name).items()}),
            )


@dataclass(frozen=True, slots=True)
class ApprovedLOEOARSet:
    """A complete approved fold set whose selection revalidates the whole publication."""

    output_dir: Path
    approved_set_path: Path
    proposal_set_sha256: str
    approved_set_sha256: str
    context: EventResidualContext
    source_context: LOEOARSourceContext
    occurrence_ids: tuple[str, ...]
    training_sha256: Mapping[str, str]
    proposal_sha256: Mapping[str, str]
    proposal_digests: Mapping[str, str]
    plot_sha256: Mapping[str, str]
    approved_paths: Mapping[str, Path]
    approved_sha256: Mapping[str, str]
    calibrations: Mapping[str, ApprovedARCalibration]
    _source: ValidatedCorrectionSource = field(repr=False, compare=False)
    _publication: LOEOPublication = field(repr=False, compare=False)
    _token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        object.__setattr__(self, "approved_set_path", Path(self.approved_set_path))
        for name in (
            "training_sha256",
            "proposal_sha256",
            "proposal_digests",
            "plot_sha256",
            "approved_sha256",
            "calibrations",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))
        object.__setattr__(
            self,
            "approved_paths",
            MappingProxyType({key: Path(value) for key, value in self.approved_paths.items()}),
        )

    def calibration_for(self, held_out_occurrence_id: str) -> ApprovedARCalibration:
        """Revalidate the complete set and return one still-tokened fold calibration."""

        if self._token is not _SET_TOKEN:
            raise TypeError("LOEO fitting requires a loaded approved AR proposal set")
        reloaded = load_approved_loeo_ar_set(
            self._source, self._publication, output_dir=self.output_dir
        )
        if (
            reloaded.proposal_set_sha256 != self.proposal_set_sha256
            or reloaded.approved_set_sha256 != self.approved_set_sha256
        ):
            raise ArtifactMismatch("approved LOEO AR set wrapper differs from its publication")
        try:
            calibration = reloaded.calibrations[held_out_occurrence_id]
        except KeyError as error:
            raise LOEOARProposalError(
                "held-out occurrence is not in the approved AR set"
            ) from error
        return require_approved_calibration(calibration)


@dataclass(frozen=True, slots=True)
class _FoldMaterial:
    fold: LOEOFold
    diagnostics: tuple[EventARDiagnostic, ...]
    calibration: ARCalibration
    diagnostic_context: EventResidualContext
    config_sha256: str
    event_sha256: str
    proposal_payload: dict[str, Any]
    plot: bytes


def _context_payload(context: EventResidualContext) -> dict[str, object]:
    if not isinstance(context, EventResidualContext):
        raise LOEOARProposalError("LOEO AR context is invalid")
    return {
        "feature_set": context.feature_set,
        "model": context.model,
        "seed": context.seed,
        "split_ids": list(context.split_ids),
    }


def _validated_publication(
    source: ValidatedCorrectionSource, publication: LOEOPublication
) -> LOEOPublication:
    if not isinstance(source, ValidatedCorrectionSource):
        raise TypeError("LOEO AR preparation requires a ValidatedCorrectionSource")
    if not isinstance(publication, LOEOPublication):
        raise TypeError("LOEO AR preparation requires a LOEOPublication")
    if publication.causal is not False or len(publication.occurrence_ids) != 10:
        raise LOEOARProposalError("LOEO AR preparation requires a causal=false ten-event set")
    current = load_loeo_universe(source, publication.context, output_dir=publication.output_dir)
    if current != publication:
        raise ArtifactMismatch("supplied LOEO publication differs from current validated inputs")
    return current


def _source_payload(
    publication: LOEOPublication, diagnostic_context: EventResidualContext
) -> dict[str, object]:
    return {
        "causal": False,
        "context": _context_payload(publication.context),
        "diagnostic_context": _context_payload(diagnostic_context),
        "loeo_generation_dir": publication.generation_dir.resolve().as_posix(),
        "loeo_manifest_path": publication.manifest_path.resolve().as_posix(),
        "loeo_manifest_sha256": file_sha256(publication.manifest_path),
        "loeo_output_dir": publication.output_dir.resolve().as_posix(),
        "universe_sha256": publication.universe_sha256,
    }


def _source_context(
    publication: LOEOPublication, diagnostic_context: EventResidualContext
) -> LOEOARSourceContext:
    return LOEOARSourceContext(
        context=publication.context,
        diagnostic_context=diagnostic_context,
        loeo_output_dir=publication.output_dir.resolve(),
        loeo_generation_dir=publication.generation_dir.resolve(),
        loeo_manifest_path=publication.manifest_path.resolve(),
        loeo_manifest_sha256=file_sha256(publication.manifest_path),
        universe_sha256=publication.universe_sha256,
    )


def _stage_payload(unsigned: dict[str, object]) -> dict[str, object]:
    return {**unsigned, "stage_identity_sha256": _sha_json(unsigned)}


def _proposal_stage_identity(publication: LOEOPublication) -> dict[str, object]:
    return _stage_payload(
        {
            "causal": False,
            "context": _context_payload(publication.context),
            "loeo_generation_dir": publication.generation_dir.resolve().as_posix(),
            "loeo_manifest_path": publication.manifest_path.resolve().as_posix(),
            "loeo_manifest_sha256": file_sha256(publication.manifest_path),
            "loeo_output_dir": publication.output_dir.resolve().as_posix(),
            "occurrence_ids": list(publication.occurrence_ids),
            "schema_version": _PROPOSAL_STAGE_SCHEMA_VERSION,
            "stage_kind": "proposal",
            "universe_sha256": publication.universe_sha256,
        }
    )


def _approval_stage_identity(prepared: LOEOARProposalSet) -> dict[str, object]:
    return _stage_payload(
        {
            "occurrence_ids": list(prepared.occurrence_ids),
            "proposal_generation_dir": prepared.generation_dir.resolve().as_posix(),
            "proposal_set_sha256": prepared.proposal_set_sha256,
            "schema_version": _APPROVAL_STAGE_SCHEMA_VERSION,
            "stage_kind": "approval",
        }
    )


def _stage_path(parent: Path, prefix: str, identity: dict[str, object]) -> Path:
    return parent / f"{prefix}{identity['stage_identity_sha256']}"


def _validate_stage_identity(path: Path, expected: dict[str, object], label: str) -> None:
    actual = _read_json(path / _STAGE_FILENAME, f"{label} identity")
    if actual != expected:
        raise LOEOARProposalError(f"{label} identity differs and is preserved")


def _summary(
    calibration: ARCalibration, diagnostics: tuple[EventARDiagnostic, ...]
) -> dict[str, object]:
    return {
        "a": calibration.a,
        "b": calibration.b,
        "diagnostics": [
            {
                "diagnostic_warning": item.diagnostic_warning,
                "ljung_box_lag": item.ljung_box_lag,
                "ljung_box_pvalue": item.ljung_box_pvalue,
                "ljung_box_statistic": item.ljung_box_statistic,
                "occurrence_id": item.occurrence_id,
                "phi": item.phi,
            }
            for item in diagnostics
        ],
        "event_count": calibration.event_count,
        "event_ids": list(calibration.event_ids),
        "formula_version": calibration.formula_version,
        "phi_center": calibration.phi_center,
        "phi_estimates": list(calibration.phi_estimates),
        "warning": any(item.diagnostic_warning for item in diagnostics),
    }


def _materialize_fold(
    source: ValidatedCorrectionSource, publication: LOEOPublication, held_out: str
) -> _FoldMaterial:
    fold = load_loeo_fold(
        source,
        publication.context,
        output_dir=publication.output_dir,
        held_out_occurrence_id=held_out,
    )
    if fold.causal is not False or len(fold.occurrence_ids) != 9:
        raise LOEOARProposalError("LOEO AR diagnostics require one physical causal=false fold")
    diagnostic_context = validate_event_residual_context(fold.frame, allow_final_split=True)
    diagnostics = diagnose_event_residuals(fold.frame, allow_final_split=True)
    event_ids = tuple(item.occurrence_id for item in diagnostics)
    expected_ids = tuple(sorted(set(publication.occurrence_ids) - {held_out}))
    if event_ids != expected_ids:
        raise LOEOARProposalError("LOEO fold diagnostic event identities differ")
    calibration = calibrate_beta_prior(
        np.asarray([item.phi for item in diagnostics], dtype=float), event_ids=event_ids
    )
    config_sha256 = _sha_json(_context_payload(diagnostic_context))
    event_sha256 = _sha_json(list(expected_ids))
    proposal_payload = ar_module._proposal_payload(
        diagnostics,
        calibration,
        residual_sha256=fold.residual_sha256,
        config_sha256=config_sha256,
        event_sha256=event_sha256,
        context=diagnostic_context,
    )
    return _FoldMaterial(
        fold,
        diagnostics,
        calibration,
        diagnostic_context,
        config_sha256,
        event_sha256,
        proposal_payload,
        render_loeo_ar_svg(held_out, diagnostics),
    )


def _proposal_lock_name(held_out: str) -> str:
    return f".{held_out}.json.lock"


def _validate_artifact_namespace(directory: Path, occurrence_ids: tuple[str, ...]) -> None:
    _require_directory(directory, f"LOEO AR {directory.name} namespace")
    expected = {f"{item}.json" for item in occurrence_ids} | {
        _proposal_lock_name(item) for item in occurrence_ids
    }
    entries = {path.name: path for path in directory.iterdir()}
    if set(entries) != expected:
        raise LOEOARProposalError(f"LOEO AR {directory.name} namespace differs")
    for name, path in entries.items():
        _require_file(path, f"LOEO AR {directory.name} artifact {name}")


def _validate_plot_namespace(directory: Path, occurrence_ids: tuple[str, ...]) -> None:
    _require_directory(directory, "LOEO AR plot namespace")
    expected = {f"{item}.svg" for item in occurrence_ids}
    entries = {path.name: path for path in directory.iterdir()}
    if set(entries) != expected:
        raise LOEOARProposalError("LOEO AR plot namespace differs")
    for name, path in entries.items():
        _require_file(path, f"LOEO AR plot {name}")


def _proposal_entry(
    material: _FoldMaterial,
    proposal_path: Path,
    plot_path: Path,
) -> dict[str, object]:
    return {
        "calibration": _summary(material.calibration, material.diagnostics),
        "config_sha256": material.config_sha256,
        "event_sha256": material.event_sha256,
        "held_out_occurrence_id": material.fold.held_out_occurrence_id,
        "occurrence_ids": list(material.fold.occurrence_ids),
        "plot_path": f"plots/{plot_path.name}",
        "plot_sha256": file_sha256(plot_path),
        "proposal_digest": material.proposal_payload["proposal_digest"],
        "proposal_path": f"proposals/{proposal_path.name}",
        "proposal_sha256": file_sha256(proposal_path),
        "training_path": material.fold.path.resolve().as_posix(),
        "training_sha256": material.fold.residual_sha256,
    }


def _set_payload(
    publication: LOEOPublication,
    diagnostic_context: EventResidualContext,
    folds: list[dict[str, object]],
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "approved": False,
        "folds": folds,
        "occurrence_ids": list(publication.occurrence_ids),
        "schema_version": _SCHEMA_VERSION,
        "source": _source_payload(publication, diagnostic_context),
    }
    return {**unsigned, "proposal_set_sha256": _sha_json(unsigned)}


def _set_content(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "proposal_set_sha256"}


def _validate_set_schema(payload: dict[str, Any]) -> None:
    required = {
        "approved",
        "folds",
        "occurrence_ids",
        "proposal_set_sha256",
        "schema_version",
        "source",
    }
    if (
        set(payload) != required
        or payload.get("approved") is not False
        or payload.get("schema_version") != _SCHEMA_VERSION
        or not isinstance(payload.get("folds"), list)
        or not isinstance(payload.get("occurrence_ids"), list)
        or not isinstance(payload.get("source"), dict)
        or payload.get("proposal_set_sha256") != _sha_json(_set_content(payload))
    ):
        raise LOEOARProposalError("LOEO AR proposal-set identity differs")


def _fold_entry_schema(entry: object) -> dict[str, Any]:
    required = {
        "calibration",
        "config_sha256",
        "event_sha256",
        "held_out_occurrence_id",
        "occurrence_ids",
        "plot_path",
        "plot_sha256",
        "proposal_digest",
        "proposal_path",
        "proposal_sha256",
        "training_path",
        "training_sha256",
    }
    if not isinstance(entry, dict) or set(entry) != required:
        raise LOEOARProposalError("LOEO AR fold-set entry schema differs")
    return entry


def _calibration_from_summary(value: object) -> ARCalibration:
    if not isinstance(value, dict):
        raise LOEOARProposalError("LOEO AR calibration summary is invalid")
    required = {
        "a",
        "b",
        "diagnostics",
        "event_count",
        "event_ids",
        "formula_version",
        "phi_center",
        "phi_estimates",
        "warning",
    }
    if set(value) != required:
        raise LOEOARProposalError("LOEO AR calibration summary schema differs")
    try:
        return ARCalibration(
            a=float(value["a"]),
            b=float(value["b"]),
            phi_center=float(value["phi_center"]),
            event_count=int(value["event_count"]),
            event_ids=tuple(value["event_ids"]),
            phi_estimates=tuple(float(item) for item in value["phi_estimates"]),
            formula_version=str(value["formula_version"]),
        )
    except (TypeError, ValueError) as error:
        raise LOEOARProposalError("LOEO AR calibration summary is invalid") from error


def _validate_generation_namespace(
    generation: Path, *, allow_approval_staging: bool = False
) -> None:
    _require_directory(generation, "LOEO AR generation")
    entries = {path.name: path for path in generation.iterdir()}
    approval_staging = {name for name in entries if name.startswith(".approval-staging-")}
    stable = set(entries) - approval_staging
    allowed = set(_GENERATION_NAMESPACE) | {"approval"}
    if (
        (stable != set(_GENERATION_NAMESPACE) and stable != allowed)
        or len(approval_staging) > 1
        or (approval_staging and not allow_approval_staging)
        or (approval_staging and "approval" in stable)
    ):
        raise LOEOARProposalError("LOEO AR generation contains unknown or partial entries")
    for name in ("proposal-set.json", "COMPLETE", _STAGE_FILENAME):
        _require_file(generation / name, f"LOEO AR generation {name}")
    for name in ("proposals", "plots"):
        _require_directory(generation / name, f"LOEO AR generation {name}")
    if "approval" in entries:
        _require_directory(entries["approval"], "LOEO AR approval namespace")
    for name in approval_staging:
        _require_directory(entries[name], "LOEO AR approval staging")


def _validate_generation(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    generation: Path,
    *,
    recompute: bool,
    allow_approval_staging: bool = False,
) -> LOEOARProposalSet:
    _validate_generation_namespace(generation, allow_approval_staging=allow_approval_staging)
    _validate_stage_identity(
        generation, _proposal_stage_identity(publication), "LOEO AR proposal stage"
    )
    payload = _read_json(generation / "proposal-set.json", "LOEO AR proposal set")
    complete = _read_json(generation / "COMPLETE", "LOEO AR completion marker")
    _validate_set_schema(payload)
    proposal_set_sha256 = str(payload["proposal_set_sha256"])
    if complete != {"proposal_set_sha256": proposal_set_sha256}:
        raise LOEOARProposalError("LOEO AR completion hash differs")
    occurrence_ids = tuple(payload["occurrence_ids"])
    if occurrence_ids != publication.occurrence_ids or len(set(occurrence_ids)) != 10:
        raise LOEOARProposalError("LOEO AR proposal-set registry differs")
    folds_payload = payload["folds"]
    if len(folds_payload) != 10:
        raise LOEOARProposalError("LOEO AR proposal set is not complete")
    held_out_ids = tuple(
        str(_fold_entry_schema(entry)["held_out_occurrence_id"]) for entry in folds_payload
    )
    if held_out_ids != occurrence_ids:
        raise LOEOARProposalError("LOEO AR held-out order or identities differ")
    _validate_artifact_namespace(generation / "proposals", occurrence_ids)
    _validate_plot_namespace(generation / "plots", occurrence_ids)

    diagnostic_context: EventResidualContext | None = None
    training_sha256: dict[str, str] = {}
    proposal_paths: dict[str, Path] = {}
    proposal_sha256: dict[str, str] = {}
    proposal_digests: dict[str, str] = {}
    plot_paths: dict[str, Path] = {}
    plot_sha256: dict[str, str] = {}
    calibrations: dict[str, ARCalibration] = {}
    for raw_entry in folds_payload:
        entry = _fold_entry_schema(raw_entry)
        held_out = str(entry["held_out_occurrence_id"])
        fold = load_loeo_fold(
            source,
            publication.context,
            output_dir=publication.output_dir,
            held_out_occurrence_id=held_out,
        )
        expected_ids = tuple(item for item in occurrence_ids if item != held_out)
        expected_diagnostic_ids = tuple(sorted(expected_ids))
        if (
            entry["training_path"] != fold.path.resolve().as_posix()
            or entry["training_sha256"] != fold.residual_sha256
            or entry["occurrence_ids"] != list(expected_ids)
        ):
            raise LOEOARProposalError("LOEO AR physical fold identity differs")
        proposal_path = generation / str(entry["proposal_path"])
        plot_path = generation / str(entry["plot_path"])
        if (
            entry["proposal_path"] != f"proposals/{held_out}.json"
            or entry["plot_path"] != f"plots/{held_out}.svg"
        ):
            raise LOEOARProposalError("LOEO AR canonical artifact path differs")
        _require_file(proposal_path, f"LOEO AR proposal {held_out}")
        _require_file(plot_path, f"LOEO AR plot {held_out}")
        if entry["proposal_sha256"] != file_sha256(proposal_path) or entry[
            "plot_sha256"
        ] != file_sha256(plot_path):
            raise LOEOARProposalError("LOEO AR proposal or plot hash differs")
        try:
            proposal_artifact = ar_module._read_artifact(proposal_path)
        except (ArtifactMismatch, ValueError) as error:
            raise LOEOARProposalError("LOEO AR fold proposal is invalid") from error
        if (
            proposal_artifact.get("approved") is not False
            or proposal_artifact.get("proposal_digest") != entry["proposal_digest"]
            or proposal_artifact.get("hashes", {}).get("residual_sha256") != fold.residual_sha256
            or proposal_artifact.get("hashes", {}).get("config_sha256") != entry["config_sha256"]
            or proposal_artifact.get("hashes", {}).get("event_sha256") != entry["event_sha256"]
        ):
            raise LOEOARProposalError("LOEO AR proposal identity differs")
        summary_calibration = _calibration_from_summary(entry["calibration"])
        if summary_calibration.event_ids != expected_diagnostic_ids:
            raise LOEOARProposalError("LOEO AR calibration event population differs")
        artifact_calibration = ar_module._calibration_from_mapping(
            proposal_artifact.get("calibration")
        )
        if artifact_calibration != summary_calibration:
            raise LOEOARProposalError("LOEO AR calibration summary differs")
        artifact_context = ar_module._context_from_mapping(proposal_artifact.get("context"))
        if diagnostic_context is None:
            diagnostic_context = artifact_context
        elif diagnostic_context != artifact_context:
            raise LOEOARProposalError("LOEO AR fold contexts differ")
        if recompute:
            material = _materialize_fold(source, publication, held_out)
            if (
                material.diagnostic_context != artifact_context
                or material.config_sha256 != entry["config_sha256"]
                or material.event_sha256 != entry["event_sha256"]
                or _canonical(material.proposal_payload) != _canonical(proposal_artifact)
                or material.plot != plot_path.read_bytes()
                or _summary(material.calibration, material.diagnostics) != entry["calibration"]
            ):
                raise LOEOARProposalError("LOEO AR rehashed semantic artifact mutation detected")
        training_sha256[held_out] = fold.residual_sha256
        proposal_paths[held_out] = proposal_path
        proposal_sha256[held_out] = str(entry["proposal_sha256"])
        proposal_digests[held_out] = str(entry["proposal_digest"])
        plot_paths[held_out] = plot_path
        plot_sha256[held_out] = str(entry["plot_sha256"])
        calibrations[held_out] = artifact_calibration
    assert diagnostic_context is not None
    if payload["source"] != _source_payload(publication, diagnostic_context):
        raise LOEOARProposalError("LOEO AR source context differs")
    return LOEOARProposalSet(
        output_dir=generation.parent,
        generation_dir=generation,
        proposal_set_path=generation / "proposal-set.json",
        proposal_set_sha256=proposal_set_sha256,
        context=publication.context,
        diagnostic_context=diagnostic_context,
        source_context=_source_context(publication, diagnostic_context),
        occurrence_ids=occurrence_ids,
        training_sha256=training_sha256,
        proposal_paths=proposal_paths,
        proposal_sha256=proposal_sha256,
        proposal_digests=proposal_digests,
        plot_paths=plot_paths,
        plot_sha256=plot_sha256,
        calibrations=calibrations,
    )


def _root_entries(root: Path) -> dict[str, Path]:
    _require_directory(root, "LOEO AR publication directory")
    entries = {path.name: path for path in root.iterdir()}
    unknown = {
        name for name in entries if name not in _ROOT_NAMESPACE and not name.startswith(".staging-")
    }
    if unknown:
        raise LOEOARProposalError("LOEO AR publication directory contains unknown entries")
    return entries


@contextmanager
def _publication_lock(root: Path) -> Iterator[None]:
    lock_path = root / ".loeo-ar.lock"
    try:
        existing_mode = lock_path.lstat().st_mode
    except FileNotFoundError:
        pass
    except OSError as error:
        raise LOEOARProposalError("LOEO AR publication lock is unsafe") from error
    else:
        if stat.S_ISLNK(existing_mode) or not stat.S_ISREG(existing_mode):
            raise LOEOARProposalError("LOEO AR publication lock is unsafe")
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except OSError as error:
        raise LOEOARProposalError("LOEO AR publication lock is unsafe") from error
    locked = False
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise LOEOARProposalError("LOEO AR publication lock is unsafe")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        locked = True
        try:
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            locked = False
    finally:
        if locked:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _write_pointer(root: Path, proposal_set_sha256: str) -> None:
    payload = {"proposal_set_sha256": proposal_set_sha256, "schema_version": _SCHEMA_VERSION}
    temporary = root / ".current.tmp"
    if temporary.exists():
        existing = _read_json(temporary, "LOEO AR temporary pointer")
        if existing != payload:
            raise LOEOARProposalError("LOEO AR temporary pointer differs and is preserved")
        temporary.unlink()
    _write_json(temporary, payload)
    os.replace(temporary, root / "current.json")
    _fsync_directory(root)


def _validate_pointer(root: Path, proposal_set_sha256: str) -> None:
    pointer = _read_json(root / "current.json", "LOEO AR current pointer")
    if pointer != {
        "proposal_set_sha256": proposal_set_sha256,
        "schema_version": _SCHEMA_VERSION,
    }:
        raise LOEOARProposalError("LOEO AR current pointer differs")


def _load_prepared_internal(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    root: Path,
    *,
    recompute: bool,
    require_pointer: bool,
    allow_approval_staging: bool = False,
) -> LOEOARProposalSet:
    entries = _root_entries(root)
    staging = [path for name, path in entries.items() if name.startswith(".staging-")]
    if staging:
        raise LOEOARProposalError("LOEO AR publication contains interrupted staging state")
    allowed = {".loeo-ar.lock", "generation", "current.json"}
    if require_pointer and set(entries) != allowed:
        raise LOEOARProposalError("LOEO AR publication is partial")
    generation = root / "generation"
    prepared = _validate_generation(
        source,
        publication,
        generation,
        recompute=recompute,
        allow_approval_staging=allow_approval_staging,
    )
    if require_pointer:
        _validate_pointer(root, prepared.proposal_set_sha256)
    return prepared


def _build_generation(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    staging: Path,
    identity: dict[str, object],
) -> str:
    _require_directory(staging, "LOEO AR staging directory")
    _validate_stage_identity(staging, identity, "LOEO AR proposal stage")
    staging_entries = {path.name: path for path in staging.iterdir()}
    if not set(staging_entries).issubset(_GENERATION_NAMESPACE):
        raise LOEOARProposalError("LOEO AR staging evidence is invalid and preserved")
    for name in ("proposal-set.json", "COMPLETE"):
        if name in staging_entries:
            _require_file(staging_entries[name], f"LOEO AR staging {name}")
    proposals = staging / "proposals"
    plots = staging / "plots"
    for directory, label in ((proposals, "proposals"), (plots, "plots")):
        if directory.name in staging_entries:
            _require_directory(directory, f"LOEO AR staging {label}")
        else:
            directory.mkdir(mode=0o700)
    proposal_names = {f"{item}.json" for item in publication.occurrence_ids} | {
        _proposal_lock_name(item) for item in publication.occurrence_ids
    }
    plot_names = {f"{item}.svg" for item in publication.occurrence_ids}
    for directory, allowed, label in (
        (proposals, proposal_names, "proposal"),
        (plots, plot_names, "plot"),
    ):
        artifacts = {path.name: path for path in directory.iterdir()}
        if not set(artifacts).issubset(allowed):
            raise LOEOARProposalError("LOEO AR staging evidence is invalid and preserved")
        for artifact in artifacts.values():
            _require_file(artifact, f"LOEO AR staging {label} artifact")
    entries: list[dict[str, object]] = []
    diagnostic_context: EventResidualContext | None = None
    for held_out in publication.occurrence_ids:
        material = _materialize_fold(source, publication, held_out)
        if diagnostic_context is None:
            diagnostic_context = material.diagnostic_context
        elif diagnostic_context != material.diagnostic_context:
            raise LOEOARProposalError("LOEO fold diagnostic contexts differ")
        proposal_path = proposals / f"{held_out}.json"
        write_ar_diagnostics(
            proposal_path,
            material.diagnostics,
            material.calibration,
            residual_sha256=material.fold.residual_sha256,
            config_sha256=material.config_sha256,
            event_sha256=material.event_sha256,
            context=material.diagnostic_context,
        )
        plot_path = plots / f"{held_out}.svg"
        if plot_path.exists():
            _require_file(plot_path, f"LOEO AR staged plot {held_out}")
            if plot_path.read_bytes() != material.plot:
                raise LOEOARProposalError("LOEO AR staged plot differs and is preserved")
        else:
            with plot_path.open("xb") as destination:
                destination.write(material.plot)
                destination.flush()
                os.fsync(destination.fileno())
        entries.append(_proposal_entry(material, proposal_path, plot_path))
    assert diagnostic_context is not None
    _fsync_directory(proposals)
    _fsync_directory(plots)
    payload = _set_payload(publication, diagnostic_context, entries)
    proposal_set_path = staging / "proposal-set.json"
    if proposal_set_path.exists():
        if _read_json(proposal_set_path, "LOEO AR staged proposal set") != payload:
            raise LOEOARProposalError("LOEO AR staged proposal set differs and is preserved")
    else:
        _write_json(proposal_set_path, payload)
    complete_payload = {"proposal_set_sha256": payload["proposal_set_sha256"]}
    complete_path = staging / "COMPLETE"
    if complete_path.exists():
        if _read_json(complete_path, "LOEO AR staged completion marker") != complete_payload:
            raise LOEOARProposalError("LOEO AR staged completion differs and is preserved")
    else:
        _write_json(complete_path, complete_payload)
    _fsync_directory(staging)
    return str(payload["proposal_set_sha256"])


def prepare_loeo_ar_proposal_set(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    *,
    output_dir: Path,
) -> LOEOARProposalSet:
    """Publish all ten training-only fold proposals; this operation never approves."""

    publication = _validated_publication(source, publication)
    root = Path(output_dir)
    if root.exists():
        _require_directory(root, "LOEO AR publication directory")
    else:
        root.mkdir(parents=True, mode=0o700)
    with _publication_lock(root):
        identity = _proposal_stage_identity(publication)
        staging_path = _stage_path(root, ".staging-", identity)
        entries = _root_entries(root)
        staging = [path for name, path in entries.items() if name.startswith(".staging-")]
        if len(staging) > 1:
            raise LOEOARProposalError("LOEO AR publication has ambiguous staging evidence")
        if staging and staging[0] != staging_path:
            raise LOEOARProposalError("LOEO AR staging identity differs and is preserved")
        if "generation" in entries:
            if staging:
                raise LOEOARProposalError("LOEO AR staging evidence is ambiguous and preserved")
            prepared = _validate_generation(
                source, publication, root / "generation", recompute=True
            )
            if "current.json" in entries:
                _validate_pointer(root, prepared.proposal_set_sha256)
            else:
                _write_pointer(root, prepared.proposal_set_sha256)
            return _load_prepared_internal(
                source, publication, root, recompute=True, require_pointer=True
            )
        if set(entries) - {".loeo-ar.lock", ".current.tmp", staging_path.name}:
            raise LOEOARProposalError("LOEO AR publication is partial or incompatible")
        if staging:
            _require_directory(staging_path, "LOEO AR staging directory")
            _validate_stage_identity(staging_path, identity, "LOEO AR proposal stage")
        else:
            staging_path.mkdir(mode=0o700)
            _write_json(staging_path / _STAGE_FILENAME, identity)
            _fsync_directory(staging_path)
            _fsync_directory(root)
        proposal_set_sha256 = _build_generation(source, publication, staging_path, identity)
        _validate_generation(source, publication, staging_path, recompute=False)
        os.replace(staging_path, root / "generation")
        _fsync_directory(root)
        _write_pointer(root, proposal_set_sha256)
    return _load_prepared_internal(source, publication, root, recompute=True, require_pointer=True)


def _approved_fold_payload(
    prepared: LOEOARProposalSet, approved_dir: Path
) -> list[dict[str, object]]:
    folds = []
    for held_out in prepared.occurrence_ids:
        path = approved_dir / f"{held_out}.json"
        payload = ar_module._read_artifact(path)
        folds.append(
            {
                "approved_artifact_digest": payload["artifact_digest"],
                "approved_path": f"approval/{held_out}.json",
                "approved_sha256": file_sha256(path),
                "held_out_occurrence_id": held_out,
                "plot_sha256": prepared.plot_sha256[held_out],
                "proposal_digest": prepared.proposal_digests[held_out],
                "proposal_sha256": prepared.proposal_sha256[held_out],
                "training_sha256": prepared.training_sha256[held_out],
            }
        )
    return folds


def _approved_set_payload(
    prepared: LOEOARProposalSet, folds: list[dict[str, object]]
) -> dict[str, object]:
    unsigned: dict[str, object] = {
        "approved": True,
        "folds": folds,
        "occurrence_ids": list(prepared.occurrence_ids),
        "proposal_set_sha256": prepared.proposal_set_sha256,
        "schema_version": _APPROVED_SCHEMA_VERSION,
    }
    return {**unsigned, "approved_set_sha256": _sha_json(unsigned)}


def _approved_set_content(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if key != "approved_set_sha256"}


def _validate_approval_namespace(directory: Path, occurrence_ids: tuple[str, ...]) -> None:
    _require_directory(directory, "LOEO AR approval namespace")
    expected = (
        _APPROVAL_NAMESPACE
        | {f"{item}.json" for item in occurrence_ids}
        | {_proposal_lock_name(item) for item in occurrence_ids}
    )
    entries = {path.name: path for path in directory.iterdir()}
    if set(entries) != expected:
        raise LOEOARProposalError("LOEO AR approval namespace differs")
    for name, path in entries.items():
        _require_file(path, f"LOEO AR approval artifact {name}")


def _load_approved_internal(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    root: Path,
    *,
    recompute: bool,
) -> ApprovedLOEOARSet:
    prepared = _load_prepared_internal(
        source, publication, root, recompute=recompute, require_pointer=True
    )
    approval = prepared.generation_dir / "approval"
    _validate_approval_namespace(approval, prepared.occurrence_ids)
    _validate_stage_identity(approval, _approval_stage_identity(prepared), "LOEO AR approval stage")
    payload = _read_json(approval / "approved-set.json", "LOEO AR approved set")
    complete = _read_json(approval / "COMPLETE", "LOEO AR approval completion marker")
    required = {
        "approved",
        "approved_set_sha256",
        "folds",
        "occurrence_ids",
        "proposal_set_sha256",
        "schema_version",
    }
    approved_set_sha256 = payload.get("approved_set_sha256")
    if (
        set(payload) != required
        or payload.get("approved") is not True
        or payload.get("schema_version") != _APPROVED_SCHEMA_VERSION
        or payload.get("proposal_set_sha256") != prepared.proposal_set_sha256
        or payload.get("occurrence_ids") != list(prepared.occurrence_ids)
        or not isinstance(payload.get("folds"), list)
        or approved_set_sha256 != _sha_json(_approved_set_content(payload))
        or complete != {"approved_set_sha256": approved_set_sha256}
    ):
        raise LOEOARProposalError("LOEO AR approved-set identity differs")
    folds = payload["folds"]
    expected_fold_keys = {
        "approved_artifact_digest",
        "approved_path",
        "approved_sha256",
        "held_out_occurrence_id",
        "plot_sha256",
        "proposal_digest",
        "proposal_sha256",
        "training_sha256",
    }
    if (
        len(folds) != 10
        or not all(isinstance(entry, dict) and set(entry) == expected_fold_keys for entry in folds)
        or tuple(entry["held_out_occurrence_id"] for entry in folds) != prepared.occurrence_ids
    ):
        raise LOEOARProposalError("LOEO AR approved fold population differs")
    calibrations: dict[str, ApprovedARCalibration] = {}
    approved_paths: dict[str, Path] = {}
    approved_sha256: dict[str, str] = {}
    for entry in folds:
        held_out = str(entry["held_out_occurrence_id"])
        approved_path = approval / f"{held_out}.json"
        if entry["approved_path"] != f"approval/{held_out}.json":
            raise LOEOARProposalError("LOEO AR approved canonical path differs")
        proposal_payload = ar_module._read_artifact(prepared.proposal_paths[held_out])
        hashes = proposal_payload["hashes"]
        try:
            calibration = load_approved_calibration(
                approved_path,
                current_residual_sha256=prepared.training_sha256[held_out],
                current_config_sha256=str(hashes["config_sha256"]),
                current_event_sha256=str(hashes["event_sha256"]),
            )
        except (ArtifactMismatch, ValueError) as error:
            raise LOEOARProposalError("LOEO AR approved fold artifact is invalid") from error
        if (
            entry["approved_sha256"] != file_sha256(approved_path)
            or entry["approved_artifact_digest"] != calibration.artifact_digest
            or entry["training_sha256"] != prepared.training_sha256[held_out]
            or entry["proposal_sha256"] != prepared.proposal_sha256[held_out]
            or entry["proposal_digest"] != prepared.proposal_digests[held_out]
            or entry["plot_sha256"] != prepared.plot_sha256[held_out]
            or calibration.calibration != prepared.calibrations[held_out]
            or calibration.context != prepared.diagnostic_context
        ):
            raise LOEOARProposalError("LOEO AR approved fold binding differs")
        calibrations[held_out] = calibration
        approved_paths[held_out] = approved_path
        approved_sha256[held_out] = str(entry["approved_sha256"])
    expected_payload = _approved_set_payload(prepared, _approved_fold_payload(prepared, approval))
    if expected_payload != payload:
        raise LOEOARProposalError("LOEO AR approved set was rehashed or substituted")
    return ApprovedLOEOARSet(
        output_dir=root,
        approved_set_path=approval / "approved-set.json",
        proposal_set_sha256=prepared.proposal_set_sha256,
        approved_set_sha256=str(approved_set_sha256),
        context=prepared.context,
        source_context=prepared.source_context,
        occurrence_ids=prepared.occurrence_ids,
        training_sha256=prepared.training_sha256,
        proposal_sha256=prepared.proposal_sha256,
        proposal_digests=prepared.proposal_digests,
        plot_sha256=prepared.plot_sha256,
        approved_paths=approved_paths,
        approved_sha256=approved_sha256,
        calibrations=calibrations,
        _source=source,
        _publication=publication,
        _token=_SET_TOKEN,
    )


def _build_approval(
    prepared: LOEOARProposalSet,
    staging: Path,
    identity: dict[str, object],
) -> None:
    _require_directory(staging, "LOEO AR approval staging")
    _validate_stage_identity(staging, identity, "LOEO AR approval stage")
    entries = {path.name: path for path in staging.iterdir()}
    allowed = (
        _APPROVAL_NAMESPACE
        | {f"{item}.json" for item in prepared.occurrence_ids}
        | {_proposal_lock_name(item) for item in prepared.occurrence_ids}
    )
    if not set(entries).issubset(allowed):
        raise LOEOARProposalError("LOEO AR approval staging evidence is invalid and preserved")
    for child in entries.values():
        _require_file(child, "LOEO AR approval staging artifact")

    set_entries = _read_json(prepared.proposal_set_path, "LOEO AR proposal set")["folds"]
    by_held = {entry["held_out_occurrence_id"]: entry for entry in set_entries}
    for held_out in prepared.occurrence_ids:
        entry = by_held[held_out]
        approve_calibration(
            prepared.proposal_paths[held_out],
            staging / f"{held_out}.json",
            current_residual_sha256=prepared.training_sha256[held_out],
            current_config_sha256=str(entry["config_sha256"]),
            current_event_sha256=str(entry["event_sha256"]),
        )
    approved_payload = _approved_set_payload(prepared, _approved_fold_payload(prepared, staging))
    approved_set_path = staging / "approved-set.json"
    if approved_set_path.exists():
        if _read_json(approved_set_path, "LOEO AR staged approved set") != approved_payload:
            raise LOEOARProposalError("LOEO AR staged approved set differs and is preserved")
    else:
        _write_json(approved_set_path, approved_payload)
    complete_payload = {"approved_set_sha256": approved_payload["approved_set_sha256"]}
    complete_path = staging / "COMPLETE"
    if complete_path.exists():
        if _read_json(complete_path, "LOEO AR staged approval completion") != complete_payload:
            raise LOEOARProposalError("LOEO AR staged approval differs and is preserved")
    else:
        _write_json(complete_path, complete_payload)
    _fsync_directory(staging)


def approve_loeo_ar_proposal_set(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    *,
    output_dir: Path,
    confirm_proposal_set_sha256: str,
) -> Path:
    """Freeze ten existing proposals only after an exact set-digest confirmation."""

    publication = _validated_publication(source, publication)
    if (
        not isinstance(confirm_proposal_set_sha256, str)
        or _DIGEST_PATTERN.fullmatch(confirm_proposal_set_sha256) is None
    ):
        raise LOEOARProposalError("proposal-set confirmation must be a lowercase SHA-256")
    root = Path(output_dir)
    _require_directory(root, "LOEO AR publication directory")
    with _publication_lock(root):
        prepared = _load_prepared_internal(
            source,
            publication,
            root,
            recompute=False,
            require_pointer=True,
            allow_approval_staging=True,
        )
        if confirm_proposal_set_sha256 != prepared.proposal_set_sha256:
            raise LOEOARProposalError("proposal-set confirmation differs")
        generation = prepared.generation_dir
        approval = generation / "approval"
        generation_entries = {path.name: path for path in generation.iterdir()}
        staging_entries = [
            path
            for name, path in generation_entries.items()
            if name.startswith(".approval-staging-")
        ]
        if "approval" in generation_entries:
            approved = _load_approved_internal(source, publication, root, recompute=False)
            return approved.approved_set_path
        identity = _approval_stage_identity(prepared)
        staging = _stage_path(generation, ".approval-staging-", identity)
        if staging_entries and staging_entries[0] != staging:
            raise LOEOARProposalError("LOEO AR approval staging identity differs and is preserved")
        if staging_entries:
            _validate_stage_identity(staging, identity, "LOEO AR approval stage")
        else:
            staging.mkdir(mode=0o700)
            _write_json(staging / _STAGE_FILENAME, identity)
            _fsync_directory(staging)
            _fsync_directory(generation)
        _build_approval(prepared, staging, identity)
        _validate_approval_namespace(staging, prepared.occurrence_ids)
        _validate_stage_identity(staging, identity, "LOEO AR approval stage")
        os.replace(staging, approval)
        _fsync_directory(generation)
    approved = _load_approved_internal(source, publication, root, recompute=False)
    return approved.approved_set_path


def load_approved_loeo_ar_set(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    *,
    output_dir: Path,
) -> ApprovedLOEOARSet:
    """Revalidate and load only a complete exact-context ten-fold approval set."""

    publication = _validated_publication(source, publication)
    root = Path(output_dir)
    _require_directory(root, "LOEO AR publication directory")
    return _load_approved_internal(source, publication, root, recompute=True)


__all__ = [
    "ApprovedLOEOARSet",
    "LOEOARProposalError",
    "LOEOARProposalSet",
    "LOEOARSourceContext",
    "approve_loeo_ar_proposal_set",
    "load_approved_loeo_ar_set",
    "prepare_loeo_ar_proposal_set",
]
