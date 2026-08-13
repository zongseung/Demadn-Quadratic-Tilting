"""Immutable identity, checkpoint, and publication state for one LOEO fold."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
import uuid
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import arviz as az
import numpy as np
import polars as pl

from hqrc_v3._loeo_contract import MODEL_OPTIONS, canonical_json, derive_loeo_seed, sha_json
from hqrc_v3._loeo_posterior import validate_h3_posterior
from hqrc_v3._loeo_products import generate_loeo_fold_products
from hqrc_v3._loeo_types import (
    LOEOFoldError,
    LOEOFoldInputs,
    LOEOFoldMaterial,
    LOEOFoldProducts,
    LOEOFoldResult,
)
from hqrc_v3.bayes.artifacts import load_hqrc_data, write_hqrc_data
from hqrc_v3.bayes.model import CYCLIC_HOUR_PARAMETERIZATION, HQRCData
from hqrc_v3.bayes.samplers import (
    PYMC_INITIALIZATION,
    SAMPLER_GEOMETRY,
    SamplingDiagnostics,
    validate_inference_data,
)
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.provenance import file_sha256

PRODUCT_FILES = {
    "hourly_predictions": "hourly_predictions.parquet",
    "metrics": "metrics.parquet",
    "posterior_summary": "posterior_summary.json",
}
CHECKPOINT_TOP = {
    "hqrc_data.current.json",
    ".hqrc_data.generations",
    "posterior.nc",
    "posterior.checkpoint.json",
}
INPUT_CHECKPOINT_TOP = {
    "hqrc_data.current.json",
    ".hqrc_data.generations",
}
DOWNSTREAM_TOP = {*PRODUCT_FILES.values(), "manifest.json"}
COMPLETE_TOP = {
    ".loeo-fold.lock",
    *CHECKPOINT_TOP,
    *DOWNSTREAM_TOP,
    "COMPLETE",
}


def same_hqrc_data(left: HQRCData, right: HQRCData) -> bool:
    return left.occurrence_ids == right.occurrence_ids and all(
        np.array_equal(getattr(left, name), getattr(right, name))
        for name in (
            "observations",
            "occurrence_index",
            "holiday_type_index",
            "tau_days",
            "hour",
            "restriction",
        )
    )


def sampler_contract(
    profile: str,
    *,
    root_seed: int,
    held_out_occurrence_id: str,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    cores: int | None = None,
    init: str | None = None,
    target_accept: float | None = None,
) -> dict[str, object]:
    derived_seed = derive_loeo_seed(root_seed, f"H3-fold-sampler:{held_out_occurrence_id}")
    if profile == "paper":
        resolved = (
            1_000 if draws is None else draws,
            1_000 if tune is None else tune,
            4 if chains is None else chains,
        )
        if resolved[0] < 1_000 or resolved[1] < 1_000 or resolved[2] != 4:
            raise LOEOFoldError("paper profile requires 4 chains and at least 1000 tune/draws")
    elif profile == "smoke":
        if draws is None or tune is None or chains is None:
            raise LOEOFoldError("smoke profile requires explicit draws, tune, and chains")
        resolved = (draws, tune, chains)
    else:
        raise LOEOFoldError("profile must be paper or smoke")
    resolved_cores = 1 if cores is None else cores
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolved
    ):
        raise LOEOFoldError("draws, tune, and chains must be positive integers")
    if (
        isinstance(resolved_cores, bool)
        or not isinstance(resolved_cores, int)
        or resolved_cores <= 0
    ):
        raise LOEOFoldError("cores must be a positive integer")
    if resolved_cores > resolved[2]:
        raise LOEOFoldError("cores must not exceed chains")
    resolved_init = PYMC_INITIALIZATION if init is None else init
    if resolved_init not in {"adapt_diag", "jitter+adapt_diag"}:
        raise LOEOFoldError("LOEO sampler init is not approved")
    resolved_target_accept = 0.99 if profile == "paper" else 0.9
    if target_accept is not None and (
        isinstance(target_accept, bool)
        or not isinstance(target_accept, (int, float))
        or not math.isfinite(float(target_accept))
        or float(target_accept) != resolved_target_accept
    ):
        raise LOEOFoldError("LOEO target_accept differs from the approved profile setting")
    return {
        "backend": "pymc",
        "root_seed": root_seed,
        "seed": derived_seed,
        "draws": resolved[0],
        "tune": resolved[1],
        "chains": resolved[2],
        "cores": resolved_cores,
        "target_accept": resolved_target_accept,
        "profile": profile,
        "init": resolved_init,
        "geometry": SAMPLER_GEOMETRY,
    }


def _context_payload(context: object) -> dict[str, object]:
    return {
        "model": context.model,
        "feature_set": context.feature_set,
        "seed": context.seed,
        "split_ids": list(context.split_ids),
    }


def _source_payload(source: ValidatedCorrectionSource) -> dict[str, Any]:
    return {
        "run_dir": source.run_dir.resolve().as_posix(),
        "source_profile": source.source_profile,
        "source_paths": {
            key: path.resolve().as_posix() for key, path in sorted(source.source_paths.items())
        },
        "source_hashes": dict(sorted(source.source_hashes.items())),
        "artifacts": {
            "residual": {
                "path": source.residual_path.resolve().as_posix(),
                "sha256": source.residual_sha256,
            },
            "oof_members": {
                "path": source.oof_members_path.resolve().as_posix(),
                "sha256": source.oof_members_sha256,
            },
            "oof_point": {
                "path": source.oof_point_path.resolve().as_posix(),
                "sha256": source.oof_point_sha256,
            },
            "final_members": {
                "path": source.final_members_path.resolve().as_posix(),
                "sha256": source.final_members_sha256,
            },
            "final_point": {
                "path": source.final_point_path.resolve().as_posix(),
                "sha256": source.final_point_sha256,
            },
        },
        "residual_manifest_json_sha256": hashlib.sha256(
            canonical_json(source.residual_manifest)
        ).hexdigest(),
        "baseline_manifest_json_sha256": hashlib.sha256(
            canonical_json(source.baseline_manifest)
        ).hexdigest(),
        "registry": [
            {
                "occurrence_id": event.occurrence_id,
                "holiday_type": event.holiday_type,
                "central_date": event.central_date.isoformat(),
                "official_start": event.official_start.isoformat(),
                "official_end": event.official_end.isoformat(),
                "restriction": event.restriction,
            }
            for event in source.events
        ],
    }


def input_identity(inputs: LOEOFoldInputs, sampler: Mapping[str, object]) -> dict[str, Any]:
    held = inputs.held_out_occurrence_id
    return {
        "schema_version": 1,
        "evaluation": "retrospective-loeo-one-fold",
        "causal": False,
        "source": _source_payload(inputs.source),
        "context": _context_payload(inputs.publication.context),
        "loeo": {
            "output_dir": inputs.publication.output_dir.resolve().as_posix(),
            "generation_dir": inputs.publication.generation_dir.resolve().as_posix(),
            "manifest_path": inputs.publication.manifest_path.resolve().as_posix(),
            "manifest_file_sha256": file_sha256(inputs.publication.manifest_path),
            "universe_path": inputs.publication.universe_path.resolve().as_posix(),
            "universe_sha256": inputs.publication.universe_sha256,
            "held_out_occurrence_id": held,
            "training_occurrence_ids": list(inputs.training_fold.occurrence_ids),
            "fold_path": inputs.training_fold.path.resolve().as_posix(),
            "fold_sha256": inputs.training_fold.residual_sha256,
        },
        "approved_loeo_ar": {
            "output_dir": inputs.approved_set.output_dir.resolve().as_posix(),
            "approved_set_path": inputs.approved_set.approved_set_path.resolve().as_posix(),
            "proposal_set_sha256": inputs.approved_set.proposal_set_sha256,
            "approved_set_sha256": inputs.approved_set.approved_set_sha256,
            "fold_approved_path": inputs.approved_set.approved_paths[held].resolve().as_posix(),
            "fold_approved_sha256": inputs.approved_set.approved_sha256[held],
            "fold_approved_artifact_digest": inputs.approved.artifact_digest,
            "fold_proposal_sha256": inputs.approved_set.proposal_sha256[held],
            "fold_proposal_digest": inputs.approved_set.proposal_digests[held],
        },
        "evaluation_scale": {
            "evaluation_split_id": inputs.evaluation_split_id,
            "scale_source_split_id": inputs.scale_source_split_id,
            "sigma_n_mw": inputs.sigma_eval,
        },
        "model": {
            "variant": "H3",
            "pooling": "partial",
            "options": asdict(MODEL_OPTIONS),
            "cyclic_hour_parameterization": CYCLIC_HOUR_PARAMETERIZATION,
        },
        "sampler": dict(sampler),
        "predictive_seed": derive_loeo_seed(
            int(sampler["root_seed"]), f"H3-fold-predictive:{held}"
        ),
        "coverage": {
            "training_rows": int(inputs.hqrc_data.observations.size),
            "held_out_rows": inputs.held_out.frame.height,
        },
    }


def _posterior_metadata_matches(
    idata: object, inputs: LOEOFoldInputs, sampler: Mapping[str, object]
) -> None:
    posterior = validate_h3_posterior(idata, inputs)
    if (
        int(posterior.sizes["chain"]) != sampler["chains"]
        or int(posterior.sizes["draw"]) != sampler["draws"]
    ):
        raise LOEOFoldError("LOEO posterior sample dimensions differ from sampler contract")
    try:
        calibration = json.loads(idata.attrs["hqrc_calibration_json"])
        model = json.loads(idata.attrs["hqrc_model_json"])
        recorded_sampler = json.loads(idata.attrs["hqrc_sampler_json"])
    except (AttributeError, KeyError, TypeError, json.JSONDecodeError) as error:
        raise LOEOFoldError("LOEO posterior provenance metadata is missing") from error
    expected_calibration = {
        "artifact_path": str(inputs.approved.artifact_path),
        "artifact_digest": inputs.approved.artifact_digest,
        "residual_sha256": inputs.approved.residual_sha256,
        "config_sha256": inputs.approved.config_sha256,
        "event_sha256": inputs.approved.event_sha256,
        "context": _context_payload(inputs.approved.context),
        "a": inputs.approved.a,
        "b": inputs.approved.b,
    }
    if calibration != expected_calibration:
        raise LOEOFoldError("LOEO posterior approved fold context differs")
    if model != {
        "variant": "H3",
        "pooling": "partial",
        "options": asdict(MODEL_OPTIONS),
        "cyclic_hour_parameterization": CYCLIC_HOUR_PARAMETERIZATION,
    }:
        raise LOEOFoldError("LOEO posterior H3 model contract differs")
    if idata.attrs.get("hqrc_backend") != sampler["backend"]:
        raise LOEOFoldError("LOEO posterior sampler backend differs")
    expected_sampler = {
        "draws": sampler["draws"],
        "tune": sampler["tune"],
        "chains": sampler["chains"],
        "cores": sampler["cores"],
        "seed": sampler["seed"],
        "target_accept": sampler["target_accept"],
        "paper_profile": sampler["profile"] == "paper",
        "init": sampler["init"],
        "geometry": sampler["geometry"],
    }
    if recorded_sampler != expected_sampler:
        raise LOEOFoldError("LOEO posterior sampler contract differs")
    if idata.attrs.get("hqrc_causal") != "false":
        raise LOEOFoldError("LOEO posterior causal label differs")


def _fsync_file(path: Path) -> None:
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def _safe_name(name: str) -> str:
    if not name or name in {".", ".."} or "/" in name:
        raise LOEOFoldError("LOEO publication filename is unsafe")
    return name


def _relative_file_fd(directory_fd: int, name: str) -> int:
    try:
        descriptor = os.open(
            _safe_name(name),
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=directory_fd,
        )
        identity = os.fstat(descriptor)
    except OSError as error:
        raise LOEOFoldError("LOEO publication file is missing or unsafe") from error
    if not stat.S_ISREG(identity.st_mode):
        os.close(descriptor)
        raise LOEOFoldError("LOEO publication file is missing or unsafe")
    return descriptor


def _relative_bytes(directory_fd: int, name: str) -> bytes:
    descriptor = _relative_file_fd(directory_fd, name)
    try:
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            return stream.read()
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _relative_sha256(directory_fd: int, name: str) -> str:
    return hashlib.sha256(_relative_bytes(directory_fd, name)).hexdigest()


def publication_entries(publication: LOEOPublicationHandle) -> frozenset[str]:
    try:
        return frozenset(os.listdir(publication.directory_fd))
    except OSError as error:
        raise LOEOFoldError("LOEO publication directory is unreadable") from error


def publication_has(publication: LOEOPublicationHandle, name: str) -> bool:
    try:
        identity = os.stat(_safe_name(name), dir_fd=publication.directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    except OSError as error:
        raise LOEOFoldError("LOEO publication entry is unsafe") from error
    return stat.S_ISREG(identity.st_mode)


def relative_sha256(publication: LOEOPublicationHandle, name: str) -> str:
    return _relative_sha256(publication.directory_fd, name)


def _materialize_relative(directory_fd: int, name: str, destination: Path) -> None:
    descriptor = _relative_file_fd(directory_fd, name)
    try:
        with os.fdopen(descriptor, "rb") as source, destination.open("xb") as output:
            descriptor = -1
            for block in iter(lambda: source.read(1024 * 1024), b""):
                output.write(block)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _relative_json(directory_fd: int, name: str, description: str) -> dict[str, Any]:
    raw = _relative_bytes(directory_fd, name)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise LOEOFoldError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise LOEOFoldError(f"{description} is not canonical JSON")
    return value


def _publish_bytes(
    publication: LOEOPublicationHandle,
    name: str,
    payload: bytes,
    *,
    boundary: str,
    directory_fd: int | None = None,
) -> tuple[int, int]:
    target_fd = publication.directory_fd if directory_fd is None else directory_fd
    temporary = f".{_safe_name(name)}.{uuid.uuid4().hex}.tmp"
    descriptor: int | None = None
    identity: os.stat_result | None = None
    temporary_unlinked = False
    try:
        descriptor = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
            dir_fd=target_fd,
        )
        identity = os.fstat(descriptor)
        if not stat.S_ISREG(identity.st_mode):
            raise LOEOFoldError("LOEO publication temporary file is unsafe")
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("LOEO publication temporary write made no progress")
            remaining = remaining[written:]
        os.fsync(descriptor)
        publication_boundary(boundary)
        os.link(
            temporary,
            name,
            src_dir_fd=target_fd,
            dst_dir_fd=target_fd,
            follow_symlinks=False,
        )
        temporary_identity = os.stat(temporary, dir_fd=target_fd, follow_symlinks=False)
        target_descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            dir_fd=target_fd,
        )
        try:
            target_identity = os.fstat(target_descriptor)
        finally:
            os.close(target_descriptor)
        expected = (identity.st_dev, identity.st_ino)
        if (
            (temporary_identity.st_dev, temporary_identity.st_ino) != expected
            or (target_identity.st_dev, target_identity.st_ino) != expected
            or not stat.S_ISREG(target_identity.st_mode)
        ):
            raise LOEOFoldError("LOEO publication link identity changed")
        os.unlink(temporary, dir_fd=target_fd)
        temporary_unlinked = True
        os.fsync(target_fd)
        return expected
    except OSError as error:
        raise LOEOFoldError("LOEO publication failed safely") from error
    finally:
        if descriptor is not None and identity is not None and not temporary_unlinked:
            try:
                descriptor_identity = os.fstat(descriptor)
                entry_identity = os.stat(temporary, dir_fd=target_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                expected = (identity.st_dev, identity.st_ino)
                if (
                    stat.S_ISREG(descriptor_identity.st_mode)
                    and (descriptor_identity.st_dev, descriptor_identity.st_ino) == expected
                    and (entry_identity.st_dev, entry_identity.st_ino) == expected
                ):
                    os.unlink(temporary, dir_fd=target_fd)
                    os.fsync(target_fd)
        if descriptor is not None:
            os.close(descriptor)


def publish_json(
    publication: LOEOPublicationHandle,
    name: str,
    value: Mapping[str, Any],
    *,
    boundary: str,
) -> tuple[int, int]:
    return _publish_bytes(
        publication,
        name,
        canonical_json(dict(value)) + b"\n",
        boundary=boundary,
    )


@contextmanager
def _external_temporary(suffix: str):
    with tempfile.TemporaryDirectory(prefix="hqrc-v3-task15d-", dir="/private/tmp") as root:
        yield Path(root) / f"serialization{suffix}"


def _copy_external_file(
    publication: LOEOPublicationHandle, source: Path, name: str, *, boundary: str
) -> None:
    with source.open("rb") as stream:
        payload = stream.read()
    _publish_bytes(publication, name, payload, boundary=boundary)


def _open_or_create_child(publication: LOEOPublicationHandle, name: str) -> int:
    name = _safe_name(name)
    try:
        os.mkdir(name, 0o700, dir_fd=publication.directory_fd)
    except FileExistsError:
        pass
    except OSError as error:
        raise LOEOFoldError("LOEO child publication namespace is unsafe") from error
    return _open_directory(
        Path(name), "LOEO child publication namespace", dir_fd=publication.directory_fd
    )


def write_hqrc_checkpoint(
    publication: LOEOPublicationHandle,
    data: HQRCData,
    *,
    settings: dict[str, object],
) -> None:
    """Serialize HQRCData externally, then publish one held-dirfd generation."""

    with tempfile.TemporaryDirectory(prefix="hqrc-v3-task15d-hqrc-", dir="/private/tmp") as root:
        external = Path(root)
        source_npz, source_metadata = write_hqrc_data(
            external / "hqrc_data.npz", data, settings=settings
        )
        source_pointer = json.loads((external / "hqrc_data.current.json").read_bytes())
        metadata = json.loads(source_metadata.read_bytes())
        generation = uuid.uuid4().hex
        npz_name = f"{generation}.npz"
        metadata_name = f"{generation}.json"
        npz_payload = source_npz.read_bytes()
        metadata["generation"] = generation
        unsigned_metadata = {
            key: value for key, value in metadata.items() if key != "artifact_sha256"
        }
        metadata["artifact_sha256"] = sha_json(unsigned_metadata)
        metadata_payload = canonical_json(metadata) + b"\n"
        pointer = {
            **source_pointer,
            "generation": generation,
            "npz": f".hqrc_data.generations/{npz_name}",
            "metadata": f".hqrc_data.generations/{metadata_name}",
            "npz_sha256": hashlib.sha256(npz_payload).hexdigest(),
            "metadata_sha256": hashlib.sha256(metadata_payload).hexdigest(),
        }
        unsigned_pointer = {key: value for key, value in pointer.items() if key != "pointer_digest"}
        pointer["pointer_digest"] = sha_json(unsigned_pointer)
        child_fd = _open_or_create_child(publication, ".hqrc_data.generations")
        try:
            _publish_bytes(
                publication,
                npz_name,
                npz_payload,
                boundary="hqrc-generation-npz-prewrite",
                directory_fd=child_fd,
            )
            _publish_bytes(
                publication,
                metadata_name,
                metadata_payload,
                boundary="hqrc-generation-metadata-prewrite",
                directory_fd=child_fd,
            )
        finally:
            os.close(child_fd)
        publish_json(
            publication,
            "hqrc_data.current.json",
            pointer,
            boundary="hqrc-data-prewrite",
        )


def require_real_directory(path: Path, description: str) -> None:
    try:
        identity = path.lstat()
    except OSError as error:
        raise LOEOFoldError(f"{description} is missing or unsafe") from error
    if not stat.S_ISDIR(identity.st_mode):
        raise LOEOFoldError(f"{description} is missing or unsafe")


def _directory_flags() -> int:
    return os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)


def _open_directory(path: Path, description: str, *, dir_fd: int | None = None) -> int:
    try:
        descriptor = os.open(
            path if dir_fd is None else path.name, _directory_flags(), dir_fd=dir_fd
        )
        identity = os.fstat(descriptor)
    except OSError as error:
        raise LOEOFoldError(f"{description} is missing or unsafe") from error
    if not stat.S_ISDIR(identity.st_mode):
        os.close(descriptor)
        raise LOEOFoldError(f"{description} is missing or unsafe")
    return descriptor


def _ensure_directory_component(parent_fd: int, name: str, description: str) -> int:
    if not name or name in {".", ".."} or "/" in name:
        raise LOEOFoldError(f"{description} is unsafe")
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    except OSError as error:
        raise LOEOFoldError(f"{description} is unsafe") from error
    return _open_directory(Path(name), description, dir_fd=parent_fd)


@dataclass(frozen=True, slots=True)
class LOEONamespace:
    """Canonical trusted-root-relative identity for one fold directory."""

    root: Path
    components: tuple[str, ...]
    path: Path
    root_identity: tuple[int, int]
    final_identity: tuple[int, int]

    def __fspath__(self) -> str:
        return str(self.path)


@dataclass(frozen=True, slots=True)
class LOEOPublicationHandle:
    """A locked final-directory descriptor plus its canonical path identity."""

    namespace: LOEONamespace
    directory_fd: int

    @property
    def path(self) -> Path:
        return self.namespace.path


def _identity(descriptor: int, description: str) -> tuple[int, int]:
    try:
        current = os.fstat(descriptor)
    except OSError as error:
        raise LOEOFoldError(f"{description} descriptor is unavailable") from error
    if not stat.S_ISDIR(current.st_mode):
        raise LOEOFoldError(f"{description} is unsafe")
    return current.st_dev, current.st_ino


def _open_namespace_final(namespace: LOEONamespace) -> int:
    descriptors: list[int] = []
    try:
        descriptors.append(_open_directory(namespace.root, "LOEO fold trusted output root"))
        if _identity(descriptors[0], "LOEO fold trusted output root") != namespace.root_identity:
            raise LOEOFoldError("LOEO fold trusted output root changed")
        for component in namespace.components:
            descriptors.append(
                _open_directory(
                    Path(component),
                    "LOEO fold namespace component",
                    dir_fd=descriptors[-1],
                )
            )
        final = descriptors[-1]
        if _identity(final, "LOEO fold result directory") != namespace.final_identity:
            raise LOEOFoldError("LOEO fold namespace identity changed")
        for descriptor in descriptors[:-1]:
            os.close(descriptor)
        return final
    except Exception:
        for descriptor in descriptors:
            try:
                os.close(descriptor)
            except OSError:
                pass
        raise


def _namespace_for_existing_directory(directory: Path) -> LOEONamespace:
    path = Path(directory)
    if not path.is_absolute() or path.parent == path:
        raise LOEOFoldError("LOEO fold result directory requires an absolute trusted parent")
    require_real_directory(path.parent, "LOEO fold trusted output root")
    root_fd = _open_directory(path.parent, "LOEO fold trusted output root")
    final_fd: int | None = None
    try:
        final_fd = _open_directory(Path(path.name), "LOEO fold result directory", dir_fd=root_fd)
        return LOEONamespace(
            root=path.parent,
            components=(path.name,),
            path=path,
            root_identity=_identity(root_fd, "LOEO fold trusted output root"),
            final_identity=_identity(final_fd, "LOEO fold result directory"),
        )
    finally:
        if final_fd is not None:
            os.close(final_fd)
        os.close(root_fd)


def guard_namespace(publication: LOEOPublicationHandle) -> None:
    """Re-walk the canonical no-follow chain and match the held final inode."""

    if not isinstance(publication, LOEOPublicationHandle):
        raise TypeError("publication must be an LOEOPublicationHandle")
    held_identity = _identity(publication.directory_fd, "LOEO fold result directory")
    if held_identity != publication.namespace.final_identity:
        raise LOEOFoldError("LOEO fold held namespace identity changed")
    current_fd = _open_namespace_final(publication.namespace)
    try:
        if _identity(current_fd, "LOEO fold result directory") != held_identity:
            raise LOEOFoldError("LOEO fold namespace identity changed")
    finally:
        os.close(current_fd)


@contextmanager
def fold_lock(directory: LOEONamespace | Path, *, lock_name: str = ".loeo-fold.lock"):
    namespace_value = (
        directory
        if isinstance(directory, LOEONamespace)
        else _namespace_for_existing_directory(Path(directory))
    )
    directory_fd = _open_namespace_final(namespace_value)
    lock_fd: int | None = None
    locked = False
    try:
        flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            lock_fd = os.open(_safe_name(lock_name), flags, 0o600, dir_fd=directory_fd)
            identity = os.fstat(lock_fd)
        except OSError as error:
            raise LOEOFoldError("LOEO fold result lock is unsafe") from error
        if not stat.S_ISREG(identity.st_mode):
            raise LOEOFoldError("LOEO fold result lock is unsafe")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
        locked = True
        publication = LOEOPublicationHandle(namespace_value, directory_fd)
        guard_namespace(publication)
        yield publication
    finally:
        if lock_fd is not None:
            try:
                if locked:
                    fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)
        os.close(directory_fd)


def publication_boundary(_: str) -> None:
    """Failure-injection hook proving checkpoint and downstream recovery boundaries."""


def checked_publication_boundary(publication: LOEOPublicationHandle, name: str) -> None:
    """Run an injectable boundary and detect a namespace swap immediately."""

    guard_namespace(publication)
    publication_boundary(name)
    guard_namespace(publication)


def _diagnostic_payload(diagnostics: SamplingDiagnostics) -> dict[str, Any]:
    return {
        "max_rhat": diagnostics.max_rhat if math.isfinite(diagnostics.max_rhat) else None,
        "min_bulk_ess": (
            diagnostics.min_bulk_ess if math.isfinite(diagnostics.min_bulk_ess) else None
        ),
        "min_tail_ess": (
            diagnostics.min_tail_ess if math.isfinite(diagnostics.min_tail_ess) else None
        ),
        "divergences": diagnostics.divergences,
    }


def write_posterior_checkpoint(
    publication: LOEOPublicationHandle,
    idata: object,
    *,
    inputs: LOEOFoldInputs,
    sampler: Mapping[str, object],
    identity_sha256: str,
) -> None:
    _posterior_metadata_matches(idata, inputs, sampler)
    diagnostics = validate_inference_data(idata, paper_profile=sampler["profile"] == "paper")
    with _external_temporary(".nc") as temporary:
        az.to_netcdf(idata, temporary)
        _fsync_file(temporary)
        _copy_external_file(publication, temporary, "posterior.nc", boundary="posterior-prewrite")
    publish_json(
        publication,
        "posterior.checkpoint.json",
        {
            "schema_version": 1,
            "state": "POSTERIOR_COMPLETE",
            "causal": False,
            "identity_sha256": identity_sha256,
            "posterior_sha256": _relative_sha256(publication.directory_fd, "posterior.nc"),
            "diagnostics": _diagnostic_payload(diagnostics),
        },
        boundary="posterior-checkpoint-prewrite",
    )


def _load_posterior_checkpoint(
    publication: LOEOPublicationHandle,
    *,
    inputs: LOEOFoldInputs,
    sampler: Mapping[str, object],
    identity_sha256: str,
):
    checkpoint = _relative_json(
        publication.directory_fd, "posterior.checkpoint.json", "LOEO posterior checkpoint"
    )
    if (
        set(checkpoint)
        != {
            "schema_version",
            "state",
            "causal",
            "identity_sha256",
            "posterior_sha256",
            "diagnostics",
        }
        or checkpoint["schema_version"] != 1
        or checkpoint["state"] != "POSTERIOR_COMPLETE"
        or checkpoint["causal"] is not False
        or checkpoint["identity_sha256"] != identity_sha256
    ):
        raise LOEOFoldError("LOEO posterior checkpoint differs")
    if _relative_sha256(publication.directory_fd, "posterior.nc") != checkpoint["posterior_sha256"]:
        raise LOEOFoldError("LOEO posterior checkpoint hash differs")
    with _external_temporary(".nc") as posterior_path:
        _materialize_relative(publication.directory_fd, "posterior.nc", posterior_path)
        try:
            idata = az.from_netcdf(posterior_path).load()
        except (OSError, ValueError) as error:
            raise LOEOFoldError("LOEO posterior checkpoint is unreadable") from error
    _posterior_metadata_matches(idata, inputs, sampler)
    diagnostics = validate_inference_data(idata, paper_profile=sampler["profile"] == "paper")
    if checkpoint["diagnostics"] != _diagnostic_payload(diagnostics):
        raise LOEOFoldError("LOEO posterior checkpoint diagnostics differ")
    return idata


def _hqrc_generation_names(
    publication: LOEOPublicationHandle,
) -> tuple[dict[str, Any], str, str]:
    pointer = _relative_json(
        publication.directory_fd, "hqrc_data.current.json", "LOEO HQRCData pointer"
    )
    generation_fd = _open_directory(
        Path(".hqrc_data.generations"),
        "LOEO HQRCData generation namespace",
        dir_fd=publication.directory_fd,
    )
    names: list[str] = []
    for key in ("npz", "metadata"):
        relative = pointer.get(key)
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise LOEOFoldError("LOEO HQRCData pointer path is unsafe")
        path = Path(relative)
        if path.parent != Path(".hqrc_data.generations"):
            raise LOEOFoldError("LOEO HQRCData pointer path is unsafe")
        names.append(path.name)
    try:
        if names[0] == names[1] or set(os.listdir(generation_fd)) != set(names):
            raise LOEOFoldError("LOEO HQRCData generation namespace differs")
        for name in names:
            descriptor = _relative_file_fd(generation_fd, name)
            os.close(descriptor)
    finally:
        os.close(generation_fd)
    return pointer, names[0], names[1]


def _load_hqrc_checkpoint(publication: LOEOPublicationHandle):
    pointer, npz_name, metadata_name = _hqrc_generation_names(publication)
    generation_fd = _open_directory(
        Path(".hqrc_data.generations"),
        "LOEO HQRCData generation namespace",
        dir_fd=publication.directory_fd,
    )
    try:
        with tempfile.TemporaryDirectory(
            prefix="hqrc-v3-task15d-load-", dir="/private/tmp"
        ) as root:
            directory = Path(root)
            child = directory / ".hqrc_data.generations"
            child.mkdir()
            _materialize_relative(generation_fd, npz_name, child / npz_name)
            _materialize_relative(generation_fd, metadata_name, child / metadata_name)
            (directory / "hqrc_data.current.json").write_bytes(canonical_json(pointer) + b"\n")
            return load_hqrc_data(directory / "hqrc_data.npz")
    finally:
        os.close(generation_fd)


def _artifact_records(publication: LOEOPublicationHandle) -> dict[str, dict[str, str]]:
    _, data_npz, data_metadata = _hqrc_generation_names(publication)
    names = {
        "hqrc_data_pointer": (
            publication.directory_fd,
            "hqrc_data.current.json",
            "hqrc_data.current.json",
        ),
        "posterior": (publication.directory_fd, "posterior.nc", "posterior.nc"),
        "posterior_checkpoint": (
            publication.directory_fd,
            "posterior.checkpoint.json",
            "posterior.checkpoint.json",
        ),
        **{
            name: (publication.directory_fd, filename, filename)
            for name, filename in PRODUCT_FILES.items()
        },
    }
    generation_fd = _open_directory(
        Path(".hqrc_data.generations"), "LOEO HQRCData namespace", dir_fd=publication.directory_fd
    )
    names["hqrc_data_npz"] = (generation_fd, data_npz, f".hqrc_data.generations/{data_npz}")
    names["hqrc_data_metadata"] = (
        generation_fd,
        data_metadata,
        f".hqrc_data.generations/{data_metadata}",
    )
    records: dict[str, dict[str, str]] = {}
    try:
        for name, (descriptor, filename, relative) in names.items():
            records[name] = {
                "path": relative,
                "sha256": _relative_sha256(descriptor, filename),
            }
    finally:
        os.close(generation_fd)
    return records


def _expected_product_for_filename(
    filename: str, products: LOEOFoldProducts, manifest: Mapping[str, Any] | None
) -> object:
    if filename == PRODUCT_FILES["hourly_predictions"]:
        return products.hourly_predictions
    if filename == PRODUCT_FILES["metrics"]:
        return products.metrics
    if filename == PRODUCT_FILES["posterior_summary"]:
        return dict(products.posterior_summary)
    if filename == "manifest.json" and manifest is not None:
        return dict(manifest)
    raise LOEOFoldError("LOEO downstream publication order differs")


def _existing_downstream_matches(
    publication: LOEOPublicationHandle, expected: object, *, filename: str
) -> bool:
    if filename.endswith(".parquet"):
        with _external_temporary(".parquet") as temporary:
            _materialize_relative(publication.directory_fd, filename, temporary)
            try:
                actual = pl.read_parquet(temporary)
            except (OSError, pl.exceptions.PolarsError) as error:
                raise LOEOFoldError("partial LOEO downstream product is unreadable") from error
        return isinstance(expected, pl.DataFrame) and _frames_equal(actual, expected)
    return (
        _relative_json(publication.directory_fd, filename, "partial LOEO downstream JSON")
        == expected
    )


def validate_downstream_prefix(
    publication: LOEOPublicationHandle,
    *,
    entries: frozenset[str],
    products: LOEOFoldProducts,
    identity: Mapping[str, Any],
) -> frozenset[str]:
    """Validate an immutable ordered downstream prefix before any recovery write."""

    order = (
        PRODUCT_FILES["hourly_predictions"],
        PRODUCT_FILES["metrics"],
        PRODUCT_FILES["posterior_summary"],
        "manifest.json",
    )
    present = {name for name in order if name in entries}
    prefix_length = len(present)
    expected_present = set(order[:prefix_length])
    if present != expected_present:
        raise LOEOFoldError("partial LOEO downstream artifacts are not a valid prefix")
    expected_manifest = (
        manifest_payload(publication, identity=identity, products=products)
        if prefix_length == len(order)
        else None
    )
    for filename in order[:prefix_length]:
        expected = _expected_product_for_filename(filename, products, expected_manifest)
        if not _existing_downstream_matches(publication, expected, filename=filename):
            raise LOEOFoldError("partial LOEO downstream artifact semantics differ")
    return frozenset(expected_present)


def write_products(
    publication: LOEOPublicationHandle,
    products: LOEOFoldProducts,
    *,
    preserve: frozenset[str] = frozenset(),
) -> None:
    for name, frame in (
        ("hourly_predictions", products.hourly_predictions),
        ("metrics", products.metrics),
    ):
        if PRODUCT_FILES[name] in preserve:
            continue
        with _external_temporary(".parquet") as temporary:
            frame.write_parquet(temporary)
            _fsync_file(temporary)
            _copy_external_file(
                publication,
                temporary,
                PRODUCT_FILES[name],
                boundary=f"{name}-prewrite",
            )
        checked_publication_boundary(publication, f"{name}-published")
    if PRODUCT_FILES["posterior_summary"] not in preserve:
        publish_json(
            publication,
            PRODUCT_FILES["posterior_summary"],
            products.posterior_summary,
            boundary="posterior-summary-prewrite",
        )
        checked_publication_boundary(publication, "posterior-summary-published")


def _read_products(publication: LOEOPublicationHandle) -> LOEOFoldProducts:
    frames = []
    for name in (PRODUCT_FILES["hourly_predictions"], PRODUCT_FILES["metrics"]):
        with _external_temporary(".parquet") as temporary:
            _materialize_relative(publication.directory_fd, name, temporary)
            try:
                frames.append(pl.read_parquet(temporary))
            except (OSError, pl.exceptions.PolarsError) as error:
                raise LOEOFoldError("LOEO fold product is unreadable") from error
    hourly, metrics = frames
    summary = _relative_json(
        publication.directory_fd,
        PRODUCT_FILES["posterior_summary"],
        "LOEO posterior summary",
    )
    return LOEOFoldProducts(hourly, metrics, summary)


def _frames_equal(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    return (
        left.columns == right.columns
        and left.schema == right.schema
        and left.equals(right, null_equal=True)
    )


def result(directory: Path, *, reused: bool, sampler_fit_count: int) -> LOEOFoldResult:
    return LOEOFoldResult(
        output_dir=directory,
        manifest_path=directory / "manifest.json",
        posterior_path=directory / "posterior.nc",
        hourly_predictions_path=directory / PRODUCT_FILES["hourly_predictions"],
        metrics_path=directory / PRODUCT_FILES["metrics"],
        posterior_summary_path=directory / PRODUCT_FILES["posterior_summary"],
        sampler_fit_count=sampler_fit_count,
        reused=reused,
    )


def manifest_payload(
    publication: LOEOPublicationHandle,
    *,
    identity: Mapping[str, Any],
    products: LOEOFoldProducts,
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "state": "COMPLETE",
        "causal": False,
        "identity": dict(identity),
        "outputs": _artifact_records(publication),
        "rows": {
            "hourly_predictions": products.hourly_predictions.height,
            "metrics": products.metrics.height,
        },
    }
    return {**unsigned, "manifest_digest": sha_json(unsigned)}


def load_complete_material(
    publication: LOEOPublicationHandle,
    *,
    identity: Mapping[str, Any],
    inputs: LOEOFoldInputs,
    sampler: Mapping[str, object],
) -> LOEOFoldMaterial:
    directory = publication.path
    if publication_entries(publication) != COMPLETE_TOP:
        raise LOEOFoldError("completed LOEO fold directory contains unknown or partial entries")
    manifest = _relative_json(publication.directory_fd, "manifest.json", "LOEO fold manifest")
    complete = _relative_json(publication.directory_fd, "COMPLETE", "LOEO fold completion marker")
    unsigned = {
        key: manifest[key]
        for key in ("schema_version", "state", "causal", "identity", "outputs", "rows")
        if key in manifest
    }
    if (
        set(manifest)
        != {
            "schema_version",
            "state",
            "causal",
            "identity",
            "outputs",
            "rows",
            "manifest_digest",
        }
        or manifest.get("schema_version") != 1
        or manifest.get("state") != "COMPLETE"
        or manifest.get("causal") is not False
        or manifest.get("identity") != dict(identity)
        or manifest.get("manifest_digest") != sha_json(unsigned)
        or complete
        != {
            "manifest_sha256": _relative_sha256(publication.directory_fd, "manifest.json"),
            "state": "COMPLETE",
            "causal": False,
        }
        or manifest.get("outputs") != _artifact_records(publication)
    ):
        raise LOEOFoldError("completed LOEO fold manifest differs")
    try:
        data, settings = _load_hqrc_checkpoint(publication)
    except (OSError, ValueError) as error:
        raise LOEOFoldError("published LOEO HQRCData differs") from error
    identity_sha256 = sha_json(identity)
    if not same_hqrc_data(data, inputs.hqrc_data) or settings != {
        "identity_sha256": identity_sha256,
        "model": identity["model"],
        "causal": False,
    }:
        raise LOEOFoldError("published LOEO HQRCData differs")
    idata = _load_posterior_checkpoint(
        publication,
        inputs=inputs,
        sampler=sampler,
        identity_sha256=identity_sha256,
    )
    expected = generate_loeo_fold_products(
        inputs,
        idata,
        predictive_seed=int(identity["predictive_seed"]),
        predictive_draws=int(sampler["draws"]) * int(sampler["chains"]),
    )
    published = _read_products(publication)
    if (
        manifest.get("rows")
        != {
            "hourly_predictions": expected.hourly_predictions.height,
            "metrics": expected.metrics.height,
        }
        or not _frames_equal(published.hourly_predictions, expected.hourly_predictions)
        or not _frames_equal(published.metrics, expected.metrics)
        or dict(published.posterior_summary) != dict(expected.posterior_summary)
    ):
        raise LOEOFoldError("published LOEO fold semantics differ")
    return LOEOFoldMaterial(
        result=result(directory, reused=True, sampler_fit_count=0),
        inputs=inputs,
        identity=dict(identity),
        sampler=dict(sampler),
        products=published,
        manifest=manifest,
        inference_data=idata,
    )


def validate_complete(
    publication: LOEOPublicationHandle,
    *,
    identity: Mapping[str, Any],
    inputs: LOEOFoldInputs,
    sampler: Mapping[str, object],
) -> LOEOFoldResult:
    """Validate a completed fold while retaining the historical result-only API."""

    return load_complete_material(
        publication,
        identity=identity,
        inputs=inputs,
        sampler=sampler,
    ).result


def load_resumable_checkpoint(
    publication: LOEOPublicationHandle,
    *,
    identity: Mapping[str, Any],
    inputs: LOEOFoldInputs,
    sampler: Mapping[str, object],
):
    entries = publication_entries(publication) - {".loeo-fold.lock"}
    if set(entries) - (CHECKPOINT_TOP | DOWNSTREAM_TOP) or not CHECKPOINT_TOP.issubset(entries):
        raise LOEOFoldError("unsafe partial LOEO fold publication")
    _hqrc_generation_names(publication)
    identity_sha256 = sha_json(identity)
    try:
        data, settings = _load_hqrc_checkpoint(publication)
    except (OSError, ValueError) as error:
        raise LOEOFoldError("checkpoint LOEO HQRCData differs") from error
    if not same_hqrc_data(data, inputs.hqrc_data) or settings != {
        "identity_sha256": identity_sha256,
        "model": identity["model"],
        "causal": False,
    }:
        raise LOEOFoldError("checkpoint LOEO HQRCData differs")
    idata = _load_posterior_checkpoint(
        publication,
        inputs=inputs,
        sampler=sampler,
        identity_sha256=identity_sha256,
    )
    products = generate_loeo_fold_products(
        inputs,
        idata,
        predictive_seed=int(identity["predictive_seed"]),
        predictive_draws=int(sampler["draws"]) * int(sampler["chains"]),
    )
    preserved = validate_downstream_prefix(
        publication,
        entries=entries,
        products=products,
        identity=identity,
    )
    return idata, products, preserved


def validate_input_checkpoint(
    publication: LOEOPublicationHandle,
    *,
    identity: Mapping[str, Any],
    inputs: LOEOFoldInputs,
) -> None:
    """Validate a cache written after HQRC data publication but before sampling.

    This is the sole resumable state that contains no posterior.  It is safe to
    continue only when the directory contains exactly the immutable HQRC-data
    checkpoint and that checkpoint matches the freshly prepared fold inputs.
    Any posterior, downstream product, foreign file, or mismatched data remains
    fail-closed.
    """

    entries = publication_entries(publication) - {".loeo-fold.lock"}
    if entries != INPUT_CHECKPOINT_TOP:
        raise LOEOFoldError("unsafe input-only LOEO fold publication")
    try:
        data, settings = _load_hqrc_checkpoint(publication)
    except (OSError, ValueError) as error:
        raise LOEOFoldError("input-only LOEO HQRCData differs") from error
    if not same_hqrc_data(data, inputs.hqrc_data) or settings != {
        "identity_sha256": sha_json(identity),
        "model": identity["model"],
        "causal": False,
    }:
        raise LOEOFoldError("input-only LOEO HQRCData differs")


def namespace(
    output_root: Path, inputs: LOEOFoldInputs, sampler: Mapping[str, object], identity: object
) -> LOEONamespace:
    context = inputs.publication.context
    components = (
        "loeo-h3",
        context.model,
        context.feature_set,
        f"seed-{context.seed}",
        inputs.held_out_occurrence_id,
        str(sampler["profile"]),
        f"identity-{sha_json(identity)}",
    )
    return secure_namespace(Path(output_root), components)


def secure_namespace(output_root: Path, components: tuple[str, ...]) -> LOEONamespace:
    """Create a canonical no-follow namespace below one absolute trusted root."""

    root = Path(output_root)
    if not root.is_absolute():
        raise LOEOFoldError("LOEO output root must be an absolute trusted boundary")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise LOEOFoldError("LOEO output root is unsafe") from error
    if not components:
        raise LOEOFoldError("LOEO output namespace components are empty")
    require_real_directory(root, "LOEO output root")
    root_fd = _open_directory(root, "LOEO output root")
    root_identity = _identity(root_fd, "LOEO trusted output root")
    current_fd = root_fd
    final_identity: tuple[int, int] | None = None
    try:
        for component in components:
            next_fd = _ensure_directory_component(current_fd, component, "LOEO namespace component")
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        final_identity = _identity(current_fd, "LOEO result directory")
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    if final_identity is None:
        raise LOEOFoldError("LOEO namespace could not be established")
    path = root.joinpath(*components)
    return LOEONamespace(root, components, path, root_identity, final_identity)


__all__ = [
    "checked_publication_boundary",
    "fold_lock",
    "guard_namespace",
    "input_identity",
    "INPUT_CHECKPOINT_TOP",
    "load_complete_material",
    "load_resumable_checkpoint",
    "manifest_payload",
    "namespace",
    "publication_entries",
    "publication_boundary",
    "publication_has",
    "publish_json",
    "relative_sha256",
    "require_real_directory",
    "result",
    "sampler_contract",
    "secure_namespace",
    "validate_complete",
    "validate_input_checkpoint",
    "validate_downstream_prefix",
    "write_posterior_checkpoint",
    "write_products",
    "write_hqrc_checkpoint",
]
