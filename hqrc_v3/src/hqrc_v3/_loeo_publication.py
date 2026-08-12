"""Immutable identity, checkpoint, and publication state for one LOEO fold."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
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
    LOEOFoldProducts,
    LOEOFoldResult,
)
from hqrc_v3.bayes.artifacts import load_hqrc_data
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
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in resolved
    ):
        raise LOEOFoldError("draws, tune, and chains must be positive integers")
    return {
        "backend": "pymc",
        "root_seed": root_seed,
        "seed": derived_seed,
        "draws": resolved[0],
        "tune": resolved[1],
        "chains": resolved[2],
        "target_accept": 0.99 if profile == "paper" else 0.9,
        "profile": profile,
        "init": PYMC_INITIALIZATION,
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_bytes(path: Path, payload: bytes) -> None:
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, canonical_json(dict(value)) + b"\n")


def _read_json(path: Path, description: str) -> dict[str, Any]:
    _require_real_file(path, description)
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise LOEOFoldError(f"{description} is unreadable") from error
    if not isinstance(value, dict) or raw != canonical_json(value) + b"\n":
        raise LOEOFoldError(f"{description} is not canonical JSON")
    return value


def _require_real_file(path: Path, description: str) -> None:
    try:
        identity = path.lstat()
    except OSError as error:
        raise LOEOFoldError(f"{description} is missing or unsafe") from error
    if not stat.S_ISREG(identity.st_mode):
        raise LOEOFoldError(f"{description} is missing or unsafe")


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
def fold_lock(directory: LOEONamespace | Path):
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
            lock_fd = os.open(".loeo-fold.lock", flags, 0o600, dir_fd=directory_fd)
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
    directory = publication.path
    _posterior_metadata_matches(idata, inputs, sampler)
    diagnostics = validate_inference_data(idata, paper_profile=sampler["profile"] == "paper")
    guard_namespace(publication)
    descriptor, name = tempfile.mkstemp(prefix=".posterior.", suffix=".nc", dir=directory)
    os.close(descriptor)
    temporary = Path(name)
    posterior_path = directory / "posterior.nc"
    try:
        az.to_netcdf(idata, temporary)
        _fsync_file(temporary)
        guard_namespace(publication)
        os.replace(temporary, posterior_path)
        _fsync_directory(directory)
        guard_namespace(publication)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    guard_namespace(publication)
    atomic_json(
        directory / "posterior.checkpoint.json",
        {
            "schema_version": 1,
            "state": "POSTERIOR_COMPLETE",
            "causal": False,
            "identity_sha256": identity_sha256,
            "posterior_sha256": file_sha256(posterior_path),
            "diagnostics": _diagnostic_payload(diagnostics),
        },
    )
    guard_namespace(publication)


def _load_posterior_checkpoint(
    directory: Path,
    *,
    inputs: LOEOFoldInputs,
    sampler: Mapping[str, object],
    identity_sha256: str,
):
    checkpoint = _read_json(directory / "posterior.checkpoint.json", "LOEO posterior checkpoint")
    posterior_path = directory / "posterior.nc"
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
    _require_real_file(posterior_path, "LOEO posterior checkpoint artifact")
    if file_sha256(posterior_path) != checkpoint["posterior_sha256"]:
        raise LOEOFoldError("LOEO posterior checkpoint hash differs")
    try:
        idata = az.from_netcdf(posterior_path)
    except (OSError, ValueError) as error:
        raise LOEOFoldError("LOEO posterior checkpoint is unreadable") from error
    _posterior_metadata_matches(idata, inputs, sampler)
    diagnostics = validate_inference_data(idata, paper_profile=sampler["profile"] == "paper")
    if checkpoint["diagnostics"] != _diagnostic_payload(diagnostics):
        raise LOEOFoldError("LOEO posterior checkpoint diagnostics differ")
    return idata


def _hqrc_generation_paths(directory: Path) -> tuple[Path, Path, Path]:
    pointer = _read_json(directory / "hqrc_data.current.json", "LOEO HQRCData pointer")
    generation_dir = directory / ".hqrc_data.generations"
    require_real_directory(generation_dir, "LOEO HQRCData generation namespace")
    paths: list[Path] = []
    for key in ("npz", "metadata"):
        relative = pointer.get(key)
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
        ):
            raise LOEOFoldError("LOEO HQRCData pointer path is unsafe")
        path = directory / relative
        if path.parent != generation_dir:
            raise LOEOFoldError("LOEO HQRCData pointer path is unsafe")
        _require_real_file(path, "LOEO HQRCData generation artifact")
        paths.append(path)
    if paths[0] == paths[1] or {path.name for path in generation_dir.iterdir()} != {
        paths[0].name,
        paths[1].name,
    }:
        raise LOEOFoldError("LOEO HQRCData generation namespace differs")
    return directory / "hqrc_data.current.json", paths[0], paths[1]


def _artifact_records(directory: Path) -> dict[str, dict[str, str]]:
    pointer, data_npz, data_metadata = _hqrc_generation_paths(directory)
    paths = {
        "hqrc_data_pointer": pointer,
        "hqrc_data_npz": data_npz,
        "hqrc_data_metadata": data_metadata,
        "posterior": directory / "posterior.nc",
        "posterior_checkpoint": directory / "posterior.checkpoint.json",
        **{name: directory / filename for name, filename in PRODUCT_FILES.items()},
    }
    records: dict[str, dict[str, str]] = {}
    for name, path in paths.items():
        _require_real_file(path, f"LOEO fold output {name}")
        records[name] = {
            "path": path.relative_to(directory).as_posix(),
            "sha256": file_sha256(path),
        }
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


def _existing_downstream_matches(path: Path, expected: object, *, filename: str) -> bool:
    _require_real_file(path, "partial LOEO downstream artifact")
    if filename.endswith(".parquet"):
        try:
            actual = pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError) as error:
            raise LOEOFoldError("partial LOEO downstream product is unreadable") from error
        return isinstance(expected, pl.DataFrame) and _frames_equal(actual, expected)
    return _read_json(path, "partial LOEO downstream JSON") == expected


def validate_downstream_prefix(
    directory: Path,
    *,
    entries: Mapping[str, Path],
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
        manifest_payload(directory, identity=identity, products=products)
        if prefix_length == len(order)
        else None
    )
    for filename in order[:prefix_length]:
        expected = _expected_product_for_filename(filename, products, expected_manifest)
        if not _existing_downstream_matches(entries[filename], expected, filename=filename):
            raise LOEOFoldError("partial LOEO downstream artifact semantics differ")
    return frozenset(expected_present)


def write_products(
    publication: LOEOPublicationHandle,
    products: LOEOFoldProducts,
    *,
    preserve: frozenset[str] = frozenset(),
) -> None:
    directory = publication.path
    temporary_paths: list[Path] = []
    try:
        for name, frame in (
            ("hourly_predictions", products.hourly_predictions),
            ("metrics", products.metrics),
        ):
            if PRODUCT_FILES[name] in preserve:
                continue
            guard_namespace(publication)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{name}.", suffix=".parquet", dir=directory
            )
            os.close(descriptor)
            temporary = Path(temporary_name)
            temporary_paths.append(temporary)
            frame.write_parquet(temporary)
            _fsync_file(temporary)
            guard_namespace(publication)
            os.replace(temporary, directory / PRODUCT_FILES[name])
            temporary_paths.remove(temporary)
            checked_publication_boundary(publication, f"{name}-published")
        if PRODUCT_FILES["posterior_summary"] not in preserve:
            guard_namespace(publication)
            atomic_json(directory / PRODUCT_FILES["posterior_summary"], products.posterior_summary)
            checked_publication_boundary(publication, "posterior-summary-published")
    finally:
        for path in temporary_paths:
            path.unlink(missing_ok=True)


def _read_products(directory: Path) -> LOEOFoldProducts:
    try:
        hourly = pl.read_parquet(directory / PRODUCT_FILES["hourly_predictions"])
        metrics = pl.read_parquet(directory / PRODUCT_FILES["metrics"])
    except (OSError, pl.exceptions.PolarsError) as error:
        raise LOEOFoldError("LOEO fold product is unreadable") from error
    summary = _read_json(directory / PRODUCT_FILES["posterior_summary"], "LOEO posterior summary")
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
    directory: Path, *, identity: Mapping[str, Any], products: LOEOFoldProducts
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "state": "COMPLETE",
        "causal": False,
        "identity": dict(identity),
        "outputs": _artifact_records(directory),
        "rows": {
            "hourly_predictions": products.hourly_predictions.height,
            "metrics": products.metrics.height,
        },
    }
    return {**unsigned, "manifest_digest": sha_json(unsigned)}


def validate_complete(
    directory: Path,
    *,
    identity: Mapping[str, Any],
    inputs: LOEOFoldInputs,
    sampler: Mapping[str, object],
) -> LOEOFoldResult:
    if {path.name for path in directory.iterdir()} != COMPLETE_TOP:
        raise LOEOFoldError("completed LOEO fold directory contains unknown or partial entries")
    manifest = _read_json(directory / "manifest.json", "LOEO fold manifest")
    complete = _read_json(directory / "COMPLETE", "LOEO fold completion marker")
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
            "manifest_sha256": file_sha256(directory / "manifest.json"),
            "state": "COMPLETE",
            "causal": False,
        }
        or manifest.get("outputs") != _artifact_records(directory)
    ):
        raise LOEOFoldError("completed LOEO fold manifest differs")
    try:
        data, settings = load_hqrc_data(directory / "hqrc_data.npz")
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
        directory,
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
    published = _read_products(directory)
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
    return result(directory, reused=True, sampler_fit_count=0)


def load_resumable_checkpoint(
    directory: Path,
    *,
    identity: Mapping[str, Any],
    inputs: LOEOFoldInputs,
    sampler: Mapping[str, object],
):
    entries = {path.name: path for path in directory.iterdir() if path.name != ".loeo-fold.lock"}
    if set(entries) - (CHECKPOINT_TOP | DOWNSTREAM_TOP) or not CHECKPOINT_TOP.issubset(entries):
        raise LOEOFoldError("unsafe partial LOEO fold publication")
    for name, path in entries.items():
        if name == ".hqrc_data.generations":
            require_real_directory(path, "partial LOEO HQRCData namespace")
        else:
            _require_real_file(path, "partial LOEO fold artifact")
    _hqrc_generation_paths(directory)
    identity_sha256 = sha_json(identity)
    try:
        data, settings = load_hqrc_data(directory / "hqrc_data.npz")
    except (OSError, ValueError) as error:
        raise LOEOFoldError("checkpoint LOEO HQRCData differs") from error
    if not same_hqrc_data(data, inputs.hqrc_data) or settings != {
        "identity_sha256": identity_sha256,
        "model": identity["model"],
        "causal": False,
    }:
        raise LOEOFoldError("checkpoint LOEO HQRCData differs")
    idata = _load_posterior_checkpoint(
        directory,
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
        directory,
        entries=entries,
        products=products,
        identity=identity,
    )
    return idata, products, preserved


def namespace(
    output_root: Path, inputs: LOEOFoldInputs, sampler: Mapping[str, object], identity: object
) -> LOEONamespace:
    root = Path(output_root)
    if not root.is_absolute():
        raise LOEOFoldError("LOEO fold output root must be an absolute trusted boundary")
    try:
        root.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise LOEOFoldError("LOEO fold output root is unsafe") from error
    require_real_directory(root, "LOEO fold output root")
    root_fd = _open_directory(root, "LOEO fold output root")
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
    root_identity = _identity(root_fd, "LOEO fold trusted output root")
    current_fd = root_fd
    final_identity: tuple[int, int] | None = None
    try:
        for component in components:
            next_fd = _ensure_directory_component(
                current_fd, component, "LOEO fold namespace component"
            )
            if current_fd != root_fd:
                os.close(current_fd)
            current_fd = next_fd
        final_identity = _identity(current_fd, "LOEO fold result directory")
    finally:
        if current_fd != root_fd:
            os.close(current_fd)
        os.close(root_fd)
    if final_identity is None:
        raise LOEOFoldError("LOEO fold namespace could not be established")
    path = root.joinpath(*components)
    return LOEONamespace(root, components, path, root_identity, final_identity)


__all__ = [
    "atomic_json",
    "checked_publication_boundary",
    "fold_lock",
    "guard_namespace",
    "input_identity",
    "load_resumable_checkpoint",
    "manifest_payload",
    "namespace",
    "publication_boundary",
    "require_real_directory",
    "result",
    "sampler_contract",
    "validate_complete",
    "validate_downstream_prefix",
    "write_posterior_checkpoint",
    "write_products",
]
