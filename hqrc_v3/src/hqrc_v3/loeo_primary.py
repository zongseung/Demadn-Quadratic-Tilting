"""Immutable orchestration for one primary H3 LOEO matrix."""

from __future__ import annotations

import hashlib
from pathlib import Path

import hqrc_v3._loeo_primary_publication as matrix_publication_io
import hqrc_v3._loeo_publication as fold_publication_io
from hqrc_v3._loeo_contract import canonical_json, derive_loeo_seed, sha_json
from hqrc_v3._loeo_primary_products import generate_loeo_primary_products
from hqrc_v3._loeo_primary_types import LOEOPrimaryError, LOEOPrimaryProducts, LOEOPrimaryResult
from hqrc_v3._loeo_types import LOEOFoldError, LOEOFoldMaterial
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.diagnostics.loeo import LOEOError, LOEOPublication
from hqrc_v3.diagnostics.loeo_ar import ApprovedLOEOARSet
from hqrc_v3.loeo_stage import (
    fit_loeo_fold,
    load_loeo_fold_material,
    validate_loeo_fold_sources,
)

_AGGREGATION_SCHEMA_VERSION = 2


def _selected_folds(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    approved_set: ApprovedLOEOARSet,
    *,
    held_out_occurrence_ids: tuple[str, ...],
    root_seed: int,
    profile: str,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    cores: int | None,
    init: str | None,
    target_accept: float | None,
    backend: str,
    device: str,
    output_root: Path,
) -> tuple[str, ...]:
    if not isinstance(source, ValidatedCorrectionSource):
        raise TypeError("LOEO primary requires a ValidatedCorrectionSource")
    if not isinstance(publication, LOEOPublication):
        raise TypeError("LOEO primary requires a LOEOPublication")
    if not isinstance(approved_set, ApprovedLOEOARSet):
        raise TypeError("LOEO primary requires an ApprovedLOEOARSet")
    if not isinstance(held_out_occurrence_ids, tuple):
        raise LOEOPrimaryError("LOEO primary selected folds must be an explicit tuple")
    if not Path(output_root).is_absolute():
        raise LOEOPrimaryError("LOEO primary output root must be an absolute trusted boundary")
    canonical = publication.occurrence_ids
    if (
        publication.causal is not False
        or len(canonical) != 10
        or approved_set.occurrence_ids != canonical
        or set(publication.fold_paths) != set(canonical)
        or set(publication.fold_sha256) != set(canonical)
        or set(approved_set.calibrations) != set(canonical)
        or set(approved_set.approved_sha256) != set(canonical)
    ):
        raise LOEOPrimaryError("LOEO primary requires one complete ten-fold approved publication")
    if profile == "paper":
        if source.source_profile != "paper" or held_out_occurrence_ids != canonical:
            raise LOEOPrimaryError("paper LOEO primary requires all ten canonical folds")
    elif profile == "smoke":
        if (
            not held_out_occurrence_ids
            or len(set(held_out_occurrence_ids)) != len(held_out_occurrence_ids)
            or tuple(event for event in canonical if event in held_out_occurrence_ids)
            != held_out_occurrence_ids
        ):
            raise LOEOPrimaryError("smoke LOEO primary requires a non-empty canonical subset")
    else:
        raise LOEOPrimaryError("LOEO primary profile must be paper or smoke")
    try:
        fold_publication_io.sampler_contract(
            profile,
            root_seed=root_seed,
            held_out_occurrence_id=held_out_occurrence_ids[0],
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            init=init,
            target_accept=target_accept,
            backend=backend,
            device=device,
        )
    except (TypeError, ValueError, LOEOFoldError) as error:
        raise LOEOPrimaryError("LOEO primary sampler contract is invalid") from error
    if profile == "paper":
        try:
            validate_loeo_fold_sources(source, publication, approved_set)
        except (TypeError, ValueError, LOEOFoldError, LOEOError) as error:
            raise LOEOPrimaryError(
                "paper LOEO primary inputs failed complete revalidation"
            ) from error
    return held_out_occurrence_ids


def _fold_reference(material: LOEOFoldMaterial) -> dict[str, object]:
    identity = dict(material.identity)
    manifest = dict(material.manifest)
    return {
        "held_out_occurrence_id": material.inputs.held_out_occurrence_id,
        "output_dir": material.result.output_dir.resolve().as_posix(),
        "identity_sha256": sha_json(identity),
        "manifest_sha256": hashlib.sha256(canonical_json(manifest) + b"\n").hexdigest(),
        "manifest_digest": manifest["manifest_digest"],
        "artifacts": manifest["outputs"],
        "sampler": dict(material.sampler),
        "predictive_seed": identity["predictive_seed"],
        "state": "COMPLETE_REVALIDATED",
    }


def _matrix_identity(
    materials: tuple[LOEOFoldMaterial, ...],
    *,
    selected_occurrence_ids: tuple[str, ...],
    root_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    init: str | None = None,
    target_accept: float | None = None,
    backend: str = "pymc",
    device: str = "cpu",
) -> dict[str, object]:
    first = dict(materials[0].identity)
    common_keys = ("source", "context", "model")
    for material in materials:
        identity = dict(material.identity)
        held = material.inputs.held_out_occurrence_id
        sampler = dict(material.sampler)
        base = fold_publication_io.sampler_contract(
            profile,
            root_seed=root_seed,
            held_out_occurrence_id=held,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            init=init,
            target_accept=target_accept,
            backend=backend,
            device=device,
        )
        candidates = [base]
        if profile == "paper" and base["target_accept"] == 0.99:
            candidates.append(
                fold_publication_io.retry_sampler_contract(
                    base,
                    variant="H3",
                    held_out_occurrence_id=held,
                )
            )
            candidates.append(
                fold_publication_io.legacy_retry_sampler_contract(
                    base,
                    variant="H3",
                    held_out_occurrence_id=held,
                )
            )
        if (
            any(identity.get(key) != first.get(key) for key in common_keys)
            or identity.get("causal") is not False
            or identity.get("sampler") != sampler
            or sampler not in candidates
            or identity.get("predictive_seed")
            != derive_loeo_seed(root_seed, f"H3-fold-predictive:{held}")
        ):
            raise LOEOPrimaryError("LOEO primary fold identity or resolved seeds differ")
    return {
        "schema_version": _AGGREGATION_SCHEMA_VERSION,
        "evaluation": "retrospective-loeo-primary-h3",
        "causal": False,
        "profile": profile,
        "publication_scope": "paper-all-ten" if profile == "paper" else "smoke-subset",
        "selected_occurrence_ids": list(selected_occurrence_ids),
        "root_seed": root_seed,
        "source": first["source"],
        "context": first["context"],
        "loeo": {
            key: first["loeo"][key]
            for key in (
                "output_dir",
                "generation_dir",
                "manifest_path",
                "manifest_file_sha256",
                "universe_path",
                "universe_sha256",
            )
        },
        "approved_loeo_ar": {
            key: first["approved_loeo_ar"][key]
            for key in (
                "output_dir",
                "approved_set_path",
                "proposal_set_sha256",
                "approved_set_sha256",
            )
        },
        "model": first["model"],
        "folds": [_fold_reference(material) for material in materials],
        "aggregation": {
            "schema_version": _AGGREGATION_SCHEMA_VERSION,
            "point_metrics": "recomputed-from-concatenated-hourly-observed-and-point",
            "probabilistic_metrics": "held-out-hour-weighted-exact-per-event-metrics",
            "groups": ["pooled", "seollal", "chuseok"],
            "psis_loo_label": "training_nine_event_psis_loo",
            "full_posterior_draws_duplicated": False,
        },
    }


def _close_materials(materials: list[LOEOFoldMaterial]) -> None:
    for material in materials:
        close = getattr(material.inference_data, "close", None)
        if callable(close):
            close()


def fit_loeo_primary(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    approved_set: ApprovedLOEOARSet,
    *,
    held_out_occurrence_ids: tuple[str, ...],
    root_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    init: str | None = None,
    target_accept: float | None = None,
    backend: str = "pymc",
    device: str = "cpu",
    output_root: Path,
) -> LOEOPrimaryResult:
    """Fit/reuse selected reviewed folds and publish one immutable H3 aggregate."""

    selected = _selected_folds(
        source,
        publication,
        approved_set,
        held_out_occurrence_ids=held_out_occurrence_ids,
        root_seed=root_seed,
        profile=profile,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        init=init,
        target_accept=target_accept,
        backend=backend,
        device=device,
        output_root=output_root,
    )
    fold_fit_counts: dict[str, int] = {}
    materials: list[LOEOFoldMaterial] = []
    try:
        for held_out in selected:
            fitted = fit_loeo_fold(
                source,
                publication,
                approved_set,
                held_out_occurrence_id=held_out,
                sampler_seed=root_seed,
                profile=profile,
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                init=init,
                target_accept=target_accept,
                backend=backend,
                device=device,
                output_root=output_root,
            )
            fold_fit_counts[held_out] = fitted.sampler_fit_count
            materials.append(
                load_loeo_fold_material(
                    source,
                    publication,
                    approved_set,
                    held_out_occurrence_id=held_out,
                    sampler_seed=root_seed,
                    profile=profile,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    init=init,
                    target_accept=target_accept,
                    backend=backend,
                    device=device,
                    output_root=output_root,
                )
            )
        material_tuple = tuple(materials)
        products = generate_loeo_primary_products(
            material_tuple,
            selected_occurrence_ids=selected,
        )
        identity = _matrix_identity(
            material_tuple,
            selected_occurrence_ids=selected,
            root_seed=root_seed,
            profile=profile,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            init=init,
            target_accept=target_accept,
            backend=backend,
            device=device,
        )
        matrix_namespace = matrix_publication_io.namespace(
            Path(output_root),
            context=publication.context,
            profile=profile,
            selected_occurrence_ids=selected,
            identity=identity,
        )
        return matrix_publication_io.publish_or_resume(
            matrix_namespace,
            identity=identity,
            products=products,
            selected_occurrence_ids=selected,
            fold_fit_counts=fold_fit_counts,
        )
    except LOEOPrimaryError:
        raise
    except LOEOFoldError as error:
        raise LOEOPrimaryError("LOEO primary fold or publication failed safely") from error
    finally:
        _close_materials(materials)


__all__ = [
    "LOEOPrimaryError",
    "LOEOPrimaryProducts",
    "LOEOPrimaryResult",
    "fit_loeo_primary",
    "generate_loeo_primary_products",
]
