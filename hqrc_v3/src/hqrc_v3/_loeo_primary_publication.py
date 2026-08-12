"""Held-dirfd immutable publication for the primary H3 LOEO matrix."""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl

import hqrc_v3._loeo_publication as publication_io
from hqrc_v3._loeo_contract import sha_json
from hqrc_v3._loeo_primary_types import LOEOPrimaryError, LOEOPrimaryProducts, LOEOPrimaryResult

PRODUCT_FILES = {
    "hourly_predictions": "hourly_predictions.parquet",
    "per_event_metrics": "per_event_metrics.parquet",
    "aggregate_metrics": "aggregate_metrics.parquet",
    "posterior_summaries": "posterior_summaries.parquet",
    "training_psis_loo": "training_psis_loo.parquet",
}
_ORDER = (*PRODUCT_FILES.values(), "manifest.json")
_LOCK = ".loeo-primary.lock"


def namespace(
    output_root: Path,
    *,
    context: object,
    profile: str,
    selected_occurrence_ids: tuple[str, ...],
    identity: Mapping[str, Any],
) -> publication_io.LOEONamespace:
    selection_digest = sha_json(list(selected_occurrence_ids))
    return publication_io.secure_namespace(
        output_root,
        (
            "loeo-primary-h3",
            context.model,
            context.feature_set,
            f"seed-{context.seed}",
            profile,
            f"selection-{selection_digest}",
            f"identity-{sha_json(identity)}",
        ),
    )


def _product_items(products: LOEOPrimaryProducts) -> tuple[tuple[str, pl.DataFrame], ...]:
    return tuple((filename, getattr(products, key)) for key, filename in PRODUCT_FILES.items())


def _read_frame(publication: publication_io.LOEOPublicationHandle, name: str) -> pl.DataFrame:
    with tempfile.TemporaryDirectory(prefix="hqrc-v3-task15e-read-", dir="/private/tmp") as root:
        path = Path(root) / "artifact.parquet"
        publication_io._materialize_relative(publication.directory_fd, name, path)
        try:
            return pl.read_parquet(path)
        except (OSError, pl.exceptions.PolarsError) as error:
            raise LOEOPrimaryError("LOEO primary aggregate Parquet is unreadable") from error


def _publish_frame(
    publication: publication_io.LOEOPublicationHandle,
    name: str,
    frame: pl.DataFrame,
) -> None:
    with tempfile.TemporaryDirectory(prefix="hqrc-v3-task15e-write-", dir="/private/tmp") as root:
        path = Path(root) / "artifact.parquet"
        frame.write_parquet(path)
        publication_io._fsync_file(path)
        publication_io._copy_external_file(
            publication,
            path,
            name,
            boundary=f"matrix-{name}-prewrite",
        )
    publication_io.checked_publication_boundary(publication, f"matrix-{name}-published")


def _artifact_records(
    publication: publication_io.LOEOPublicationHandle,
) -> dict[str, dict[str, str]]:
    return {
        key: {
            "path": filename,
            "sha256": publication_io.relative_sha256(publication, filename),
        }
        for key, filename in PRODUCT_FILES.items()
    }


def manifest_payload(
    publication: publication_io.LOEOPublicationHandle,
    *,
    identity: Mapping[str, Any],
    products: LOEOPrimaryProducts,
) -> dict[str, Any]:
    unsigned: dict[str, Any] = {
        "schema_version": 1,
        "state": "COMPLETE",
        "evaluation": "retrospective-loeo-primary-h3",
        "causal": False,
        "identity": dict(identity),
        "outputs": _artifact_records(publication),
        "rows": {key: getattr(products, key).height for key in PRODUCT_FILES},
    }
    return {**unsigned, "manifest_digest": sha_json(unsigned)}


def _frames_equal(left: pl.DataFrame, right: pl.DataFrame) -> bool:
    return (
        left.columns == right.columns
        and left.schema == right.schema
        and left.equals(right, null_equal=True)
    )


def _validate_product(
    publication: publication_io.LOEOPublicationHandle,
    filename: str,
    expected: pl.DataFrame,
) -> None:
    if not _frames_equal(_read_frame(publication, filename), expected):
        raise LOEOPrimaryError("LOEO primary aggregate product semantics differ")


def _result(
    directory: Path,
    *,
    selected_occurrence_ids: tuple[str, ...],
    fold_fit_counts: Mapping[str, int],
    reused: bool,
) -> LOEOPrimaryResult:
    return LOEOPrimaryResult(
        output_dir=directory,
        manifest_path=directory / "manifest.json",
        hourly_predictions_path=directory / PRODUCT_FILES["hourly_predictions"],
        per_event_metrics_path=directory / PRODUCT_FILES["per_event_metrics"],
        aggregate_metrics_path=directory / PRODUCT_FILES["aggregate_metrics"],
        posterior_summaries_path=directory / PRODUCT_FILES["posterior_summaries"],
        training_psis_loo_path=directory / PRODUCT_FILES["training_psis_loo"],
        selected_occurrence_ids=selected_occurrence_ids,
        fold_sampler_fit_counts=dict(fold_fit_counts),
        sampler_fit_count=sum(fold_fit_counts.values()),
        reused=reused,
    )


def validate_complete(
    publication: publication_io.LOEOPublicationHandle,
    *,
    identity: Mapping[str, Any],
    products: LOEOPrimaryProducts,
    selected_occurrence_ids: tuple[str, ...],
    fold_fit_counts: Mapping[str, int],
) -> LOEOPrimaryResult:
    expected_entries = {_LOCK, *_ORDER, "COMPLETE"}
    if publication_io.publication_entries(publication) != expected_entries:
        raise LOEOPrimaryError("completed LOEO primary publication is partial or unknown")
    for filename, frame in _product_items(products):
        _validate_product(publication, filename, frame)
    manifest = publication_io._relative_json(
        publication.directory_fd, "manifest.json", "LOEO primary manifest"
    )
    expected_manifest = manifest_payload(publication, identity=identity, products=products)
    complete = publication_io._relative_json(
        publication.directory_fd, "COMPLETE", "LOEO primary completion marker"
    )
    if manifest != expected_manifest or complete != {
        "manifest_sha256": publication_io.relative_sha256(publication, "manifest.json"),
        "state": "COMPLETE",
        "causal": False,
    }:
        raise LOEOPrimaryError("completed LOEO primary manifest differs")
    return _result(
        publication.path,
        selected_occurrence_ids=selected_occurrence_ids,
        fold_fit_counts=fold_fit_counts,
        reused=True,
    )


def publish_or_resume(
    matrix_namespace: publication_io.LOEONamespace,
    *,
    identity: Mapping[str, Any],
    products: LOEOPrimaryProducts,
    selected_occurrence_ids: tuple[str, ...],
    fold_fit_counts: Mapping[str, int],
) -> LOEOPrimaryResult:
    with publication_io.fold_lock(matrix_namespace, lock_name=_LOCK) as publication:
        publication_io.guard_namespace(publication)
        if publication_io.publication_has(publication, "COMPLETE"):
            return validate_complete(
                publication,
                identity=identity,
                products=products,
                selected_occurrence_ids=selected_occurrence_ids,
                fold_fit_counts=fold_fit_counts,
            )
        entries = publication_io.publication_entries(publication) - {_LOCK}
        present = tuple(name for name in _ORDER if name in entries)
        if set(entries) != set(_ORDER[: len(present)]) or present != _ORDER[: len(present)]:
            raise LOEOPrimaryError("partial LOEO primary publication is not an ordered prefix")
        product_by_name = dict(_product_items(products))
        for filename in present:
            if filename == "manifest.json":
                expected_manifest = manifest_payload(
                    publication, identity=identity, products=products
                )
                if (
                    publication_io._relative_json(
                        publication.directory_fd, filename, "partial LOEO primary manifest"
                    )
                    != expected_manifest
                ):
                    raise LOEOPrimaryError("partial LOEO primary manifest semantics differ")
            else:
                _validate_product(publication, filename, product_by_name[filename])
        for filename in _ORDER[len(present) :]:
            if filename == "manifest.json":
                publication_io.publish_json(
                    publication,
                    filename,
                    manifest_payload(publication, identity=identity, products=products),
                    boundary="matrix-manifest-prewrite",
                )
            else:
                _publish_frame(publication, filename, product_by_name[filename])
        publication_io.publish_json(
            publication,
            "COMPLETE",
            {
                "manifest_sha256": publication_io.relative_sha256(publication, "manifest.json"),
                "state": "COMPLETE",
                "causal": False,
            },
            boundary="matrix-complete-prewrite",
        )
        publication_io.checked_publication_boundary(publication, "matrix-complete-published")
        return _result(
            matrix_namespace.path,
            selected_occurrence_ids=selected_occurrence_ids,
            fold_fit_counts=fold_fit_counts,
            reused=False,
        )


__all__ = ["PRODUCT_FILES", "manifest_payload", "namespace", "publish_or_resume"]
