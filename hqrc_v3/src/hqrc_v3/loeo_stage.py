"""Immutable one-fold retrospective H3 LOEO preparation and orchestration."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

import hqrc_v3._loeo_publication as publication_io
from hqrc_v3._loeo_contract import (
    MODEL_OPTIONS,
    derive_loeo_seed,
)
from hqrc_v3._loeo_contract import (
    sha_json as _sha_json,
)
from hqrc_v3._loeo_products import generate_loeo_fold_products
from hqrc_v3._loeo_types import (
    LOEOFoldError,
    LOEOFoldInputs,
    LOEOFoldProducts,
    LOEOFoldResult,
)
from hqrc_v3.bayes.artifacts import write_hqrc_data
from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.bayes.samplers import sample_hqrc
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.diagnostics.ar import ApprovedARCalibration, validate_event_residual_context
from hqrc_v3.diagnostics.loeo import (
    LOEOFold,
    LOEOHeldOut,
    LOEOPublication,
    load_loeo_event,
    load_loeo_fold,
    load_loeo_universe,
)
from hqrc_v3.diagnostics.loeo_ar import ApprovedLOEOARSet
from hqrc_v3.provenance import file_sha256

_HOLIDAY_INDEX = {"seollal": 0, "chuseok": 1}


def _build_hqrc_data(fold: LOEOFold, approved: ApprovedARCalibration) -> HQRCData:
    if fold.causal is not False or len(fold.occurrence_ids) != 9:
        raise LOEOFoldError("LOEO training requires one physical causal=false nine-event fold")
    if fold.held_out_occurrence_id in fold.occurrence_ids:
        raise LOEOFoldError("LOEO training fold contains its held-out occurrence")
    try:
        actual_context = validate_event_residual_context(fold.frame, allow_final_split=True)
    except (TypeError, ValueError) as error:
        raise LOEOFoldError("LOEO training fold residual context is invalid") from error
    if actual_context != approved.context:
        raise LOEOFoldError("LOEO fold context differs from its approved calibration")
    if tuple(sorted(fold.occurrence_ids)) != approved.calibration.event_ids:
        raise LOEOFoldError("LOEO approved event ids differ from the physical training fold")
    required = {
        "standardized_residual",
        "occurrence_id",
        "holiday_type",
        "tau_days",
        "hour",
        "restriction",
        "causal",
    }
    if not required.issubset(fold.frame.columns):
        raise LOEOFoldError("LOEO physical training fold schema differs")
    if fold.frame["causal"].unique().to_list() != [False]:
        raise LOEOFoldError("LOEO physical training fold must be causal=false")
    ordered = fold.frame
    if tuple(ordered["occurrence_id"].unique(maintain_order=True).to_list()) != fold.occurrence_ids:
        raise LOEOFoldError("LOEO physical training occurrence ordering differs")
    event_index = {event_id: index for index, event_id in enumerate(fold.occurrence_ids)}
    try:
        return HQRCData(
            observations=ordered["standardized_residual"].to_numpy(),
            occurrence_index=np.asarray(
                [event_index[value] for value in ordered["occurrence_id"].to_list()],
                dtype=np.int64,
            ),
            holiday_type_index=np.asarray(
                [_HOLIDAY_INDEX[value] for value in ordered["holiday_type"].to_list()],
                dtype=np.int64,
            ),
            tau_days=ordered["tau_days"].to_numpy(),
            hour=ordered["hour"].to_numpy(),
            restriction=ordered["restriction"].to_numpy(),
            occurrence_ids=fold.occurrence_ids,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise LOEOFoldError("LOEO physical training fold cannot form HQRCData") from error


def _evaluation_scale(held_out: LOEOHeldOut) -> tuple[float, int]:
    frame = held_out.frame
    required = {
        "occurrence_id",
        "split_id",
        "sigma_n_mw",
        "restriction",
        "causal",
    }
    if not required.issubset(frame.columns) or frame.is_empty():
        raise LOEOFoldError("LOEO held-out event schema differs")
    if frame["occurrence_id"].unique().to_list() != [held_out.occurrence_id]:
        raise LOEOFoldError("LOEO evaluation contains a non-held-out event")
    if frame["causal"].unique().to_list() != [False]:
        raise LOEOFoldError("LOEO held-out event must be causal=false")
    year_text = held_out.occurrence_id.rsplit("-", 1)[-1]
    if not year_text.isdigit():
        raise LOEOFoldError("LOEO held-out occurrence year is invalid")
    year = int(year_text)
    expected_split = f"oof-{year}" if year < 2024 else "final-2024"
    if frame["split_id"].unique().to_list() != [expected_split]:
        raise LOEOFoldError("LOEO held-out evaluation split differs")
    scales = frame["sigma_n_mw"].unique()
    if scales.len() != 1:
        raise LOEOFoldError("LOEO evaluation scale is not unique")
    scale = scales.item()
    if (
        isinstance(scale, bool)
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise LOEOFoldError("LOEO evaluation scale must be finite and positive")
    restrictions = frame["restriction"].unique()
    if restrictions.len() != 1 or restrictions.item() not in (0, 1):
        raise LOEOFoldError("LOEO held-out restriction metadata differs")
    return float(scale), int(restrictions.item())


def prepare_loeo_fold_inputs(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    approved_set: ApprovedLOEOARSet,
    *,
    held_out_occurrence_id: str,
) -> LOEOFoldInputs:
    """Prepare one fully source-revalidated fold without invoking a sampler."""

    if not isinstance(source, ValidatedCorrectionSource):
        raise TypeError("LOEO fold preparation requires a ValidatedCorrectionSource")
    if not isinstance(publication, LOEOPublication):
        raise TypeError("LOEO fold preparation requires a LOEOPublication")
    if not isinstance(approved_set, ApprovedLOEOARSet):
        raise TypeError("LOEO fold preparation requires an ApprovedLOEOARSet")
    if not isinstance(held_out_occurrence_id, str) or not held_out_occurrence_id.strip():
        raise LOEOFoldError("LOEO held-out occurrence id must be nonblank")
    current = load_loeo_universe(source, publication.context, output_dir=publication.output_dir)
    if current != publication:
        raise LOEOFoldError("supplied LOEO publication differs from current validated inputs")
    if (
        publication.causal is not False
        or len(publication.occurrence_ids) != 10
        or approved_set.occurrence_ids != publication.occurrence_ids
        or approved_set.context != publication.context
        or approved_set.source_context.loeo_manifest_sha256
        != file_sha256(publication.manifest_path)
        or approved_set.source_context.universe_sha256 != publication.universe_sha256
    ):
        raise LOEOFoldError("approved LOEO AR set differs from the physical publication")
    try:
        approved = approved_set.calibration_for(held_out_occurrence_id)
        training = load_loeo_fold(
            source,
            publication.context,
            output_dir=publication.output_dir,
            held_out_occurrence_id=held_out_occurrence_id,
        )
        held_out = load_loeo_event(
            source,
            publication.context,
            output_dir=publication.output_dir,
            occurrence_id=held_out_occurrence_id,
        )
    except (TypeError, ValueError) as error:
        raise LOEOFoldError("LOEO fold inputs failed complete source revalidation") from error
    if (
        approved_set.training_sha256.get(held_out_occurrence_id) != training.residual_sha256
        or approved.residual_sha256 != training.residual_sha256
    ):
        raise LOEOFoldError("LOEO fold approval is not bound to the physical training artifact")
    hqrc_data = _build_hqrc_data(training, approved)
    scale, restriction = _evaluation_scale(held_out)
    return LOEOFoldInputs(
        source=source,
        publication=publication,
        approved_set=approved_set,
        approved=approved,
        held_out_occurrence_id=held_out_occurrence_id,
        training_fold=training,
        held_out=held_out,
        hqrc_data=hqrc_data,
        sigma_eval=scale,
        restriction=restriction,
        causal=False,
    )


def fit_loeo_fold(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    approved_set: ApprovedLOEOARSet,
    *,
    held_out_occurrence_id: str,
    sampler_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    output_root: Path,
) -> LOEOFoldResult:
    """Fit or strictly reuse exactly one immutable H3 partial-pooling LOEO fold."""

    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved_set,
        held_out_occurrence_id=held_out_occurrence_id,
    )
    sampler = publication_io.sampler_contract(
        profile,
        root_seed=sampler_seed,
        held_out_occurrence_id=held_out_occurrence_id,
        draws=draws,
        tune=tune,
        chains=chains,
    )
    if profile == "paper" and inputs.source.source_profile != "paper":
        raise LOEOFoldError("paper sampler requires a paper-profile correction source")
    identity = publication_io.input_identity(inputs, sampler)
    identity_sha256 = _sha_json(identity)
    directory = publication_io.namespace(Path(output_root), inputs, sampler, identity)
    with publication_io.fold_lock(directory):
        complete = directory / "COMPLETE"
        manifest = directory / "manifest.json"
        if complete.exists() or complete.is_symlink():
            if not manifest.exists() or manifest.is_symlink():
                raise LOEOFoldError("partial LOEO fold publication")
            return publication_io.validate_complete(
                directory, identity=identity, inputs=inputs, sampler=sampler
            )
        existing = {path.name for path in directory.iterdir() if path.name != ".loeo-fold.lock"}
        if existing:
            idata = publication_io.load_resumable_checkpoint(
                directory,
                identity=identity,
                inputs=inputs,
                sampler=sampler,
            )
            fit_count = 0
        else:
            write_hqrc_data(
                directory / "hqrc_data.npz",
                inputs.hqrc_data,
                settings={
                    "identity_sha256": identity_sha256,
                    "model": identity["model"],
                    "causal": False,
                },
            )
            publication_io.publication_boundary("hqrc-data-published")
            idata = sample_hqrc(
                inputs.hqrc_data,
                inputs.approved,
                variant="H3",
                pooling="partial",
                options=MODEL_OPTIONS,
                draws=int(sampler["draws"]),
                tune=int(sampler["tune"]),
                chains=int(sampler["chains"]),
                seed=int(sampler["seed"]),
                backend="pymc",
                paper_profile=profile == "paper",
            )
            idata.attrs["hqrc_causal"] = "false"
            publication_io.write_posterior_checkpoint(
                directory,
                idata,
                inputs=inputs,
                sampler=sampler,
                identity_sha256=identity_sha256,
            )
            fit_count = 1
        publication_io.publication_boundary("posterior-checkpointed")
        products = generate_loeo_fold_products(
            inputs,
            idata,
            predictive_seed=int(identity["predictive_seed"]),
            predictive_draws=int(sampler["draws"]) * int(sampler["chains"]),
        )
        publication_io.write_products(directory, products)
        payload = publication_io.manifest_payload(directory, identity=identity, products=products)
        publication_io.atomic_json(directory / "manifest.json", payload)
        publication_io.publication_boundary("manifest-published")
        publication_io.atomic_json(
            directory / "COMPLETE",
            {
                "manifest_sha256": file_sha256(directory / "manifest.json"),
                "state": "COMPLETE",
                "causal": False,
            },
        )
        publication_io.publication_boundary("complete-published")
    return publication_io.result(directory, reused=False, sampler_fit_count=fit_count)


def load_loeo_fold_result(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    approved_set: ApprovedLOEOARSet,
    *,
    held_out_occurrence_id: str,
    sampler_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    output_root: Path,
) -> LOEOFoldResult:
    """Load and semantically revalidate one completed result without fitting."""

    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved_set,
        held_out_occurrence_id=held_out_occurrence_id,
    )
    sampler = publication_io.sampler_contract(
        profile,
        root_seed=sampler_seed,
        held_out_occurrence_id=held_out_occurrence_id,
        draws=draws,
        tune=tune,
        chains=chains,
    )
    identity = publication_io.input_identity(inputs, sampler)
    directory = publication_io.namespace(Path(output_root), inputs, sampler, identity)
    publication_io.require_real_directory(directory, "completed LOEO fold result")
    with publication_io.fold_lock(directory):
        return publication_io.validate_complete(
            directory, identity=identity, inputs=inputs, sampler=sampler
        )


__all__ = [
    "LOEOFoldError",
    "LOEOFoldInputs",
    "LOEOFoldProducts",
    "LOEOFoldResult",
    "derive_loeo_seed",
    "fit_loeo_fold",
    "generate_loeo_fold_products",
    "load_loeo_fold_result",
    "prepare_loeo_fold_inputs",
]
