"""Portable, resumable orchestration for the manuscript H1--H3 LOEO matrix.

The baseline publication is deliberately a common immutable source: its ten
model/feature streams must all be complete before one standardized-residual
artifact can be trusted.  Once that source exists, this module processes HQRC
contexts in a fixed model order.  A context is fully completed (LOEO universe,
ACF-derived AR proposal, optional exact-digest approval, ten H3 folds, and
aggregate products) before the next context starts.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

from hqrc_v3.baselines.config import MODEL_NAMES
from hqrc_v3.correction_source import ValidatedCorrectionSource, validate_correction_source
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.diagnostics.loeo import publish_loeo_universe
from hqrc_v3.diagnostics.loeo_ar import (
    approve_loeo_ar_proposal_set,
    load_approved_loeo_ar_set,
    prepare_loeo_ar_proposal_set,
)
from hqrc_v3.loeo_ablation import fit_loeo_ablation
from hqrc_v3.loeo_primary import fit_loeo_primary

Progress = Callable[[str], None]


class PaperPipelineError(ValueError):
    """Raised when the requested manuscript pipeline scope is invalid."""


@dataclass(frozen=True, slots=True)
class PipelineContextResult:
    """Materialized state for one model/feature/seed HQRC context."""

    context: EventResidualContext
    loeo_dir: Path
    ar_dir: Path
    proposal_set_sha256: str
    status: str
    primary_dir: Path | None = None
    variant_output_dirs: tuple[tuple[str, Path], ...] = ()
    sampler_fit_count: int | None = None
    reused: bool | None = None


def _emit(progress: Progress | None, message: str) -> None:
    if progress is not None:
        progress(message)


def _canonical_contexts(
    source: ValidatedCorrectionSource,
    *,
    models: Iterable[str],
    feature_sets: Iterable[str],
) -> tuple[EventResidualContext, ...]:
    requested_models = tuple(models)
    requested_features = tuple(feature_sets)
    if (
        not requested_models
        or len(set(requested_models)) != len(requested_models)
        or any(model not in MODEL_NAMES for model in requested_models)
    ):
        raise PaperPipelineError("models must be a non-empty unique subset of manuscript models")
    if (
        not requested_features
        or len(set(requested_features)) != len(requested_features)
        or any(feature_set not in {"B0", "B1"} for feature_set in requested_features)
    ):
        raise PaperPipelineError("feature sets must be a non-empty unique subset of B0/B1")

    available = {
        (context.model, context.feature_set): context for context in source.available_contexts
    }
    selected: list[EventResidualContext] = []
    # The explicit manuscript order is part of execution provenance.  Never
    # depend on a filesystem or dictionary iteration order for model sequencing.
    for model in MODEL_NAMES:
        if model not in requested_models:
            continue
        for feature_set in ("B0", "B1"):
            if feature_set not in requested_features:
                continue
            try:
                selected.append(available[(model, feature_set)])
            except KeyError as error:
                raise PaperPipelineError(
                    f"validated residual source lacks {model}/{feature_set}"
                ) from error
    return tuple(selected)


def run_paper_loeo_pipeline(
    *,
    source_run_dir: Path,
    config_path: Path,
    output_root: Path,
    models: Iterable[str],
    feature_sets: Iterable[str],
    variants: Iterable[str] = ("H1", "H2", "H3"),
    root_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    init: str | None = None,
    target_accept: float | None = None,
    approve_derived_ar: bool = False,
    progress: Progress | None = None,
) -> tuple[PipelineContextResult, ...]:
    """Run/reuse selected H1--H3 contexts sequentially from one validated source.

    With ``approve_derived_ar=False`` this intentionally stops after publishing
    ACF/PACF plots and proposal JSON for every selected context.  Passing the
    explicit opt-in is the reproducible confirmation that those data-derived
    proposal digests are the values to freeze; only then are NUTS and LOEO
    aggregate products executed.
    """

    if profile not in {"paper", "smoke"}:
        raise PaperPipelineError("profile must be paper or smoke")
    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise PaperPipelineError("root seed must be a non-negative integer")
    selected_variants = tuple(variants)
    if (
        not selected_variants
        or len(set(selected_variants)) != len(selected_variants)
        or any(variant not in {"H1", "H2", "H3"} for variant in selected_variants)
        or tuple(variant for variant in ("H1", "H2", "H3") if variant in selected_variants)
        != selected_variants
    ):
        raise PaperPipelineError("variants must be a unique manuscript-ordered subset of H1/H2/H3")
    output = Path(output_root).expanduser().resolve()
    if not output.is_absolute():  # pragma: no cover - resolve guarantees this on supported hosts
        raise PaperPipelineError("output root must resolve to an absolute path")

    _emit(progress, "[source] validating immutable baseline and standardized residuals")
    source = validate_correction_source(
        run_dir=Path(source_run_dir).expanduser().resolve(),
        config_path=Path(config_path).expanduser().resolve(),
        profile=profile,
    )
    contexts = _canonical_contexts(source, models=models, feature_sets=feature_sets)
    results: list[PipelineContextResult] = []
    for index, context in enumerate(contexts, start=1):
        label = f"{context.model}/{context.feature_set}/seed-{context.seed}"
        context_root = output / context.model / context.feature_set / f"seed-{context.seed}"
        loeo_dir = context_root / "loeo"
        ar_dir = context_root / "loeo-ar"
        _emit(progress, f"[{index}/{len(contexts)} {label}] LOEO universe")
        publication = publish_loeo_universe(source, context, output_dir=loeo_dir)
        _emit(progress, f"[{index}/{len(contexts)} {label}] ACF/PACF and AR prior proposal")
        proposal = prepare_loeo_ar_proposal_set(source, publication, output_dir=ar_dir)

        if not approve_derived_ar:
            results.append(
                PipelineContextResult(
                    context=context,
                    loeo_dir=loeo_dir,
                    ar_dir=ar_dir,
                    proposal_set_sha256=proposal.proposal_set_sha256,
                    status="AR_REVIEW_REQUIRED",
                )
            )
            _emit(
                progress,
                f"[{index}/{len(contexts)} {label}] AR proposal ready; H1--H3 sampling skipped",
            )
            continue

        _emit(progress, f"[{index}/{len(contexts)} {label}] freezing approved AR proposal")
        approve_loeo_ar_proposal_set(
            source,
            publication,
            output_dir=ar_dir,
            confirm_proposal_set_sha256=proposal.proposal_set_sha256,
        )
        approved = load_approved_loeo_ar_set(source, publication, output_dir=ar_dir)
        variant_outputs: list[tuple[str, Path]] = []
        fit_count = 0
        reused = True
        primary = None
        for variant in selected_variants:
            _emit(
                progress,
                f"[{index}/{len(contexts)} {label}] {variant} NUTS LOEO (10 folds, sequential)",
            )
            if variant in {"H1", "H2"}:
                ablation = fit_loeo_ablation(
                    source,
                    publication,
                    approved,
                    variant=variant,
                    held_out_occurrence_ids=publication.occurrence_ids,
                    root_seed=root_seed,
                    profile=profile,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    init=init,
                    target_accept=target_accept,
                    output_root=output,
                )
                variant_outputs.append((variant, ablation.output_dir))
                fit_count += ablation.sampler_fit_count
                reused = reused and ablation.reused
            else:
                primary = fit_loeo_primary(
                    source,
                    publication,
                    approved,
                    held_out_occurrence_ids=publication.occurrence_ids,
                    root_seed=root_seed,
                    profile=profile,
                    draws=draws,
                    tune=tune,
                    chains=chains,
                    cores=cores,
                    init=init,
                    target_accept=target_accept,
                    output_root=output,
                )
                variant_outputs.append((variant, primary.output_dir))
                fit_count += primary.sampler_fit_count
                reused = reused and primary.reused
        results.append(
            PipelineContextResult(
                context=context,
                loeo_dir=loeo_dir,
                ar_dir=ar_dir,
                proposal_set_sha256=proposal.proposal_set_sha256,
                status="COMPLETE",
                primary_dir=None if primary is None else primary.output_dir,
                variant_output_dirs=tuple(variant_outputs),
                sampler_fit_count=fit_count,
                reused=reused,
            )
        )
        _emit(progress, f"[{index}/{len(contexts)} {label}] complete")
    return tuple(results)


__all__ = [
    "PaperPipelineError",
    "PipelineContextResult",
    "run_paper_loeo_pipeline",
]
