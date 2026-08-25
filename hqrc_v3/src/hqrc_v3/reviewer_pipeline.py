"""Sequential, resumable orchestration for the AR-free HQT reviewer experiment."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from hqrc_v3.baselines.config import MODEL_NAMES, PaperBaselineConfig
from hqrc_v3.baselines.paper import run_paper_final_stage, run_paper_oof_stage
from hqrc_v3.contracts import ForecastMatrix
from hqrc_v3.evaluation.reviewer import ReviewerReportInputs, build_reviewer_report
from hqrc_v3.legacy_hqt_causal import run_legacy_hqt_causal_2024
from hqrc_v3.legacy_hqt_loeo import run_legacy_hqt_loeo
from hqrc_v3.residual_stage import prepare_standardized_residual_artifact

REVIEWER_FEATURE_SUITE = ("B0", "B1W")
_HQT_FEATURE_SUITE = ("B1W",)
_SAMPLER_INITIALIZATIONS = frozenset(("adapt_diag", "jitter+adapt_diag"))


class ReviewerPipelineError(ValueError):
    """Raised before execution when the reviewer command scope is invalid."""


@dataclass(frozen=True, slots=True)
class ReviewerPipelineOptions:
    """Validated source inputs and execution policy for one reviewer run."""

    matrices: Mapping[str, ForecastMatrix]
    baseline_config: PaperBaselineConfig
    artifact_hashes: Mapping[str, str]
    data_path: Path
    config_path: Path
    model_config_path: Path
    event_registry_path: Path
    holiday_calendar_path: Path
    temporary_holiday_availability_path: Path
    source_run_dir: Path
    cache_dir: Path
    output_root: Path
    baseline_seed: int
    root_seed: int
    models: tuple[str, ...]
    profile: Literal["paper", "smoke"]
    draws: int | None = None
    tune: int | None = None
    chains: int | None = None
    cores: int | None = None
    init: str = "adapt_diag"
    target_accept: float | None = None
    smoke_boosting_rounds: int | None = None


@dataclass(frozen=True, slots=True)
class ReviewerPipelineResult:
    """Reviewer namespaces and aggregate fit/reuse accounting."""

    source_run_dir: Path
    loeo_root: Path
    causal_root: Path
    paper_root: Path
    baseline_fit_count: int
    baseline_cache_hit_count: int
    hqt_fit_count: int


def _positive_integer(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value > 0


def _validate_options(options: ReviewerPipelineOptions) -> None:
    if not isinstance(options, ReviewerPipelineOptions):
        raise TypeError("options must be ReviewerPipelineOptions")
    if set(options.matrices) != set(REVIEWER_FEATURE_SUITE):
        raise ReviewerPipelineError("reviewer baseline matrices must be exactly B0 and B1W")
    if (
        not isinstance(options.models, tuple)
        or not options.models
        or len(set(options.models)) != len(options.models)
        or any(model not in MODEL_NAMES for model in options.models)
        or options.models != tuple(model for model in MODEL_NAMES if model in options.models)
    ):
        raise ReviewerPipelineError("reviewer models must be a canonical manuscript-model subset")
    if isinstance(options.baseline_seed, bool) or not isinstance(options.baseline_seed, int):
        raise ReviewerPipelineError("reviewer baseline seed must be an integer")
    if isinstance(options.root_seed, bool) or not isinstance(options.root_seed, int):
        raise ReviewerPipelineError("reviewer root seed must be an integer")
    if options.init not in _SAMPLER_INITIALIZATIONS:
        raise ReviewerPipelineError("reviewer sampler initialization is unsupported")
    if options.target_accept is not None and (
        isinstance(options.target_accept, bool)
        or not isinstance(options.target_accept, (int, float))
        or not math.isfinite(float(options.target_accept))
        or not 0 < float(options.target_accept) < 1
    ):
        raise ReviewerPipelineError("reviewer target_accept must be between zero and one")
    if options.cores is not None and (
        not _positive_integer(options.cores)
        or (options.chains is not None and options.cores > options.chains)
    ):
        raise ReviewerPipelineError("reviewer cores must be positive and no greater than chains")

    if options.profile == "paper":
        resolved_draws = 1_000 if options.draws is None else options.draws
        resolved_tune = 1_000 if options.tune is None else options.tune
        resolved_chains = 4 if options.chains is None else options.chains
        resolved_target = 0.99 if options.target_accept is None else options.target_accept
        if options.models != MODEL_NAMES:
            raise ReviewerPipelineError(
                "paper reviewer profile requires all five manuscript models"
            )
        if (
            not _positive_integer(resolved_draws)
            or not _positive_integer(resolved_tune)
            or resolved_draws < 1_000
            or resolved_tune < 1_000
            or resolved_chains < 4
            or resolved_target != 0.99
        ):
            raise ReviewerPipelineError(
                "paper reviewer profile requires at least 4 chains, at least 1000 tune/draws, "
                "and target_accept=0.99"
            )
        if options.smoke_boosting_rounds is not None:
            raise ReviewerPipelineError(
                "smoke boosting rounds cannot alter the paper reviewer profile"
            )
    elif options.profile == "smoke":
        if not all(
            _positive_integer(value) for value in (options.draws, options.tune, options.chains)
        ):
            raise ReviewerPipelineError(
                "smoke reviewer profile requires explicit positive draws, tune, and chains"
            )
        if not _positive_integer(options.smoke_boosting_rounds):
            raise ReviewerPipelineError(
                "smoke reviewer profile requires explicit positive smoke boosting rounds"
            )
    else:
        raise ReviewerPipelineError("reviewer profile must be paper or smoke")

    source = Path(options.source_run_dir).expanduser().resolve()
    output = Path(options.output_root).expanduser().resolve()
    namespaces = (source, output / "retrospective-loeo", output / "causal-2024", output / "paper")
    if len(set(namespaces)) != len(namespaces):
        raise ReviewerPipelineError(
            "reviewer source, LOEO, causal, and paper namespaces must be separate"
        )


def _baseline_stage_inputs(
    options: ReviewerPipelineOptions,
    *,
    source_run_dir: Path,
    cache_dir: Path,
) -> dict[str, object]:
    return {
        "matrices": options.matrices,
        "config": options.baseline_config,
        "run_dir": source_run_dir,
        "cache_dir": cache_dir,
        "artifact_hashes": options.artifact_hashes,
        "classical_seed": options.baseline_seed,
        "models": options.models,
        "feature_sets": REVIEWER_FEATURE_SUITE,
        "profile": options.profile,
        "smoke_boosting_rounds": options.smoke_boosting_rounds,
    }


def _emit(progress: Callable[[str], None] | None, message: str) -> None:
    if progress is not None:
        progress(message)


def run_hqt_reviewer_pipeline(
    options: ReviewerPipelineOptions,
    *,
    progress: Callable[[str], None] | None = None,
) -> ReviewerPipelineResult:
    """Run or strictly reuse the six reviewer stages in their fixed order."""

    _validate_options(options)
    source_run_dir = Path(options.source_run_dir).expanduser().resolve()
    cache_dir = Path(options.cache_dir).expanduser().resolve()
    output_root = Path(options.output_root).expanduser().resolve()
    loeo_root = output_root / "retrospective-loeo"
    causal_root = output_root / "causal-2024"
    paper_root = output_root / "paper"
    baseline_inputs = _baseline_stage_inputs(
        options,
        source_run_dir=source_run_dir,
        cache_dir=cache_dir,
    )

    _emit(progress, "[baseline] reviewer OOF: B0 then B1W")
    oof = run_paper_oof_stage(**baseline_inputs, oof_years=None)

    _emit(progress, "[baseline] reviewer final-2024")
    final = run_paper_final_stage(**baseline_inputs)

    _emit(progress, "[residuals] reviewer B0/B1W publication")
    prepare_standardized_residual_artifact(
        run_dir=source_run_dir,
        data_path=Path(options.data_path).expanduser().resolve(),
        config_path=Path(options.config_path).expanduser().resolve(),
        model_config_path=Path(options.model_config_path).expanduser().resolve(),
        event_registry_path=Path(options.event_registry_path).expanduser().resolve(),
        holiday_calendar_path=Path(options.holiday_calendar_path).expanduser().resolve(),
        temporary_holiday_availability_path=Path(options.temporary_holiday_availability_path)
        .expanduser()
        .resolve(),
        profile=options.profile,
    )

    loeo = run_legacy_hqt_loeo(
        source_run_dir=source_run_dir,
        config_path=Path(options.config_path).expanduser().resolve(),
        output_root=loeo_root,
        models=options.models,
        feature_sets=_HQT_FEATURE_SUITE,
        held_out_occurrence_ids=None,
        root_seed=options.root_seed,
        profile=options.profile,
        draws=options.draws,
        tune=options.tune,
        chains=options.chains,
        cores=options.cores,
        init=options.init,
        target_accept=options.target_accept,
    )
    for item in loeo:
        _emit(
            progress,
            f"[HQT] {item.context.model}/B1W retrospective LOEO complete "
            f"fits={item.sampler_fit_count} reused={item.reused_fold_count}",
        )

    causal = run_legacy_hqt_causal_2024(
        source_run_dir=source_run_dir,
        config_path=Path(options.config_path).expanduser().resolve(),
        output_root=causal_root,
        models=options.models,
        feature_set="B1W",
        root_seed=options.root_seed,
        profile=options.profile,
        draws=options.draws,
        tune=options.tune,
        chains=options.chains,
        cores=options.cores,
        init=options.init,
        target_accept=options.target_accept,
    )
    for item in causal:
        _emit(
            progress,
            f"[HQT] {item.context.model}/B1W causal-2024 complete "
            f"fits={item.sampler_fit_count} reused={int(item.reused)}",
        )

    report = build_reviewer_report(
        ReviewerReportInputs(
            final_predictions_path=Path(final.point_path),
            loeo_context_dirs=tuple(Path(item.output_dir) for item in loeo),
            causal_context_dirs=tuple(Path(item.output_dir) for item in causal),
            root_seed=options.root_seed,
        ),
        output_root=paper_root,
    )
    _emit(progress, "[report] reviewer tables and figures complete")

    return ReviewerPipelineResult(
        source_run_dir=source_run_dir,
        loeo_root=loeo_root,
        causal_root=causal_root,
        paper_root=Path(report.output_root),
        baseline_fit_count=int(oof.fit_count) + int(final.fit_count),
        baseline_cache_hit_count=int(oof.cache_hit_count) + int(final.cache_hit_count),
        hqt_fit_count=sum(int(item.sampler_fit_count) for item in (*loeo, *causal)),
    )


__all__ = [
    "REVIEWER_FEATURE_SUITE",
    "ReviewerPipelineError",
    "ReviewerPipelineOptions",
    "ReviewerPipelineResult",
    "run_hqt_reviewer_pipeline",
]
