"""Correction variants and explicit LOEO evaluation orchestration."""

from hqrc_v3.correction_stage import (
    CausalCorrectionError,
    CausalCorrectionInputs,
    CausalCorrectionProducts,
    CausalCorrectionResult,
    fit_causal_2024_correction,
    prepare_causal_correction_inputs,
)
from hqrc_v3.corrections.similar_day import SimilarDayError, same_holiday_profile
from hqrc_v3.corrections.variants import (
    CorrectionContractError,
    NonEventBlockPool,
    PSISLOOSummary,
    VariantContext,
    build_non_event_block_pool,
    psis_loo_summary,
    run_event_loeo,
    run_variant,
)

__all__ = [
    "CorrectionContractError",
    "CausalCorrectionError",
    "CausalCorrectionInputs",
    "CausalCorrectionProducts",
    "CausalCorrectionResult",
    "NonEventBlockPool",
    "PSISLOOSummary",
    "SimilarDayError",
    "VariantContext",
    "build_non_event_block_pool",
    "fit_causal_2024_correction",
    "psis_loo_summary",
    "prepare_causal_correction_inputs",
    "run_event_loeo",
    "run_variant",
    "same_holiday_profile",
]
