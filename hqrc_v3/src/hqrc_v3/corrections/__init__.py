"""Correction variants and explicit LOEO evaluation orchestration."""

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
    "NonEventBlockPool",
    "PSISLOOSummary",
    "SimilarDayError",
    "VariantContext",
    "build_non_event_block_pool",
    "psis_loo_summary",
    "run_event_loeo",
    "run_variant",
    "same_holiday_profile",
]
