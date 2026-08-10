"""Correction variants and explicit LOEO evaluation orchestration."""

from hqrc_v3.corrections.similar_day import SimilarDayError, same_holiday_profile
from hqrc_v3.corrections.variants import (
    CorrectionContractError,
    PSISLOOSummary,
    VariantContext,
    psis_loo_summary,
    run_event_loeo,
    run_variant,
)

__all__ = [
    "CorrectionContractError",
    "PSISLOOSummary",
    "SimilarDayError",
    "VariantContext",
    "psis_loo_summary",
    "run_event_loeo",
    "run_variant",
    "same_holiday_profile",
]
