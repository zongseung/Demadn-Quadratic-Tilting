"""Typed contracts for one immutable primary-H3 LOEO matrix."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from hqrc_v3._loeo_types import LOEOFoldError


class LOEOPrimaryError(LOEOFoldError):
    """Raised when a primary-matrix input or publication cannot be trusted."""


@dataclass(frozen=True, slots=True)
class LOEOPrimaryProducts:
    """Deterministic aggregate products without duplicated posterior draws."""

    hourly_predictions: pl.DataFrame
    per_event_metrics: pl.DataFrame
    aggregate_metrics: pl.DataFrame
    posterior_summaries: pl.DataFrame
    training_psis_loo: pl.DataFrame


@dataclass(frozen=True, slots=True)
class LOEOPrimaryResult:
    """One completed or strictly reused primary-H3 matrix publication."""

    output_dir: Path
    manifest_path: Path
    hourly_predictions_path: Path
    per_event_metrics_path: Path
    aggregate_metrics_path: Path
    posterior_summaries_path: Path
    training_psis_loo_path: Path
    selected_occurrence_ids: tuple[str, ...]
    fold_sampler_fit_counts: Mapping[str, int]
    sampler_fit_count: int
    reused: bool


__all__ = ["LOEOPrimaryError", "LOEOPrimaryProducts", "LOEOPrimaryResult"]
