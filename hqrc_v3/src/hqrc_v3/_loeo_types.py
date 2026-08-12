"""Shared typed contracts for the one-fold LOEO implementation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.diagnostics.ar import ApprovedARCalibration
from hqrc_v3.diagnostics.loeo import LOEOFold, LOEOHeldOut, LOEOPublication
from hqrc_v3.diagnostics.loeo_ar import ApprovedLOEOARSet


class LOEOFoldError(ValueError):
    """Raised when a one-fold LOEO input or immutable result cannot be trusted."""


@dataclass(frozen=True, slots=True)
class LOEOFoldInputs:
    """Fully source-revalidated inputs for one held-out event and nine-event fit."""

    source: ValidatedCorrectionSource
    publication: LOEOPublication
    approved_set: ApprovedLOEOARSet
    approved: ApprovedARCalibration
    held_out_occurrence_id: str
    training_fold: LOEOFold
    held_out: LOEOHeldOut
    hqrc_data: HQRCData
    sigma_eval: float
    restriction: int
    causal: bool = False


@dataclass(frozen=True, slots=True)
class LOEOFoldProducts:
    """Fold-local hourly products, metrics, and compact posterior diagnostics."""

    hourly_predictions: pl.DataFrame
    metrics: pl.DataFrame
    posterior_summary: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class LOEOFoldResult:
    """One completed or strictly reused immutable fold publication."""

    output_dir: Path
    manifest_path: Path
    posterior_path: Path
    hourly_predictions_path: Path
    metrics_path: Path
    posterior_summary_path: Path
    sampler_fit_count: int
    reused: bool


__all__ = ["LOEOFoldError", "LOEOFoldInputs", "LOEOFoldProducts", "LOEOFoldResult"]
