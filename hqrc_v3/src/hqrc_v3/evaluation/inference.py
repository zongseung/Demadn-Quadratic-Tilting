"""Occurrence-unit inference with deterministic bootstrap and Holm adjustment."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import stats


class InferenceContractError(ValueError):
    """Raised when event-level comparisons do not form a valid paired analysis."""


@dataclass(frozen=True)
class EventBootstrapResult:
    samples: np.ndarray
    lower: float
    median: float
    upper: float
    resampled_units: str
    seed: int


@dataclass(frozen=True)
class WilcoxonEventResult:
    statistic: float
    p_value: float
    n_events: int


@dataclass(frozen=True)
class HACDMResult:
    statistic: float
    p_value: float
    bandwidth: int
    event_block_seed: int
    n_hours: int
    bootstrap_draws: int
    resampled_units: str


def _finite_vector(value: object, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=float)
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise InferenceContractError(f"{name} must be a finite non-empty vector")
    return result


def bootstrap_event_median(
    event_values: Mapping[str, np.ndarray], *, draws: int, seed: int
) -> EventBootstrapResult:
    """Bootstrap occurrence ids, keeping every hourly observation in each selected event."""

    if not isinstance(event_values, Mapping) or not event_values:
        raise InferenceContractError("event_values must be a non-empty occurrence mapping")
    identifiers = tuple(event_values)
    if any(not isinstance(item, str) or not item.strip() for item in identifiers):
        raise InferenceContractError("event identifiers must be nonblank strings")
    values = [_finite_vector(event_values[item], name=f"event {item}") for item in identifiers]
    if not isinstance(draws, int) or isinstance(draws, bool) or draws <= 0:
        raise InferenceContractError("draws must be a positive integer")
    rng = np.random.default_rng(seed)
    # Each occurrence gets one summary before resampling; hourly rows are never
    # treated as independent bootstrap units.
    event_summary = np.asarray([value.mean() for value in values])
    index = rng.integers(0, len(identifiers), size=(draws, len(identifiers)))
    samples = np.median(event_summary[index], axis=1)
    return EventBootstrapResult(
        samples=samples,
        lower=float(np.quantile(samples, 0.025)),
        median=float(np.median(event_summary)),
        upper=float(np.quantile(samples, 0.975)),
        resampled_units="event",
        seed=int(seed),
    )


def wilcoxon_event_test(reference: np.ndarray, candidate: np.ndarray) -> WilcoxonEventResult:
    """Apply paired Wilcoxon signed-rank to occurrence summaries, never hours."""

    left = _finite_vector(reference, name="reference")
    right = _finite_vector(candidate, name="candidate")
    if left.shape != right.shape:
        raise InferenceContractError("paired event summaries must have equal length")
    if np.allclose(left, right, rtol=0.0, atol=0.0):
        return WilcoxonEventResult(0.0, 1.0, int(left.size))
    result = stats.wilcoxon(left, right, alternative="two-sided", method="auto")
    return WilcoxonEventResult(float(result.statistic), float(result.pvalue), int(left.size))


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm-adjust p-values within precisely the supplied comparison family."""

    if not isinstance(p_values, Mapping) or not p_values:
        raise InferenceContractError("p_values must be a non-empty comparison family")
    items = list(p_values.items())
    if any(
        not isinstance(name, str)
        or not name.strip()
        or not math.isfinite(value)
        or not 0 <= value <= 1
        for name, value in items
    ):
        raise InferenceContractError("p-values must be finite values in [0, 1] with nonblank names")
    ordered = sorted(items, key=lambda item: (item[1], item[0]))
    adjusted: dict[str, float] = {}
    running = 0.0
    total = len(ordered)
    for rank, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - rank) * float(value)))
        adjusted[name] = running
    return {name: adjusted[name] for name in p_values}


def hac_dm_test(
    reference_loss: np.ndarray,
    candidate_loss: np.ndarray,
    *,
    event_ids: Sequence[str],
    bandwidth: int,
    event_block_seed: int,
    bootstrap_draws: int = 2_000,
) -> HACDMResult:
    """Secondary HAC-DM with an actual deterministic whole-event block bootstrap p-value."""

    reference = _finite_vector(reference_loss, name="reference_loss")
    candidate = _finite_vector(candidate_loss, name="candidate_loss")
    if reference.shape != candidate.shape or len(event_ids) != reference.size:
        raise InferenceContractError("losses and event_ids must be aligned")
    if not isinstance(bandwidth, int) or isinstance(bandwidth, bool) or bandwidth < 0:
        raise InferenceContractError("bandwidth must be a non-negative integer")
    if not isinstance(event_block_seed, int) or isinstance(event_block_seed, bool):
        raise InferenceContractError("event_block_seed must be an integer")
    if (
        not isinstance(bootstrap_draws, int)
        or isinstance(bootstrap_draws, bool)
        or bootstrap_draws <= 0
    ):
        raise InferenceContractError("bootstrap_draws must be a positive integer")
    if any(not isinstance(event_id, str) or not event_id.strip() for event_id in event_ids):
        raise InferenceContractError("event_ids must be nonblank strings")
    differential = reference - candidate
    centered = differential - differential.mean()
    long_run = float(np.mean(centered**2))
    for lag in range(1, min(bandwidth, differential.size - 1) + 1):
        covariance = float(np.mean(centered[lag:] * centered[:-lag]))
        long_run += 2.0 * (1.0 - lag / (bandwidth + 1.0)) * covariance
    if long_run <= np.finfo(float).eps:
        statistic = (
            0.0
            if abs(float(differential.mean())) <= np.finfo(float).eps
            else math.copysign(math.inf, differential.mean())
        )
        p_value = 1.0 if statistic == 0.0 else 0.0
    else:
        statistic = float(math.sqrt(differential.size) * differential.mean() / math.sqrt(long_run))
        blocks = [
            centered[np.asarray(event_ids) == event_id] for event_id in dict.fromkeys(event_ids)
        ]
        rng = np.random.default_rng(event_block_seed)
        bootstrap = np.empty(bootstrap_draws)
        for draw in range(bootstrap_draws):
            chosen = rng.integers(0, len(blocks), size=len(blocks))
            resampled = np.concatenate([blocks[index] for index in chosen])
            bootstrap[draw] = math.sqrt(resampled.size) * resampled.mean() / math.sqrt(long_run)
        p_value = float(
            (1 + np.count_nonzero(np.abs(bootstrap) >= abs(statistic))) / (bootstrap_draws + 1)
        )
    return HACDMResult(
        statistic,
        p_value,
        bandwidth,
        event_block_seed,
        int(differential.size),
        bootstrap_draws,
        "event",
    )
