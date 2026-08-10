"""Posterior predictive utilities for event-reset HQRC corrections."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

Pooling = Literal["complete", "partial", "none"]


class PredictiveShapeError(ValueError):
    """Raised when predictive inputs cannot describe finite joint trajectories."""


def _finite_array(value: object, *, name: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if (ndim is not None and array.ndim != ndim) or array.size == 0 or not np.isfinite(array).all():
        raise PredictiveShapeError(f"{name} must be a finite non-empty {ndim or 'array'}")
    return array


def corrected_predictive_draws(
    baseline: np.ndarray, sigma_n: float, q: np.ndarray, e: np.ndarray
) -> np.ndarray:
    """Return ``baseline + sigma_N * (q + e)`` without a second bootstrap noise term."""

    baseline_values = _finite_array(baseline, name="baseline", ndim=1)
    q_values = _finite_array(q, name="q", ndim=2)
    e_values = _finite_array(e, name="e", ndim=2)
    if q_values.shape != e_values.shape or q_values.shape[1] != baseline_values.size:
        raise PredictiveShapeError("q/e draws must match baseline timestamps")
    if (
        not isinstance(sigma_n, (int, float, np.number))
        or not math.isfinite(sigma_n)
        or sigma_n <= 0
    ):
        raise PredictiveShapeError("sigma_n must be finite and positive")
    return baseline_values[None, :] + float(sigma_n) * (q_values + e_values)


def baseline_bootstrap_draws(non_event_blocks: np.ndarray, *, draws: int, seed: int) -> np.ndarray:
    """Resample complete non-event residual horizons for the uncorrected H0 baseline."""

    blocks = _finite_array(non_event_blocks, name="non_event_blocks", ndim=2)
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws <= 0:
        raise PredictiveShapeError("draws must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise PredictiveShapeError("seed must be an integer")
    index = np.random.default_rng(int(seed)).integers(0, blocks.shape[0], size=int(draws))
    return blocks[index].copy()


def _posterior_array(posterior: Mapping[str, object], name: str) -> np.ndarray:
    if name not in posterior:
        raise PredictiveShapeError(f"posterior is missing {name!r}")
    values = getattr(posterior[name], "values", posterior[name])
    return _finite_array(values, name=f"posterior {name}")


def _holiday_index(holiday_type: int | str) -> int:
    if isinstance(holiday_type, (int, np.integer)) and not isinstance(holiday_type, bool):
        if int(holiday_type) in (0, 1):
            return int(holiday_type)
    if isinstance(holiday_type, str):
        normalized = holiday_type.strip().lower()
        if normalized in {"seollal", "lunar-new-year", "lunar_new_year", "0"}:
            return 0
        if normalized in {"chuseok", "harvest", "1"}:
            return 1
    raise PredictiveShapeError("holiday_type must identify Seollal (0) or Chuseok (1)")


def _draw_rows(values: np.ndarray, *, trailing_dimensions: int, draws: int, rng) -> np.ndarray:
    if values.ndim < trailing_dimensions + 1:
        raise PredictiveShapeError("posterior sample array has too few dimensions")
    flattened = values.reshape((-1,) + values.shape[-trailing_dimensions:])
    return flattened[rng.integers(0, flattened.shape[0], size=draws)]


def draw_new_event_correction(
    posterior: Mapping[str, object],
    *,
    holiday_type: int | str,
    pooling: Pooling,
    draws: int,
    seed: int,
    restriction: int = 0,
    training_occurrence_indices: Sequence[int] | None = None,
) -> np.ndarray:
    """Draw new-event quadratic coefficients from the requested pooling population.

    No-pooling intentionally samples *only* coefficient rows explicitly supplied as
    training occurrences.  It therefore cannot reuse a held-out occurrence's fitted
    coefficient when predicting that occurrence.
    """

    if pooling not in {"complete", "partial", "none"}:
        raise PredictiveShapeError("pooling must be complete, partial, or none")
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws <= 0:
        raise PredictiveShapeError("draws must be a positive integer")
    if restriction not in (0, 1):
        raise PredictiveShapeError("restriction must be binary")
    holiday = _holiday_index(holiday_type)
    rng = np.random.default_rng(int(seed))

    if pooling == "complete":
        beta_type = _posterior_array(posterior, "beta_type")
        rows = _draw_rows(beta_type, trailing_dimensions=2, draws=int(draws), rng=rng)
        if rows.shape[1] <= holiday:
            raise PredictiveShapeError("beta_type does not contain the requested holiday type")
        return rows[:, holiday, :]

    if pooling == "none":
        beta = _posterior_array(posterior, "beta")
        rows = _draw_rows(beta, trailing_dimensions=2, draws=int(draws), rng=rng)
        if training_occurrence_indices is None:
            raise PredictiveShapeError("no-pooling prediction requires training occurrence indices")
        allowed = np.asarray(training_occurrence_indices, dtype=int)
        if (
            allowed.ndim != 1
            or allowed.size == 0
            or (allowed < 0).any()
            or (allowed >= rows.shape[1]).any()
        ):
            raise PredictiveShapeError(
                "training occurrence indices must be valid non-empty indices"
            )
        components = rng.choice(allowed, size=int(draws), replace=True)
        return rows[np.arange(int(draws)), components, :]

    mu = _posterior_array(posterior, "mu")
    rows = _draw_rows(mu, trailing_dimensions=2, draws=int(draws), rng=rng)
    if rows.shape[1] <= holiday:
        raise PredictiveShapeError("mu does not contain the requested holiday type")
    location = rows[:, holiday, :]
    if "delta" in posterior:
        delta = _draw_rows(
            _posterior_array(posterior, "delta"), trailing_dimensions=2, draws=int(draws), rng=rng
        )
        if delta.shape[1] <= holiday or delta.shape[2] != location.shape[1]:
            raise PredictiveShapeError("delta shape is incompatible with mu")
        location = location + int(restriction) * delta[:, holiday, :]
    if "between_scale" in posterior:
        scale = _draw_rows(
            _posterior_array(posterior, "between_scale"),
            trailing_dimensions=2,
            draws=int(draws),
            rng=rng,
        )
        if scale.shape[1] <= holiday or scale.shape[2] != location.shape[1]:
            raise PredictiveShapeError("between_scale shape is incompatible with mu")
        if (scale < 0).any():
            raise PredictiveShapeError("between_scale must be non-negative")
        location = location + rng.normal(size=location.shape) * scale[:, holiday, :]
    return location


def simulate_stationary_ar1(
    *, phi: np.ndarray | float, sigma: np.ndarray | float, horizon: int, seed: int
) -> np.ndarray:
    """Simulate one joint stationary AR(1) trajectory per posterior draw.

    Every call begins with its own stationary initial state; callers invoke it once
    per event, so no innovation is ever carried across event boundaries.
    """

    phi_values = _finite_array(phi, name="phi").reshape(-1)
    sigma_values = _finite_array(sigma, name="sigma").reshape(-1)
    if phi_values.size not in (1, sigma_values.size) and sigma_values.size != 1:
        raise PredictiveShapeError("phi and sigma must have matching draw counts")
    count = max(phi_values.size, sigma_values.size)
    phi_values = np.broadcast_to(phi_values, (count,))
    sigma_values = np.broadcast_to(sigma_values, (count,))
    if (np.abs(phi_values) >= 1).any() or (sigma_values < 0).any():
        raise PredictiveShapeError("phi must be inside (-1, 1) and sigma non-negative")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
        raise PredictiveShapeError("horizon must be a positive integer")
    rng = np.random.default_rng(int(seed))
    result = np.empty((count, int(horizon)), dtype=float)
    stationary_sd = sigma_values / np.sqrt(1.0 - phi_values**2)
    result[:, 0] = rng.normal(scale=stationary_sd)
    for index in range(1, int(horizon)):
        result[:, index] = phi_values * result[:, index - 1] + rng.normal(scale=sigma_values)
    return result
