"""Joint posterior selection and event-reset posterior predictive simulators."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

Pooling = Literal["complete", "partial", "none"]
Innovation = Literal["normal_ar1", "student_t_ar1", "normal_ar2"]


class PredictiveShapeError(ValueError):
    """Raised when a predictive input is not a finite, jointly sampled trajectory."""


def _finite(value: object, name: str) -> np.ndarray:
    array = np.asarray(getattr(value, "values", value), dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        raise PredictiveShapeError(f"{name} must be finite and non-empty")
    return array


def _sample_matrix(value: object, *, name: str, trailing: int) -> np.ndarray:
    array = _finite(value, f"posterior {name}")
    if array.ndim < trailing + 1:
        raise PredictiveShapeError(f"posterior {name} has too few dimensions")
    if trailing == 0:
        return array.reshape(-1)
    return array.reshape((-1,) + array.shape[-trailing:])


def select_posterior_indices(
    posterior: Mapping[str, object], *, specifications: Mapping[str, int], draws: int, seed: int
) -> np.ndarray:
    """Choose one posterior row per predictive draw and validate every required variable.

    `specifications` maps each variable to its number of non-sample trailing
    dimensions.  All variables must have exactly the same flattened chain/draw
    count; this makes mixing independent posterior rows impossible.
    """

    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws <= 0:
        raise PredictiveShapeError("draws must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise PredictiveShapeError("seed must be an integer")
    if not specifications:
        raise PredictiveShapeError("at least one posterior variable is required")
    counts: list[int] = []
    for name, trailing in specifications.items():
        if name not in posterior or not isinstance(trailing, int) or trailing < 0:
            raise PredictiveShapeError(f"posterior is missing required variable {name!r}")
        counts.append(_sample_matrix(posterior[name], name=name, trailing=trailing).shape[0])
    if len(set(counts)) != 1:
        raise PredictiveShapeError("posterior variables must have identical sample counts")
    return np.random.default_rng(int(seed)).integers(0, counts[0], size=int(draws))


def posterior_values(
    posterior: Mapping[str, object], name: str, *, trailing: int, indices: np.ndarray
) -> np.ndarray:
    """Return values at the already-selected shared posterior sample indices."""

    values = _sample_matrix(posterior[name], name=name, trailing=trailing)
    chosen = np.asarray(indices)
    if chosen.ndim != 1 or chosen.size == 0 or not np.issubdtype(chosen.dtype, np.integer):
        raise PredictiveShapeError("posterior indices must be a non-empty integer vector")
    if (chosen < 0).any() or (chosen >= values.shape[0]).any():
        raise PredictiveShapeError("posterior index is outside the available sample range")
    return values[chosen]


def corrected_predictive_draws(
    baseline: np.ndarray, sigma_n: float, q: np.ndarray, e: np.ndarray
) -> np.ndarray:
    """Return exactly ``baseline + sigma_N * (q + e)``; never add bootstrap noise."""

    baseline_values = _finite(baseline, "baseline")
    q_values = _finite(q, "q")
    e_values = _finite(e, "e")
    if baseline_values.ndim != 1 or q_values.ndim != 2 or e_values.ndim != 2:
        raise PredictiveShapeError("baseline must be 1D and q/e must be 2D")
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
    """Low-level complete-block sampler retained for internal/test use only."""

    blocks = _finite(non_event_blocks, "non_event_blocks")
    if blocks.ndim != 2 or blocks.shape[1] != 24:
        raise PredictiveShapeError("non-event blocks must be finite complete 24-hour trajectories")
    indices = select_bootstrap_indices(blocks.shape[0], draws=draws, seed=seed)
    return blocks[indices].copy()


def select_bootstrap_indices(count: int, *, draws: int, seed: int) -> np.ndarray:
    if not isinstance(count, int) or count <= 0:
        raise PredictiveShapeError("block count must be positive")
    if not isinstance(draws, (int, np.integer)) or isinstance(draws, bool) or draws <= 0:
        raise PredictiveShapeError("draws must be a positive integer")
    if not isinstance(seed, (int, np.integer)) or isinstance(seed, bool):
        raise PredictiveShapeError("seed must be an integer")
    return np.random.default_rng(int(seed)).integers(0, count, size=int(draws))


def bootstrap_horizon_draws(
    blocks: np.ndarray, *, horizon: int, draws: int, seed: int
) -> np.ndarray:
    """Concatenate independent complete 24-hour blocks and truncate to the target horizon."""

    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
        raise PredictiveShapeError("horizon must be a positive integer")
    blocks = _finite(blocks, "non_event_blocks")
    if blocks.ndim != 2 or blocks.shape[1] != 24:
        raise PredictiveShapeError("non-event blocks must be finite complete 24-hour trajectories")
    per_draw = math.ceil(int(horizon) / 24)
    indices = np.random.default_rng(int(seed)).integers(
        0, blocks.shape[0], size=(int(draws), per_draw)
    )
    return blocks[indices].reshape(int(draws), -1)[:, : int(horizon)]


def _holiday_index(holiday_type: int | str) -> int:
    if isinstance(holiday_type, (int, np.integer)) and not isinstance(holiday_type, bool):
        if int(holiday_type) in (0, 1):
            return int(holiday_type)
    if isinstance(holiday_type, str):
        normalized = holiday_type.strip().lower()
        if normalized in {"seollal", "0"}:
            return 0
        if normalized in {"chuseok", "1"}:
            return 1
    raise PredictiveShapeError("holiday_type must identify Seollal (0) or Chuseok (1)")


def correction_specifications(
    posterior: Mapping[str, object], *, pooling: Pooling, restriction: int
) -> dict[str, int]:
    if pooling == "complete":
        return {"beta_type": 2}
    if pooling == "none":
        return {"beta": 2}
    specifications = {"mu": 2}
    if restriction:
        specifications["delta"] = 2
    if "between_cholesky" in posterior:
        specifications["between_cholesky"] = 3
    else:
        specifications["between_scale"] = 2
    return specifications


def draw_new_event_correction(
    posterior: Mapping[str, object],
    *,
    holiday_type: int | str,
    pooling: Pooling,
    draws: int,
    seed: int,
    restriction: int = 0,
    training_occurrence_indices: Sequence[int] | None = None,
    sample_indices: np.ndarray | None = None,
) -> np.ndarray:
    """Draw a new-event coefficient from one shared posterior row per draw."""

    if pooling not in {"complete", "partial", "none"} or restriction not in (0, 1):
        raise PredictiveShapeError("pooling and restriction are invalid")
    specs = correction_specifications(posterior, pooling=pooling, restriction=restriction)
    if sample_indices is None:
        indices = select_posterior_indices(posterior, specifications=specs, draws=draws, seed=seed)
    else:
        select_posterior_indices(posterior, specifications=specs, draws=1, seed=seed)
        indices = np.asarray(sample_indices)
    if len(indices) != draws:
        raise PredictiveShapeError("shared posterior indices must match requested draws")
    holiday = _holiday_index(holiday_type)
    if pooling == "complete":
        beta_type = posterior_values(posterior, "beta_type", trailing=2, indices=indices)
        if beta_type.shape[1] <= holiday:
            raise PredictiveShapeError("beta_type lacks the requested holiday type")
        return beta_type[:, holiday, :]
    if pooling == "none":
        beta = posterior_values(posterior, "beta", trailing=2, indices=indices)
        allowed = np.asarray(training_occurrence_indices, dtype=int)
        if (
            allowed.ndim != 1
            or allowed.size == 0
            or (allowed < 0).any()
            or (allowed >= beta.shape[1]).any()
        ):
            raise PredictiveShapeError("no-pooling requires valid training occurrence indices")
        components = np.random.default_rng(int(seed) + 1).choice(allowed, size=draws)
        return beta[np.arange(draws), components]
    mu = posterior_values(posterior, "mu", trailing=2, indices=indices)
    if mu.shape[1] <= holiday:
        raise PredictiveShapeError("mu lacks the requested holiday type")
    location = mu[:, holiday, :]
    if restriction:
        delta = posterior_values(posterior, "delta", trailing=2, indices=indices)
        if delta.shape != mu.shape:
            raise PredictiveShapeError("delta must have the same shape as mu")
        location = location + delta[:, holiday, :]
    epsilon = np.random.default_rng(int(seed) + 2).normal(size=location.shape)
    if "between_cholesky" in posterior:
        cholesky = posterior_values(posterior, "between_cholesky", trailing=3, indices=indices)
        if cholesky.shape[1] <= holiday or cholesky.shape[2:] != (
            location.shape[1],
            location.shape[1],
        ):
            raise PredictiveShapeError("between_cholesky is incompatible with mu")
        return location + np.einsum("dij,dj->di", cholesky[:, holiday], epsilon)
    scale = posterior_values(posterior, "between_scale", trailing=2, indices=indices)
    if scale.shape != mu.shape or (scale < 0).any():
        raise PredictiveShapeError("between_scale must be non-negative and match mu")
    return location + scale[:, holiday, :] * epsilon


def _vectors(phi: object, sigma: object, *, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    phi_values, sigma_values = _finite(phi, "phi").reshape(-1), _finite(sigma, "sigma").reshape(-1)
    if (
        phi_values.shape != sigma_values.shape
        or (np.abs(phi_values) >= 1).any()
        or (sigma_values < 0).any()
    ):
        raise PredictiveShapeError("phi/sigma must be aligned, stable, and non-negative")
    if not isinstance(horizon, (int, np.integer)) or isinstance(horizon, bool) or horizon <= 0:
        raise PredictiveShapeError("horizon must be a positive integer")
    return phi_values, sigma_values


def simulate_stationary_ar1(*, phi: object, sigma: object, horizon: int, seed: int) -> np.ndarray:
    """Joint normal AR(1) paths with a stationary reset at every event call."""

    phi_values, sigma_values = _vectors(phi, sigma, horizon=horizon)
    rng = np.random.default_rng(int(seed))
    result = np.empty((phi_values.size, int(horizon)))
    result[:, 0] = rng.normal(scale=sigma_values / np.sqrt(1 - phi_values**2))
    for step in range(1, int(horizon)):
        result[:, step] = phi_values * result[:, step - 1] + rng.normal(scale=sigma_values)
    return result


def simulate_student_t_ar1(
    *, phi: object, sigma: object, nu: object, horizon: int, seed: int
) -> np.ndarray:
    """Match the model's Student-t AR(1) likelihood, including its stationary reset scale."""

    phi_values, sigma_values = _vectors(phi, sigma, horizon=horizon)
    nu_values = _finite(nu, "nu").reshape(-1)
    if nu_values.shape != phi_values.shape or (nu_values <= 2).any():
        raise PredictiveShapeError("nu must align with phi and exceed two")
    rng = np.random.default_rng(int(seed))
    result = np.empty((phi_values.size, int(horizon)))
    result[:, 0] = rng.standard_t(nu_values) * sigma_values / np.sqrt(1 - phi_values**2)
    for step in range(1, int(horizon)):
        result[:, step] = (
            phi_values * result[:, step - 1] + rng.standard_t(nu_values) * sigma_values
        )
    return result


def simulate_stationary_ar2(
    *, ar2_phi: object, sigma: object, horizon: int, seed: int
) -> np.ndarray:
    """Simulate stable normal AR(2), resetting its stationary bivariate initial state per event."""

    phi = _finite(ar2_phi, "ar2_phi")
    sigma_values = _finite(sigma, "sigma").reshape(-1)
    if phi.ndim != 2 or phi.shape[1] != 2 or phi.shape[0] != sigma_values.size:
        raise PredictiveShapeError("ar2_phi must have shape (draw, 2) aligned with sigma")
    phi1, phi2 = phi[:, 0], phi[:, 1]
    denominator = (1 + phi2) * ((1 - phi2) ** 2 - phi1**2)
    if (np.abs(phi2) >= 1).any() or (denominator <= 0).any() or (sigma_values < 0).any():
        raise PredictiveShapeError("ar2_phi must be stationary and sigma non-negative")
    if not isinstance(horizon, (int, np.integer)) or horizon <= 0:
        raise PredictiveShapeError("horizon must be a positive integer")
    gamma0 = sigma_values**2 * (1 - phi2) / denominator
    gamma1 = phi1 * gamma0 / (1 - phi2)
    rng = np.random.default_rng(int(seed))
    result = np.empty((phi.shape[0], int(horizon)))
    result[:, 0] = rng.normal(scale=np.sqrt(gamma0))
    if horizon > 1:
        result[:, 1] = (gamma1 / gamma0) * result[:, 0] + rng.normal(
            scale=np.sqrt(gamma0 - gamma1**2 / gamma0)
        )
    for step in range(2, int(horizon)):
        result[:, step] = (
            phi1 * result[:, step - 1] + phi2 * result[:, step - 2] + rng.normal(scale=sigma_values)
        )
    return result
