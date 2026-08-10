"""H0--H5 correction runners and explicit event-unit LOEO orchestration."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import arviz as az
import numpy as np
import polars as pl

from hqrc_v3.bayes.predictive import (
    PredictiveShapeError,
    baseline_bootstrap_draws,
    corrected_predictive_draws,
    draw_new_event_correction,
    simulate_stationary_ar1,
)
from hqrc_v3.corrections.similar_day import same_holiday_profile

Variant = Literal["H0", "H1", "H2", "H3", "H4", "H5"]


class CorrectionContractError(ValueError):
    """Raised when a correction context or LOEO backend violates a contract."""


@dataclass(frozen=True)
class VariantContext:
    """Inputs for one occurrence, with one row for every target timestamp."""

    timestamps: Sequence[object]
    baseline: np.ndarray
    observed: np.ndarray
    sigma_n: float
    holiday_type: int | str
    tau_days: np.ndarray
    hour: np.ndarray
    posterior: Mapping[str, object] = field(default_factory=dict)
    non_event_blocks: np.ndarray | None = None
    training_profiles: Mapping[str, np.ndarray] | None = None
    similar_day_training: pl.DataFrame | None = None
    day_positions: np.ndarray | None = None
    restriction: int = 0
    pooling: Literal["complete", "partial", "none"] = "partial"
    training_occurrence_indices: Sequence[int] | None = None
    draws: int = 1_000
    seed: int = 0

    def validate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        timestamp_values = list(self.timestamps)
        baseline = np.asarray(self.baseline, dtype=float)
        observed = np.asarray(self.observed, dtype=float)
        tau = np.asarray(self.tau_days, dtype=float)
        hour = np.asarray(self.hour)
        count = len(timestamp_values)
        if count == 0 or any(
            value.ndim != 1 or value.size != count for value in (baseline, observed, tau, hour)
        ):
            raise CorrectionContractError(
                "timestamps, baseline, observed, tau_days, and hour must align"
            )
        if (
            not np.isfinite(baseline).all()
            or not np.isfinite(observed).all()
            or not np.isfinite(tau).all()
        ):
            raise CorrectionContractError("baseline, observed, and tau_days must be finite")
        if not np.issubdtype(hour.dtype, np.integer) or ((hour < 0) | (hour > 23)).any():
            raise CorrectionContractError("hour must be integer-valued from 0 through 23")
        if not math.isfinite(self.sigma_n) or self.sigma_n <= 0:
            raise CorrectionContractError("sigma_n must be finite and positive")
        if self.restriction not in (0, 1) or self.pooling not in {"complete", "partial", "none"}:
            raise CorrectionContractError("restriction must be binary and pooling must be declared")
        if not isinstance(self.draws, int) or isinstance(self.draws, bool) or self.draws <= 0:
            raise CorrectionContractError("draws must be a positive integer")
        return baseline, observed, tau, hour.astype(int, copy=False)


def _posterior_values(
    posterior: Mapping[str, object],
    name: str,
    draws: int,
    seed: int,
    *,
    trailing_dimensions: int = 0,
) -> np.ndarray:
    if name not in posterior:
        raise CorrectionContractError(f"posterior is missing {name!r}")
    raw = getattr(posterior[name], "values", posterior[name])
    values = np.asarray(raw, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise CorrectionContractError(f"posterior {name} must be finite")
    if values.ndim < trailing_dimensions:
        raise CorrectionContractError(f"posterior {name} has too few dimensions")
    trailing = values.shape[-trailing_dimensions:] if trailing_dimensions else ()
    flattened = values.reshape((-1,) + trailing)
    rng = np.random.default_rng(seed)
    return flattened[rng.integers(0, len(flattened), size=draws)]


def _holiday_index(value: int | str) -> int:
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return int(value)
    if isinstance(value, str):
        return (
            0
            if value.strip().lower() in {"seollal", "0"}
            else 1
            if value.strip().lower() in {"chuseok", "1"}
            else -1
        )
    return -1


def _shape_q(
    variant: Variant, context: VariantContext, tau: np.ndarray, hour: np.ndarray
) -> np.ndarray:
    draws = context.draws
    if variant == "H1":
        coefficients = draw_new_event_correction(
            context.posterior,
            holiday_type=context.holiday_type,
            pooling=context.pooling,
            draws=draws,
            seed=context.seed,
            restriction=context.restriction,
            training_occurrence_indices=context.training_occurrence_indices,
        )
        return np.broadcast_to(coefficients[:, :1], (draws, tau.size)).copy()
    coefficients = draw_new_event_correction(
        context.posterior,
        holiday_type=context.holiday_type,
        pooling=context.pooling,
        draws=draws,
        seed=context.seed,
        restriction=context.restriction,
        training_occurrence_indices=context.training_occurrence_indices,
    )
    if variant == "H4":
        if coefficients.shape[1] < 1:
            raise CorrectionContractError("H4 requires an event-level intercept coefficient")
        profile_name = "day_effect" if "day_effect" in context.posterior else "day_profile"
        if profile_name not in context.posterior:
            raise CorrectionContractError("H4 requires a smoothed unrestricted day profile")
        profile = _posterior_values(
            context.posterior, profile_name, draws, context.seed + 11, trailing_dimensions=2
        )
        holiday = _holiday_index(context.holiday_type)
        positions = (
            np.asarray(context.day_positions, dtype=int)
            if context.day_positions is not None
            else np.unique(np.floor(tau).astype(int))
        )
        day = np.floor(tau).astype(int)
        if (
            profile.ndim != 3
            or profile.shape[1] <= holiday
            or positions.ndim != 1
            or positions.size != profile.shape[2]
            or np.unique(positions).size != positions.size
            or (positions.size > 1 and not np.all(np.diff(positions) > 0))
            or np.setdiff1d(day, positions).size
        ):
            raise CorrectionContractError("day profile is incompatible with target positions")
        lookup = np.searchsorted(positions, day)
        q = coefficients[:, [0]] + profile[:, holiday, lookup]
    else:
        if coefficients.shape[1] < 3:
            raise CorrectionContractError("quadratic correction requires three coefficients")
        q = coefficients[:, [0]] + coefficients[:, [1]] * tau + coefficients[:, [2]] * tau**2
    if variant == "H2":
        return q
    holiday = _holiday_index(context.holiday_type)
    if holiday < 0:
        raise CorrectionContractError("unknown holiday type")
    if "gamma" not in context.posterior:
        raise CorrectionContractError(f"{variant} requires a circular hourly profile")
    gamma = _posterior_values(
        context.posterior, "gamma", draws, context.seed + 13, trailing_dimensions=2
    )
    if gamma.ndim != 3 or gamma.shape[1] <= holiday or gamma.shape[2] != 24:
        raise CorrectionContractError("gamma is incompatible with the hourly profile")
    q = q + gamma[:, holiday, hour]
    return q


def _ar_draws(context: VariantContext, horizon: int) -> np.ndarray:
    if "phi" not in context.posterior or "sigma_r" not in context.posterior:
        return np.zeros((context.draws, horizon), dtype=float)
    phi = _posterior_values(context.posterior, "phi", context.draws, context.seed + 17)
    sigma = _posterior_values(context.posterior, "sigma_r", context.draws, context.seed + 19)
    try:
        return simulate_stationary_ar1(
            phi=phi, sigma=sigma, horizon=horizon, seed=context.seed + 23
        )
    except PredictiveShapeError as error:
        raise CorrectionContractError(str(error)) from error


def _h5_q(context: VariantContext, tau: np.ndarray, hour: np.ndarray) -> np.ndarray:
    if context.similar_day_training is not None:
        target = pl.DataFrame(
            {
                "holiday_type": [context.holiday_type] * tau.size,
                "relative_day": np.floor(tau).astype(int),
                "hour": hour,
            }
        )
        profile = same_holiday_profile(context.similar_day_training, target)
    elif context.training_profiles:
        profiles = [np.asarray(value, dtype=float) for value in context.training_profiles.values()]
        if not profiles or any(
            value.shape != tau.shape or not np.isfinite(value).all() for value in profiles
        ):
            raise CorrectionContractError("H5 training profiles must be finite and target-aligned")
        # Equal occurrence weighting is deliberate; the mapping must already exclude
        # the held-out occurrence and contains no target residuals.
        profile = np.mean(np.stack(profiles), axis=0)
    else:
        raise CorrectionContractError("H5 requires same-holiday training profiles")
    return np.broadcast_to(profile, (context.draws, tau.size)).copy()


def run_variant(variant: Variant, context: VariantContext) -> pl.DataFrame:
    """Run H0--H5 and return a timestamp-preserving long prediction artifact."""

    if variant not in {"H0", "H1", "H2", "H3", "H4", "H5"}:
        raise CorrectionContractError("variant must be H0 through H5")
    baseline, observed, tau, hour = context.validate()
    if variant == "H0":
        if context.non_event_blocks is None:
            raise CorrectionContractError(
                "H0 requires horizon-preserving non-event residual blocks"
            )
        try:
            residual = baseline_bootstrap_draws(
                context.non_event_blocks, draws=context.draws, seed=context.seed
            )
        except PredictiveShapeError as error:
            raise CorrectionContractError(str(error)) from error
        if residual.shape[1] != baseline.size:
            raise CorrectionContractError(
                "H0 non-event residual blocks must preserve the forecast horizon"
            )
        draws = baseline[None, :] + residual
        point = baseline
    else:
        q = _h5_q(context, tau, hour) if variant == "H5" else _shape_q(variant, context, tau, hour)
        draws = corrected_predictive_draws(
            baseline, context.sigma_n, q, _ar_draws(context, baseline.size)
        )
        point = baseline + context.sigma_n * q.mean(axis=0)
    return pl.DataFrame(
        {
            "target_timestamp": list(context.timestamps),
            "observed_mw": observed,
            "baseline_mw": baseline,
            "point_forecast_mw": point,
            "predictive_draws": [row.tolist() for row in draws.T],
            "variant": [variant] * baseline.size,
        }
    )


class LOEOBackend(Protocol):
    """Injected, explicitly staged LOEO workflow; it must own approved fold artifacts."""

    def fit_predict(
        self, held_out: str, fit_events: Sequence[str], ar_events: Sequence[str], *, causal: bool
    ) -> Any: ...

    def load_approved_fold_calibration(self, held_out: str, ar_events: Sequence[str]) -> Any: ...

    def fit_correction(self, fit_events: Sequence[str], calibration: Any) -> Any: ...

    def predict_heldout(
        self, fitted: Any, held_out: str, frame: object, *, causal: bool
    ) -> Any: ...


def _to_frame(value: Any) -> pl.DataFrame:
    if isinstance(value, pl.DataFrame):
        return value
    try:
        return pl.DataFrame(value)
    except (TypeError, ValueError) as error:
        raise CorrectionContractError(
            "LOEO backend must return a Polars-compatible frame"
        ) from error


def _run_loeo_fold(
    backend: LOEOBackend,
    *,
    held_out: str,
    fit_events: tuple[str, ...],
    event_frame: object,
) -> Any:
    """Use an explicitly approved staged backend, or its equivalent atomic adapter."""

    staged_names = ("load_approved_fold_calibration", "fit_correction", "predict_heldout")
    if all(callable(getattr(backend, name, None)) for name in staged_names):
        calibration = backend.load_approved_fold_calibration(held_out, fit_events)
        fitted = backend.fit_correction(fit_events, calibration)
        return backend.predict_heldout(fitted, held_out, event_frame, causal=False)
    fit_predict = getattr(backend, "fit_predict", None)
    if callable(fit_predict):
        return fit_predict(held_out, fit_events, fit_events, causal=False)
    raise CorrectionContractError(
        "LOEO backend must load an approved fold calibration before correction fitting"
    )


def run_event_loeo(event_frames: Mapping[str, object], backend: LOEOBackend) -> pl.DataFrame:
    """Evaluate event-unit LOEO without ever auto-approving a fold calibration.

    The backend is injected so its `fit_predict` implementation can load the
    pre-approved fold calibration and fit artifacts.  This module never performs a
    calibration or approves a diagnostic artifact implicitly.
    """

    if not isinstance(event_frames, Mapping) or len(event_frames) < 2:
        raise CorrectionContractError("LOEO requires at least two uniquely named event frames")
    event_ids = tuple(event_frames)
    if any(not isinstance(item, str) or not item.strip() for item in event_ids):
        raise CorrectionContractError("LOEO occurrence ids must be nonblank strings")
    results: list[pl.DataFrame] = []
    for held_out in event_ids:
        fit_events = tuple(event_id for event_id in event_ids if event_id != held_out)
        produced = _to_frame(
            _run_loeo_fold(
                backend,
                held_out=held_out,
                fit_events=fit_events,
                event_frame=event_frames[held_out],
            )
        )
        if produced.is_empty() or "target_timestamp" not in produced.columns:
            raise CorrectionContractError(
                "LOEO backend result must contain non-empty target timestamps"
            )
        results.append(
            produced.with_columns(
                pl.lit(held_out).alias("held_out_occurrence"), pl.lit(False).alias("causal")
            )
        )
    return pl.concat(results, how="diagonal_relaxed")


@dataclass(frozen=True)
class PSISLOOSummary:
    elpd_loo: float
    standard_error: float
    pareto_k: np.ndarray


def psis_loo_summary(inference_data: Any, *, var_name: str = "event") -> PSISLOOSummary:
    """Preserve ArviZ ELPD, standard error, and pointwise Pareto-k diagnostics."""

    try:
        result = az.loo(inference_data, var_name=var_name, pointwise=True)
        pareto_k = np.asarray(result.pareto_k, dtype=float)
    except Exception as error:  # ArviZ has version-specific exception classes.
        raise CorrectionContractError(
            f"unable to compute PSIS-LOO for {var_name!r}: {error}"
        ) from error
    if (
        not np.isfinite(float(result.elpd_loo))
        or not np.isfinite(float(result.se))
        or not np.isfinite(pareto_k).all()
    ):
        raise CorrectionContractError("PSIS-LOO output must be finite")
    return PSISLOOSummary(float(result.elpd_loo), float(result.se), pareto_k)
