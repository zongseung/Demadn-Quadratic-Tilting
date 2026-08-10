"""Fail-closed H0--H5 runners and staged 10-occurrence LOEO evaluation."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Literal, Protocol

import arviz as az
import numpy as np
import polars as pl

from hqrc_v3.bayes.predictive import (
    PredictiveShapeError,
    bootstrap_horizon_draws,
    corrected_predictive_draws,
    correction_specifications,
    draw_new_event_correction,
    posterior_values,
    select_posterior_indices,
    simulate_stationary_ar1,
    simulate_stationary_ar2,
    simulate_student_t_ar1,
)
from hqrc_v3.corrections.similar_day import same_holiday_profile
from hqrc_v3.diagnostics.ar import ApprovedARCalibration, require_approved_calibration

Variant = Literal["H0", "H1", "H2", "H3", "H4", "H5"]
Innovation = Literal["normal_ar1", "student_t_ar1", "normal_ar2"]
_LOEO_IDS = frozenset(
    f"{holiday}-{year}" for holiday in ("seollal", "chuseok") for year in range(2020, 2025)
)
_BLOCK_POOL_TOKEN = object()


class CorrectionContractError(ValueError):
    """Raised when correction evaluation would violate a time or provenance contract."""


@dataclass(frozen=True)
class NonEventBlockPool:
    """Validated complete non-event 24-hour residual trajectories."""

    blocks: np.ndarray
    _token: object | None = field(default=None, repr=False, compare=False)


def build_non_event_block_pool(frame: pl.DataFrame) -> NonEventBlockPool:
    """Construct H0's only admissible block source from complete non-event residual days."""

    required = {"origin", "target_timestamp", "horizon", "residual_mw", "is_event"}
    if not isinstance(frame, pl.DataFrame) or required - set(frame.columns):
        raise CorrectionContractError(
            "non-event pool requires origin/timestamp/horizon/residual/is_event"
        )
    if frame.is_empty() or frame["is_event"].dtype != pl.Boolean:
        raise CorrectionContractError("non-event pool requires a non-empty boolean is_event column")
    subset = frame.filter(~pl.col("is_event"))
    if (
        subset.is_empty()
        or not subset["residual_mw"].cast(pl.Float64, strict=True).is_finite().all()
    ):
        raise CorrectionContractError("non-event residual blocks must be finite")
    blocks: list[np.ndarray] = []
    for origin in subset["origin"].unique().sort().to_list():
        block = subset.filter(pl.col("origin") == origin).sort("horizon")
        horizon = block["horizon"].to_numpy()
        timestamps = block["target_timestamp"].to_list()
        if (
            block.height != 24
            or not np.array_equal(horizon, np.arange(1, 25))
            or any(
                timestamps[index] != timestamps[0] + timedelta(hours=index)
                for index in range(len(timestamps))
            )
        ):
            raise CorrectionContractError(
                "each non-event H0 block must be one complete hourly horizon"
            )
        blocks.append(block["residual_mw"].to_numpy().astype(float))
    if not blocks:
        raise CorrectionContractError("non-event pool has no complete horizons")
    return NonEventBlockPool(np.stack(blocks), _BLOCK_POOL_TOKEN)


@dataclass(frozen=True)
class VariantContext:
    """One event's exact target grid and all pre-approved predictive inputs."""

    timestamps: Sequence[datetime]
    baseline: np.ndarray
    observed: np.ndarray
    sigma_n: float
    holiday_type: int | str
    tau_days: np.ndarray
    hour: np.ndarray
    posterior: Mapping[str, object] = field(default_factory=dict)
    non_event_pool: NonEventBlockPool | None = None
    similar_day_training: pl.DataFrame | None = None
    held_out_occurrence_id: str | None = None
    day_positions: np.ndarray | None = None
    restriction: int = 0
    pooling: Literal["complete", "partial", "none"] = "partial"
    innovation: Innovation = "normal_ar1"
    training_occurrence_indices: Sequence[int] | None = None
    draws: int = 1_000
    seed: int = 0

    def validate(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        values = [
            np.asarray(self.baseline, dtype=float),
            np.asarray(self.observed, dtype=float),
            np.asarray(self.tau_days, dtype=float),
            np.asarray(self.hour),
        ]
        timestamps = list(self.timestamps)
        if not timestamps or any(not isinstance(item, datetime) for item in timestamps):
            raise CorrectionContractError("timestamps must be non-empty naive datetimes")
        if any(item.tzinfo is not None for item in timestamps) or any(
            timestamps[index] != timestamps[0] + timedelta(hours=index)
            for index in range(len(timestamps))
        ):
            raise CorrectionContractError("timestamps must be unique, chronological, and hourly")
        if any(item.ndim != 1 or item.size != len(timestamps) for item in values):
            raise CorrectionContractError(
                "timestamps, baseline, observed, tau_days, and hour must align"
            )
        baseline, observed, tau, hour = values
        if not all(np.isfinite(item).all() for item in (baseline, observed, tau)):
            raise CorrectionContractError("baseline, observed, and tau_days must be finite")
        if not np.issubdtype(hour.dtype, np.integer) or ((hour < 0) | (hour > 23)).any():
            raise CorrectionContractError("hour must be integer-valued from 0 through 23")
        absolute_hours = np.rint(tau * 24).astype(int)
        if not np.allclose(tau * 24, absolute_hours, rtol=0, atol=1e-4) or not np.array_equal(
            np.diff(absolute_hours), np.ones(max(0, len(absolute_hours) - 1), dtype=int)
        ):
            raise CorrectionContractError("tau_days must advance in exact one-hour positions")
        if not np.array_equal(absolute_hours % 24, hour) or not np.array_equal(
            np.asarray([item.hour for item in timestamps]), hour
        ):
            raise CorrectionContractError("hour must agree with tau_days and target timestamps")
        if not math.isfinite(self.sigma_n) or self.sigma_n <= 0:
            raise CorrectionContractError("sigma_n must be finite and positive")
        if self.restriction not in (0, 1) or self.pooling not in {"complete", "partial", "none"}:
            raise CorrectionContractError("restriction/pooling are invalid")
        if self.innovation not in {"normal_ar1", "student_t_ar1", "normal_ar2"}:
            raise CorrectionContractError("innovation must be a declared HQRC sensitivity")
        if not isinstance(self.draws, int) or isinstance(self.draws, bool) or self.draws <= 0:
            raise CorrectionContractError("draws must be a positive integer")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise CorrectionContractError("seed must be an integer")
        return baseline, observed, tau, hour.astype(int)


def _holiday_index(value: int | str) -> int:
    if value in (0, "seollal", "Seollal"):
        return 0
    if value in (1, "chuseok", "Chuseok"):
        return 1
    raise CorrectionContractError("holiday_type must identify Seollal or Chuseok")


def _specifications(context: VariantContext, variant: Variant) -> dict[str, int]:
    specifications = (
        {}
        if variant == "H5"
        else correction_specifications(
            context.posterior, pooling=context.pooling, restriction=context.restriction
        )
    )
    if variant in {"H3", "H4"}:
        specifications["gamma"] = 2
    if variant == "H4":
        specifications["day_effect"] = 2
    if variant != "H0":
        specifications["sigma_r"] = 0
        if context.innovation == "normal_ar2":
            specifications["ar2_phi"] = 1
        else:
            specifications["phi"] = 0
        if context.innovation == "student_t_ar1":
            specifications["nu"] = 0
    return specifications


def _shape_q(
    variant: Variant,
    context: VariantContext,
    tau: np.ndarray,
    hour: np.ndarray,
    indices: np.ndarray,
) -> np.ndarray:
    beta = draw_new_event_correction(
        context.posterior,
        holiday_type=context.holiday_type,
        pooling=context.pooling,
        draws=context.draws,
        seed=context.seed,
        restriction=context.restriction,
        training_occurrence_indices=context.training_occurrence_indices,
        sample_indices=indices,
    )
    if variant == "H1":
        return np.broadcast_to(beta[:, :1], (context.draws, tau.size)).copy()
    holiday = _holiday_index(context.holiday_type)
    if variant == "H4":
        profile = posterior_values(context.posterior, "day_effect", trailing=2, indices=indices)
        positions = np.asarray(
            context.day_positions
            if context.day_positions is not None
            else np.unique(np.floor(tau)),
            dtype=int,
        )
        days = np.floor(tau).astype(int)
        if (
            profile.ndim != 3
            or profile.shape[1] <= holiday
            or profile.shape[2] != positions.size
            or not np.array_equal(positions, np.unique(positions))
            or np.setdiff1d(days, positions).size
        ):
            raise CorrectionContractError("H4 day_effect is incompatible with target day positions")
        q = beta[:, :1] + profile[:, holiday, np.searchsorted(positions, days)]
    else:
        if beta.shape[1] < 3:
            raise CorrectionContractError("quadratic corrections require three coefficient terms")
        q = beta[:, [0]] + beta[:, [1]] * tau + beta[:, [2]] * tau**2
    if variant == "H2":
        return q
    gamma = posterior_values(context.posterior, "gamma", trailing=2, indices=indices)
    if gamma.ndim != 3 or gamma.shape[1] <= holiday or gamma.shape[2] != 24:
        raise CorrectionContractError("hourly profile gamma is incompatible with target hours")
    return q + gamma[:, holiday, hour]


def _ar_draws(context: VariantContext, horizon: int, indices: np.ndarray) -> np.ndarray:
    sigma = posterior_values(context.posterior, "sigma_r", trailing=0, indices=indices)
    try:
        if context.innovation == "normal_ar1":
            return simulate_stationary_ar1(
                phi=posterior_values(context.posterior, "phi", trailing=0, indices=indices),
                sigma=sigma,
                horizon=horizon,
                seed=context.seed + 31,
            )
        if context.innovation == "student_t_ar1":
            return simulate_student_t_ar1(
                phi=posterior_values(context.posterior, "phi", trailing=0, indices=indices),
                sigma=sigma,
                nu=posterior_values(context.posterior, "nu", trailing=0, indices=indices),
                horizon=horizon,
                seed=context.seed + 31,
            )
        return simulate_stationary_ar2(
            ar2_phi=posterior_values(context.posterior, "ar2_phi", trailing=1, indices=indices),
            sigma=sigma,
            horizon=horizon,
            seed=context.seed + 31,
        )
    except PredictiveShapeError as error:
        raise CorrectionContractError(str(error)) from error


def _h5_q(context: VariantContext, tau: np.ndarray, hour: np.ndarray) -> np.ndarray:
    if context.similar_day_training is None or not context.held_out_occurrence_id:
        raise CorrectionContractError(
            "H5 requires training DataFrame and explicit held-out occurrence id"
        )
    target = pl.DataFrame(
        {
            "holiday_type": [context.holiday_type] * tau.size,
            "relative_day": np.floor(tau).astype(int),
            "hour": hour,
        }
    )
    try:
        profile = same_holiday_profile(
            context.similar_day_training,
            target,
            held_out_occurrence_id=context.held_out_occurrence_id,
        )
    except ValueError as error:
        raise CorrectionContractError(str(error)) from error
    return np.broadcast_to(profile, (context.draws, tau.size)).copy()


def run_variant(variant: Variant, context: VariantContext) -> pl.DataFrame:
    """Run one correction without changing a single target timestamp."""

    if variant not in {"H0", "H1", "H2", "H3", "H4", "H5"}:
        raise CorrectionContractError("variant must be H0 through H5")
    baseline, observed, tau, hour = context.validate()
    if variant == "H0":
        if (
            not isinstance(context.non_event_pool, NonEventBlockPool)
            or context.non_event_pool._token is not _BLOCK_POOL_TOKEN
        ):
            raise CorrectionContractError("H0 requires a validated non-event block pool")
        draws = baseline[None] + bootstrap_horizon_draws(
            context.non_event_pool.blocks,
            horizon=len(baseline),
            draws=context.draws,
            seed=context.seed,
        )
        point = baseline
    else:
        specs = _specifications(context, variant)
        indices = select_posterior_indices(
            context.posterior, specifications=specs, draws=context.draws, seed=context.seed
        )
        q = (
            _h5_q(context, tau, hour)
            if variant == "H5"
            else _shape_q(variant, context, tau, hour, indices)
        )
        draws = corrected_predictive_draws(
            baseline, context.sigma_n, q, _ar_draws(context, len(baseline), indices)
        )
        point = baseline + context.sigma_n * q.mean(axis=0)
    return pl.DataFrame(
        {
            "target_timestamp": list(context.timestamps),
            "observed_mw": observed,
            "baseline_mw": baseline,
            "point_forecast_mw": point,
            "predictive_draws": [row.tolist() for row in draws.T],
            "variant": [variant] * len(baseline),
        }
    )


class LOEOBackend(Protocol):
    def load_approved_fold_calibration(
        self, held_out: str, ar_events: Sequence[str]
    ) -> ApprovedARCalibration: ...
    def fit_correction(
        self, fit_events: Sequence[str], calibration: ApprovedARCalibration
    ) -> Any: ...
    def predict_heldout(
        self, fitted: Any, held_out: str, frame: pl.DataFrame, *, causal: bool
    ) -> pl.DataFrame: ...


def _valid_event_frame(frame: object, *, name: str) -> pl.DataFrame:
    if (
        not isinstance(frame, pl.DataFrame)
        or frame.is_empty()
        or "target_timestamp" not in frame.columns
    ):
        raise CorrectionContractError(
            f"{name} must be a non-empty Polars frame with target_timestamp"
        )
    times = frame["target_timestamp"].to_list()
    if (
        any(not isinstance(item, datetime) for item in times)
        or times != sorted(times)
        or len(set(times)) != len(times)
    ):
        raise CorrectionContractError(f"{name} target timestamps must be unique and ordered")
    return frame


def run_event_loeo(event_frames: Mapping[str, pl.DataFrame], backend: LOEOBackend) -> pl.DataFrame:
    """Run only the ten registered occurrences using approved fold calibrations."""

    if set(event_frames) != _LOEO_IDS:
        raise CorrectionContractError(
            "LOEO requires exactly the ten Seollal/Chuseok 2020--2024 occurrence ids"
        )
    if not all(
        callable(getattr(backend, name, None))
        for name in ("load_approved_fold_calibration", "fit_correction", "predict_heldout")
    ):
        raise CorrectionContractError("LOEO requires a staged approved-calibration backend")
    checked = {
        event_id: _valid_event_frame(frame, name=f"input {event_id}")
        for event_id, frame in event_frames.items()
    }
    results: list[pl.DataFrame] = []
    for held_out in sorted(checked):
        fit_events = tuple(event_id for event_id in sorted(checked) if event_id != held_out)
        calibration = require_approved_calibration(
            backend.load_approved_fold_calibration(held_out, fit_events)
        )
        fitted = backend.fit_correction(fit_events, calibration)
        output = _valid_event_frame(
            backend.predict_heldout(fitted, held_out, checked[held_out], causal=False),
            name=f"output {held_out}",
        )
        if output["target_timestamp"].to_list() != checked[held_out]["target_timestamp"].to_list():
            raise CorrectionContractError(
                "LOEO output timestamps must exactly equal the held-out input"
            )
        results.append(
            output.with_columns(
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
    try:
        result = az.loo(inference_data, var_name=var_name, pointwise=True)
        pareto_k = np.asarray(result.pareto_k, dtype=float)
    except Exception as error:
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
