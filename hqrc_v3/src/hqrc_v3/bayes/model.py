"""Hierarchical holiday residual correction with event-reset AR innovations."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from hqrc_v3.diagnostics.ar import ApprovedARCalibration, require_approved_calibration

Variant = Literal["H1", "H2", "H3", "H4"]
Pooling = Literal["complete", "partial", "none"]
Innovation = Literal["normal_ar1", "student_t_ar1", "normal_ar2"]


@dataclass(frozen=True)
class HQRCData:
    """Validated long-form standardized event residuals grouped into reset segments.

    Rows must be ordered by occurrence and an occurrence may occur in only one
    contiguous block.  That representation makes event-reset AR likelihoods
    explicit and prevents an accidental transition across event boundaries.
    """

    observations: np.ndarray
    occurrence_index: np.ndarray
    holiday_type_index: np.ndarray
    tau_days: np.ndarray
    hour: np.ndarray
    restriction: np.ndarray
    occurrence_ids: tuple[str, ...]
    segments: tuple[slice, ...] = field(init=False, repr=False)
    occurrence_holiday_type: np.ndarray = field(init=False, repr=False)
    occurrence_restriction: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        values = {
            "observations": self.observations,
            "occurrence_index": self.occurrence_index,
            "holiday_type_index": self.holiday_type_index,
            "tau_days": self.tau_days,
            "hour": self.hour,
            "restriction": self.restriction,
        }
        arrays = {name: np.asarray(value) for name, value in values.items()}
        if any(array.ndim != 1 for array in arrays.values()):
            raise ValueError("HQRCData inputs must be one-dimensional arrays")
        count = arrays["observations"].size
        if count == 0 or any(array.size != count for array in arrays.values()):
            raise ValueError("HQRCData inputs must be aligned non-empty arrays")
        if not np.isfinite(arrays["observations"].astype(float)).all() or not np.isfinite(
            arrays["tau_days"].astype(float)
        ).all():
            raise ValueError("HQRC observations and tau_days must be finite")
        for name in ("occurrence_index", "holiday_type_index", "hour", "restriction"):
            array = arrays[name]
            if not np.issubdtype(array.dtype, np.integer) or array.dtype == np.dtype(bool):
                raise ValueError(f"HQRCData {name} must be integer-valued")
        occurrence = arrays["occurrence_index"].astype(np.int64, copy=False)
        holiday_type = arrays["holiday_type_index"].astype(np.int64, copy=False)
        hour = arrays["hour"].astype(np.int64, copy=False)
        restriction = arrays["restriction"].astype(np.int64, copy=False)
        if (occurrence < 0).any() or tuple(np.unique(occurrence)) != tuple(
            range(len(self.occurrence_ids))
        ):
            raise ValueError("occurrence_index must be dense and match occurrence_ids")
        if not self.occurrence_ids or any(
            not isinstance(value, str) or not value.strip() for value in self.occurrence_ids
        ):
            raise ValueError("occurrence_ids must be non-empty strings")
        if len(set(self.occurrence_ids)) != len(self.occurrence_ids):
            raise ValueError("occurrence_ids must be unique")
        if ((holiday_type < 0) | (holiday_type > 1)).any():
            raise ValueError("holiday_type_index must be 0 (seollal) or 1 (chuseok)")
        if ((hour < 0) | (hour > 23)).any():
            raise ValueError("hour must be from 0 through 23")
        if not np.isin(restriction, (0, 1)).all():
            raise ValueError("restriction must be binary")

        seen: set[int] = set()
        starts: list[int] = []
        ends: list[int] = []
        occurrence_types = np.empty(len(self.occurrence_ids), dtype=np.int64)
        occurrence_restriction = np.empty(len(self.occurrence_ids), dtype=np.int64)
        start = 0
        for row in range(1, count + 1):
            if row != count and occurrence[row] == occurrence[start]:
                continue
            current = int(occurrence[start])
            if current in seen:
                raise ValueError("occurrence rows must form contiguous segments")
            seen.add(current)
            block = slice(start, row)
            if np.unique(holiday_type[block]).size != 1 or np.unique(restriction[block]).size != 1:
                raise ValueError("each occurrence must have one holiday type and restriction flag")
            block_tau = arrays["tau_days"][block].astype(float, copy=False)
            if block_tau.size > 1 and not np.allclose(
                np.diff(block_tau), 1.0 / 24.0, rtol=0.0, atol=1e-10
            ):
                raise ValueError("occurrence tau_days must be strictly ordered in one-hour steps")
            block_hour = hour[block]
            if block_hour.size > 1 and not np.array_equal(
                (block_hour[1:] - block_hour[:-1]) % 24,
                np.ones(block_hour.size - 1, dtype=np.int64),
            ):
                raise ValueError("occurrence hour values must advance one hour modulo 24")
            starts.append(start)
            ends.append(row)
            occurrence_types[current] = holiday_type[start]
            occurrence_restriction[current] = restriction[start]
            start = row
        if tuple(sorted(seen)) != tuple(range(len(self.occurrence_ids))):
            raise ValueError("each occurrence_id must have one segment")
        object.__setattr__(self, "observations", arrays["observations"].astype(float, copy=False))
        object.__setattr__(self, "occurrence_index", occurrence)
        object.__setattr__(self, "holiday_type_index", holiday_type)
        object.__setattr__(self, "tau_days", arrays["tau_days"].astype(float, copy=False))
        object.__setattr__(self, "hour", hour)
        object.__setattr__(self, "restriction", restriction)
        object.__setattr__(
            self, "segments", tuple(slice(begin, end) for begin, end in zip(starts, ends))
        )
        object.__setattr__(self, "occurrence_holiday_type", occurrence_types)
        object.__setattr__(self, "occurrence_restriction", occurrence_restriction)


@dataclass(frozen=True)
class HQRCModelOptions:
    """Predeclared covariance and innovation sensitivity choices."""

    covariance: Literal["full", "diagonal"] = "full"
    include_restriction: bool = True
    lkj_eta: float = 2.0
    between_scale_prior: float = 1.0
    innovation: Innovation = "normal_ar1"

    def validate(self) -> None:
        if self.covariance not in {"full", "diagonal"}:
            raise ValueError("covariance must be 'full' or 'diagonal'")
        if not isinstance(self.include_restriction, bool):
            raise TypeError("include_restriction must be a boolean")
        if not math.isfinite(self.lkj_eta) or self.lkj_eta <= 0:
            raise ValueError("lkj_eta must be finite and positive")
        if not math.isfinite(self.between_scale_prior) or self.between_scale_prior <= 0:
            raise ValueError("between_scale_prior must be finite and positive")
        if self.innovation not in {"normal_ar1", "student_t_ar1", "normal_ar2"}:
            raise ValueError("innovation must be normal_ar1, student_t_ar1, or normal_ar2")


def stationary_ar1_logp_numpy(segments: Sequence[np.ndarray], phi: float, sigma: float) -> float:
    """Return the stationary AR(1) log density, restarting independently per event."""

    if not math.isfinite(phi) or abs(phi) >= 1:
        raise ValueError("phi must be finite and strictly inside (-1, 1)")
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("sigma must be finite and positive")
    stationary_sd = sigma / math.sqrt(1.0 - phi**2)
    total = 0.0
    for segment in segments:
        values = np.asarray(segment, dtype=float)
        if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
            raise ValueError("each AR segment must be a finite non-empty vector")
        total += -0.5 * (
            (values[0] / stationary_sd) ** 2 + math.log(2 * math.pi * stationary_sd**2)
        )
        innovation = values[1:] - phi * values[:-1]
        total += float(
            -0.5 * np.sum((innovation / sigma) ** 2 + math.log(2 * math.pi * sigma**2))
        )
    return total


def _validate_calibration(calibration: ApprovedARCalibration) -> None:
    trusted = require_approved_calibration(calibration)
    values = trusted.calibration
    if (
        not math.isfinite(values.a)
        or not math.isfinite(values.b)
        or values.a <= 1
        or values.b <= 1
    ):
        raise ValueError("ARCalibration Beta parameters must be finite and greater than one")


def _intrinsic_random_walk(
    pm, pt, *, name: str, positions: int, scale_name: str, scale: float, circular: bool = False
):
    """Build an identified sum-zero RW1 from innovations, never iid level effects."""

    sigma = pm.HalfNormal(scale_name, sigma=scale, shape=2)
    if positions == 1:
        return pm.Deterministic(name, pt.zeros((2, 1)))
    innovation = pm.Normal(
        f"{name}_innovation", mu=0.0, sigma=sigma[:, None], shape=(2, positions - 1)
    )
    if circular:
        pm.Potential(
            f"{name}_circular_random_walk",
            pm.logp(pm.Normal.dist(mu=0.0, sigma=sigma), -pt.sum(innovation, axis=1)).sum(),
        )
    path = pt.concatenate((pt.zeros((2, 1)), pt.cumsum(innovation, axis=1)), axis=1)
    return pm.Deterministic(name, path - pt.mean(path, axis=1, keepdims=True))


def _cyclic_hour_profile(pm, pt, *, name: str):
    profile = _intrinsic_random_walk(
        pm, pt, name=name, positions=24, scale_name="sigma_gamma", scale=0.5, circular=True
    )
    return profile


def _occurrence_coefficients(
    pm,
    pt,
    data: HQRCData,
    *,
    n_coefficients: int,
    pooling: Pooling,
    options: HQRCModelOptions,
):
    occurrence_type = data.occurrence_holiday_type
    occurrence_restriction = data.occurrence_restriction.astype(float)
    count = len(data.occurrence_ids)
    if pooling == "complete":
        by_type = pm.Normal("beta_type", mu=0.0, sigma=2.0, shape=(2, n_coefficients))
        return pm.Deterministic("beta", by_type[occurrence_type])
    if pooling == "none":
        return pm.Normal("beta", mu=0.0, sigma=2.0, shape=(count, n_coefficients))

    mu = pm.Normal("mu", mu=0.0, sigma=2.0, shape=(2, n_coefficients))
    location = mu[occurrence_type]
    if options.include_restriction:
        delta = pm.Normal("delta", mu=0.0, sigma=2.0, shape=(2, n_coefficients))
        location = location + delta[occurrence_type] * occurrence_restriction[:, None]
    offset = pm.Normal("beta_offset", mu=0.0, sigma=1.0, shape=(count, n_coefficients))
    if options.covariance == "diagonal":
        scale = pm.HalfNormal(
            "between_scale", sigma=options.between_scale_prior, shape=(2, n_coefficients)
        )
        return pm.Deterministic("beta", location + offset * scale[occurrence_type])

    cholesky = []
    scales = []
    for holiday_type in range(2):
        chol, _, stds = pm.LKJCholeskyCov(
            f"between_cov_{holiday_type}",
            n=n_coefficients,
            eta=options.lkj_eta,
            sd_dist=pm.HalfNormal.dist(sigma=options.between_scale_prior),
            compute_corr=True,
        )
        cholesky.append(chol)
        scales.append(stds)
    pm.Deterministic("between_scale", pt.stack(scales))
    transformed = pt.stack(
        [pt.dot(cholesky[int(occurrence_type[index])], offset[index]) for index in range(count)]
    )
    return pm.Deterministic("beta", location + transformed)


def _ar_likelihood(
    pm,
    pt,
    residual,
    data: HQRCData,
    calibration: ApprovedARCalibration,
    innovation: Innovation,
):
    sigma = pm.HalfNormal("sigma_r", sigma=1.0)
    if innovation == "normal_ar2":
        pacf = pm.Uniform("ar2_pacf", lower=-0.95, upper=0.95, shape=2)
        phi2 = pacf[1]
        phi1 = pacf[0] * (1.0 - phi2)
        pm.Deterministic("ar2_phi", pt.stack((phi1, phi2)))
        gamma0 = sigma**2 * (1.0 - phi2) / (
            (1.0 + phi2) * ((1.0 - phi2) ** 2 - phi1**2)
        )
        gamma1 = phi1 * gamma0 / (1.0 - phi2)
        covariance = pt.stack(((gamma0, gamma1), (gamma1, gamma0)))
        event_terms = []
        for segment in data.segments:
            values = residual[segment]
            if segment.stop - segment.start == 1:
                event_terms.append(
                    pm.logp(pm.Normal.dist(mu=0.0, sigma=pt.sqrt(gamma0)), values[0])
                )
            else:
                initial = pm.logp(pm.MvNormal.dist(mu=pt.zeros(2), cov=covariance), values[:2])
                innovation_term = (
                    pm.logp(
                        pm.Normal.dist(mu=phi1 * values[1:-1] + phi2 * values[:-2], sigma=sigma),
                        values[2:],
                    ).sum()
                )
                event_terms.append(initial + innovation_term)
        pointwise = pm.Deterministic("event_log_likelihood", pt.stack(event_terms))
        pm.Potential("event_reset_ar2", pointwise.sum())
        return

    u_phi = pm.Beta("u_phi", alpha=calibration.a, beta=calibration.b)
    phi = pm.Deterministic("phi", 2.0 * u_phi - 1.0)
    stationary_scale = sigma / pt.sqrt(1.0 - phi**2)
    if innovation == "student_t_ar1":
        nu_minus_two = pm.Exponential("nu_minus_two", lam=1.0 / 10.0)
        nu = pm.Deterministic("nu", nu_minus_two + 2.0)
        def distribution(mu, scale):
            return pm.StudentT.dist(nu=nu, mu=mu, sigma=scale)
    else:
        def distribution(mu, scale):
            return pm.Normal.dist(mu=mu, sigma=scale)
    event_terms = []
    for segment in data.segments:
        values = residual[segment]
        event_terms.append(
            pm.logp(distribution(0.0, stationary_scale), values[0])
            + pm.logp(distribution(phi * values[:-1], sigma), values[1:]).sum()
        )
    pointwise = pm.Deterministic("event_log_likelihood", pt.stack(event_terms))
    pm.Potential("event_reset_ar1", pointwise.sum())


def build_hqrc_model(
    data: HQRCData,
    calibration: ApprovedARCalibration,
    variant: Variant = "H3",
    pooling: Pooling = "partial",
    options: HQRCModelOptions | None = None,
):
    """Build H1--H4 using only approved AR calibration and event-reset likelihoods."""

    if not isinstance(data, HQRCData):
        raise TypeError("data must be HQRCData")
    _validate_calibration(calibration)
    if variant not in {"H1", "H2", "H3", "H4"}:
        raise ValueError("variant must be H1, H2, H3, or H4")
    if pooling not in {"complete", "partial", "none"}:
        raise ValueError("pooling must be complete, partial, or none")
    selected_options = options or HQRCModelOptions()
    selected_options.validate()

    import pymc as pm
    import pytensor.tensor as pt

    n_coefficients = 3 if variant in {"H2", "H3"} else 1
    with pm.Model() as model:
        beta = _occurrence_coefficients(
            pm,
            pt,
            data,
            n_coefficients=n_coefficients,
            pooling=pooling,
            options=selected_options,
        )
        occurrence = data.occurrence_index
        if variant == "H1":
            mean = beta[occurrence, 0]
        elif variant in {"H2", "H3"}:
            design = np.column_stack(
                (np.ones(data.observations.size), data.tau_days, data.tau_days**2)
            )
            mean = pt.sum(beta[occurrence] * design, axis=1)
        else:
            unique_days = np.unique(np.floor(data.tau_days).astype(np.int64))
            day_lookup = np.searchsorted(unique_days, np.floor(data.tau_days).astype(np.int64))
            day_effect = _intrinsic_random_walk(
                pm,
                pt,
                name="day_effect",
                positions=len(unique_days),
                scale_name="sigma_day",
                scale=1.0,
            )
            mean = beta[occurrence, 0] + day_effect[data.holiday_type_index, day_lookup]
        if variant in {"H3", "H4"}:
            gamma = _cyclic_hour_profile(pm, pt, name="gamma")
            mean = mean + gamma[data.holiday_type_index, data.hour]
        residual = pt.as_tensor_variable(data.observations) - mean
        _ar_likelihood(pm, pt, residual, data, calibration, selected_options.innovation)
    return model
