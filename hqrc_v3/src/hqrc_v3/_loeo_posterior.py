"""Central semantic validation for one-fold H3 posterior checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hqrc_v3._loeo_types import LOEOFoldError, LOEOFoldInputs
from hqrc_v3.bayes.model import event_reset_ar1_logp_numpy

_DIMS: dict[str, tuple[str, ...]] = {
    "mu": ("mu_dim_0", "mu_dim_1"),
    "delta": ("delta_dim_0", "delta_dim_1"),
    "beta_offset": ("beta_offset_dim_0", "beta_offset_dim_1"),
    "gamma_innovation_raw": (
        "gamma_innovation_raw_dim_0",
        "gamma_innovation_raw_dim_1",
    ),
    "between_cov_0": ("between_cov_0_dim_0",),
    "between_cov_1": ("between_cov_1_dim_0",),
    "sigma_gamma": ("sigma_gamma_dim_0",),
    "sigma_r": (),
    "u_phi": (),
    "between_cov_0_corr": ("between_cov_0_corr_dim_0", "between_cov_0_corr_dim_1"),
    "between_cov_0_stds": ("between_cov_0_stds_dim_0",),
    "between_cov_1_corr": ("between_cov_1_corr_dim_0", "between_cov_1_corr_dim_1"),
    "between_cov_1_stds": ("between_cov_1_stds_dim_0",),
    "between_cholesky": (
        "between_cholesky_dim_0",
        "between_cholesky_dim_1",
        "between_cholesky_dim_2",
    ),
    "between_scale": ("between_scale_dim_0", "between_scale_dim_1"),
    "beta": ("beta_dim_0", "beta_dim_1"),
    "gamma_innovation": ("gamma_innovation_dim_0", "gamma_innovation_dim_1"),
    "gamma": ("gamma_dim_0", "gamma_dim_1"),
    "phi": (),
    "event_log_likelihood": ("event_log_likelihood_dim_0",),
}
_SHAPES = {
    "mu": (2, 3),
    "delta": (2, 3),
    "beta_offset": (9, 3),
    "gamma_innovation_raw": (2, 23),
    "between_cov_0": (6,),
    "between_cov_1": (6,),
    "sigma_gamma": (2,),
    "sigma_r": (),
    "u_phi": (),
    "between_cov_0_corr": (3, 3),
    "between_cov_0_stds": (3,),
    "between_cov_1_corr": (3, 3),
    "between_cov_1_stds": (3,),
    "between_cholesky": (2, 3, 3),
    "between_scale": (2, 3),
    "beta": (9, 3),
    "gamma_innovation": (2, 23),
    "gamma": (2, 24),
    "phi": (),
    "event_log_likelihood": (9,),
}


def posterior_mapping(idata: object) -> Mapping[str, object]:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or not hasattr(posterior, "data_vars"):
        raise LOEOFoldError("LOEO posterior InferenceData is missing")
    return posterior


def _close(actual: object, expected: object) -> bool:
    return np.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def _validate_schema(posterior: object, chains: int, draws: int) -> None:
    if set(posterior.data_vars) != set(_DIMS):
        raise LOEOFoldError("LOEO H3 posterior variable namespace differs")
    expected_coords: dict[str, int] = {"chain": chains, "draw": draws}
    for name, trailing_dims in _DIMS.items():
        variable = posterior[name]
        if tuple(variable.dims) != ("chain", "draw", *trailing_dims):
            raise LOEOFoldError(f"LOEO H3 posterior variable {name} dimensions differ")
        if tuple(variable.shape) != (chains, draws, *_SHAPES[name]):
            raise LOEOFoldError(f"LOEO H3 posterior variable {name} shape differs")
        for dimension, size in zip(trailing_dims, _SHAPES[name]):
            expected_coords[dimension] = size
        if not np.isfinite(np.asarray(variable)).all():
            raise LOEOFoldError(f"LOEO H3 posterior variable {name} differs")
    if set(posterior.coords) != set(expected_coords):
        raise LOEOFoldError(
            "LOEO H3 posterior coordinate namespace differs: "
            f"extra={sorted(set(posterior.coords) - set(expected_coords))}, "
            f"missing={sorted(set(expected_coords) - set(posterior.coords))}"
        )
    for name, size in expected_coords.items():
        coordinate = posterior.coords[name]
        if tuple(coordinate.dims) != (name,) or not np.array_equal(
            coordinate.values, np.arange(size)
        ):
            raise LOEOFoldError(f"LOEO H3 posterior coordinate {name} differs")


def _validate_covariance(posterior: object) -> None:
    cholesky = np.asarray(posterior["between_cholesky"], dtype=float)
    if (np.diagonal(cholesky, axis1=-2, axis2=-1) <= 0).any() or not _close(
        cholesky, np.tril(cholesky)
    ):
        raise LOEOFoldError("LOEO posterior full covariance draws differ")
    lower = np.tril_indices(3)
    for holiday in range(2):
        packed = np.asarray(posterior[f"between_cov_{holiday}"], dtype=float)
        stds = np.asarray(posterior[f"between_cov_{holiday}_stds"], dtype=float)
        corr = np.asarray(posterior[f"between_cov_{holiday}_corr"], dtype=float)
        chol = cholesky[..., holiday, :, :]
        covariance = chol @ np.swapaxes(chol, -1, -2)
        reconstructed_stds = np.sqrt(np.diagonal(covariance, axis1=-2, axis2=-1))
        reconstructed_corr = covariance / (
            reconstructed_stds[..., :, None] * reconstructed_stds[..., None, :]
        )
        if (
            (stds <= 0).any()
            or not _close(packed, chol[..., lower[0], lower[1]])
            or not _close(stds, reconstructed_stds)
            or not _close(corr, reconstructed_corr)
            or not _close(np.asarray(posterior["between_scale"])[..., holiday, :], stds)
        ):
            raise LOEOFoldError("LOEO posterior LKJ covariance deterministics differ")


def _validate_deterministics(posterior: object, inputs: LOEOFoldInputs) -> None:
    raw = np.asarray(posterior["gamma_innovation_raw"], dtype=float)
    sigma_gamma = np.asarray(posterior["sigma_gamma"], dtype=float)
    innovation = np.asarray(posterior["gamma_innovation"], dtype=float)
    expected_innovation = raw * sigma_gamma[..., :, None]
    path = np.concatenate(
        (np.zeros((*innovation.shape[:-1], 1)), np.cumsum(innovation, axis=-1)), axis=-1
    )
    expected_gamma = path - path.mean(axis=-1, keepdims=True)
    if not _close(innovation, expected_innovation) or not _close(
        posterior["gamma"], expected_gamma
    ):
        raise LOEOFoldError("LOEO posterior cyclic gamma deterministics differ")

    data = inputs.hqrc_data
    holiday = data.occurrence_holiday_type
    restriction = data.occurrence_restriction.astype(float)
    mu = np.asarray(posterior["mu"], dtype=float)
    delta = np.asarray(posterior["delta"], dtype=float)
    offset = np.asarray(posterior["beta_offset"], dtype=float)
    chol = np.asarray(posterior["between_cholesky"], dtype=float)[..., holiday, :, :]
    transformed = np.einsum("...eij,...ej->...ei", chol, offset)
    expected_beta = (
        mu[..., holiday, :]
        + delta[..., holiday, :] * restriction[None, None, :, None]
        + transformed
    )
    beta = np.asarray(posterior["beta"], dtype=float)
    if not _close(beta, expected_beta):
        raise LOEOFoldError("LOEO posterior beta deterministic differs")

    design = np.column_stack((np.ones(data.observations.size), data.tau_days, data.tau_days**2))
    mean = np.sum(beta[..., data.occurrence_index, :] * design, axis=-1)
    gamma = np.asarray(posterior["gamma"], dtype=float)
    mean += gamma[..., data.holiday_type_index, data.hour]
    residual = np.asarray(data.observations)[None, None, :] - mean
    segments = [residual[..., segment] for segment in data.segments]
    expected_likelihood = event_reset_ar1_logp_numpy(
        segments, posterior["phi"], posterior["sigma_r"]
    )
    if not _close(posterior["event_log_likelihood"], expected_likelihood):
        raise LOEOFoldError("LOEO posterior event likelihood deterministic differs")


def _validate_log_likelihood(idata: object, posterior: object) -> None:
    group = getattr(idata, "log_likelihood", None)
    expected_dim = "event_log_likelihood_dim_0"
    if (
        group is None
        or set(group.data_vars) != {"event"}
        or tuple(group["event"].dims) != ("chain", "draw", expected_dim)
        or set(group.coords) != {"chain", "draw", expected_dim}
        or not np.array_equal(group["event"], posterior["event_log_likelihood"])
    ):
        raise LOEOFoldError("LOEO log_likelihood group mirror differs")
    for name in ("chain", "draw", expected_dim):
        if not np.array_equal(group.coords[name], posterior.coords[name]):
            raise LOEOFoldError("LOEO log_likelihood coordinate mirror differs")


def validate_h3_posterior(idata: object, inputs: LOEOFoldInputs) -> Mapping[str, object]:
    """Fail closed unless a posterior is the exact finite H3 nine-event contract."""

    posterior = posterior_mapping(idata)
    if len(inputs.hqrc_data.occurrence_ids) != 9:
        raise LOEOFoldError("LOEO posterior requires exactly nine training events")
    try:
        chains = int(posterior.sizes["chain"])
        draws = int(posterior.sizes["draw"])
    except (KeyError, TypeError, ValueError) as error:
        raise LOEOFoldError("LOEO posterior sample dimensions differ") from error
    if chains <= 0 or draws <= 0:
        raise LOEOFoldError("LOEO posterior sample dimensions differ")
    _validate_schema(posterior, chains, draws)
    u_phi = np.asarray(posterior["u_phi"], dtype=float)
    phi = np.asarray(posterior["phi"], dtype=float)
    if (
        (u_phi <= 0).any()
        or (u_phi >= 1).any()
        or (np.abs(phi) >= 1).any()
        or not _close(phi, 2.0 * u_phi - 1.0)
        or (np.asarray(posterior["sigma_r"]) <= 0).any()
        or (np.asarray(posterior["sigma_gamma"]) <= 0).any()
    ):
        raise LOEOFoldError("LOEO posterior support or phi transform differs")
    _validate_covariance(posterior)
    _validate_deterministics(posterior, inputs)
    _validate_log_likelihood(idata, posterior)
    return posterior


__all__ = ["posterior_mapping", "validate_h3_posterior"]
