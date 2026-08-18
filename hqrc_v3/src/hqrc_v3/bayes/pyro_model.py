"""Shared float64 PyTorch/Pyro implementation of the paper H1--H3 model."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
from torch import Tensor

from hqrc_v3.bayes.model import HQRCData, HQRCModelOptions, Pooling, Variant
from hqrc_v3.diagnostics.ar import ApprovedARCalibration, require_approved_calibration

_DTYPE = torch.float64


class PyroModelError(ValueError):
    """Raised when the Pyro backend is asked to run a non-paper model."""


def _float64(value: object, *, device: torch.device | None = None) -> Tensor:
    return torch.as_tensor(value, dtype=_DTYPE, device=device)


def centered_cyclic_hour_profile_torch(innovations: Tensor) -> Tensor:
    """Convert free RW1 innovations into an identified sum-zero cyclic profile."""

    values = _float64(innovations, device=innovations.device)
    zeros = torch.zeros((*values.shape[:-1], 1), dtype=_DTYPE, device=values.device)
    path = torch.cat((zeros, torch.cumsum(values, dim=-1)), dim=-1)
    return path - path.mean(dim=-1, keepdim=True)


def hqrc_mean_torch(
    data: HQRCData,
    *,
    beta: Tensor,
    variant: Variant,
    gamma: Tensor | None = None,
) -> Tensor:
    """Evaluate the H1--H3 mean equation at fixed Torch coefficient values."""

    if variant not in {"H1", "H2", "H3"}:
        raise PyroModelError("Pyro HQRC supports only H1, H2, and H3")
    coefficients = _float64(beta, device=beta.device)
    device = coefficients.device
    occurrence = torch.as_tensor(data.occurrence_index, dtype=torch.long, device=device)
    if variant == "H1":
        mean = coefficients[..., occurrence, 0]
    else:
        tau = _float64(data.tau_days, device=device)
        design = torch.stack((torch.ones_like(tau), tau, tau.square()), dim=-1)
        mean = (coefficients[..., occurrence, :] * design).sum(dim=-1)
    if variant == "H3":
        if gamma is None:
            raise PyroModelError("H3 requires a cyclic hour profile")
        profile = _float64(gamma, device=device)
        holiday_type = torch.as_tensor(data.holiday_type_index, dtype=torch.long, device=device)
        hour = torch.as_tensor(data.hour, dtype=torch.long, device=device)
        mean = mean + profile[..., holiday_type, hour]
    return mean


def event_reset_ar1_logp_torch(
    residual: Tensor,
    data: HQRCData,
    *,
    phi: Tensor | float,
    sigma: Tensor | float,
) -> Tensor:
    """Return stationary normal AR(1) log densities, restarting at each event."""

    values = _float64(residual, device=residual.device)
    phi_value = _float64(phi, device=values.device)
    sigma_value = _float64(sigma, device=values.device)
    stationary_sd = sigma_value / torch.sqrt(1.0 - phi_value.square())

    def normal_logp(value: Tensor, scale: Tensor) -> Tensor:
        return -0.5 * ((value / scale).square() + torch.log(2.0 * torch.pi * scale.square()))

    terms = []
    for segment in data.segments:
        event = values[..., segment]
        term = normal_logp(event[..., 0], stationary_sd)
        innovations = event[..., 1:] - phi_value[..., None] * event[..., :-1]
        term = term + normal_logp(innovations, sigma_value[..., None]).sum(dim=-1)
        terms.append(term)
    return torch.stack(terms, dim=-1)


def _stack_sites(samples: Mapping[str, Tensor], name: str, *, dimension: int) -> Tensor:
    values = [
        _float64(samples[f"{name}_{index}"], device=samples[f"{name}_{index}"].device)
        for index in range(2)
    ]
    return torch.stack(values, dim=dimension)


def reconstruct_pyro_deterministics(
    samples: Mapping[str, Tensor], data: HQRCData, variant: Variant
) -> dict[str, Tensor]:
    """Reconstruct PyMC-compatible deterministic variables from Pyro sample sites."""

    if variant not in {"H1", "H2", "H3"}:
        raise PyroModelError("Pyro HQRC supports only H1, H2, and H3")
    coefficients = 1 if variant == "H1" else 3
    mu = _float64(samples["mu"], device=samples["mu"].device)
    device = mu.device
    delta = _float64(samples["delta"], device=device)
    offset = _float64(samples["beta_offset"], device=device)
    sampled_scale = _stack_sites(samples, "between_scale", dimension=-2).to(device)
    if coefficients == 1:
        identity = torch.ones((*sampled_scale.shape[:-1], 1, 1), dtype=_DTYPE, device=device)
        correlation_cholesky = identity
    else:
        correlation_cholesky = _stack_sites(samples, "between_corr_cholesky", dimension=-3).to(
            device
        )
    between_cholesky = sampled_scale.unsqueeze(-1) * correlation_cholesky
    covariance = between_cholesky @ between_cholesky.transpose(-1, -2)
    between_scale = torch.sqrt(torch.diagonal(covariance, dim1=-2, dim2=-1))
    correlation = covariance / (between_scale.unsqueeze(-1) * between_scale.unsqueeze(-2))
    lower = torch.tril_indices(coefficients, coefficients, device=device)

    occurrence_type = torch.as_tensor(data.occurrence_holiday_type, dtype=torch.long, device=device)
    restriction = _float64(data.occurrence_restriction, device=device)
    location = mu[..., occurrence_type, :] + (delta[..., occurrence_type, :] * restriction[:, None])
    selected_cholesky = between_cholesky[..., occurrence_type, :, :]
    beta = location + torch.einsum("...nij,...nj->...ni", selected_cholesky, offset)

    result = {
        "between_scale": between_scale,
        "between_cholesky": between_cholesky,
        "beta": beta,
    }
    for holiday_type in range(2):
        result.update(
            {
                f"between_cov_{holiday_type}": between_cholesky[
                    ..., holiday_type, lower[0], lower[1]
                ],
                f"between_cov_{holiday_type}_corr": correlation[..., holiday_type, :, :],
                f"between_cov_{holiday_type}_stds": between_scale[..., holiday_type, :],
            }
        )
    gamma = None
    if variant == "H3":
        sigma_gamma = _float64(samples["sigma_gamma"], device=device)
        raw = _float64(samples["gamma_innovation_raw"], device=device)
        innovation = raw * sigma_gamma.unsqueeze(-1)
        gamma = centered_cyclic_hour_profile_torch(innovation)
        result.update({"gamma_innovation": innovation, "gamma": gamma})

    u_phi = _float64(samples["u_phi"], device=device)
    phi = 2.0 * u_phi - 1.0
    mean = hqrc_mean_torch(data, beta=beta, variant=variant, gamma=gamma)
    observations = _float64(data.observations, device=device)
    residual = observations - mean
    event_log_likelihood = event_reset_ar1_logp_torch(
        residual,
        data,
        phi=phi,
        sigma=_float64(samples["sigma_r"], device=device),
    )
    result.update({"phi": phi, "event_log_likelihood": event_log_likelihood})
    return result


def _paper_options(
    variant: Variant, pooling: Pooling, options: HQRCModelOptions | None
) -> HQRCModelOptions:
    if variant not in {"H1", "H2", "H3"}:
        raise PyroModelError("Pyro HQRC supports only H1, H2, and H3")
    if pooling != "partial":
        raise PyroModelError("Pyro HQRC requires partial pooling")
    selected = options or HQRCModelOptions()
    selected.validate()
    if selected.covariance != "full":
        raise PyroModelError("Pyro HQRC requires full covariance")
    if not selected.include_restriction:
        raise PyroModelError("Pyro HQRC requires restriction effects")
    if selected.innovation != "normal_ar1":
        raise PyroModelError("Pyro HQRC requires normal_ar1 innovations")
    return selected


def build_pyro_hqrc_model(
    data: HQRCData,
    calibration: ApprovedARCalibration,
    *,
    variant: Variant = "H3",
    pooling: Pooling = "partial",
    options: HQRCModelOptions | None = None,
    device: str = "cpu",
) -> Callable[[], None]:
    """Build the exact paper-only Pyro H1--H3 model on one explicit device."""

    if not isinstance(data, HQRCData):
        raise TypeError("data must be HQRCData")
    selected = _paper_options(variant, pooling, options)
    approved = require_approved_calibration(calibration)
    if (
        not math.isfinite(approved.a)
        or not math.isfinite(approved.b)
        or approved.a <= 1.0
        or approved.b <= 1.0
    ):
        raise PyroModelError("approved AR calibration parameters must exceed one")

    import pyro
    import pyro.distributions as dist

    torch_device = torch.device(device)
    coefficients = 1 if variant == "H1" else 3
    zero = torch.tensor(0.0, dtype=_DTYPE, device=torch_device)
    one = torch.tensor(1.0, dtype=_DTYPE, device=torch_device)
    two = torch.tensor(2.0, dtype=_DTYPE, device=torch_device)

    def model() -> None:
        samples: dict[str, Tensor] = {
            "mu": pyro.sample("mu", dist.Normal(zero, two).expand((2, coefficients)).to_event(2)),
            "delta": pyro.sample(
                "delta", dist.Normal(zero, two).expand((2, coefficients)).to_event(2)
            ),
            "beta_offset": pyro.sample(
                "beta_offset",
                dist.Normal(zero, one).expand((len(data.occurrence_ids), coefficients)).to_event(2),
            ),
        }
        for holiday_type in range(2):
            samples[f"between_scale_{holiday_type}"] = pyro.sample(
                f"between_scale_{holiday_type}",
                dist.HalfNormal(
                    torch.tensor(selected.between_scale_prior, dtype=_DTYPE, device=torch_device)
                )
                .expand((coefficients,))
                .to_event(1),
            )
            if coefficients > 1:
                samples[f"between_corr_cholesky_{holiday_type}"] = pyro.sample(
                    f"between_corr_cholesky_{holiday_type}",
                    dist.LKJCholesky(
                        coefficients,
                        torch.tensor(selected.lkj_eta, dtype=_DTYPE, device=torch_device),
                    ),
                )
        if variant == "H3":
            samples["sigma_gamma"] = pyro.sample(
                "sigma_gamma",
                dist.HalfNormal(_float64(0.5, device=torch_device)).expand((2,)).to_event(1),
            )
            samples["gamma_innovation_raw"] = pyro.sample(
                "gamma_innovation_raw", dist.Normal(zero, one).expand((2, 23)).to_event(2)
            )
        samples["sigma_r"] = pyro.sample("sigma_r", dist.HalfNormal(one))
        samples["u_phi"] = pyro.sample(
            "u_phi",
            dist.Beta(
                _float64(approved.a, device=torch_device),
                _float64(approved.b, device=torch_device),
            ),
        )

        deterministic = reconstruct_pyro_deterministics(samples, data, variant)
        for name, value in deterministic.items():
            pyro.deterministic(name, value)
        if variant == "H3":
            innovation = deterministic["gamma_innovation"]
            sigma_gamma = samples["sigma_gamma"]
            closure = dist.Normal(zero, sigma_gamma).log_prob(-innovation.sum(dim=-1))
            pyro.factor("gamma_circular_random_walk", (closure + sigma_gamma.log()).sum())
        pyro.factor("event_reset_ar1", deterministic["event_log_likelihood"].sum())

    return model
