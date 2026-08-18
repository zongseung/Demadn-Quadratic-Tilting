"""Sampler selection and transparent diagnostics for HQRC Bayesian models."""

from __future__ import annotations

import importlib.metadata
import json
import math
import time
from dataclasses import asdict, dataclass
from typing import Literal

import arviz as az
import numpy as np
import xarray as xr

from hqrc_v3.accelerators import ResolvedDevice, resolve_device
from hqrc_v3.bayes.model import (
    CYCLIC_HOUR_PARAMETERIZATION,
    HQRCData,
    HQRCModelOptions,
    Pooling,
    Variant,
    build_hqrc_model,
)
from hqrc_v3.diagnostics.ar import ApprovedARCalibration, require_approved_calibration


class SamplingError(RuntimeError):
    """Raised when an unavailable sampler or paper-profile diagnostic gate fails."""


PYMC_INITIALIZATION = "adapt_diag"
PYMC_INITIALIZATION_CHOICES = frozenset({"adapt_diag", "jitter+adapt_diag"})
SAMPLER_GEOMETRY = "noncentered-cyclic-hour-rw1-v1"
_PRIVATE_PYRO_SITES = frozenset(
    {
        "between_scale_0",
        "between_scale_1",
        "between_corr_cholesky_0",
        "between_corr_cholesky_1",
    }
)


@dataclass(frozen=True)
class SamplingDiagnostics:
    max_rhat: float
    min_bulk_ess: float
    min_tail_ess: float
    divergences: int


def _summary_value(summary, name: str, reducer, default: float) -> float:
    if name not in summary:
        return default
    value = float(reducer(np.asarray(summary[name], dtype=float)))
    return value if math.isfinite(value) else default


def validate_inference_data(idata: az.InferenceData, *, paper_profile: bool) -> SamplingDiagnostics:
    """Return diagnostics and enforce the stricter paper-run thresholds when requested."""

    if not isinstance(idata, az.InferenceData) or not hasattr(idata, "posterior"):
        raise SamplingError("sampler did not return ArviZ InferenceData with posterior draws")
    summary = az.summary(idata, kind="diagnostics")
    if not hasattr(idata, "sample_stats") or "diverging" not in idata.sample_stats:
        if paper_profile:
            raise SamplingError("paper_profile requires sample_stats.diverging diagnostics")
        divergences = 0
    else:
        raw_diverging = np.asarray(idata.sample_stats["diverging"])
        if (
            raw_diverging.dtype.kind not in {"b", "i", "u"}
            or not np.isfinite(raw_diverging.astype(float)).all()
        ):
            raise SamplingError("sample_stats.diverging must be finite boolean/integer diagnostics")
        divergences = int(raw_diverging.sum())
    diagnostics = SamplingDiagnostics(
        max_rhat=_summary_value(summary, "r_hat", np.nanmax, math.inf),
        min_bulk_ess=_summary_value(summary, "ess_bulk", np.nanmin, 0.0),
        min_tail_ess=_summary_value(summary, "ess_tail", np.nanmin, 0.0),
        divergences=divergences,
    )
    if paper_profile and (
        diagnostics.max_rhat > 1.01
        or diagnostics.min_bulk_ess < 400
        or diagnostics.min_tail_ess < 400
        or diagnostics.divergences
    ):
        raise SamplingError(str(diagnostics))
    return diagnostics


def build_pyro_hqrc_model(*args, **kwargs):
    """Load the Torch-backed model only after the Pyro backend is selected."""

    from hqrc_v3.bayes.pyro_model import build_pyro_hqrc_model as build

    return build(*args, **kwargs)


def reconstruct_pyro_deterministics(*args, **kwargs):
    """Load deterministic reconstruction only on the selected Pyro path."""

    from hqrc_v3.bayes.pyro_model import reconstruct_pyro_deterministics as reconstruct

    return reconstruct(*args, **kwargs)


def _run_pyro_chain(
    model,
    *,
    draws: int,
    tune: int,
    seed: int,
    target_accept: float,
    num_chains: int,
):
    import pyro
    from pyro.infer import MCMC, NUTS

    pyro.set_rng_seed(seed)
    kernel = NUTS(model, target_accept_prob=target_accept)
    mcmc = MCMC(
        kernel,
        num_samples=draws,
        warmup_steps=tune,
        num_chains=num_chains,
        disable_progbar=True,
    )
    mcmc.run()
    divergences = np.zeros(draws, dtype=np.int8)
    for index in mcmc.diagnostics().get("divergences", {}).get("chain 0", ()):
        divergences[int(index)] = 1
    return mcmc.get_samples(group_by_chain=False), divergences


def _numpy(value) -> np.ndarray:
    return np.asarray(value.detach().cpu().numpy())


def _sample_pyro(
    data: HQRCData,
    calibration: ApprovedARCalibration,
    *,
    variant: Variant,
    pooling: Pooling,
    options: HQRCModelOptions | None,
    draws: int,
    tune: int,
    seed: int,
    target_accept: float,
    device: str,
) -> tuple[az.InferenceData, ResolvedDevice]:
    resolved = resolve_device(device)
    model = build_pyro_hqrc_model(
        data,
        calibration,
        variant=variant,
        pooling=pooling,
        options=options,
        device=resolved.logical_device,
    )
    chain_posteriors: list[dict[str, np.ndarray]] = []
    chain_divergences = []
    for chain in range(4):
        samples, divergences = _run_pyro_chain(
            model,
            draws=draws,
            tune=tune,
            seed=seed + chain,
            target_accept=target_accept,
            num_chains=1,
        )
        deterministic = reconstruct_pyro_deterministics(samples, data, variant)
        public_samples = {
            name: _numpy(value)
            for name, value in samples.items()
            if name not in _PRIVATE_PYRO_SITES
        }
        public_samples.update({name: _numpy(value) for name, value in deterministic.items()})
        chain_posteriors.append(public_samples)
        chain_divergences.append(np.asarray(divergences, dtype=np.int8))
    posterior = {
        name: np.stack([chain[name] for chain in chain_posteriors]) for name in chain_posteriors[0]
    }
    idata = az.from_dict(
        posterior=posterior,
        sample_stats={"diverging": np.stack(chain_divergences).astype(np.int8)},
    )
    return idata, resolved


def sample_hqrc(
    data: HQRCData,
    calibration: ApprovedARCalibration,
    *,
    variant: Variant = "H3",
    pooling: Pooling = "partial",
    options: HQRCModelOptions | None = None,
    draws: int = 1_000,
    tune: int = 1_000,
    chains: int = 4,
    cores: int = 1,
    seed: int = 11,
    init: str = PYMC_INITIALIZATION,
    target_accept: float | None = None,
    backend: Literal["pymc", "nutpie", "pyro"] = "pymc",
    device: str = "cpu",
    paper_profile: bool = False,
):
    """Sample with PyMC by default, loading accelerator backends only when selected."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in (draws, tune, chains, cores)
    ):
        raise ValueError("draws, tune, chains, and cores must be positive integers")
    if cores > chains:
        raise ValueError("cores must not exceed chains")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if init not in PYMC_INITIALIZATION_CHOICES:
        raise ValueError("init must be an approved PyMC initialization")
    if paper_profile and (chains != 4 or draws < 1_000 or tune < 1_000):
        raise ValueError("paper_profile requires exactly 4 chains and at least 1000 tune/draws")
    if backend == "pyro" and chains != 4:
        raise ValueError("pyro requires exactly 4 sequential chains")
    trusted_calibration = require_approved_calibration(calibration)
    model = None
    if backend in {"pymc", "nutpie"}:
        model = build_hqrc_model(data, trusted_calibration, variant, pooling, options)
    started = time.perf_counter()
    resolved_target_accept = 0.99 if paper_profile else 0.9
    if target_accept is not None:
        if (
            isinstance(target_accept, bool)
            or not isinstance(target_accept, (int, float))
            or not math.isfinite(float(target_accept))
            or float(target_accept) != resolved_target_accept
        ):
            raise ValueError("target_accept differs from the approved profile setting")
    resolved = None
    if backend == "pymc":
        import pymc as pm

        with model:
            idata = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                cores=cores,
                random_seed=seed,
                progressbar=False,
                compute_convergence_checks=False,
                target_accept=resolved_target_accept,
                init=init,
            )
    elif backend == "nutpie":
        if init != PYMC_INITIALIZATION:
            raise ValueError("nutpie does not support a PyMC initialization override")
        try:
            import nutpie
        except ImportError as error:
            raise SamplingError("nutpie backend requested but nutpie is not installed") from error
        idata = nutpie.sample_pymc(
            model,
            draws=draws,
            tune=tune,
            chains=chains,
            seed=seed,
            target_accept=resolved_target_accept,
        )
    elif backend == "pyro":
        if init != PYMC_INITIALIZATION:
            raise ValueError("pyro does not support a PyMC initialization override")
        idata, resolved = _sample_pyro(
            data,
            trusted_calibration,
            variant=variant,
            pooling=pooling,
            options=options,
            draws=draws,
            tune=tune,
            seed=seed,
            target_accept=resolved_target_accept,
            device=device,
        )
    else:
        raise ValueError("backend must be 'pymc', 'nutpie', or 'pyro'")
    if "event_log_likelihood" not in idata.posterior:
        raise SamplingError("HQRC posterior is missing occurrence-level log likelihood")
    idata.add_groups(
        {
            "log_likelihood": xr.Dataset(
                {"event": idata.posterior["event_log_likelihood"].rename("event")}
            )
        }
    )
    diagnostics = validate_inference_data(idata, paper_profile=paper_profile)
    if backend != "pyro":
        idata.attrs.update(
            {
                "hqrc_backend": backend,
                "hqrc_elapsed_seconds": time.perf_counter() - started,
                "hqrc_pymc_version": importlib.metadata.version("pymc"),
                "hqrc_arviz_version": importlib.metadata.version("arviz"),
                "hqrc_diagnostics_json": json.dumps(asdict(diagnostics), sort_keys=True),
                "hqrc_sampler_json": json.dumps(
                    {
                        "draws": draws,
                        "tune": tune,
                        "chains": chains,
                        "cores": cores,
                        "seed": seed,
                        "target_accept": resolved_target_accept,
                        "paper_profile": paper_profile,
                        "init": init if backend == "pymc" else "nutpie-default",
                        "geometry": SAMPLER_GEOMETRY,
                    },
                    sort_keys=True,
                ),
                "hqrc_model_json": json.dumps(
                    {
                        "variant": variant,
                        "pooling": pooling,
                        "options": asdict(options or HQRCModelOptions()),
                        "cyclic_hour_parameterization": CYCLIC_HOUR_PARAMETERIZATION,
                    },
                    sort_keys=True,
                ),
                "hqrc_calibration_json": json.dumps(
                    {
                        "artifact_path": str(trusted_calibration.artifact_path),
                        "artifact_digest": trusted_calibration.artifact_digest,
                        "residual_sha256": trusted_calibration.residual_sha256,
                        "config_sha256": trusted_calibration.config_sha256,
                        "event_sha256": trusted_calibration.event_sha256,
                        "context": asdict(trusted_calibration.context),
                        "a": trusted_calibration.a,
                        "b": trusted_calibration.b,
                    },
                    sort_keys=True,
                ),
            }
        )
        if backend == "nutpie":
            idata.attrs["hqrc_nutpie_version"] = importlib.metadata.version("nutpie")
        return idata

    assert resolved is not None
    sampler_metadata = {
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": cores,
        "seed": seed,
        "target_accept": resolved_target_accept,
        "paper_profile": paper_profile,
        "init": "pyro-default",
        "geometry": SAMPLER_GEOMETRY,
        "resolved_device_kind": resolved.kind,
        "logical_device": resolved.logical_device,
        "physical_device": resolved.physical_device,
        "dtype": "float64",
        "chain_execution": "sequential",
        "capability_probe": asdict(resolved.probe),
        "fallback_reason": resolved.fallback_reason,
    }
    attrs = {
        "hqrc_backend": backend,
        "hqrc_elapsed_seconds": time.perf_counter() - started,
        "hqrc_arviz_version": importlib.metadata.version("arviz"),
        "hqrc_diagnostics_json": json.dumps(asdict(diagnostics), sort_keys=True),
        "hqrc_sampler_json": json.dumps(sampler_metadata, sort_keys=True),
        "hqrc_model_json": json.dumps(
            {
                "variant": variant,
                "pooling": pooling,
                "options": asdict(options or HQRCModelOptions()),
                "cyclic_hour_parameterization": CYCLIC_HOUR_PARAMETERIZATION,
            },
            sort_keys=True,
        ),
        "hqrc_calibration_json": json.dumps(
            {
                "artifact_path": str(trusted_calibration.artifact_path),
                "artifact_digest": trusted_calibration.artifact_digest,
                "residual_sha256": trusted_calibration.residual_sha256,
                "config_sha256": trusted_calibration.config_sha256,
                "event_sha256": trusted_calibration.event_sha256,
                "context": asdict(trusted_calibration.context),
                "a": trusted_calibration.a,
                "b": trusted_calibration.b,
            },
            sort_keys=True,
        ),
    }
    attrs.update(
        {
            "hqrc_torch_version": importlib.metadata.version("torch"),
            "hqrc_pyro_version": importlib.metadata.version("pyro-ppl"),
            "hqrc_device": resolved.logical_device,
            "hqrc_physical_device": resolved.physical_device or "",
            "hqrc_dtype": "float64",
            "hqrc_device_probe": resolved.probe.detail,
            "hqrc_device_fallback_reason": resolved.fallback_reason or "",
            "hqrc_chain_execution": "sequential",
        }
    )
    idata.attrs.update(attrs)
    return idata
