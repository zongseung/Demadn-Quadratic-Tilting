"""Lazy accelerator selection and HQRC capability probing."""

from __future__ import annotations

import importlib
import os
import platform
import re
from dataclasses import dataclass
from typing import Literal


class DeviceResolutionError(RuntimeError):
    """Raised when an explicitly requested accelerator cannot run HQRC."""


@dataclass(frozen=True)
class DeviceProbe:
    success: bool
    detail: str


@dataclass(frozen=True)
class ResolvedDevice:
    kind: Literal["cpu", "cuda", "mps"]
    logical_device: str
    physical_device: str | None
    probe: DeviceProbe
    fallback_reason: str | None = None


def probe_hqrc_device(device: ResolvedDevice, *, torch_module=None) -> DeviceProbe:
    """Exercise NUTS-required float64 LKJ and stationary AR(1) gradients."""

    torch = torch_module or importlib.import_module("torch")
    try:
        cholesky = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [0.2, 0.9797958971132712, 0.0],
                [-0.1, 0.15, 0.9836157786453001],
            ],
            dtype=torch.float64,
            device=device.logical_device,
            requires_grad=True,
        )
        concentration = torch.tensor(2.0, dtype=torch.float64, device=device.logical_device)
        lkj_logp = torch.distributions.LKJCholesky(3, concentration).log_prob(cholesky)
        residual = torch.tensor(
            [0.1, -0.2, 0.3, -0.1],
            dtype=torch.float64,
            device=device.logical_device,
            requires_grad=True,
        )
        phi = torch.tensor(
            0.25,
            dtype=torch.float64,
            device=device.logical_device,
            requires_grad=True,
        )
        sigma = torch.tensor(
            0.8,
            dtype=torch.float64,
            device=device.logical_device,
            requires_grad=True,
        )
        stationary = sigma / (1.0 - phi.square()).sqrt()
        normal = torch.distributions.Normal
        ar_logp = None
        for event in (residual[:2], residual[2:]):
            event_logp = normal(0.0, stationary).log_prob(event[0])
            event_logp = event_logp + normal(phi * event[:-1], sigma).log_prob(event[1:]).sum()
            ar_logp = event_logp if ar_logp is None else ar_logp + event_logp
        assert ar_logp is not None
        objective = lkj_logp + ar_logp
        objective.backward()
        gradients = (cholesky.grad, residual.grad, phi.grad, sigma.grad)
        if not bool(torch.isfinite(objective).all()) or any(
            gradient is None or not bool(torch.isfinite(gradient).all()) for gradient in gradients
        ):
            raise RuntimeError("missing or non-finite float64 HQRC device-probe gradient")
    except Exception as error:
        return DeviceProbe(False, str(error))
    return DeviceProbe(True, "float64-gradient-lkj-ar")


def _available(torch, kind: str) -> bool:
    backend = torch.cuda if kind == "cuda" else torch.backends.mps
    return bool(backend.is_available())


def _candidate(kind: Literal["cuda", "mps"], torch, *, logical_device: str) -> ResolvedDevice:
    physical = os.environ.get("HQRC_PHYSICAL_DEVICE") if kind == "cuda" else None
    if physical is not None:
        if torch.cuda.device_count() != 1:
            raise DeviceResolutionError(
                "HQRC_PHYSICAL_DEVICE requires exactly one visible CUDA device"
            )
        if logical_device != "cuda:0":
            raise DeviceResolutionError("an isolated CUDA child must use logical cuda:0")
    return ResolvedDevice(
        kind=kind,
        logical_device=logical_device,
        physical_device=physical,
        probe=DeviceProbe(False, "not-run"),
    )


def _cpu(*, fallback_reason: str | None = None, probe: DeviceProbe | None = None):
    return ResolvedDevice(
        kind="cpu",
        logical_device="cpu",
        physical_device=None,
        probe=probe or DeviceProbe(True, "cpu"),
        fallback_reason=fallback_reason,
    )


def _probed_cpu(torch, *, fallback_reason: str | None = None) -> ResolvedDevice:
    candidate = _cpu(probe=DeviceProbe(False, "not-run"))
    probe = probe_hqrc_device(candidate, torch_module=torch)
    if not probe.success:
        raise DeviceResolutionError(f"cpu HQRC probe failed: {probe.detail}")
    return _cpu(fallback_reason=fallback_reason, probe=probe)


def _resolve_isolated_cuda(request: str, torch, physical: str) -> ResolvedDevice:
    if request not in {"auto", "cuda", "cuda:0"}:
        raise DeviceResolutionError("an isolated CUDA child must use logical cuda:0")
    if not _available(torch, "cuda"):
        raise DeviceResolutionError("isolated CUDA child: CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise DeviceResolutionError("HQRC_PHYSICAL_DEVICE requires exactly one visible CUDA device")
    candidate = ResolvedDevice(
        kind="cuda",
        logical_device="cuda:0",
        physical_device=physical,
        probe=DeviceProbe(False, "not-run"),
    )
    probe = probe_hqrc_device(candidate, torch_module=torch)
    if not probe.success:
        raise DeviceResolutionError(f"cuda HQRC probe failed: {probe.detail}")
    return ResolvedDevice("cuda", "cuda:0", physical, probe)


def resolve_device(
    request: str, *, torch_module=None, platform_name: str | None = None
) -> ResolvedDevice:
    """Resolve and capability-probe one HQRC worker device lazily."""

    cuda_match = re.fullmatch(r"cuda:(0|[1-9][0-9]*)", request)
    if request not in {"auto", "cpu", "cuda", "mps"} and cuda_match is None:
        raise DeviceResolutionError("device must be 'auto', 'cpu', 'cuda', 'cuda:N', or 'mps'")
    try:
        torch = torch_module or importlib.import_module("torch")
    except Exception as error:
        raise DeviceResolutionError(f"{request} is not available: {error}") from error
    physical = os.environ.get("HQRC_PHYSICAL_DEVICE")
    if physical is not None:
        return _resolve_isolated_cuda(request, torch, physical)
    if request == "cpu":
        return _probed_cpu(torch)

    selected = "mps" if (platform_name or platform.system()) == "Darwin" else "cuda"
    kind = selected if request == "auto" else ("cuda" if request.startswith("cuda") else request)
    if not _available(torch, kind):
        reason = f"{kind} is not available"
        if request == "auto":
            return _probed_cpu(torch, fallback_reason=reason)
        raise DeviceResolutionError(reason)

    logical_device = request if cuda_match else ("cuda:0" if kind == "cuda" else "mps")
    candidate = _candidate(kind, torch, logical_device=logical_device)
    probe = probe_hqrc_device(candidate, torch_module=torch)
    if not probe.success:
        if request == "auto":
            return _probed_cpu(torch, fallback_reason=probe.detail)
        raise DeviceResolutionError(f"{kind} HQRC probe failed: {probe.detail}")
    return ResolvedDevice(
        kind=candidate.kind,
        logical_device=candidate.logical_device,
        physical_device=candidate.physical_device,
        probe=probe,
    )
