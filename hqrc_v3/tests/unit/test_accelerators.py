from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from hqrc_v3.accelerators import (
    DeviceProbe,
    DeviceResolutionError,
    ResolvedDevice,
    probe_hqrc_device,
    resolve_device,
)


class _Backend:
    def __init__(self) -> None:
        self.available = False

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1


class _FakeLKJ:
    def __init__(self, owner, *args, **kwargs) -> None:
        self.owner = owner
        self.real = torch.distributions.LKJCholesky(*args, **kwargs)

    def log_prob(self, value):
        if self.owner.lkj_log_prob_error:
            raise RuntimeError("LKJ log_prob unsupported")
        result = self.real.log_prob(value)
        return result.detach() if self.owner.detach_lkj_log_prob else result


class _FakeNormal:
    def __init__(self, owner, *args, **kwargs) -> None:
        self.owner = owner
        self.real = torch.distributions.Normal(*args, **kwargs)

    def log_prob(self, value):
        self.owner.normal_log_prob_calls.append((self.real.loc, value))
        return self.real.log_prob(value)


class _FakeDistributions:
    def __init__(self, owner) -> None:
        self.owner = owner

    def LKJCholesky(self, *args, **kwargs):
        return _FakeLKJ(self.owner, *args, **kwargs)

    def Normal(self, *args, **kwargs):
        return _FakeNormal(self.owner, *args, **kwargs)


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _Backend()
        self.mps = _Backend()
        self.backends = SimpleNamespace(mps=self.mps)
        self.distributions = _FakeDistributions(self)
        self.float64 = torch.float64
        self.probe_error: Exception | None = None
        self.probe_error_device: str | None = None
        self.lkj_log_prob_error = False
        self.detach_lkj_log_prob = False
        self.break_ar_gradients = False
        self.normal_log_prob_calls = []

    def tensor(self, *args, **kwargs):
        requested_device = kwargs.get("device")
        if self.probe_error is not None and requested_device in {
            "mps",
            self.probe_error_device,
        }:
            raise self.probe_error
        if self.break_ar_gradients and kwargs.get("requires_grad"):
            value = args[0]
            if value == [0.1, -0.2, 0.3, -0.1] or (
                isinstance(value, (float, int)) and value in {0.25, 0.8}
            ):
                kwargs["requires_grad"] = False
        kwargs["device"] = "cpu"
        return torch.tensor(*args, **kwargs)

    def isfinite(self, value):
        return torch.isfinite(value)


@pytest.fixture
def fake_torch() -> _FakeTorch:
    return _FakeTorch()


def test_importing_accelerators_does_not_import_torch():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import hqrc_v3.accelerators; assert 'torch' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cpu_resolution_runs_the_hqrc_probe(fake_torch):
    resolved = resolve_device("cpu", torch_module=fake_torch)

    assert resolved.kind == "cpu"
    assert resolved.logical_device == "cpu"
    assert resolved.probe == DeviceProbe(True, "float64-gradient-lkj-ar")


@pytest.mark.parametrize("mode", ["raises", "detached"])
def test_probe_rejects_broken_lkj_log_prob_gradient(fake_torch, mode):
    fake_torch.lkj_log_prob_error = mode == "raises"
    fake_torch.detach_lkj_log_prob = mode == "detached"

    probe = probe_hqrc_device(
        ResolvedDevice("cpu", "cpu", None, DeviceProbe(False, "not-run")),
        torch_module=fake_torch,
    )

    assert not probe.success


def test_probe_rejects_missing_ar_gradients(fake_torch):
    fake_torch.break_ar_gradients = True

    probe = probe_hqrc_device(
        ResolvedDevice("cpu", "cpu", None, DeviceProbe(False, "not-run")),
        torch_module=fake_torch,
    )

    assert not probe.success


def test_probe_ar_density_restarts_at_event_boundaries(fake_torch):
    probe = probe_hqrc_device(
        ResolvedDevice("cpu", "cpu", None, DeviceProbe(False, "not-run")),
        torch_module=fake_torch,
    )

    assert probe.success
    assert len(fake_torch.normal_log_prob_calls) == 4
    assert [value.numel() for _location, value in fake_torch.normal_log_prob_calls] == [
        1,
        1,
        1,
        1,
    ]


def test_auto_prefers_cuda_and_records_physical_identity(fake_torch, monkeypatch):
    monkeypatch.setenv("HQRC_PHYSICAL_DEVICE", "1")
    fake_torch.cuda.available = True

    resolved = resolve_device("auto", torch_module=fake_torch, platform_name="Windows")

    assert resolved.kind == "cuda"
    assert resolved.logical_device == "cuda:0"
    assert resolved.physical_device == "1"
    assert resolved.probe.success


def test_physical_cuda_mapping_requires_one_visible_device(fake_torch, monkeypatch):
    monkeypatch.setenv("HQRC_PHYSICAL_DEVICE", "1")
    fake_torch.cuda.available = True
    monkeypatch.setattr(fake_torch.cuda, "device_count", lambda: 2)

    with pytest.raises(DeviceResolutionError, match="exactly one visible CUDA device"):
        resolve_device("cuda", torch_module=fake_torch, platform_name="Linux")


def test_child_accepts_explicit_logical_cuda_zero(fake_torch, monkeypatch):
    monkeypatch.setenv("HQRC_PHYSICAL_DEVICE", "1")
    fake_torch.cuda.available = True

    resolved = resolve_device("cuda:0", torch_module=fake_torch, platform_name="Windows")

    assert resolved.logical_device == "cuda:0"
    assert resolved.physical_device == "1"


@pytest.mark.parametrize("device_name", ["cpu", "mps", "cuda:1"])
def test_isolated_cuda_child_rejects_nonzero_logical_device(fake_torch, monkeypatch, device_name):
    monkeypatch.setenv("HQRC_PHYSICAL_DEVICE", "1")
    fake_torch.cuda.available = True
    fake_torch.mps.available = True

    with pytest.raises(DeviceResolutionError, match="logical cuda:0"):
        resolve_device(device_name, torch_module=fake_torch, platform_name="Darwin")


def test_isolated_cuda_child_auto_fails_when_cuda_is_unavailable(fake_torch, monkeypatch):
    monkeypatch.setenv("HQRC_PHYSICAL_DEVICE", "1")

    with pytest.raises(DeviceResolutionError, match="CUDA.*not available"):
        resolve_device("auto", torch_module=fake_torch, platform_name="Windows")


def test_isolated_cuda_child_auto_fails_when_cuda_probe_fails(fake_torch, monkeypatch):
    monkeypatch.setenv("HQRC_PHYSICAL_DEVICE", "1")
    fake_torch.cuda.available = True
    fake_torch.probe_error = RuntimeError("CUDA probe failed")
    fake_torch.probe_error_device = "cuda:0"

    with pytest.raises(DeviceResolutionError, match="CUDA probe failed"):
        resolve_device("auto", torch_module=fake_torch, platform_name="Windows")


def test_isolated_cuda_child_auto_uses_cuda_even_on_darwin(fake_torch, monkeypatch):
    monkeypatch.setenv("HQRC_PHYSICAL_DEVICE", "1")
    fake_torch.cuda.available = True

    resolved = resolve_device("auto", torch_module=fake_torch, platform_name="Darwin")

    assert resolved.kind == "cuda"
    assert resolved.logical_device == "cuda:0"
    assert resolved.physical_device == "1"


def test_auto_mps_probe_failure_falls_back_but_explicit_mps_fails(fake_torch):
    fake_torch.mps.available = True
    fake_torch.probe_error = RuntimeError("unsupported op")

    resolved = resolve_device("auto", torch_module=fake_torch, platform_name="Darwin")

    assert resolved.kind == "cpu"
    assert "unsupported op" in resolved.fallback_reason
    assert resolved.probe.success
    with pytest.raises(DeviceResolutionError, match="unsupported op"):
        resolve_device("mps", torch_module=fake_torch, platform_name="Darwin")


@pytest.mark.parametrize("device_name", ["cuda", "mps"])
def test_explicit_unavailable_accelerator_fails_closed(fake_torch, device_name):
    with pytest.raises(DeviceResolutionError, match="not available"):
        resolve_device(device_name, torch_module=fake_torch, platform_name="Darwin")
