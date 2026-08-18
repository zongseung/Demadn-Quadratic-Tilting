from __future__ import annotations

import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

from hqrc_v3.accelerators import DeviceResolutionError, resolve_device


class _Backend:
    def __init__(self) -> None:
        self.available = False

    def is_available(self) -> bool:
        return self.available

    def device_count(self) -> int:
        return 1


class _FakeTorch:
    def __init__(self) -> None:
        self.cuda = _Backend()
        self.mps = _Backend()
        self.backends = SimpleNamespace(mps=self.mps)
        self.distributions = torch.distributions
        self.float64 = torch.float64
        self.probe_error: Exception | None = None

    def tensor(self, *args, **kwargs):
        if self.probe_error is not None and kwargs.get("device") == "mps":
            raise self.probe_error
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
    assert resolved.probe == type(resolved.probe)(True, "float64-gradient-lkj-ar")


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
