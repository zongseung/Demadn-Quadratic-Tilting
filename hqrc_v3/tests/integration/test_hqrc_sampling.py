from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import arviz as az
import numpy as np
import pymc as pm
import pytest
import torch

import hqrc_v3.bayes.samplers as sampler_module
from hqrc_v3.accelerators import DeviceProbe, ResolvedDevice
from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.bayes.samplers import (
    SamplingDiagnostics,
    SamplingError,
    _run_pyro_chain,
    sample_hqrc,
    validate_inference_data,
)
from hqrc_v3.diagnostics.ar import (
    EventResidualContext,
    approve_calibration,
    calibrate_beta_prior,
    load_approved_calibration,
    write_ar_diagnostics,
)


@pytest.mark.slow
def test_tiny_hqrc_sampling_returns_finite_posterior(tmp_path):
    data = HQRCData(
        observations=np.array([0.1, 0.2, 0.0, 0.2, 0.3, 0.1, -0.1, 0.0]),
        occurrence_index=np.repeat(np.arange(2), 4),
        holiday_type_index=np.repeat(np.arange(2), 4),
        tau_days=np.tile(np.arange(4, dtype=float) / 24.0, 2),
        hour=np.tile(np.arange(4), 2),
        restriction=np.repeat(np.array([0, 1]), 4),
        occurrence_ids=("a", "b"),
    )
    calibration = calibrate_beta_prior(np.array([0.25, 0.35]), event_ids=("a", "b"))
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibration,
        residual_sha256="residual-hash",
        config_sha256="config-hash",
        event_sha256="event-hash",
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
    )
    approved = approve_calibration(
        proposal,
        tmp_path / "approved.json",
        current_residual_sha256="residual-hash",
        current_config_sha256="config-hash",
        current_event_sha256="event-hash",
    )
    calibration = load_approved_calibration(
        approved,
        current_residual_sha256="residual-hash",
        current_config_sha256="config-hash",
        current_event_sha256="event-hash",
    )
    idata = sample_hqrc(data, calibration, draws=30, tune=30, chains=2, seed=5)
    assert np.isfinite(idata.posterior["phi"]).all()
    assert "log_likelihood" in idata.groups()
    assert np.isfinite(az.loo(idata, var_name="event").elpd_loo)
    assert json.loads(idata.attrs["hqrc_sampler_json"])["target_accept"] == 0.9
    assert (
        json.loads(idata.attrs["hqrc_calibration_json"])["artifact_digest"]
        == calibration.artifact_digest
    )
    assert json.loads(idata.attrs["hqrc_calibration_json"])["context"] == {
        "feature_set": "B0",
        "model": "model",
        "seed": 5,
        "split_ids": ["oof-2020"],
    }
    netcdf_path = tmp_path / "posterior.nc"
    az.to_netcdf(idata, netcdf_path)
    round_trip = az.from_netcdf(netcdf_path)
    assert json.loads(round_trip.attrs["hqrc_model_json"])["variant"] == "H3"
    assert validate_inference_data(idata, paper_profile=False).divergences >= 0


def test_paper_profile_rejects_smoke_sampler_limits_before_model_build():
    with pytest.raises(ValueError, match="4 chains"):
        sample_hqrc(None, None, draws=30, tune=30, chains=2, paper_profile=True)


def test_pymc_sampler_uses_fixed_stable_initialization_and_records_geometry(tmp_path, monkeypatch):
    data = HQRCData(
        observations=np.array([0.1, 0.2, 0.0, 0.2, 0.3, 0.1, -0.1, 0.0]),
        occurrence_index=np.repeat(np.arange(2), 4),
        holiday_type_index=np.repeat(np.arange(2), 4),
        tau_days=np.tile(np.arange(4, dtype=float) / 24.0, 2),
        hour=np.tile(np.arange(4), 2),
        restriction=np.repeat(np.array([0, 1]), 4),
        occurrence_ids=("a", "b"),
    )
    calibration = calibrate_beta_prior(np.array([0.25, 0.35]), event_ids=("a", "b"))
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibration,
        residual_sha256="residual-hash",
        config_sha256="config-hash",
        event_sha256="event-hash",
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
    )
    approved = approve_calibration(
        proposal,
        tmp_path / "approved.json",
        current_residual_sha256="residual-hash",
        current_config_sha256="config-hash",
        current_event_sha256="event-hash",
    )
    approved = load_approved_calibration(
        approved,
        current_residual_sha256="residual-hash",
        current_config_sha256="config-hash",
        current_event_sha256="event-hash",
    )
    captured = {}

    def fake_sample(**kwargs):
        captured.update(kwargs)
        rng = np.random.default_rng(31)
        return az.from_dict(
            posterior={"event_log_likelihood": rng.normal(size=(2, 20, 2))},
            sample_stats={"diverging": np.zeros((2, 20), dtype=np.int8)},
        )

    monkeypatch.setattr(pm, "sample", fake_sample)
    idata = sample_hqrc(data, approved, draws=20, tune=20, chains=2, seed=5)
    sampler = json.loads(idata.attrs["hqrc_sampler_json"])
    model = json.loads(idata.attrs["hqrc_model_json"])

    assert captured["init"] == "adapt_diag"
    assert captured["cores"] == 1
    assert sampler["cores"] == 1
    assert sampler["init"] == "adapt_diag"
    assert sampler["geometry"] == "noncentered-cyclic-hour-rw1-v1"
    assert model["cyclic_hour_parameterization"] == "noncentered-rw1-v1"


def test_sampler_records_requested_cores_and_rejects_invalid_values(tmp_path, monkeypatch):
    data = HQRCData(
        observations=np.array([0.1, 0.2]),
        occurrence_index=np.array([0, 0]),
        holiday_type_index=np.array([0, 0]),
        tau_days=np.array([0.0, 1.0 / 24.0]),
        hour=np.array([0, 1]),
        restriction=np.array([0, 0]),
        occurrence_ids=("a",),
    )
    approved = SimpleNamespace(
        artifact_path=tmp_path / "approved.json",
        artifact_digest="digest",
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="event",
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
        a=0.1,
        b=0.2,
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.require_approved_calibration", lambda value: value)
    monkeypatch.setattr(
        "hqrc_v3.bayes.samplers.build_hqrc_model",
        lambda *_args: __import__("contextlib").nullcontext(),
    )
    captured = {}
    monkeypatch.setattr(
        pm,
        "sample",
        lambda **kwargs: (
            captured.update(kwargs)
            or az.from_dict(
                posterior={"event_log_likelihood": np.zeros((2, 2, 1))},
                sample_stats={"diverging": np.zeros((2, 2), dtype=np.int8)},
            )
        ),
    )
    monkeypatch.setattr(
        "hqrc_v3.bayes.samplers.validate_inference_data",
        lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 1.0, 1.0, 0),
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.importlib.metadata.version", lambda _: "test")
    idata = sample_hqrc(data, approved, draws=2, tune=2, chains=2, cores=2)
    assert captured["cores"] == 2
    assert json.loads(idata.attrs["hqrc_sampler_json"])["cores"] == 2
    for invalid in (0, True, 3):
        with pytest.raises(ValueError):
            sample_hqrc(data, approved, draws=2, tune=2, chains=2, cores=invalid)


def test_sample_hqrc_records_explicit_jitter_initialization(tmp_path, monkeypatch):
    data = HQRCData(
        observations=np.array([0.1, 0.2]),
        occurrence_index=np.array([0, 0]),
        holiday_type_index=np.array([0, 0]),
        tau_days=np.array([0.0, 1.0 / 24.0]),
        hour=np.array([0, 1]),
        restriction=np.array([0, 0]),
        occurrence_ids=("a",),
    )
    approved = SimpleNamespace(
        artifact_path=tmp_path / "approved.json",
        artifact_digest="digest",
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="event",
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
        a=0.1,
        b=0.2,
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.require_approved_calibration", lambda value: value)
    monkeypatch.setattr(
        "hqrc_v3.bayes.samplers.build_hqrc_model",
        lambda *_args: __import__("contextlib").nullcontext(),
    )
    captured = {}
    monkeypatch.setattr(
        pm,
        "sample",
        lambda **kwargs: (
            captured.update(kwargs)
            or az.from_dict(
                posterior={"event_log_likelihood": np.zeros((4, 2, 1))},
                sample_stats={"diverging": np.zeros((4, 2), dtype=np.int8)},
            )
        ),
    )
    monkeypatch.setattr(
        "hqrc_v3.bayes.samplers.validate_inference_data",
        lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 800.0, 700.0, 0),
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.importlib.metadata.version", lambda _: "test")
    idata = sample_hqrc(
        data,
        approved,
        draws=1000,
        tune=1000,
        chains=4,
        cores=2,
        init="jitter+adapt_diag",
        target_accept=0.99,
        paper_profile=True,
    )
    assert captured["init"] == "jitter+adapt_diag"
    assert captured["target_accept"] == 0.99
    assert json.loads(idata.attrs["hqrc_sampler_json"])["init"] == "jitter+adapt_diag"


def test_paper_diagnostics_fail_closed_without_divergence_statistics():
    idata = az.from_dict(posterior={"phi": np.zeros((4, 8))})
    with pytest.raises(SamplingError, match="diverging"):
        validate_inference_data(idata, paper_profile=True)


@pytest.mark.parametrize("cores", [2, 3, 5])
def test_pyro_rejects_unsupported_core_topology_before_model_build(cores):
    with pytest.raises(ValueError, match="cores=1 or cores=4"):
        sample_hqrc(None, None, backend="pyro", chains=4, cores=cores)


@pytest.mark.parametrize(
    ("device_request", "kind", "logical_device", "physical_device"),
    [
        ("cuda:0", "cuda", "cuda:0", "0"),
        ("mps", "mps", "mps", None),
        ("auto", "cuda", "cuda:0", "1"),
    ],
)
def test_parallel_pyro_rejects_non_cpu_resolved_device(
    device_request, kind, logical_device, physical_device, monkeypatch
):
    resolved = ResolvedDevice(
        kind=kind,
        logical_device=logical_device,
        physical_device=physical_device,
        probe=DeviceProbe(success=True, detail="test-probe"),
    )
    monkeypatch.setattr(sampler_module, "resolve_device", lambda _request: resolved)

    with pytest.raises(SamplingError, match="CPU"):
        sampler_module._sample_pyro(
            None,
            None,
            variant="H1",
            pooling="partial",
            options=None,
            draws=4,
            tune=4,
            seed=17,
            target_accept=0.9,
            device=device_request,
            cores=4,
        )


def test_pyro_runs_four_chains_through_spawn_pool_and_preserves_chain_order(tmp_path, monkeypatch):
    data = HQRCData(
        observations=np.linspace(-0.2, 0.2, 18),
        occurrence_index=np.repeat(np.arange(9), 2),
        holiday_type_index=np.repeat(np.arange(9) % 2, 2),
        tau_days=np.tile(np.array([0.0, 1 / 24]), 9),
        hour=np.tile(np.array([0, 1]), 9),
        restriction=np.repeat(np.arange(9) % 2, 2),
        occurrence_ids=tuple(f"event-{index}" for index in range(9)),
    )
    approved = SimpleNamespace(
        artifact_path=tmp_path / "approved.json",
        artifact_digest="digest",
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="event",
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
        a=2.0,
        b=3.0,
    )
    probe = DeviceProbe(success=True, detail="float64-gradient-lkj-ar")
    resolved = ResolvedDevice(
        kind="cpu",
        logical_device="cpu",
        physical_device=None,
        probe=probe,
    )
    monkeypatch.setattr(sampler_module, "require_approved_calibration", lambda value: value)
    monkeypatch.setattr(sampler_module, "resolve_device", lambda _request: resolved)
    monkeypatch.setattr(
        sampler_module,
        "build_pyro_hqrc_model",
        lambda *_args, **_kwargs: pytest.fail("parallel parent built the nested Pyro model"),
    )
    monkeypatch.setattr(
        sampler_module,
        "validate_inference_data",
        lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 800.0, 700.0, 0),
    )
    versions = {"arviz": "test-arviz", "torch": "test-torch", "pyro-ppl": "test-pyro"}
    monkeypatch.setattr(sampler_module.importlib.metadata, "version", versions.__getitem__)
    captured = {}

    class FakeSpawnPool:
        def imap_unordered(self, worker, requests, chunksize):
            captured["worker"] = worker
            captured["requests"] = tuple(requests)
            captured["chunksize"] = chunksize
            return iter(
                SimpleNamespace(
                    chain=request.chain,
                    posterior={
                        "phi": np.full(request.draws, request.chain, dtype=float),
                        "event_log_likelihood": np.full(
                            (request.draws, 9), request.chain, dtype=float
                        ),
                    },
                    divergences=np.zeros(request.draws, dtype=np.int8),
                )
                for request in reversed(captured["requests"])
            )

        def close(self):
            captured["closed"] = True

        def terminate(self):
            captured["terminated"] = True

        def join(self):
            captured["joined"] = True

    class FakeSpawnContext:
        def Pool(self, **kwargs):
            captured["pool"] = kwargs
            return FakeSpawnPool()

    def fake_get_context(method):
        captured["start_method"] = method
        return FakeSpawnContext()

    monkeypatch.setattr(sampler_module.multiprocessing, "get_context", fake_get_context)

    idata = sample_hqrc(
        data,
        approved,
        draws=4,
        tune=3,
        chains=4,
        cores=4,
        seed=17,
        backend="pyro",
        device="cpu",
    )

    assert captured["start_method"] == "spawn"
    assert captured["pool"] == {"processes": 4}
    assert captured["worker"] is sampler_module._run_initialized_pyro_chain_worker
    assert captured["chunksize"] == 1
    assert captured["closed"] is captured["joined"] is True
    assert "terminated" not in captured
    requests = captured["requests"]
    assert [request.chain for request in requests] == [0, 1, 2, 3]
    assert [request.seed for request in requests] == [17, 18, 19, 20]
    assert all(request.approved_ar_path == approved.artifact_path for request in requests)
    assert all(request.residual_sha256 == "residual" for request in requests)
    assert all(request.config_sha256 == "config" for request in requests)
    assert all(request.event_sha256 == "event" for request in requests)
    np.testing.assert_array_equal(idata.posterior["phi"][:, 0], np.arange(4))
    assert idata.posterior["event_log_likelihood"].shape == (4, 4, 9)
    sampler = json.loads(idata.attrs["hqrc_sampler_json"])
    assert sampler["cores"] == 4
    assert sampler["chain_execution"] == "parallel"
    assert idata.attrs["hqrc_chain_execution"] == "parallel"


def test_parallel_pyro_pool_terminates_and_joins_on_first_child_failure(monkeypatch):
    captured = {}
    original = RuntimeError("chain 1 failed")

    class FailingPool:
        def imap_unordered(self, worker, requests, chunksize):
            del worker, chunksize
            requests = tuple(requests)

            def results():
                yield SimpleNamespace(chain=requests[-1].chain)
                raise original

            return results()

        def close(self):
            captured["closed"] = True

        def terminate(self):
            captured["terminated"] = True

        def join(self):
            captured["joined"] = True

    class FakeSpawnContext:
        def Pool(self, **kwargs):
            captured["pool"] = kwargs
            return FailingPool()

    monkeypatch.setattr(
        sampler_module.multiprocessing,
        "get_context",
        lambda method: FakeSpawnContext() if method == "spawn" else pytest.fail(method),
    )

    with pytest.raises(RuntimeError, match="chain 1 failed") as raised:
        sampler_module._run_parallel_pyro_chains(
            tuple(SimpleNamespace(chain=chain) for chain in range(4))
        )

    assert raised.value is original
    assert captured == {
        "pool": {"processes": 4},
        "terminated": True,
        "joined": True,
    }


@pytest.mark.slow
def test_pyro_four_chain_spawn_runs_real_model(tmp_path):
    occurrence_ids = tuple(f"event-{index}" for index in range(9))
    data = HQRCData(
        observations=np.linspace(-0.2, 0.2, 18),
        occurrence_index=np.repeat(np.arange(9), 2),
        holiday_type_index=np.repeat(np.arange(9) % 2, 2),
        tau_days=np.tile(np.array([0.0, 1 / 24]), 9),
        hour=np.tile(np.array([0, 1]), 9),
        restriction=np.repeat(np.arange(9) % 2, 2),
        occurrence_ids=occurrence_ids,
    )
    calibration = calibrate_beta_prior(np.linspace(0.2, 0.4, 9), event_ids=occurrence_ids)
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibration,
        residual_sha256="residual-hash",
        config_sha256="config-hash",
        event_sha256="event-hash",
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
    )
    approved_path = approve_calibration(
        proposal,
        tmp_path / "approved.json",
        current_residual_sha256="residual-hash",
        current_config_sha256="config-hash",
        current_event_sha256="event-hash",
    )
    approved = load_approved_calibration(
        approved_path,
        current_residual_sha256="residual-hash",
        current_config_sha256="config-hash",
        current_event_sha256="event-hash",
    )

    idata = sample_hqrc(
        data,
        approved,
        variant="H1",
        draws=4,
        tune=4,
        chains=4,
        cores=4,
        seed=17,
        backend="pyro",
        device="cpu",
    )

    assert idata.posterior.sizes["chain"] == 4
    assert idata.posterior.sizes["draw"] == 4
    assert all(np.isfinite(np.asarray(value)).all() for value in idata.posterior.data_vars.values())
    assert idata.sample_stats["diverging"].dtype.kind in {"i", "u"}
    sampler = json.loads(idata.attrs["hqrc_sampler_json"])
    assert sampler["cores"] == 4
    assert sampler["chain_execution"] == "parallel"
    assert idata.attrs["hqrc_chain_execution"] == "parallel"


def test_run_pyro_chain_constructs_one_chain_mcmc_and_integer_divergences(monkeypatch):
    captured = {}
    model = object()

    class FakeNUTS:
        def __init__(self, selected_model, *, target_accept_prob):
            captured["nuts"] = (selected_model, target_accept_prob)

    class FakeMCMC:
        def __init__(self, kernel, **kwargs):
            captured["mcmc"] = (kernel, kwargs)

        def run(self):
            captured["ran"] = True

        def diagnostics(self):
            return {"divergences": {"chain 0": [1, 3]}}

        def get_samples(self, *, group_by_chain):
            captured["group_by_chain"] = group_by_chain
            return {"mu": "samples"}

    fake_pyro = SimpleNamespace(set_rng_seed=lambda seed: captured.setdefault("seed", seed))
    fake_infer = SimpleNamespace(MCMC=FakeMCMC, NUTS=FakeNUTS)
    monkeypatch.setitem(sys.modules, "pyro", fake_pyro)
    monkeypatch.setitem(sys.modules, "pyro.infer", fake_infer)

    samples, divergences = _run_pyro_chain(
        model,
        draws=4,
        tune=3,
        seed=17,
        target_accept=0.9,
        num_chains=1,
    )

    kernel, mcmc_kwargs = captured["mcmc"]
    assert isinstance(kernel, FakeNUTS)
    assert captured["seed"] == 17
    assert captured["nuts"] == (model, 0.9)
    assert mcmc_kwargs == {
        "num_samples": 4,
        "warmup_steps": 3,
        "num_chains": 1,
        "disable_progbar": True,
    }
    assert captured["ran"] is True
    assert captured["group_by_chain"] is False
    assert samples == {"mu": "samples"}
    np.testing.assert_array_equal(divergences, np.array([0, 1, 0, 1], dtype=np.int8))


def test_pyro_runs_four_sequential_chains_and_exposes_pymc_posterior(tmp_path, monkeypatch):
    data = HQRCData(
        observations=np.linspace(-0.2, 0.2, 18),
        occurrence_index=np.repeat(np.arange(9), 2),
        holiday_type_index=np.repeat(np.arange(9) % 2, 2),
        tau_days=np.tile(np.array([0.0, 1 / 24]), 9),
        hour=np.tile(np.array([0, 1]), 9),
        restriction=np.repeat(np.arange(9) % 2, 2),
        occurrence_ids=tuple(f"event-{index}" for index in range(9)),
    )
    approved = SimpleNamespace(
        artifact_path=tmp_path / "approved.json",
        artifact_digest="digest",
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="event",
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
        a=2.0,
        b=3.0,
    )
    probe = DeviceProbe(success=True, detail="float64-gradient-lkj-ar")
    resolved = ResolvedDevice(
        kind="cuda",
        logical_device="cuda:0",
        physical_device="1",
        probe=probe,
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.require_approved_calibration", lambda value: value)
    pymc_builds = []
    monkeypatch.setattr(
        "hqrc_v3.bayes.samplers.build_hqrc_model",
        lambda *_args: pymc_builds.append(True) or __import__("contextlib").nullcontext(),
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.resolve_device", lambda _request: resolved)
    model = object()
    monkeypatch.setattr("hqrc_v3.bayes.samplers.build_pyro_hqrc_model", lambda *a, **k: model)
    calls = []

    def fake_chain(selected_model, **kwargs):
        calls.append((selected_model, kwargs))
        draws = kwargs["draws"]

        def repeated(value):
            return value.to(torch.float64).expand((draws, *value.shape)).clone()

        samples = {
            "mu": repeated(torch.zeros((2, 3))),
            "delta": repeated(torch.zeros((2, 3))),
            "beta_offset": repeated(torch.zeros((9, 3))),
            "between_scale_0": repeated(torch.ones(3)),
            "between_scale_1": repeated(torch.ones(3)),
            "between_corr_cholesky_0": repeated(torch.eye(3)),
            "between_corr_cholesky_1": repeated(torch.eye(3)),
            "sigma_gamma": repeated(torch.ones(2)),
            "gamma_innovation_raw": repeated(torch.zeros((2, 23))),
            "sigma_r": repeated(torch.tensor(0.8)),
            "u_phi": repeated(torch.tensor(0.625)),
        }
        return samples, np.arange(draws, dtype=np.int8) % 2

    monkeypatch.setattr("hqrc_v3.bayes.samplers._run_pyro_chain", fake_chain)
    monkeypatch.setattr(
        "hqrc_v3.bayes.samplers.validate_inference_data",
        lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 800.0, 700.0, 4),
    )
    versions = {"arviz": "test-arviz", "torch": "test-torch", "pyro-ppl": "test-pyro"}
    monkeypatch.setattr("hqrc_v3.bayes.samplers.importlib.metadata.version", versions.__getitem__)

    idata = sample_hqrc(
        data,
        approved,
        draws=4,
        tune=3,
        chains=4,
        cores=1,
        seed=17,
        backend="pyro",
        device="auto",
    )

    assert [call[1]["seed"] for call in calls] == [17, 18, 19, 20]
    assert not pymc_builds
    assert all(call[0] is model and call[1]["num_chains"] == 1 for call in calls)
    assert idata.posterior.sizes["chain"] == 4
    assert idata.posterior.sizes["draw"] == 4
    assert idata.posterior["beta_offset"].shape == (4, 4, 9, 3)
    assert idata.posterior["event_log_likelihood"].shape == (4, 4, 9)
    assert idata.posterior["event_log_likelihood"].dims == (
        "chain",
        "draw",
        "event_log_likelihood_dim_0",
    )
    assert set(idata.posterior) == {
        "mu",
        "delta",
        "beta_offset",
        "sigma_gamma",
        "gamma_innovation_raw",
        "sigma_r",
        "u_phi",
        "between_cov_0",
        "between_cov_1",
        "between_cov_0_corr",
        "between_cov_0_stds",
        "between_cov_1_corr",
        "between_cov_1_stds",
        "between_cholesky",
        "between_scale",
        "beta",
        "gamma_innovation",
        "gamma",
        "phi",
        "event_log_likelihood",
    }
    assert idata.sample_stats["diverging"].dtype.kind in {"i", "u"}
    np.testing.assert_array_equal(
        idata.log_likelihood["event"], idata.posterior["event_log_likelihood"]
    )
    assert idata.attrs["hqrc_device"] == "cuda:0"
    assert idata.attrs["hqrc_physical_device"] == "1"
    assert idata.attrs["hqrc_dtype"] == "float64"
    assert idata.attrs["hqrc_device_probe"] == "float64-gradient-lkj-ar"
    assert idata.attrs["hqrc_chain_execution"] == "sequential"
    assert json.loads(idata.attrs["hqrc_sampler_json"])["cores"] == 1

    fallback = ResolvedDevice(
        kind="cpu",
        logical_device="cpu",
        physical_device=None,
        probe=probe,
        fallback_reason="unsupported op",
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.resolve_device", lambda _request: fallback)
    fallback_idata = sample_hqrc(
        data,
        approved,
        draws=4,
        tune=3,
        chains=4,
        cores=1,
        seed=17,
        backend="pyro",
        device="auto",
    )
    assert fallback_idata.attrs["hqrc_device_fallback_reason"] == "unsupported op"


def test_default_pymc_sampler_metadata_bytes_are_unchanged(tmp_path, monkeypatch):
    data = HQRCData(
        observations=np.array([0.1, 0.2]),
        occurrence_index=np.array([0, 0]),
        holiday_type_index=np.array([0, 0]),
        tau_days=np.array([0.0, 1 / 24]),
        hour=np.array([0, 1]),
        restriction=np.array([0, 0]),
        occurrence_ids=("a",),
    )
    approved = SimpleNamespace(
        artifact_path=tmp_path / "approved.json",
        artifact_digest="digest",
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="event",
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
        a=2.0,
        b=3.0,
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.require_approved_calibration", lambda value: value)
    monkeypatch.setattr(
        "hqrc_v3.bayes.samplers.build_hqrc_model",
        lambda *_args: __import__("contextlib").nullcontext(),
    )
    monkeypatch.setattr(
        pm,
        "sample",
        lambda **_kwargs: az.from_dict(
            posterior={"event_log_likelihood": np.zeros((2, 2, 1))},
            sample_stats={"diverging": np.zeros((2, 2), dtype=np.int8)},
        ),
    )
    monkeypatch.setattr(
        "hqrc_v3.bayes.samplers.validate_inference_data",
        lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 1.0, 1.0, 0),
    )
    monkeypatch.setattr("hqrc_v3.bayes.samplers.importlib.metadata.version", lambda _: "test")

    idata = sample_hqrc(data, approved, draws=2, tune=2, chains=2)

    assert idata.attrs["hqrc_sampler_json"] == (
        '{"chains": 2, "cores": 1, "draws": 2, '
        '"geometry": "noncentered-cyclic-hour-rw1-v1", "init": "adapt_diag", '
        '"paper_profile": false, "seed": 11, "target_accept": 0.9, "tune": 2}'
    )
    assert set(idata.attrs) == {
        "hqrc_backend",
        "hqrc_elapsed_seconds",
        "hqrc_pymc_version",
        "hqrc_arviz_version",
        "hqrc_diagnostics_json",
        "hqrc_sampler_json",
        "hqrc_model_json",
        "hqrc_calibration_json",
    }
    assert list(idata.attrs) == [
        "hqrc_backend",
        "hqrc_elapsed_seconds",
        "hqrc_pymc_version",
        "hqrc_arviz_version",
        "hqrc_diagnostics_json",
        "hqrc_sampler_json",
        "hqrc_model_json",
        "hqrc_calibration_json",
    ]
