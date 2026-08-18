from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from hqrc_v3.bayes.model import HQRCData, HQRCModelOptions, event_reset_ar1_logp_numpy
from hqrc_v3.bayes.pyro_model import (
    PyroModelError,
    build_pyro_hqrc_model,
    centered_cyclic_hour_profile_torch,
    event_reset_ar1_logp_torch,
    hqrc_mean_torch,
    reconstruct_pyro_deterministics,
)
from hqrc_v3.diagnostics.ar import (
    EventResidualContext,
    approve_calibration,
    calibrate_beta_prior,
    load_approved_calibration,
    write_ar_diagnostics,
)


@pytest.fixture
def tiny_data() -> HQRCData:
    return HQRCData(
        observations=np.array([0.2, -0.1]),
        occurrence_index=np.array([0, 1]),
        holiday_type_index=np.array([0, 1]),
        tau_days=np.array([0.0, 0.5]),
        hour=np.array([0, 12]),
        restriction=np.array([0, 1]),
        occurrence_ids=("seollal-a", "chuseok-b"),
    )


@pytest.fixture
def ar_data() -> HQRCData:
    return HQRCData(
        observations=np.zeros(4),
        occurrence_index=np.array([0, 0, 1, 1]),
        holiday_type_index=np.array([0, 0, 1, 1]),
        tau_days=np.array([0.0, 1.0 / 24.0, 0.5, 13.0 / 24.0]),
        hour=np.array([0, 1, 12, 13]),
        restriction=np.array([0, 0, 1, 1]),
        occurrence_ids=("seollal-a", "chuseok-b"),
    )


@pytest.fixture
def approved_calibration(tmp_path):
    hashes = {"residual": "residual-hash", "config": "config-hash", "event": "event-hash"}
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibrate_beta_prior(np.array([0.25, 0.35, 0.45]), event_ids=("a", "b", "c")),
        residual_sha256=hashes["residual"],
        config_sha256=hashes["config"],
        event_sha256=hashes["event"],
        context=EventResidualContext("model", "B0", 5, ("oof-2020",)),
    )
    approved = approve_calibration(
        proposal,
        tmp_path / "approved.json",
        current_residual_sha256=hashes["residual"],
        current_config_sha256=hashes["config"],
        current_event_sha256=hashes["event"],
    )
    return load_approved_calibration(
        approved,
        current_residual_sha256=hashes["residual"],
        current_config_sha256=hashes["config"],
        current_event_sha256=hashes["event"],
    )


def _beta_fixture() -> torch.Tensor:
    return torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0]], dtype=torch.float64)


def _gamma_fixture() -> torch.Tensor:
    gamma = torch.zeros((2, 24), dtype=torch.float64)
    gamma[0, 0], gamma[0, 1] = 0.25, -0.25
    gamma[1, 12], gamma[1, 13] = -0.25, 0.25
    return gamma


@pytest.mark.parametrize(
    ("variant", "expected"),
    [("H1", [1.0, 1.0]), ("H2", [1.0, 1.75]), ("H3", [1.25, 1.50])],
)
def test_torch_mean_matches_hand_calculated_fixture(variant, expected, tiny_data):
    actual = hqrc_mean_torch(
        tiny_data, beta=_beta_fixture(), variant=variant, gamma=_gamma_fixture()
    )
    assert actual.dtype == torch.float64
    assert actual.cpu().tolist() == pytest.approx(expected)


def test_cyclic_hour_profile_is_cumulative_and_sum_zero():
    innovations = torch.tensor([[1.0, 2.0, -1.0], [0.0, 0.0, 0.0]])

    actual = centered_cyclic_hour_profile_torch(innovations)

    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), [[-1.5, -0.5, 1.5, 0.5], [0.0, 0.0, 0.0, 0.0]])
    assert actual.sum(dim=-1).tolist() == pytest.approx([0.0, 0.0])


def test_event_reset_ar1_matches_numpy_reference(ar_data):
    residual = torch.tensor([0.4, -0.2, 1.1, 0.3], dtype=torch.float64)
    torch_value = event_reset_ar1_logp_torch(residual, ar_data, phi=0.25, sigma=0.8)
    numpy_value = event_reset_ar1_logp_numpy(
        [np.array([0.4, -0.2]), np.array([1.1, 0.3])], phi=0.25, sigma=0.8
    )

    assert torch_value.dtype == torch.float64
    np.testing.assert_allclose(
        torch_value.detach().cpu().numpy(), numpy_value, rtol=1e-10, atol=1e-10
    )


def test_reconstruction_matches_literal_partial_pooling_fixture(tiny_data):
    samples = {
        "mu": torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]], dtype=torch.float64),
        "delta": torch.tensor([[9.0, 9.0, 9.0], [0.5, -0.5, 0.25]], dtype=torch.float64),
        "beta_offset": torch.zeros((2, 3), dtype=torch.float64),
        "between_scale_0": torch.ones(3, dtype=torch.float64),
        "between_scale_1": torch.ones(3, dtype=torch.float64),
        "between_corr_cholesky_0": torch.eye(3, dtype=torch.float64),
        "between_corr_cholesky_1": torch.eye(3, dtype=torch.float64),
        "sigma_gamma": torch.tensor([1.0, 1.0], dtype=torch.float64),
        "gamma_innovation_raw": torch.zeros((2, 23), dtype=torch.float64),
        "sigma_r": torch.tensor(0.8, dtype=torch.float64),
        "u_phi": torch.tensor(0.625, dtype=torch.float64),
    }

    actual = reconstruct_pyro_deterministics(samples, tiny_data, "H3")

    np.testing.assert_allclose(actual["beta"].numpy(), [[1.0, 2.0, 3.0], [1.0, 0.5, 1.75]])
    assert actual["phi"].item() == pytest.approx(0.25)
    np.testing.assert_allclose(actual["gamma"].numpy(), np.zeros((2, 24)))
    expected_logp = event_reset_ar1_logp_numpy(
        [np.array([-0.8]), np.array([-1.7875])], phi=0.25, sigma=0.8
    )
    np.testing.assert_allclose(
        actual["event_log_likelihood"].numpy(), expected_logp, rtol=1e-10, atol=1e-10
    )
    assert all(value.dtype == torch.float64 for value in actual.values())


@pytest.mark.parametrize(
    ("variant", "pooling", "options", "message"),
    [
        ("H4", "partial", HQRCModelOptions(), "Pyro HQRC supports only H1, H2, and H3"),
        ("H3", "complete", HQRCModelOptions(), "Pyro HQRC requires partial pooling"),
        (
            "H3",
            "partial",
            HQRCModelOptions(covariance="diagonal"),
            "Pyro HQRC requires full covariance",
        ),
        (
            "H3",
            "partial",
            HQRCModelOptions(include_restriction=False),
            "Pyro HQRC requires restriction effects",
        ),
        (
            "H3",
            "partial",
            HQRCModelOptions(innovation="student_t_ar1"),
            "Pyro HQRC requires normal_ar1 innovations",
        ),
    ],
)
def test_model_rejects_nonpaper_profiles_with_exact_errors(
    tiny_data, approved_calibration, variant, pooling, options, message
):
    with pytest.raises(PyroModelError) as error:
        build_pyro_hqrc_model(
            tiny_data,
            approved_calibration,
            variant=variant,
            pooling=pooling,
            options=options,
        )

    assert str(error.value) == message


def test_h3_cpu_model_has_exact_sites_and_finite_double_log_density(
    tiny_data, approved_calibration
):
    pyro = pytest.importorskip("pyro")
    model = build_pyro_hqrc_model(tiny_data, approved_calibration, device="cpu")

    trace = pyro.poutine.trace(model).get_trace()
    trace.compute_log_prob()

    assert {
        "mu",
        "delta",
        "beta_offset",
        "between_scale_0",
        "between_scale_1",
        "between_corr_cholesky_0",
        "between_corr_cholesky_1",
        "sigma_gamma",
        "gamma_innovation_raw",
        "sigma_r",
        "u_phi",
        "beta",
        "gamma",
        "phi",
        "event_log_likelihood",
        "event_reset_ar1",
    } <= set(trace.nodes)
    tensors = [
        node["value"]
        for node in trace.nodes.values()
        if node.get("type") == "sample" and isinstance(node.get("value"), torch.Tensor)
    ]
    assert tensors and all(value.dtype == torch.float64 for value in tensors)
    assert all(torch.isfinite(value).all() for value in tensors)
    assert math.isfinite(float(trace.log_prob_sum()))
