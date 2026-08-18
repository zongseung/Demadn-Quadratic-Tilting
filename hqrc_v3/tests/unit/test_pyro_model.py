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


def _batched_covariance_samples() -> dict[str, torch.Tensor]:
    root_87 = math.sqrt(0.87)
    root_75 = math.sqrt(0.75)
    root_7775 = math.sqrt(0.7775)
    base = {
        "mu": torch.tensor([[1.0, 2.0, 3.0], [0.5, 1.0, 1.5]]),
        "delta": torch.tensor([[9.0, 9.0, 9.0], [0.5, -0.5, 0.25]]),
        "beta_offset": torch.tensor([[1.0, -2.0, 0.5], [-1.0, 0.25, 2.0]]),
        "between_scale_0": torch.tensor([2.0, 3.0, 4.0]),
        "between_scale_1": torch.tensor([0.5, 1.5, 2.5]),
        "between_corr_cholesky_0": torch.tensor(
            [[1.0, 0.0, 0.0], [0.6, 0.8, 0.0], [-0.2, 0.3, root_87]]
        ),
        "between_corr_cholesky_1": torch.tensor(
            [[1.0, 0.0, 0.0], [-0.5, root_75, 0.0], [0.25, 0.4, root_7775]]
        ),
        "sigma_gamma": torch.tensor([1.0, 1.0]),
        "gamma_innovation_raw": torch.zeros((2, 23)),
        "sigma_r": torch.tensor(0.8),
        "u_phi": torch.tensor(0.625),
    }
    return {
        name: value.to(torch.float64).expand((2, 3, *value.shape)).clone()
        for name, value in base.items()
    }


def test_batched_reconstruction_matches_complete_authoritative_covariance_schema(tiny_data):
    samples = _batched_covariance_samples()
    actual = reconstruct_pyro_deterministics(samples, tiny_data, "H3")
    expected_shapes = {
        "between_cov_0": (2, 3, 6),
        "between_cov_1": (2, 3, 6),
        "between_cov_0_corr": (2, 3, 3, 3),
        "between_cov_0_stds": (2, 3, 3),
        "between_cov_1_corr": (2, 3, 3, 3),
        "between_cov_1_stds": (2, 3, 3),
        "between_cholesky": (2, 3, 2, 3, 3),
        "between_scale": (2, 3, 2, 3),
        "beta": (2, 3, 2, 3),
        "gamma_innovation": (2, 3, 2, 23),
        "gamma": (2, 3, 2, 24),
        "phi": (2, 3),
        "event_log_likelihood": (2, 3, 2),
    }
    assert set(actual) == set(expected_shapes)
    assert {name: tuple(value.shape) for name, value in actual.items()} == expected_shapes
    public_sample_shapes = {
        "mu": (2, 3, 2, 3),
        "delta": (2, 3, 2, 3),
        "beta_offset": (2, 3, 2, 3),
        "gamma_innovation_raw": (2, 3, 2, 23),
        "sigma_gamma": (2, 3, 2),
        "sigma_r": (2, 3),
        "u_phi": (2, 3),
    }
    posterior = {name: samples[name] for name in public_sample_shapes} | actual
    assert set(posterior) == set(public_sample_shapes) | set(expected_shapes)
    assert {name: tuple(value.shape) for name, value in posterior.items()} == (
        public_sample_shapes | expected_shapes
    )

    root_87 = math.sqrt(0.87)
    root_75 = math.sqrt(0.75)
    root_7775 = math.sqrt(0.7775)
    expected_cholesky = np.array(
        [
            [[2.0, 0.0, 0.0], [1.8, 2.4, 0.0], [-0.8, 1.2, 4.0 * root_87]],
            [
                [0.5, 0.0, 0.0],
                [-0.75, 1.5 * root_75, 0.0],
                [0.625, 1.0, 2.5 * root_7775],
            ],
        ]
    )
    expected_corr = (
        np.array([[1.0, 0.6, -0.2], [0.6, 1.0, 0.12], [-0.2, 0.12, 1.0]]),
        np.array(
            [
                [1.0, -0.5, 0.25],
                [-0.5, 1.0, -0.125 + 0.4 * root_75],
                [0.25, -0.125 + 0.4 * root_75, 1.0],
            ]
        ),
    )
    expected_beta = np.array(
        [
            [3.0, -1.0, -0.2 + 2.0 * root_87],
            [0.5, 1.25 + 0.375 * root_75, 1.375 + 5.0 * root_7775],
        ]
    )
    expected = {
        "between_cholesky": expected_cholesky,
        "between_scale": [[2.0, 3.0, 4.0], [0.5, 1.5, 2.5]],
        "between_cov_0": [2.0, 1.8, 2.4, -0.8, 1.2, 4.0 * root_87],
        "between_cov_1": [0.5, -0.75, 1.5 * root_75, 0.625, 1.0, 2.5 * root_7775],
        "beta": expected_beta,
    }
    for name, value in expected.items():
        np.testing.assert_allclose(actual[name].numpy(), np.broadcast_to(value, actual[name].shape))
    for holiday, scales in enumerate(([2.0, 3.0, 4.0], [0.5, 1.5, 2.5])):
        np.testing.assert_allclose(
            actual[f"between_cov_{holiday}_corr"].numpy(),
            np.broadcast_to(expected_corr[holiday], actual[f"between_cov_{holiday}_corr"].shape),
        )
        np.testing.assert_allclose(
            actual[f"between_cov_{holiday}_stds"].numpy(),
            np.broadcast_to(scales, actual[f"between_cov_{holiday}_stds"].shape),
        )
    assert actual["phi"].unique().item() == pytest.approx(0.25)
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


def _conditioned_sites(variant: str) -> dict[str, torch.Tensor]:
    coefficients = 1 if variant == "H1" else 3
    values = {
        "mu": torch.zeros((2, coefficients), dtype=torch.float64),
        "delta": torch.zeros((2, coefficients), dtype=torch.float64),
        "beta_offset": torch.zeros((2, coefficients), dtype=torch.float64),
        "between_scale_0": torch.ones(coefficients, dtype=torch.float64),
        "between_scale_1": torch.ones(coefficients, dtype=torch.float64),
        "sigma_r": torch.tensor(0.8, dtype=torch.float64),
        "u_phi": torch.tensor(0.625, dtype=torch.float64),
    }
    if coefficients == 3:
        values.update(
            {
                "between_corr_cholesky_0": torch.eye(3, dtype=torch.float64),
                "between_corr_cholesky_1": torch.eye(3, dtype=torch.float64),
            }
        )
    if variant == "H3":
        values.update(
            {
                "sigma_gamma": torch.tensor([0.25, 1.75], dtype=torch.float64),
                "gamma_innovation_raw": torch.zeros((2, 23), dtype=torch.float64),
            }
        )
    return values


@pytest.mark.parametrize(("variant", "coefficients"), [("H1", 1), ("H2", 3), ("H3", 3)])
def test_conditioned_cpu_models_have_exact_shapes_priors_and_factors(
    tiny_data, approved_calibration, variant, coefficients
):
    pyro = pytest.importorskip("pyro")
    model = build_pyro_hqrc_model(tiny_data, approved_calibration, variant=variant, device="cpu")
    conditioned = pyro.poutine.condition(model, data=_conditioned_sites(variant))

    trace = pyro.poutine.trace(conditioned).get_trace()
    trace.compute_log_prob()

    expected_sites = {
        "mu",
        "delta",
        "beta_offset",
        "between_scale_0",
        "between_scale_1",
        "sigma_r",
        "u_phi",
        "between_cov_0",
        "between_cov_1",
        "between_cov_0_corr",
        "between_cov_0_stds",
        "between_cov_1_corr",
        "between_cov_1_stds",
        "beta",
        "phi",
        "event_log_likelihood",
        "event_reset_ar1",
    }
    if coefficients == 3:
        expected_sites.update({"between_corr_cholesky_0", "between_corr_cholesky_1"})
    if variant == "H3":
        expected_sites.update({"sigma_gamma", "gamma_innovation_raw"})
    assert expected_sites <= set(trace.nodes)
    assert tuple(trace.nodes["beta"]["value"].shape) == (2, coefficients)
    assert tuple(trace.nodes["between_cholesky"]["value"].shape) == (
        2,
        coefficients,
        coefficients,
    )
    assert tuple(trace.nodes["between_cov_0"]["value"].shape) == (
        coefficients * (coefficients + 1) // 2,
    )
    assert ("gamma" in trace.nodes) is (variant == "H3")
    assert ("gamma_circular_random_walk" in trace.nodes) is (variant == "H3")
    assert ("between_corr_cholesky_0" in trace.nodes) is (coefficients == 3)

    scale_distribution = trace.nodes["between_scale_0"]["fn"]
    assert scale_distribution.base_dist.scale.tolist() == pytest.approx([1.0] * coefficients)
    assert trace.nodes["sigma_r"]["fn"].scale.item() == pytest.approx(1.0)
    if coefficients == 3:
        assert trace.nodes["between_corr_cholesky_0"]["fn"].concentration.item() == pytest.approx(
            2.0
        )
    phi_distribution = trace.nodes["u_phi"]["fn"]
    assert phi_distribution.concentration1.item() == pytest.approx(approved_calibration.a)
    assert phi_distribution.concentration0.item() == pytest.approx(approved_calibration.b)
    assert trace.nodes["phi"]["value"].item() == pytest.approx(0.25)

    expected_event_logp = event_reset_ar1_logp_numpy(
        [np.array([0.2]), np.array([-0.1])], phi=0.25, sigma=0.8
    )
    np.testing.assert_allclose(
        trace.nodes["event_log_likelihood"]["value"].numpy(), expected_event_logp
    )
    assert trace.nodes["event_reset_ar1"]["log_prob"].item() == pytest.approx(
        expected_event_logp.sum()
    )
    if variant == "H3":
        assert trace.nodes["sigma_gamma"]["fn"].base_dist.scale.tolist() == pytest.approx(
            [0.5, 0.5]
        )
        assert trace.nodes["gamma_circular_random_walk"]["log_prob"].item() == pytest.approx(
            -math.log(2.0 * math.pi)
        )
    tensors = [
        node["value"]
        for node in trace.nodes.values()
        if node.get("type") == "sample" and isinstance(node.get("value"), torch.Tensor)
    ]
    assert tensors and all(value.dtype == torch.float64 for value in tensors)
    assert all(torch.isfinite(value).all() for value in tensors)
    assert math.isfinite(float(trace.log_prob_sum()))
