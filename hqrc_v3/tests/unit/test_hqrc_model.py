from __future__ import annotations

from dataclasses import replace

import numpy as np
import pymc as pm
import pytensor.tensor as pt
import pytest

from hqrc_v3.bayes.model import (
    HQRCData,
    HQRCModelOptions,
    _intrinsic_random_walk,
    build_hqrc_model,
)
from hqrc_v3.diagnostics.ar import (
    EventResidualContext,
    approve_calibration,
    calibrate_beta_prior,
    load_approved_calibration,
    write_ar_diagnostics,
)
from hqrc_v3.provenance import ArtifactMismatch


@pytest.fixture
def approved_calibration(tmp_path):
    calibration = calibrate_beta_prior(np.array([0.25, 0.35, 0.45]), event_ids=("a", "b", "c"))
    hashes = {"residual": "residual-hash", "config": "config-hash", "event": "event-hash"}
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibration,
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


@pytest.fixture
def tiny_hqrc_data():
    observations = []
    occurrence_index = []
    holiday_type_index = []
    tau_days = []
    hour = []
    restriction = []
    for occurrence, holiday_type in enumerate((0, 1, 0, 1)):
        for step in range(4):
            observations.append(0.1 * occurrence + 0.05 * step)
            occurrence_index.append(occurrence)
            holiday_type_index.append(holiday_type)
            tau_days.append(-1.0 + step / 24.0)
            hour.append(step)
            restriction.append(occurrence % 2)
    return HQRCData(
        observations=np.array(observations),
        occurrence_index=np.array(occurrence_index),
        holiday_type_index=np.array(holiday_type_index),
        tau_days=np.array(tau_days),
        hour=np.array(hour),
        restriction=np.array(restriction),
        occurrence_ids=("a", "b", "c", "d"),
    )


def test_h3_contains_required_random_variables(tiny_hqrc_data, approved_calibration):
    model = build_hqrc_model(
        tiny_hqrc_data,
        approved_calibration,
        variant="H3",
        pooling="partial",
        options=HQRCModelOptions(),
    )
    assert {"mu", "delta", "between_scale", "gamma", "phi", "sigma_r"} <= set(
        model.named_vars
    )
    assert "u_phi" in model.named_vars


def test_h3_noncenters_cyclic_hour_random_walk_without_changing_output_shape(
    tiny_hqrc_data, approved_calibration
):
    model = build_hqrc_model(
        tiny_hqrc_data,
        approved_calibration,
        variant="H3",
        pooling="partial",
        options=HQRCModelOptions(),
    )

    free_names = {variable.name for variable in model.free_RVs}
    assert "gamma_innovation_raw" in free_names
    assert "gamma_innovation" not in free_names
    assert model.named_vars["gamma_innovation_raw"].type.shape == (2, 23)
    assert model.named_vars["gamma_innovation"].type.shape == (2, 23)
    assert model.named_vars["gamma"].type.shape == (2, 24)


@pytest.mark.parametrize("sigma", [0.05, 0.5, 1.7])
def test_noncentered_cyclic_rw_density_is_exact_change_of_variables(sigma):
    raw = np.linspace(-1.1, 0.9, 23)
    innovation = sigma * raw
    log_two_pi = np.log(2.0 * np.pi)

    def normal_logp(value, scale):
        return -0.5 * (value / scale) ** 2 - np.log(scale) - 0.5 * log_two_pi

    centered = np.sum(normal_logp(innovation, sigma)) + normal_logp(
        -np.sum(innovation), sigma
    )
    transformed_centered = centered + (raw.size + 1) * np.log(sigma)
    noncentered = (
        np.sum(normal_logp(raw, 1.0))
        + normal_logp(-np.sum(innovation), sigma)
        + np.log(sigma)
    )

    np.testing.assert_allclose(noncentered, transformed_centered, rtol=0.0, atol=1e-12)


def test_noncentered_cyclic_closure_does_not_change_halfnormal_scale_prior():
    with pm.Model() as model:
        _intrinsic_random_walk(
            pm,
            pt,
            name="test_cycle",
            positions=24,
            scale_name="test_sigma",
            scale=0.5,
            circular=True,
            noncentered=True,
        )
    potential_logp = model.compile_logp(vars=model.potentials, jacobian=False)
    point = model.initial_point()
    point["test_cycle_innovation_raw"] = np.tile(np.linspace(-0.8, 0.6, 23), (2, 1))
    point["test_sigma_log__"] = np.log(np.array([0.08, 0.2]))
    narrow = potential_logp(point)
    point["test_sigma_log__"] = np.log(np.array([0.8, 1.4]))
    wide = potential_logp(point)

    np.testing.assert_allclose(narrow, wide, rtol=0.0, atol=1e-12)


def test_model_rejects_untrusted_direct_calibration(tiny_hqrc_data):
    direct = calibrate_beta_prior(np.array([0.25, 0.35]), event_ids=("a", "b"))
    with pytest.raises(TypeError, match="approved"):
        build_hqrc_model(tiny_hqrc_data, direct)


def test_model_revalidates_approved_artifact_against_immutable_contents(
    tiny_hqrc_data, approved_calibration
):
    forged = replace(
        approved_calibration,
        calibration=calibrate_beta_prior(np.array([0.25, 0.35]), event_ids=("a", "b")),
    )
    with pytest.raises(ArtifactMismatch, match="does not match"):
        build_hqrc_model(tiny_hqrc_data, forged)


def test_diagonal_no_restriction_sensitivity_omits_correlation_and_delta(
    tiny_hqrc_data, approved_calibration
):
    model = build_hqrc_model(
        tiny_hqrc_data,
        approved_calibration,
        variant="H3",
        pooling="partial",
        options=HQRCModelOptions(covariance="diagonal", include_restriction=False),
    )
    assert "delta" not in model.named_vars
    assert not any("corr" in name for name in model.named_vars)


@pytest.mark.parametrize("variant", ["H1", "H2", "H3", "H4"])
@pytest.mark.parametrize("pooling", ["complete", "partial", "none"])
def test_variants_and_pooling_structures_build(
    tiny_hqrc_data, approved_calibration, variant, pooling
):
    model = build_hqrc_model(tiny_hqrc_data, approved_calibration, variant, pooling)
    assert "phi" in model.named_vars
    assert ("gamma" in model.named_vars) is (variant in {"H3", "H4"})
    assert ("mu" in model.named_vars) is (pooling == "partial")


@pytest.mark.parametrize("innovation", ["student_t_ar1", "normal_ar2"])
def test_predeclared_ar_sensitivities_build(tiny_hqrc_data, approved_calibration, innovation):
    model = build_hqrc_model(
        tiny_hqrc_data,
        approved_calibration,
        "H3",
        "partial",
        HQRCModelOptions(innovation=innovation),
    )
    expected = "nu_minus_two" if innovation == "student_t_ar1" else "ar2_pacf"
    assert expected in model.named_vars


@pytest.mark.parametrize("innovation", ["normal_ar1", "student_t_ar1", "normal_ar2"])
def test_every_innovation_exposes_occurrence_log_likelihood(
    tiny_hqrc_data, approved_calibration, innovation
):
    model = build_hqrc_model(
        tiny_hqrc_data,
        approved_calibration,
        options=HQRCModelOptions(innovation=innovation),
    )
    assert model.named_vars["event_log_likelihood"].type.shape == (4,)


@pytest.mark.parametrize(
    "options", [HQRCModelOptions(lkj_eta=0), HQRCModelOptions(between_scale_prior=0)]
)
def test_model_options_reject_nonpositive_values(options):
    with pytest.raises(ValueError):
        options.validate()


def test_data_rejects_noncontiguous_occurrence_segments():
    with pytest.raises(ValueError, match="contiguous"):
        HQRCData(
            observations=np.ones(3),
            occurrence_index=np.array([0, 1, 0]),
            holiday_type_index=np.array([0, 1, 0]),
            tau_days=np.array([0.0, 1.0 / 24.0, 0.0]),
            hour=np.array([0, 1, 0]),
            restriction=np.zeros(3, dtype=int),
            occurrence_ids=("a", "b"),
        )


@pytest.mark.parametrize(
    ("tau_days", "hour", "message"),
    [
        (np.array([0.0, 2.0 / 24.0]), np.array([0, 2]), "one-hour"),
        (np.array([0.0, 1.0 / 24.0]), np.array([0, 2]), "hour"),
    ],
)
def test_data_rejects_nonhourly_or_shuffled_within_occurrence(tau_days, hour, message):
    with pytest.raises(ValueError, match=message):
        HQRCData(
            observations=np.ones(2),
            occurrence_index=np.zeros(2, dtype=int),
            holiday_type_index=np.zeros(2, dtype=int),
            tau_days=tau_days,
            hour=hour,
            restriction=np.zeros(2, dtype=int),
            occurrence_ids=("a",),
        )


def test_data_rejects_tau_hour_mismatch_even_when_steps_are_hourly():
    with pytest.raises(ValueError, match="align"):
        HQRCData(
            observations=np.ones(2),
            occurrence_index=np.zeros(2, dtype=int),
            holiday_type_index=np.zeros(2, dtype=int),
            tau_days=np.array([0.5, 13.0 / 24.0]),
            hour=np.array([0, 1]),
            restriction=np.zeros(2, dtype=int),
            occurrence_ids=("a",),
        )


def test_data_accepts_float32_hourly_tau_with_matching_hour():
    data = HQRCData(
        observations=np.ones(3),
        occurrence_index=np.zeros(3, dtype=int),
        holiday_type_index=np.zeros(3, dtype=int),
        tau_days=np.array([-1.0, -23.0 / 24.0, -22.0 / 24.0], dtype=np.float32),
        hour=np.array([0, 1, 2]),
        restriction=np.zeros(3, dtype=int),
        occurrence_ids=("a",),
    )
    assert data.hour.tolist() == [0, 1, 2]


def test_h4_uses_integer_day_position_and_intrinsic_random_walk(
    tiny_hqrc_data, approved_calibration
):
    model = build_hqrc_model(tiny_hqrc_data, approved_calibration, variant="H4")
    assert model.named_vars["day_effect"].type.shape == (2, 1)
    assert "day_effect_raw" not in model.named_vars
    assert "sigma_day" in model.named_vars
    assert "gamma_raw" not in model.named_vars
    assert "gamma_innovation" in model.named_vars
