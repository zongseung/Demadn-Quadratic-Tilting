from __future__ import annotations

import numpy as np
import pytest
from hqrc_v3.bayes.model import HQRCData, HQRCModelOptions, build_hqrc_model
from hqrc_v3.diagnostics.ar import calibrate_beta_prior


@pytest.fixture
def approved_calibration():
    return calibrate_beta_prior(np.array([0.25, 0.35, 0.45]), event_ids=("a", "b", "c"))


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
            tau_days.append(float(step - 1))
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
            tau_days=np.zeros(3),
            hour=np.arange(3),
            restriction=np.zeros(3, dtype=int),
            occurrence_ids=("a", "b"),
        )
