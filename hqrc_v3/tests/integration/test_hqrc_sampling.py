from __future__ import annotations

import json

import arviz as az
import numpy as np
import pytest
from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.bayes.samplers import SamplingError, sample_hqrc, validate_inference_data
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
    netcdf_path = tmp_path / "posterior.nc"
    az.to_netcdf(idata, netcdf_path)
    round_trip = az.from_netcdf(netcdf_path)
    assert json.loads(round_trip.attrs["hqrc_model_json"])["variant"] == "H3"
    assert validate_inference_data(idata, paper_profile=False).divergences >= 0


def test_paper_profile_rejects_smoke_sampler_limits_before_model_build():
    with pytest.raises(ValueError, match="4 chains"):
        sample_hqrc(None, None, draws=30, tune=30, chains=2, paper_profile=True)


def test_paper_diagnostics_fail_closed_without_divergence_statistics():
    idata = az.from_dict(posterior={"phi": np.zeros((4, 8))})
    with pytest.raises(SamplingError, match="diverging"):
        validate_inference_data(idata, paper_profile=True)
