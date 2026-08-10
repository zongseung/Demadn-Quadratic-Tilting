from __future__ import annotations

import numpy as np
import pytest
from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.bayes.samplers import sample_hqrc, validate_inference_data
from hqrc_v3.diagnostics.ar import calibrate_beta_prior


@pytest.mark.slow
def test_tiny_hqrc_sampling_returns_finite_posterior():
    data = HQRCData(
        observations=np.array([0.1, 0.2, 0.0, 0.2, 0.3, 0.1, -0.1, 0.0]),
        occurrence_index=np.repeat(np.arange(2), 4),
        holiday_type_index=np.repeat(np.arange(2), 4),
        tau_days=np.tile(np.arange(4, dtype=float), 2),
        hour=np.tile(np.arange(4), 2),
        restriction=np.repeat(np.array([0, 1]), 4),
        occurrence_ids=("a", "b"),
    )
    calibration = calibrate_beta_prior(np.array([0.25, 0.35]), event_ids=("a", "b"))
    idata = sample_hqrc(data, calibration, draws=30, tune=30, chains=2, seed=5)
    assert np.isfinite(idata.posterior["phi"]).all()
    assert validate_inference_data(idata, paper_profile=False).divergences >= 0
