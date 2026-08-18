from __future__ import annotations

import numpy as np
import pytest
from hqrc_v3.bayes.artifacts import write_hqrc_data
from hqrc_v3.bayes.benchmark import run_sampler_worker, write_sampler_request
from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.diagnostics.ar import (
    EventResidualContext,
    approve_calibration,
    calibrate_beta_prior,
    write_ar_diagnostics,
)
from hqrc_v3.provenance import file_sha256


@pytest.mark.slow
def test_real_pymc_sampler_runs_in_fresh_process(tmp_path):
    data = HQRCData(
        observations=np.array([0.1, 0.2, 0.0, 0.2, 0.3, 0.1, -0.1, 0.0]),
        occurrence_index=np.repeat(np.arange(2), 4),
        holiday_type_index=np.repeat(np.arange(2), 4),
        tau_days=np.tile(np.arange(4, dtype=float) / 24.0, 2),
        hour=np.tile(np.arange(4), 2),
        restriction=np.repeat(np.array([0, 1]), 4),
        occurrence_ids=("a", "b"),
    )
    npz, metadata = write_hqrc_data(tmp_path / "data.npz", data, settings={"profile": "smoke"})
    residual, config, event = tmp_path / "residual", tmp_path / "config", tmp_path / "event"
    for path in (residual, config, event):
        path.write_text(path.name)
    hashes = [file_sha256(path) for path in (residual, config, event)]
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibrate_beta_prior(np.array([0.2, 0.4]), event_ids=("a", "b")),
        residual_sha256=hashes[0],
        config_sha256=hashes[1],
        event_sha256=hashes[2],
        context=EventResidualContext("model", "B0", 1, ("oof-2020",)),
    )
    approved = approve_calibration(
        proposal,
        tmp_path / "approved.json",
        current_residual_sha256=hashes[0],
        current_config_sha256=hashes[1],
        current_event_sha256=hashes[2],
    )
    request = write_sampler_request(
        tmp_path / "request.json",
        hqrc_npz=npz,
        hqrc_metadata=metadata,
        approved_ar=approved,
        residual_sha256=hashes[0],
        config_sha256=hashes[1],
        event_sha256=hashes[2],
        variant="H3",
        pooling="partial",
        options={"covariance": "diagonal"},
        backend="pymc",
        seed=5,
        draws=10,
        tune=10,
        chains=2,
        profile="smoke",
    )
    result = run_sampler_worker(request, tmp_path / "result.json", timeout_seconds=180)
    assert result.backend == "pymc" and result.pid > 0
    assert result.wall_seconds > 0 and result.peak_rss_mb > 0
