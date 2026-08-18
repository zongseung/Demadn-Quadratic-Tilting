from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from hqrc_v3.bayes.samplers import PYMC_INITIALIZATION, SAMPLER_GEOMETRY
from hqrc_v3.correction_stage import (
    fit_causal_2024_correction,
    prepare_causal_correction_inputs,
)
from hqrc_v3.provenance import file_sha256


@pytest.mark.slow
def test_real_causal_correction_reuses_final_and_runs_reduced_pymc(tmp_path):
    """Opt-in smoke over a real completed baseline/residual/approval publication."""

    run_value = os.environ.get("HQRC_V3_REAL_RUN_DIR")
    approved_value = os.environ.get("HQRC_V3_REAL_APPROVED_AR")
    if not run_value and not approved_value:
        pytest.skip("set HQRC_V3_REAL_RUN_DIR and HQRC_V3_REAL_APPROVED_AR")
    if not run_value or not approved_value:
        pytest.fail("both real correction environment variables are required")
    run = Path(run_value).resolve()
    approved = Path(approved_value).resolve()
    residual_manifest_path = run / "inputs/standardized_residuals_manifest.json"
    residual_manifest = json.loads(residual_manifest_path.read_bytes())
    source_profile = residual_manifest["profile"]
    config_path = Path(residual_manifest["inputs"]["experiment_config"]["path"])
    protected = (
        run / "predictions/baseline_manifest.json",
        run / "predictions/final_2024_members.parquet",
        run / "predictions/final_2024.parquet",
        residual_manifest_path,
        run / "inputs/standardized_residuals.parquet",
        approved,
    )
    before = {path: file_sha256(path) for path in protected}

    inputs = prepare_causal_correction_inputs(
        run_dir=run,
        config_path=config_path,
        approved_ar_path=approved,
        profile="smoke",
    )
    assert len(inputs.hqrc_data.occurrence_ids) == 8
    assert inputs.source_profile == source_profile
    assert inputs.hqrc_data.observations.size == 1_032
    assert inputs.prediction.occurrence_ids == ("seollal-2024", "chuseok-2024")
    assert inputs.prediction.event_frame.height == 264

    result = fit_causal_2024_correction(
        run_dir=run,
        config_path=config_path,
        approved_ar_path=approved,
        sampler_seed=20260811,
        profile="smoke",
        draws=5,
        tune=5,
        chains=2,
        output_root=tmp_path,
    )

    assert result.sampler_fit_count == 1 and not result.reused
    assert result.output_dir.is_relative_to(tmp_path)
    assert result.output_dir.parts[-4:] == (
        "smoke",
        "sampler-seed-20260811",
        f"init-{PYMC_INITIALIZATION}-geometry-{SAMPLER_GEOMETRY}",
        "draws-5-tune-5-chains-2",
    )
    event = pl.read_parquet(result.event_predictions_path)
    full = pl.read_parquet(result.full_period_point_predictions_path)
    assert event.height == 264
    assert set(event["occurrence_id"]) == {"seollal-2024", "chuseok-2024"}
    outside = full.filter(~pl.col("is_hqrc_event"))
    assert np.array_equal(
        outside["baseline_mw"].to_numpy(), outside["corrected_point_mw"].to_numpy()
    )
    manifest = json.loads(result.manifest_path.read_bytes())
    assert manifest["identity"]["source_profile"] == source_profile
    assert manifest["identity"]["sampler_profile"] == "smoke"
    assert manifest["identity"]["sampler"]["init"] == PYMC_INITIALIZATION
    assert manifest["identity"]["sampler"]["geometry"] == SAMPLER_GEOMETRY
    reused = fit_causal_2024_correction(
        run_dir=run,
        config_path=config_path,
        approved_ar_path=approved,
        sampler_seed=20260811,
        profile="smoke",
        draws=5,
        tune=5,
        chains=2,
        output_root=tmp_path,
    )
    assert reused.reused and reused.sampler_fit_count == 0
    assert reused.output_dir == result.output_dir
    assert {path: file_sha256(path) for path in protected} == before
