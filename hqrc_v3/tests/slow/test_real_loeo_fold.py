"""Opt-in reduced PyMC smoke for one actual XGBoost-B1 LOEO fold."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pytest
from hqrc_v3.bayes.artifacts import load_hqrc_data
from hqrc_v3.correction_source import validate_correction_source
from hqrc_v3.diagnostics.loeo import load_loeo_universe
from hqrc_v3.diagnostics.loeo_ar import load_approved_loeo_ar_set
from hqrc_v3.loeo_stage import fit_loeo_fold, prepare_loeo_fold_inputs
from hqrc_v3.provenance import file_sha256

_PROPOSAL_SHA = "db54721c0899f29312abe3d47d08977b33e02f53315d629c17fd319f6bfa83cd"
_APPROVED_SHA = "cb385e3a5a261819612d333b05c287462b99f790767f0aadcc4f795aae68581c"


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in sorted(root.rglob("*")):
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise AssertionError(f"protected tree contains symlink: {candidate}")
        if stat.S_ISREG(mode):
            result[candidate.relative_to(root).as_posix()] = file_sha256(candidate)
    return result


@pytest.mark.slow
def test_real_xgboost_b1_one_fold_runs_reduced_pymc_without_mutating_inputs(tmp_path: Path):
    run_value = os.environ.get("HQRC_V3_REAL_RUN_DIR")
    loeo_value = os.environ.get("HQRC_V3_REAL_LOEO_DIR")
    ar_value = os.environ.get("HQRC_V3_REAL_LOEO_AR_DIR")
    paper_value = os.environ.get("HQRC_V3_REAL_PAPER_ARTIFACT_DIR")
    values = (run_value, loeo_value, ar_value, paper_value)
    if not any(values):
        pytest.skip(
            "set HQRC_V3_REAL_RUN_DIR, HQRC_V3_REAL_LOEO_DIR, "
            "HQRC_V3_REAL_LOEO_AR_DIR, and HQRC_V3_REAL_PAPER_ARTIFACT_DIR"
        )
    if not all(values):
        pytest.fail("all four real LOEO environment variables are required")
    run = Path(run_value).resolve()
    loeo_dir = Path(loeo_value).resolve()
    ar_dir = Path(ar_value).resolve()
    paper_dir = Path(paper_value).resolve()
    residual_manifest = json.loads(
        (run / "inputs/standardized_residuals_manifest.json").read_bytes()
    )
    config_path = Path(residual_manifest["inputs"]["experiment_config"]["path"])
    protected_before = {
        "source": _tree_hashes(run),
        "loeo": _tree_hashes(loeo_dir),
        "ar": _tree_hashes(ar_dir),
        "paper": _tree_hashes(paper_dir),
    }

    source = validate_correction_source(
        run_dir=run,
        config_path=config_path,
        profile="smoke",
    )
    matches = [
        context
        for context in source.available_contexts
        if (context.model, context.feature_set, context.seed) == ("xgboost", "B1", 7)
    ]
    assert len(matches) == 1
    context = matches[0]
    publication = load_loeo_universe(source, context, output_dir=loeo_dir)
    approved = load_approved_loeo_ar_set(source, publication, output_dir=ar_dir)
    assert approved.proposal_set_sha256 == _PROPOSAL_SHA
    assert approved.approved_set_sha256 == _APPROVED_SHA

    held_out = "seollal-2024"
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id=held_out,
    )
    assert len(inputs.hqrc_data.occurrence_ids) == 9
    assert held_out not in inputs.hqrc_data.occurrence_ids
    assert held_out not in inputs.approved.calibration.event_ids
    assert set(inputs.approved.calibration.event_ids) == set(inputs.hqrc_data.occurrence_ids)
    assert inputs.hqrc_data.observations.size == 1_152
    assert inputs.held_out.frame.height == 144
    assert inputs.sigma_eval == pytest.approx(2290.0780128512224)

    result = fit_loeo_fold(
        source,
        publication,
        approved,
        held_out_occurrence_id=held_out,
        sampler_seed=20260812,
        profile="smoke",
        draws=3,
        tune=3,
        chains=2,
        output_root=tmp_path,
    )
    assert result.sampler_fit_count == 1 and result.reused is False
    assert result.output_dir.is_relative_to(tmp_path)
    hourly = pl.read_parquet(result.hourly_predictions_path)
    assert hourly.height == 144
    assert hourly["occurrence_id"].unique().to_list() == [held_out]
    assert hourly["causal"].unique().to_list() == [False]
    summary = json.loads(result.posterior_summary_path.read_bytes())
    assert summary["phi"]["sample_count"] == 6
    assert summary["phi"]["parameterization"].startswith("phi=2*u_phi-1")
    posterior = az.from_netcdf(result.posterior_path)
    assert {"u_phi", "phi"}.issubset(posterior.posterior)
    np.testing.assert_allclose(posterior.posterior["phi"], 2.0 * posterior.posterior["u_phi"] - 1.0)
    assert posterior.attrs["hqrc_causal"] == "false"
    calibration_metadata = json.loads(posterior.attrs["hqrc_calibration_json"])
    assert calibration_metadata["a"] == inputs.approved.a
    assert calibration_metadata["b"] == inputs.approved.b
    assert calibration_metadata["residual_sha256"] == inputs.training_fold.residual_sha256
    manifest = json.loads(result.manifest_path.read_bytes())
    assert manifest["identity"]["approved_loeo_ar"]["proposal_set_sha256"] == _PROPOSAL_SHA
    assert manifest["identity"]["approved_loeo_ar"]["approved_set_sha256"] == _APPROVED_SHA
    assert manifest["identity"]["model"]["variant"] == "H3"
    assert manifest["identity"]["model"]["pooling"] == "partial"
    assert manifest["identity"]["causal"] is False
    assert manifest["causal"] is False
    assert summary["causal"] is False
    assert pl.read_parquet(result.metrics_path)["causal"].to_list() == [False]
    _, hqrc_settings = load_hqrc_data(result.output_dir / "hqrc_data.npz")
    assert hqrc_settings["causal"] is False
    assert (
        json.loads((result.output_dir / "posterior.checkpoint.json").read_bytes())["causal"]
        is False
    )
    assert json.loads((result.output_dir / "COMPLETE").read_bytes())["causal"] is False
    posterior.close()

    reused = fit_loeo_fold(
        source,
        publication,
        approved,
        held_out_occurrence_id=held_out,
        sampler_seed=20260812,
        profile="smoke",
        draws=3,
        tune=3,
        chains=2,
        output_root=tmp_path,
    )
    assert reused.reused is True and reused.sampler_fit_count == 0
    assert reused.output_dir == result.output_dir
    assert {
        "source": _tree_hashes(run),
        "loeo": _tree_hashes(loeo_dir),
        "ar": _tree_hashes(ar_dir),
        "paper": _tree_hashes(paper_dir),
    } == protected_before
