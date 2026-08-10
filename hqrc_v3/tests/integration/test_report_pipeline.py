import json

import pytest
from hqrc_v3.diagnostics.ar import approve_calibration
from hqrc_v3.evaluation.reports import (
    ReportContractError,
    build_report,
    prepare_synthetic_run,
    run_synthetic_pipeline,
)
from hqrc_v3.provenance import file_sha256


def _ready_run(tmp_path):
    run = tmp_path / "run"
    proposal = prepare_synthetic_run(output_dir=run, seed=3)
    approved = approve_calibration(
        proposal,
        run / "ar_diagnostics/approved.json",
        current_residual_sha256=file_sha256(run / "inputs/standardized_residuals.parquet"),
        current_config_sha256=file_sha256(run / "inputs/resolved_config.toml"),
        current_event_sha256=file_sha256(run / "inputs/event_registry.csv"),
    )
    run_synthetic_pipeline(output_dir=run, seed=3, approved_path=approved)
    return run


def test_report_invalidates_stale_complete_before_rejecting_tampered_input(tmp_path):
    run = _ready_run(tmp_path)
    (run / "inputs/resolved_config.toml").write_text("changed")
    with pytest.raises(ReportContractError, match="hash mismatch"):
        build_report(run)
    assert not (run / "COMPLETE").exists()


def test_report_rejects_updated_manifest_with_cross_run_approval(tmp_path):
    run = _ready_run(tmp_path)
    (run / "inputs/standardized_residuals.parquet").write_bytes(b"wrong run")
    manifest = json.loads((run / "manifest.json").read_text())
    # A malicious manifest rewrite cannot make the approved calibration compatible.
    digest = file_sha256(run / "inputs/standardized_residuals.parquet")
    manifest["residuals"]["sha256"] = digest
    manifest["input_artifacts"]["standardized_residuals"]["sha256"] = digest
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReportContractError, match="AR-approval provenance"):
        build_report(run)
    assert not (run / "COMPLETE").exists()
