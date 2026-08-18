from hqrc_v3.diagnostics.ar import approve_calibration
from hqrc_v3.evaluation.reports import prepare_synthetic_run, run_synthetic_pipeline
from hqrc_v3.provenance import file_sha256


def _approve(run, proposal):
    return approve_calibration(
        proposal,
        run / "ar_diagnostics/approved.json",
        current_residual_sha256=file_sha256(run / "inputs/standardized_residuals.parquet"),
        current_config_sha256=file_sha256(run / "inputs/resolved_config.toml"),
        current_event_sha256=file_sha256(run / "inputs/event_registry.csv"),
    )


def test_synthetic_end_to_end_requires_explicit_approval_and_writes_required_artifacts(tmp_path):
    run = tmp_path / "run"
    proposal = prepare_synthetic_run(output_dir=run, seed=3)
    approved = _approve(run, proposal)
    run_synthetic_pipeline(output_dir=run, seed=3, approved_path=approved)
    for path in (
        "manifest.json",
        "predictions/oof.parquet",
        "inputs/standardized_residuals.parquet",
        "ar_diagnostics/approved.json",
        "posterior/smoke.nc",
        "metrics/event_metrics.parquet",
        "benchmarks/samplers.json",
        "figures/event_metrics.svg",
        "COMPLETE",
    ):
        assert (run / path).is_file()
