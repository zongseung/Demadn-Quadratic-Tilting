from hqrc_v3.evaluation.reports import run_synthetic_pipeline


def test_synthetic_end_to_end_writes_required_artifacts(tmp_path):
    run = run_synthetic_pipeline(output_dir=tmp_path / "run", seed=3)
    for path in (
        "manifest.json",
        "predictions/oof.parquet",
        "ar_diagnostics/approved.json",
        "metrics/event_metrics.parquet",
        "posterior/diagnostics.json",
        "benchmarks/samplers.json",
        "figures/report_coverage.svg",
        "COMPLETE",
    ):
        assert (run / path).is_file()
