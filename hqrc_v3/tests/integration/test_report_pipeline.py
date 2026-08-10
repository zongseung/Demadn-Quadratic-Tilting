import json

import polars as pl
import pytest
from hqrc_v3.evaluation.reports import (
    ReportContractError,
    SamplerBenchmark,
    build_report,
    run_synthetic_pipeline,
    write_benchmark,
)


def test_report_refuses_missing_ar_approval(tmp_path):
    (tmp_path / "predictions").mkdir()
    (tmp_path / "metrics").mkdir()
    prediction = tmp_path / "predictions/oof.parquet"
    metrics = tmp_path / "metrics/event_metrics.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(prediction)
    pl.DataFrame({"event_id": ["a"]}).write_parquet(metrics)
    from hqrc_v3.provenance import file_sha256

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "data_sha256": "d",
                "config_sha256": "c",
                "event_sha256": "e",
                "profile": "smoke",
                "artifacts": {
                    "predictions/oof.parquet": file_sha256(prediction),
                    "metrics/event_metrics.parquet": file_sha256(metrics),
                },
            }
        )
    )
    with pytest.raises(ReportContractError, match="approved AR calibration"):
        build_report(tmp_path)


def test_benchmark_records_unavailable_optional_backend(monkeypatch, tmp_path):
    monkeypatch.setattr("hqrc_v3.evaluation.reports.find_spec", lambda _: None)
    pymc = SamplerBenchmark(
        backend="pymc",
        wall_seconds=10.0,
        peak_rss_mb=10.0,
        min_bulk_ess_per_second=10.0,
        min_tail_ess_per_second=10.0,
        max_rhat=1.0,
        divergences=0,
    )
    output = write_benchmark(tmp_path / "benchmarks.json", [pymc])
    payload = json.loads(output.read_text())
    nutpie = next(item for item in payload["benchmarks"] if item["backend"] == "nutpie")
    assert nutpie["status"] == "not-installed"
    assert nutpie["eligible_default"] is False


def test_paper_failure_never_publishes_complete(tmp_path):
    run = run_synthetic_pipeline(output_dir=tmp_path / "run", seed=3)
    (run / "COMPLETE").unlink()
    manifest = json.loads((run / "manifest.json").read_text())
    manifest.update(profile="paper", paper_diagnostics_passed=False)
    (run / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ReportContractError, match="paper diagnostics"):
        build_report(run, profile="paper")
    assert not (run / "COMPLETE").exists()
