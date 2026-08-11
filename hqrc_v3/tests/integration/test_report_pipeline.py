import hashlib
import json
import sys
from pathlib import Path

import pytest
from hqrc_v3.bayes.benchmark import benchmark_sampler_processes
from hqrc_v3.diagnostics.ar import approve_calibration
from hqrc_v3.evaluation.reports import (
    ReportContractError,
    build_report,
    prepare_synthetic_run,
    run_synthetic_pipeline,
)
from hqrc_v3.provenance import file_sha256

FAKE_SAMPLER = Path(__file__).parents[1] / "fixtures" / "fake_sampler_worker.py"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


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


def _rebind_sampler_manifest(run):
    sampler = run / "benchmarks/samplers.json"
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["input_artifacts"]["sampler_benchmark"]["sha256"] = file_sha256(sampler)
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))


def _replace_sampler_and_rebind_manifest(run, raw):
    (run / "benchmarks/samplers.json").write_bytes(raw)
    _rebind_sampler_manifest(run)


def _write_production_sampler_benchmark(run, tmp_path):
    inputs = tmp_path / "sampler-inputs"
    inputs.mkdir(exist_ok=True)
    files = {}
    for name in ("data.npz", "data.json"):
        path = inputs / name
        path.write_text(name)
        files[name] = path
    files["approved.json"] = run / "ar_diagnostics/approved.json"
    output = benchmark_sampler_processes(
        run / "benchmarks/samplers.json",
        request_directory=tmp_path / "sampler-workers",
        timeout_seconds=5,
        request_kwargs={
            "hqrc_npz": files["data.npz"],
            "hqrc_metadata": files["data.json"],
            "approved_ar": files["approved.json"],
            "residual_sha256": file_sha256(
                run / "inputs/standardized_residuals.parquet"
            ),
            "config_sha256": file_sha256(run / "inputs/resolved_config.toml"),
            "event_sha256": file_sha256(run / "inputs/event_registry.csv"),
            "variant": "H3",
            "pooling": "partial",
            "options": {"covariance": "full"},
            "seed": 7,
            "draws": 20,
            "tune": 20,
            "chains": 2,
            "profile": "smoke",
        },
        worker_commands={
            "pymc": (sys.executable, str(FAKE_SAMPLER), "ok"),
            "nutpie": (sys.executable, str(FAKE_SAMPLER), "ok"),
        },
    )
    _rebind_sampler_manifest(run)
    return output


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


@pytest.mark.parametrize(
    "invalid_version", [True, 1.0, 2.0], ids=("boolean", "wrong-float", "equal-float")
)
def test_report_manifest_rejects_noninteger_version(tmp_path, invalid_version):
    run = _ready_run(tmp_path)
    manifest_path = run / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["schema_version"] = invalid_version
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(ReportContractError, match="versioned strict schema"):
        build_report(run)
    assert not (run / "COMPLETE").exists()


@pytest.mark.parametrize("raw", [b"not-json", b"{}"])
def test_report_rejects_malformed_hash_rebound_sampler_benchmark(tmp_path, raw):
    run = _ready_run(tmp_path)
    _replace_sampler_and_rebind_manifest(run, raw)
    with pytest.raises(ReportContractError, match="sampler benchmark"):
        build_report(run)
    assert not (run / "COMPLETE").exists()


def test_report_rejects_tampered_sampler_values_even_after_manifest_rebind(tmp_path):
    run = _ready_run(tmp_path)
    sampler = run / "benchmarks/samplers.json"
    payload = json.loads(sampler.read_text())
    pymc = next(item for item in payload["benchmarks"] if item["backend"] == "pymc")
    pymc["wall_seconds"] = -1.0
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    _replace_sampler_and_rebind_manifest(run, raw)
    with pytest.raises(ReportContractError, match="sampler benchmark"):
        build_report(run)
    assert not (run / "COMPLETE").exists()


def test_report_rejects_noncanonical_sampler_json_even_after_manifest_rebind(tmp_path):
    run = _ready_run(tmp_path)
    sampler = run / "benchmarks/samplers.json"
    payload = json.loads(sampler.read_text())
    _replace_sampler_and_rebind_manifest(run, json.dumps(payload, indent=2).encode())
    with pytest.raises(ReportContractError, match="sampler benchmark.*canonical"):
        build_report(run)
    assert not (run / "COMPLETE").exists()


def test_report_rejects_boolean_legacy_sampler_version_after_redigest(tmp_path):
    run = _ready_run(tmp_path)
    sampler = run / "benchmarks/samplers.json"
    payload = json.loads(sampler.read_bytes())
    payload.pop("benchmark_digest")
    payload["schema_version"] = True
    payload["benchmark_digest"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _replace_sampler_and_rebind_manifest(run, _canonical(payload) + b"\n")

    with pytest.raises(ReportContractError, match="legacy sampler benchmark.*invalid"):
        build_report(run)
    assert not (run / "COMPLETE").exists()


def test_report_accepts_process_orchestrator_sampler_artifact(tmp_path):
    run = _ready_run(tmp_path)
    artifact = _write_production_sampler_benchmark(run, tmp_path)
    payload = json.loads(artifact.read_text())
    assert payload["benchmarks"]["pymc"]["status"] == "ok"
    assert payload["benchmarks"]["nutpie"]["status"] == "ok"

    assert build_report(run) == run / "reports"
    assert (run / "COMPLETE").is_file()


def test_report_recomputes_production_sampler_eligibility_after_redigest(tmp_path):
    run = _ready_run(tmp_path)
    artifact = _write_production_sampler_benchmark(run, tmp_path)
    payload = json.loads(artifact.read_text())
    assert payload["nutpie_eligible_default"] is False
    payload.pop("benchmark_digest")
    payload["nutpie_eligible_default"] = True
    payload["benchmark_digest"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _replace_sampler_and_rebind_manifest(run, _canonical(payload) + b"\n")

    with pytest.raises(ReportContractError, match="sampler benchmark|eligibility"):
        build_report(run)
    assert not (run / "COMPLETE").exists()


def test_report_rejects_boolean_production_sampler_version_after_redigest(tmp_path):
    run = _ready_run(tmp_path)
    artifact = _write_production_sampler_benchmark(run, tmp_path)
    payload = json.loads(artifact.read_text())
    payload.pop("benchmark_digest")
    payload["schema_version"] = True
    payload["benchmark_digest"] = hashlib.sha256(_canonical(payload)).hexdigest()
    _replace_sampler_and_rebind_manifest(run, _canonical(payload) + b"\n")

    with pytest.raises(ReportContractError, match="sampler benchmark.*version"):
        build_report(run)
    assert not (run / "COMPLETE").exists()


def test_sampler_gate_rejects_negative_posterior_distance():
    from hqrc_v3.evaluation.reports import SamplerBenchmark, sampler_eligible_default

    measured = SamplerBenchmark("pymc", 1.0, 1.0, 1.0, 1.0, 1.0, 0)
    with pytest.raises(ReportContractError, match="non-negative"):
        sampler_eligible_default(measured, measured, mean_distance_sd=-0.01)


def test_sampler_gate_accepts_exact_distance_and_twenty_percent_boundaries():
    from hqrc_v3.evaluation.reports import SamplerBenchmark, sampler_eligible_default

    pymc = SamplerBenchmark("pymc", 10.0, 2.0, 100.0, 90.0, 1.0, 0)
    faster = SamplerBenchmark("nutpie", 8.0, 2.0, 100.0, 90.0, 1.01, 0)
    higher_ess = SamplerBenchmark("nutpie", 10.0, 2.0, 120.0, 90.0, 1.01, 0)
    assert sampler_eligible_default(pymc, faster, mean_distance_sd=0.1)
    assert sampler_eligible_default(pymc, higher_ess, mean_distance_sd=0.1)
