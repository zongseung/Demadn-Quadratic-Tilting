from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import polars as pl
import pytest
from hqrc_v3.evaluation.data_benchmark import (
    DataBenchmarkError,
    benchmark_polars_data,
    load_data_benchmark,
    load_data_benchmark_request,
    run_data_benchmark_worker,
    write_data_benchmark_request,
)
from hqrc_v3.provenance import file_sha256

FAKE = Path(__file__).parents[1] / "fixtures" / "fake_data_benchmark_worker.py"


def _canonical(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _parquet(tmp_path: Path, *, nullable: bool = False) -> Path:
    path = tmp_path / "numeric.parquet"
    second = [4.0, None, 2.0, 1.0] if nullable else [4.0, 3.0, 2.0, 1.0]
    pl.DataFrame(
        {
            "load": pl.Series([1.0, 2.0, 3.0, 4.0], dtype=pl.Float64),
            "temperature": pl.Series(second, dtype=pl.Float64),
            "ignored": [1, 2, 3, 4],
        }
    ).write_parquet(path)
    return path


def _request(tmp_path: Path, *, nullable: bool = False) -> Path:
    parquet = _parquet(tmp_path, nullable=nullable)
    return write_data_benchmark_request(
        tmp_path / "data-request.json",
        parquet_path=parquet,
        parquet_sha256=file_sha256(parquet),
        columns=("load", "temperature"),
        repetitions=3,
        workers=2,
        seed=19,
        workload="polars-numeric-summary-v1",
    )


def _sampler_benchmark(tmp_path: Path, *, wall_seconds: float = 2.0) -> Path:
    payload = {
        "schema_version": 1,
        "benchmarks": {
            "pymc": {
                "status": "ok",
                "backend": "pymc",
                "pid": 101,
                "request_digest": "a" * 64,
                "wall_seconds": wall_seconds,
                "peak_rss_mb": 20.0,
                "min_bulk_ess_per_second": 50.0,
                "min_tail_ess_per_second": 40.0,
                "max_rhat": 1.01,
                "divergences": 0,
                "versions": {"python": "test", "pymc": "test", "arviz": "test"},
            },
            "nutpie": {"status": "not-installed", "eligible_default": False},
        },
        "maximum_mean_distance_sd": None,
        "posterior_audit": {},
        "nutpie_eligible_default": False,
    }
    payload["benchmark_digest"] = _digest(payload)
    path = tmp_path / "sampler-benchmark.json"
    path.write_text(_canonical(payload) + "\n")
    return path


def test_real_polars_workers_measure_copy_ownership_and_parallelism(tmp_path):
    request = _request(tmp_path)
    sampler = _sampler_benchmark(tmp_path)
    output = benchmark_polars_data(
        tmp_path / "data-benchmark.json",
        request_path=request,
        sampler_benchmark_path=sampler,
        sampler_sha256=file_sha256(sampler),
        worker_directory=tmp_path / "worker-results",
        timeout_seconds=15,
    )

    payload = load_data_benchmark(output)
    assert payload["request"]["artifact_sha256"] == file_sha256(tmp_path / "numeric.parquet")
    assert payload["request"]["rows"] == 4
    assert payload["request"]["columns"] == ["load", "temperature"]
    assert payload["zero_copy"]["owns_data"] is False
    assert payload["zero_copy"]["writeable"] is False
    assert payload["zero_copy"]["owned_bytes_per_repetition"] == 0
    assert payload["explicit_copy"]["owns_data"] is True
    assert payload["explicit_copy"]["allocation_bytes_per_repetition"] == 4 * 2 * 8
    assert payload["zero_copy"]["checksum"] == payload["explicit_copy"]["checksum"]
    assert payload["processing"]["serial"]["requested_threads"] == 1
    assert payload["processing"]["serial"]["actual_threads"] == 1
    assert payload["processing"]["parallel"]["requested_threads"] == 2
    assert payload["processing"]["parallel"]["actual_threads"] == 2
    assert payload["processing"]["serial"]["pid"] != os.getpid()
    assert payload["processing"]["parallel"]["pid"] != os.getpid()
    assert payload["processing"]["serial"]["pid"] != payload["processing"]["parallel"]["pid"]
    assert (
        payload["processing"]["serial"]["checksum"]
        == payload["processing"]["parallel"]["checksum"]
    )
    assert len(payload["processing"]["serial"]["timings_seconds"]) == 3
    assert payload["derived"]["data_processing_share_of_measured_total"] > 0
    assert 0 <= payload["derived"]["explicit_copy_share_of_data_processing"] <= 1
    assert payload["decision"]["polars_already_rust"] is True
    assert payload["decision"]["custom_rust_implemented"] is False
    assert "not-measured" not in _canonical(payload)

    payload.pop("benchmark_digest")
    payload["derived"]["data_processing_share_of_measured_total"] = 0.75
    payload["benchmark_digest"] = _digest(payload)
    output.write_text(_canonical(payload) + "\n")
    with pytest.raises(DataBenchmarkError, match="measured share"):
        load_data_benchmark(output)


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"repetitions": 2}, "repetitions"),
        ({"workers": 1}, "workers"),
        ({"seed": -1}, "seed"),
        ({"columns": ("load", "load")}, "columns"),
        ({"workload": "unknown"}, "workload"),
    ],
)
def test_request_rejects_invalid_arguments(tmp_path, overrides, match):
    parquet = _parquet(tmp_path)
    kwargs = {
        "parquet_path": parquet,
        "parquet_sha256": file_sha256(parquet),
        "columns": ("load", "temperature"),
        "repetitions": 3,
        "workers": 2,
        "seed": 19,
        "workload": "polars-numeric-summary-v1",
    }
    kwargs.update(overrides)
    with pytest.raises(DataBenchmarkError, match=match):
        write_data_benchmark_request(tmp_path / "request.json", **kwargs)


def test_request_rejects_bound_hash_schema_canonical_and_input_tamper(tmp_path):
    parquet = _parquet(tmp_path)
    with pytest.raises(DataBenchmarkError, match="caller-bound"):
        write_data_benchmark_request(
            tmp_path / "bad.json",
            parquet_path=parquet,
            parquet_sha256="0" * 64,
            columns=("load",),
            repetitions=3,
            workers=2,
            seed=0,
            workload="polars-numeric-summary-v1",
        )

    request = _request(tmp_path)
    original = json.loads(request.read_text())
    wrong_schema = dict(original)
    wrong_schema.pop("request_digest")
    wrong_schema["unexpected"] = True
    wrong_schema["request_digest"] = _digest(wrong_schema)
    request.write_text(_canonical(wrong_schema) + "\n")
    with pytest.raises(DataBenchmarkError, match="schema"):
        load_data_benchmark_request(request)

    request.write_text(json.dumps(original, indent=2) + "\n")
    with pytest.raises(DataBenchmarkError, match="canonical"):
        load_data_benchmark_request(request)

    request.write_text(_canonical(original) + "\n")
    with (tmp_path / "numeric.parquet").open("ab") as output:
        output.write(b"tamper")
    with pytest.raises(DataBenchmarkError, match="input digest"):
        load_data_benchmark_request(request)


def test_real_worker_rejects_nullable_float_that_would_force_copy(tmp_path):
    with pytest.raises(DataBenchmarkError, match="zero-copy"):
        run_data_benchmark_worker(
            _request(tmp_path, nullable=True),
            tmp_path / "result.json",
            requested_threads=1,
            timeout_seconds=15,
        )


@pytest.mark.parametrize(
    ("mode", "timeout", "match"),
    [
        ("bad-digest", 5, "digest"),
        ("bad-pid", 5, "PID"),
        ("extra-key", 5, "schema"),
        ("nonzero", 5, "exit"),
        ("malformed", 5, "JSON"),
        ("sleep", 0.2, "timeout"),
    ],
)
def test_parent_fails_closed_for_data_worker_errors(tmp_path, mode, timeout, match):
    with pytest.raises(DataBenchmarkError, match=match):
        run_data_benchmark_worker(
            _request(tmp_path),
            tmp_path / "result.json",
            requested_threads=1,
            timeout_seconds=timeout,
            worker_command=(sys.executable, str(FAKE), mode),
        )


def test_orchestrator_rejects_serial_parallel_checksum_mismatch(tmp_path):
    sampler = _sampler_benchmark(tmp_path)
    with pytest.raises(DataBenchmarkError, match="workload checksum"):
        benchmark_polars_data(
            tmp_path / "benchmark.json",
            request_path=_request(tmp_path),
            sampler_benchmark_path=sampler,
            sampler_sha256=file_sha256(sampler),
            worker_directory=tmp_path / "worker-results",
            timeout_seconds=5,
            worker_commands={
                1: (sys.executable, str(FAKE), "ok"),
                2: (sys.executable, str(FAKE), "checksum-mismatch"),
            },
        )


def test_orchestrator_revalidates_sampler_schema_digest_and_bound_hash(tmp_path):
    sampler = _sampler_benchmark(tmp_path)
    request = _request(tmp_path)
    with pytest.raises(DataBenchmarkError, match="caller-bound sampler"):
        benchmark_polars_data(
            tmp_path / "benchmark.json",
            request_path=request,
            sampler_benchmark_path=sampler,
            sampler_sha256="0" * 64,
            worker_directory=tmp_path / "worker-results",
            timeout_seconds=5,
        )

    payload = json.loads(sampler.read_text())
    payload.pop("benchmark_digest")
    payload["unexpected"] = True
    payload["benchmark_digest"] = _digest(payload)
    sampler.write_text(_canonical(payload) + "\n")
    with pytest.raises(DataBenchmarkError, match="sampler benchmark schema"):
        benchmark_polars_data(
            tmp_path / "benchmark.json",
            request_path=request,
            sampler_benchmark_path=sampler,
            sampler_sha256=file_sha256(sampler),
            worker_directory=tmp_path / "worker-results",
            timeout_seconds=5,
        )

    sampler = _sampler_benchmark(tmp_path)
    payload = json.loads(sampler.read_text())
    payload["benchmarks"]["pymc"]["wall_seconds"] = 99.0
    sampler.write_text(_canonical(payload) + "\n")
    with pytest.raises(DataBenchmarkError, match="sampler benchmark digest"):
        benchmark_polars_data(
            tmp_path / "benchmark.json",
            request_path=request,
            sampler_benchmark_path=sampler,
            sampler_sha256=file_sha256(sampler),
            worker_directory=tmp_path / "worker-results",
            timeout_seconds=5,
        )
