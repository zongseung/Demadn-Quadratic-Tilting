from datetime import datetime
from pathlib import Path

import pytest

from hqrc_v3.data import audit_hourly_data, read_hourly_data
from hqrc_v3.evaluation.data_benchmark import (
    run_data_benchmark_worker,
    write_data_benchmark_request,
)
from hqrc_v3.provenance import file_sha256


@pytest.mark.slow
def test_real_51144_row_source_runs_serial_and_parallel_polars_workers(tmp_path):
    source = Path(__file__).resolve().parents[3] / "power_demand_final.csv"
    if not source.is_file():
        pytest.skip("repository power_demand_final.csv source is absent")
    audited = audit_hourly_data(
        read_hourly_data(source),
        expected_start=datetime(2019, 1, 1),
        expected_end=datetime(2024, 10, 31, 23),
        expected_rows=51_144,
    )
    parquet = tmp_path / "real-numeric.parquet"
    columns = ("relative_humidity", "temperature_c", "load_mw")
    audited.select(columns).write_parquet(parquet)
    request = write_data_benchmark_request(
        tmp_path / "request.json",
        parquet_path=parquet,
        parquet_sha256=file_sha256(parquet),
        columns=columns,
        repetitions=3,
        workers=2,
        seed=20240810,
        workload="polars-numeric-summary-v1",
    )

    serial = run_data_benchmark_worker(
        request,
        tmp_path / "serial.json",
        requested_threads=1,
        timeout_seconds=30,
    )
    parallel = run_data_benchmark_worker(
        request,
        tmp_path / "parallel.json",
        requested_threads=2,
        timeout_seconds=30,
    )

    assert serial["dimensions"]["rows"] == parallel["dimensions"]["rows"] == 51_144
    assert serial["threads"]["actual"] == 1
    assert parallel["threads"]["actual"] == 2
    assert serial["zero_copy"]["checksum"] == parallel["zero_copy"]["checksum"]
    assert serial["workload"]["checksum"] == parallel["workload"]["checksum"]
