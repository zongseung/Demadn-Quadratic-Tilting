from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from hqrc_v3.contracts import DataContractError
from hqrc_v3.data import audit_hourly_data, read_hourly_data


def test_audit_rejects_one_missing_hour(hourly_frame):
    broken = hourly_frame.filter(pl.col("timestamp") != datetime(2023, 1, 2, 5))

    with pytest.raises(DataContractError, match="one-hour continuity"):
        audit_hourly_data(broken, expected_start=None, expected_end=None, expected_rows=None)


def test_read_hourly_data_maps_the_explicit_korean_source_columns(tmp_path):
    source = tmp_path / "hourly.csv"
    source.write_text(
        "일시,hm,ta,power demand(MW)\n"
        "2023-01-01 00:00:00,45.0,10.0,100.0\n",
        encoding="utf-8",
    )

    loaded = read_hourly_data(source).collect()

    assert loaded.columns == ["timestamp", "relative_humidity", "temperature_c", "load_mw"]
    assert loaded.item(0, "load_mw") == 100.0
