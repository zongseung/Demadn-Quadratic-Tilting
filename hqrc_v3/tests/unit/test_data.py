from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl
import pytest
from hqrc_v3.contracts import DataContractError
from hqrc_v3.data import (
    audit_hourly_data,
    load_temporary_holiday_availability,
    read_hourly_data,
)


def test_audit_rejects_one_missing_hour(hourly_frame):
    broken = hourly_frame.filter(pl.col("timestamp") != datetime(2023, 1, 2, 5))

    with pytest.raises(DataContractError, match="one-hour continuity"):
        audit_hourly_data(broken, expected_start=None, expected_end=None, expected_rows=None)


@pytest.mark.parametrize(
    ("broken", "message"),
    (
        (
            lambda frame: pl.concat([frame, frame.head(1)]),
            "timestamps must be unique",
        ),
        (
            lambda frame: frame.reverse(),
            "timestamps must be sorted ascending",
        ),
        (
            lambda frame: frame.with_columns(pl.lit(float("inf")).alias("load_mw")),
            "load_mw must be finite and positive",
        ),
        (
            lambda frame: frame.with_columns(pl.lit(0.0).alias("load_mw")),
            "load_mw must be finite and positive",
        ),
        (
            lambda frame: frame.with_columns(pl.lit(float("nan")).alias("temperature_c")),
            "weather values must be finite",
        ),
    ),
)
def test_audit_rejects_invalid_timestamp_load_and_weather(hourly_frame, broken, message):
    with pytest.raises(DataContractError, match=message):
        audit_hourly_data(
            broken(hourly_frame), expected_start=None, expected_end=None, expected_rows=None
        )


@pytest.mark.parametrize(
    ("expected_start", "expected_end", "expected_rows", "message"),
    (
        (datetime(2023, 1, 1, 1), None, None, "does not start"),
        (None, datetime(2023, 1, 10, 22), None, "does not end"),
        (None, None, 239, "expected 239 hourly rows"),
    ),
)
def test_audit_rejects_wrong_expected_bounds(
    hourly_frame, expected_start, expected_end, expected_rows, message
):
    with pytest.raises(DataContractError, match=message):
        audit_hourly_data(
            hourly_frame,
            expected_start=expected_start,
            expected_end=expected_end,
            expected_rows=expected_rows,
        )


def test_read_hourly_data_maps_the_explicit_korean_source_columns(tmp_path):
    source = tmp_path / "hourly.csv"
    source.write_text(
        "일시,hm,ta,power demand(MW),holiday_name,is_holiday_dummies\n"
        "2023-01-01 00:00:00,45.0,10.0,100.0,,0\n",
        encoding="utf-8",
    )

    loaded = read_hourly_data(source).collect()

    assert loaded.columns == [
        "timestamp",
        "relative_humidity",
        "temperature_c",
        "load_mw",
        "source_holiday_name",
        "source_public_holiday",
    ]
    assert loaded.item(0, "load_mw") == 100.0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda frame: frame.with_columns(
                pl.when(pl.col("timestamp") == datetime(2023, 1, 8, 3))
                .then(0)
                .otherwise(pl.col("source_public_holiday"))
                .alias("source_public_holiday")
            ),
            "date-level public-holiday flag",
        ),
        (
            lambda frame: frame.with_columns(
                pl.when(pl.col("timestamp").dt.date() == datetime(2023, 1, 1).date())
                .then(pl.lit("not a holiday"))
                .otherwise(pl.col("source_holiday_name"))
                .alias("source_holiday_name")
            ),
            "name/flag",
        ),
        (
            lambda frame: frame.with_columns(pl.lit(2).alias("source_public_holiday")),
            "binary",
        ),
    ],
)
def test_audit_rejects_inconsistent_source_holiday_fields(
    hourly_frame, mutation, message
):
    with pytest.raises(DataContractError, match=message):
        audit_hourly_data(
            mutation(hourly_frame), expected_start=None, expected_end=None, expected_rows=None
        )


def test_temporary_holiday_availability_registry_is_exact_and_prior_to_each_date():
    registry = load_temporary_holiday_availability(
        Path(__file__).resolve().parents[2] / "configs/temporary_holiday_availability.csv"
    )

    assert [row.holiday_date.isoformat() for row in registry] == [
        "2020-08-17",
        "2023-10-02",
        "2024-10-01",
    ]
    assert all(row.known_on < row.holiday_date for row in registry)


def test_read_rejects_noninteger_raw_holiday_flag(tmp_path):
    source = tmp_path / "float-flag.csv"
    source.write_text(
        "일시,hm,ta,power demand(MW),holiday_name,is_holiday_dummies\n"
        "2023-01-01 00:00:00,45.0,10.0,100.0,,0.0\n",
        encoding="utf-8",
    )

    with pytest.raises(DataContractError, match="integer dtype"):
        read_hourly_data(source)
