"""Polars-first ingestion and audit of the immutable hourly input."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import polars as pl

from hqrc_v3.contracts import DataContractError

CANONICAL_COLUMNS = ("timestamp", "relative_humidity", "temperature_c", "load_mw")
SOURCE_RENAME_MAP = {
    "일시": "timestamp",
    "hm": "relative_humidity",
    "ta": "temperature_c",
    "power demand(MW)": "load_mw",
}


def read_hourly_data(path: Path) -> pl.LazyFrame:
    """Scan a raw CSV and explicitly map its documented Korean source schema."""

    try:
        raw = pl.scan_csv(path)
        source_columns = set(raw.collect_schema().names())
    except (OSError, pl.exceptions.PolarsError) as error:
        raise DataContractError(f"unable to read hourly data {path}: {error}") from error

    missing = set(SOURCE_RENAME_MAP) - source_columns
    if missing:
        raise DataContractError(f"hourly source is missing required columns: {sorted(missing)}")

    return raw.select(
        pl.col("일시").cast(pl.String).str.to_datetime(strict=True).alias("timestamp"),
        pl.col("hm").cast(pl.Float64).alias("relative_humidity"),
        pl.col("ta").cast(pl.Float64).alias("temperature_c"),
        pl.col("power demand(MW)").cast(pl.Float64).alias("load_mw"),
    )


def _collect(frame: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame:
    if isinstance(frame, pl.LazyFrame):
        return frame.collect()
    if isinstance(frame, pl.DataFrame):
        return frame
    raise DataContractError("hourly data must be a Polars DataFrame or LazyFrame")


def audit_hourly_data(
    frame: pl.DataFrame | pl.LazyFrame,
    *,
    expected_start: datetime | None,
    expected_end: datetime | None,
    expected_rows: int | None,
) -> pl.DataFrame:
    """Validate the canonical hourly schema without changing its row order."""

    hourly = _collect(frame)
    missing = set(CANONICAL_COLUMNS) - set(hourly.columns)
    if missing:
        raise DataContractError(f"hourly data is missing canonical columns: {sorted(missing)}")
    if hourly.is_empty():
        raise DataContractError("hourly data must not be empty")
    if hourly["timestamp"].null_count() > 0:
        raise DataContractError("hourly timestamps must not be null")
    if hourly["timestamp"].n_unique() != hourly.height:
        raise DataContractError("hourly timestamps must be unique")
    if not hourly["timestamp"].is_sorted():
        raise DataContractError("hourly timestamps must be sorted ascending")

    hour_deltas = hourly["timestamp"].diff().drop_nulls().dt.total_hours()
    if not hour_deltas.eq(1).all():
        raise DataContractError("hourly timestamps must have one-hour continuity")

    if expected_rows is not None and hourly.height != expected_rows:
        raise DataContractError(f"expected {expected_rows} hourly rows, found {hourly.height}")
    if expected_start is not None and hourly.item(0, "timestamp") != expected_start:
        raise DataContractError("hourly data does not start at the expected timestamp")
    if expected_end is not None and hourly.item(-1, "timestamp") != expected_end:
        raise DataContractError("hourly data does not end at the expected timestamp")

    invalid_load = hourly.select(
        (pl.col("load_mw").is_null() | ~pl.col("load_mw").is_finite()).any()
    ).item()
    if invalid_load or hourly.select((pl.col("load_mw") <= 0).any()).item():
        raise DataContractError("load_mw must be finite and positive")

    invalid_weather = hourly.select(
        (
            pl.col("temperature_c").is_null()
            | ~pl.col("temperature_c").is_finite()
            | pl.col("relative_humidity").is_null()
            | ~pl.col("relative_humidity").is_finite()
        ).any()
    ).item()
    if invalid_weather:
        raise DataContractError("weather values must be finite after the configured policy")
    return hourly
