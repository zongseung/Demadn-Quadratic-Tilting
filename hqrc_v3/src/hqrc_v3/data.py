"""Polars-first ingestion and audit of the immutable hourly input."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import polars as pl

from hqrc_v3.contracts import DataContractError

CANONICAL_COLUMNS = (
    "timestamp",
    "relative_humidity",
    "temperature_c",
    "load_mw",
    "source_holiday_name",
    "source_public_holiday",
)
SOURCE_RENAME_MAP = {
    "일시": "timestamp",
    "hm": "relative_humidity",
    "ta": "temperature_c",
    "power demand(MW)": "load_mw",
    "holiday_name": "source_holiday_name",
    "is_holiday_dummies": "source_public_holiday",
}
FIXED_START = datetime(2019, 1, 1)
FIXED_END = datetime(2024, 10, 31, 23)
FIXED_ROWS = 51_144
FIXED_PUBLIC_HOLIDAY_DATES = 107
FIXED_SUBSTITUTE_OR_TEMPORARY_DATES = 14
_AVAILABILITY_COLUMNS = ("holiday_date", "known_on", "source_url")
_INTEGER_DTYPES = {
    pl.Int8,
    pl.Int16,
    pl.Int32,
    pl.Int64,
    pl.UInt8,
    pl.UInt16,
    pl.UInt32,
    pl.UInt64,
}
_TEMPORARY_AVAILABILITY = {
    date(2020, 8, 17): (
        date(2020, 7, 21),
        "https://www.korea.kr/news/policyNewsView.do?newsId=148874895",
    ),
    date(2023, 10, 2): (
        date(2023, 8, 31),
        "https://www.korea.kr/news/policyNewsView.do?newsId=148919605",
    ),
    date(2024, 10, 1): (
        date(2024, 9, 3),
        "https://www.korea.kr/news/policyNewsView.do?newsId=148933400",
    ),
}


@dataclass(frozen=True)
class TemporaryHolidayAvailability:
    """Versioned proof that an exceptional holiday was known before forecast day."""

    holiday_date: date
    known_on: date
    source_url: str

    def __post_init__(self) -> None:
        if self.known_on >= self.holiday_date:
            raise DataContractError("temporary holiday must be known before its holiday date")
        if not self.source_url.startswith("https://www.korea.kr/"):
            raise DataContractError("temporary holiday source must be an official Korea URL")


def load_temporary_holiday_availability(
    path: Path,
) -> tuple[TemporaryHolidayAvailability, ...]:
    """Load the exact three source-backed temporary-holiday availability rows."""

    try:
        with Path(path).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None or tuple(reader.fieldnames) != _AVAILABILITY_COLUMNS:
                raise DataContractError(
                    "temporary holiday availability columns differ from the frozen schema"
                )
            rows = tuple(
                TemporaryHolidayAvailability(
                    holiday_date=date.fromisoformat((row["holiday_date"] or "").strip()),
                    known_on=date.fromisoformat((row["known_on"] or "").strip()),
                    source_url=(row["source_url"] or "").strip(),
                )
                for row in reader
            )
    except OSError as error:
        raise DataContractError(
            f"unable to read temporary holiday availability {path}: {error}"
        ) from error
    except (TypeError, ValueError) as error:
        raise DataContractError(f"invalid temporary holiday availability: {error}") from error
    actual = {row.holiday_date: (row.known_on, row.source_url) for row in rows}
    if len(actual) != len(rows) or actual != _TEMPORARY_AVAILABILITY:
        raise DataContractError("temporary holiday availability differs from the frozen registry")
    return tuple(sorted(rows, key=lambda row: row.holiday_date))


def substitute_or_temporary_expr() -> pl.Expr:
    """Return the fixed source-name predicate for B1 substitute/temporary dates."""

    name = pl.col("source_holiday_name").str.strip_chars()
    day = pl.col("timestamp").dt.date()
    return (pl.col("source_public_holiday") == 1) & (
        name.str.starts_with("Alternative holiday")
        | (name == "Temporary Public Holiday")
        | ((day == date(2024, 10, 1)) & (name == "Armed Forces Day"))
    )


def temporary_holiday_expr() -> pl.Expr:
    """Return only exceptional temporary dates requiring availability evidence."""

    name = pl.col("source_holiday_name").str.strip_chars()
    day = pl.col("timestamp").dt.date()
    return (pl.col("source_public_holiday") == 1) & (
        (name == "Temporary Public Holiday")
        | ((day == date(2024, 10, 1)) & (name == "Armed Forces Day"))
    )


def read_hourly_data(path: Path) -> pl.LazyFrame:
    """Scan a raw CSV and explicitly map its documented Korean source schema."""

    try:
        raw = pl.scan_csv(path)
        source_schema = raw.collect_schema()
        source_columns = set(source_schema.names())
    except (OSError, pl.exceptions.PolarsError) as error:
        raise DataContractError(f"unable to read hourly data {path}: {error}") from error

    missing = set(SOURCE_RENAME_MAP) - source_columns
    if missing:
        raise DataContractError(f"hourly source is missing required columns: {sorted(missing)}")
    if source_schema["is_holiday_dummies"] not in _INTEGER_DTYPES:
        raise DataContractError("source public-holiday flag must have an integer dtype")

    return raw.select(
        pl.col("일시").cast(pl.String).str.to_datetime(strict=True).alias("timestamp"),
        pl.col("hm").cast(pl.Float64).alias("relative_humidity"),
        pl.col("ta").cast(pl.Float64).alias("temperature_c"),
        pl.col("power demand(MW)").cast(pl.Float64).alias("load_mw"),
        pl.col("holiday_name")
        .cast(pl.String)
        .fill_null("")
        .str.strip_chars()
        .alias("source_holiday_name"),
        pl.col("is_holiday_dummies")
        .cast(pl.Int8, strict=True)
        .alias("source_public_holiday"),
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
    expected_public_holiday_dates: int | None = None,
    expected_substitute_or_temporary_dates: int | None = None,
    temporary_holiday_availability: tuple[TemporaryHolidayAvailability, ...] | None = None,
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

    if hourly["source_public_holiday"].null_count() > 0:
        raise DataContractError("source public-holiday flag must be non-null and binary")
    if hourly.schema["source_public_holiday"] not in _INTEGER_DTYPES:
        raise DataContractError("source public-holiday flag must have an integer dtype")
    if not hourly["source_public_holiday"].is_in([0, 1]).all():
        raise DataContractError("source public-holiday flag must be non-null and binary")
    normalized = hourly.with_columns(
        pl.col("source_holiday_name")
        .cast(pl.String, strict=True)
        .fill_null("")
        .str.strip_chars()
        .alias("source_holiday_name"),
        pl.col("timestamp").dt.date().alias("_source_date"),
    )
    daily = normalized.group_by("_source_date").agg(
        pl.len().alias("rows"),
        pl.col("source_public_holiday").n_unique().alias("flag_values"),
        pl.col("source_holiday_name").n_unique().alias("name_values"),
        pl.col("source_public_holiday").first().alias("flag"),
        pl.col("source_holiday_name").first().alias("name"),
    )
    if not daily.filter(pl.col("rows") != 24).is_empty():
        raise DataContractError("each source date must contain exactly 24 hourly rows")
    if not daily.filter(pl.col("flag_values") != 1).is_empty():
        raise DataContractError("date-level public-holiday flag must be replicated consistently")
    if not daily.filter(pl.col("name_values") != 1).is_empty():
        raise DataContractError("date-level holiday name must be replicated consistently")
    if not daily.filter((pl.col("flag") == 1) != (pl.col("name") != "")).is_empty():
        raise DataContractError("source holiday name/flag pairs are inconsistent")

    fixed_source = (
        normalized.height == FIXED_ROWS
        and normalized.item(0, "timestamp") == FIXED_START
        and normalized.item(-1, "timestamp") == FIXED_END
    )
    public_dates = daily.filter(pl.col("flag") == 1).height
    expected_public = (
        FIXED_PUBLIC_HOLIDAY_DATES
        if fixed_source and expected_public_holiday_dates is None
        else expected_public_holiday_dates
    )
    if expected_public is not None and public_dates != expected_public:
        raise DataContractError(
            f"expected {expected_public} public-holiday dates, found {public_dates}"
        )
    substitute_dates = (
        normalized.filter(substitute_or_temporary_expr())
        .select("_source_date")
        .unique()
        .height
    )
    expected_substitute = (
        FIXED_SUBSTITUTE_OR_TEMPORARY_DATES
        if fixed_source and expected_substitute_or_temporary_dates is None
        else expected_substitute_or_temporary_dates
    )
    if expected_substitute is not None and substitute_dates != expected_substitute:
        raise DataContractError(
            "expected "
            f"{expected_substitute} substitute/temporary dates, found {substitute_dates}"
        )
    if temporary_holiday_availability is not None:
        minimum = normalized.item(0, "timestamp").date()
        maximum = normalized.item(-1, "timestamp").date()
        expected_temporary_dates = {
            row.holiday_date
            for row in temporary_holiday_availability
            if minimum <= row.holiday_date <= maximum
        }
        actual_temporary_dates = set(
            normalized.filter(temporary_holiday_expr())["_source_date"].unique().to_list()
        )
        if actual_temporary_dates != expected_temporary_dates:
            raise DataContractError(
                "source temporary holidays differ from the availability registry"
            )
    return normalized.drop("_source_date")
