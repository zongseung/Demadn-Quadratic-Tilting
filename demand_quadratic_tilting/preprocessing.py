"""workalendar 기반 한국 공휴일 전처리 모듈 (Polars)."""

from __future__ import annotations

from datetime import date, timedelta
from typing import Optional

import polars as pl
from workalendar.asia import SouthKorea

from .constants import (
    CHUSEOK_CORE,
    CHUSEOK_LABELS,
    SEOLLAL_CORE,
    SEOLLAL_LABELS,
)


def _build_holiday_df(start_year: int, end_year: int) -> pl.DataFrame:
    """workalendar로 start_year~end_year 한국 공휴일 DataFrame 생성."""
    cal = SouthKorea()
    rows: list[dict] = []
    for year in range(start_year, end_year + 1):
        for dt, name in cal.holidays(year):
            rows.append({"date": dt, "holiday_name": name})
    return pl.DataFrame(rows).with_columns(pl.col("date").cast(pl.Date))


def _classify_holiday(name: str) -> str:
    if name in CHUSEOK_LABELS:
        return "Chuseok"
    if name in SEOLLAL_LABELS:
        return "Seollal"
    return "other"


def annotate_holidays(
    df: pl.DataFrame,
    datetime_col: str = "datetime",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pl.DataFrame:
    """datetime 컬럼이 있는 DataFrame에 holiday_name, holiday_type 컬럼 추가.

    Parameters
    ----------
    df : pl.DataFrame
        datetime_col 컬럼을 포함한 시계열 데이터.
    datetime_col : str
        datetime 컬럼명 (Datetime 또는 Date 타입).
    start_year, end_year : int, optional
        공휴일 조회 범위. None이면 데이터에서 추출.

    Returns
    -------
    pl.DataFrame
        holiday_name (str), holiday_type ("Chuseok"|"Seollal"|"other"|"non-event") 추가.
    """
    dt_series = df[datetime_col]
    if dt_series.dtype == pl.Date:
        date_col_expr = pl.col(datetime_col)
    else:
        date_col_expr = pl.col(datetime_col).dt.date()

    df = df.with_columns(date_col_expr.alias("_date_key"))

    if start_year is None:
        start_year = int(df["_date_key"].min().year)  # type: ignore[union-attr]
    if end_year is None:
        end_year = int(df["_date_key"].max().year)  # type: ignore[union-attr]

    hol_df = _build_holiday_df(start_year, end_year)

    df = df.join(hol_df, left_on="_date_key", right_on="date", how="left")

    df = df.with_columns(
        pl.col("holiday_name").fill_null("non-event"),
    ).with_columns(
        pl.col("holiday_name")
        .map_elements(_classify_holiday, return_dtype=pl.Utf8)
        .alias("holiday_type"),
    ).drop("_date_key")

    return df


def filter_major_holidays(df: pl.DataFrame) -> pl.DataFrame:
    """추석/설날만 필터링."""
    return df.filter(pl.col("holiday_type").is_in(["Chuseok", "Seollal"]))
