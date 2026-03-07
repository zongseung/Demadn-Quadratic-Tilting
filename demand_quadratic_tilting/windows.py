"""Holiday window / tau construction (Polars)."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import polars as pl

from .constants import CHUSEOK_CORE, CHUSEOK_LABELS, SEOLLAL_CORE, SEOLLAL_LABELS


def build_holiday_windows_and_tau(
    df: pl.DataFrame,
    holiday_name_col: str = "holiday_name",
    datetime_col: str = "datetime",
    center_hour: int = 0,
    tau_unit: str = "1h",
    pre_pad_days: int = 0,
    post_pad_days: int = 0,
) -> Tuple[
    Dict[str, List[datetime]],
    Dict[Tuple[str, datetime], int],
    Dict[datetime, str],
    Dict[datetime, str],
    float,
]:
    """라벨 기반 윈도우/τ 생성.

    Returns
    -------
    windows : dict[event_id, list[datetime]]
    tau_map : dict[(event_id, datetime), int]
    event_id_of_date : dict[datetime, event_id]
    type_of_date : dict[datetime, holiday_type]
    tau_unit_hours : float
    """
    tau_unit_td = timedelta(hours=_parse_timedelta_hours(tau_unit))
    tau_unit_hours = tau_unit_td.total_seconds() / 3600.0

    dt_series = df[datetime_col].to_list()
    name_series = df[holiday_name_col].cast(pl.Utf8).to_list()

    # 날짜-이름 매핑
    date_name_map: Dict[datetime, str] = {}
    for d, n in zip(dt_series, name_series):
        date_name_map[d] = n

    windows: Dict[str, List[datetime]] = {}
    tau_map: Dict[Tuple[str, datetime], int] = {}
    event_id_of_date: Dict[datetime, str] = {}
    type_of_date: Dict[datetime, str] = {}

    for d, name in zip(dt_series, name_series):
        if name in CHUSEOK_CORE:
            tname, labels = "Chuseok", CHUSEOK_LABELS
        elif name in SEOLLAL_CORE:
            tname, labels = "Seollal", SEOLLAL_LABELS
        else:
            continue

        y = d.year
        eid = f"{y}_{tname}"
        if eid in windows:
            continue

        d0 = d.replace(hour=center_hour, minute=0, second=0, microsecond=0)

        # 같은 해, 같은 라벨 그룹의 날짜 수집
        stamps = sorted({
            dd for dd, nn in date_name_map.items()
            if dd.year == y and nn in labels
        })
        if not stamps:
            stamps = [d0]

        if pre_pad_days > 0 or post_pad_days > 0:
            date_set = {s.replace(hour=0, minute=0, second=0, microsecond=0) for s in stamps}
            min_date = min(date_set)
            max_date = max(date_set)
            start_pad = min_date - timedelta(days=pre_pad_days)
            end_pad = max_date + timedelta(days=post_pad_days)
            pad_stamps = [
                dd for dd in dt_series
                if start_pad <= dd.replace(hour=0, minute=0, second=0, microsecond=0) <= end_pad
            ]
            stamps = sorted(set(stamps) | set(pad_stamps))

        windows[eid] = stamps

        for dd in stamps:
            delta = dd - d0
            tau_float = delta.total_seconds() / tau_unit_td.total_seconds()
            tau_int = int(round(tau_float))
            tau_map[(eid, dd)] = tau_int
            event_id_of_date[dd] = eid
            type_of_date[dd] = tname

    return windows, tau_map, event_id_of_date, type_of_date, tau_unit_hours


def _parse_timedelta_hours(s: str) -> float:
    """'1h' -> 1.0, '30m' -> 0.5 등 간단 파싱."""
    s = s.strip().lower()
    if s.endswith("h"):
        return float(s[:-1])
    if s.endswith("m"):
        return float(s[:-1]) / 60.0
    if s.endswith("d"):
        return float(s[:-1]) * 24.0
    return float(s)
