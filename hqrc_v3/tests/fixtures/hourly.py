"""Small, complete hourly inputs for forecast-sample contracts."""

from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from hqrc_v3.events import EventOccurrence


@pytest.fixture
def hourly_frame() -> pl.DataFrame:
    """Ten consecutive days leave three valid daily forecast origins."""

    start = datetime(2023, 1, 1)
    timestamps = [start + timedelta(hours=offset) for offset in range(10 * 24)]
    return pl.DataFrame(
        {
            "timestamp": timestamps,
            "load_mw": [100.0 + (offset % 24) for offset in range(len(timestamps))],
            "temperature_c": [10.0 + (offset % 8) for offset in range(len(timestamps))],
            "relative_humidity": [45.0 + (offset % 10) for offset in range(len(timestamps))],
        }
    )


@pytest.fixture
def holiday_calendar() -> tuple[EventOccurrence, ...]:
    return (
        EventOccurrence(
            occurrence_id="seollal-2023",
            holiday_type="seollal",
            central_date=datetime(2023, 1, 9).date(),
            official_start=datetime(2023, 1, 8).date(),
            official_end=datetime(2023, 1, 10).date(),
            restriction=0,
        ),
        EventOccurrence(
            occurrence_id="chuseok-2023",
            holiday_type="chuseok",
            central_date=datetime(2023, 9, 29).date(),
            official_start=datetime(2023, 9, 28).date(),
            official_end=datetime(2023, 9, 30).date(),
            restriction=0,
        ),
    )
