"""Validated holiday occurrence registries used by HQRC v3."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Literal, cast

HolidayType = Literal["seollal", "chuseok"]
_HOLIDAY_TYPES = frozenset(("seollal", "chuseok"))
_REQUIRED_COLUMNS = frozenset(
    (
        "occurrence_id",
        "holiday_type",
        "central_date",
        "official_start",
        "official_end",
        "restriction",
    )
)


class EventRegistryError(ValueError):
    """Raised when a registry cannot serve as an unambiguous event contract."""


@dataclass(frozen=True)
class EventOccurrence:
    occurrence_id: str
    holiday_type: HolidayType
    central_date: date
    official_start: date
    official_end: date
    restriction: int

    def __post_init__(self) -> None:
        if not self.occurrence_id:
            raise EventRegistryError("occurrence_id must not be blank")
        if self.holiday_type not in _HOLIDAY_TYPES:
            raise EventRegistryError(f"unknown holiday_type: {self.holiday_type}")
        if self.official_start > self.official_end:
            raise EventRegistryError("official_start must not be after official_end")
        if not self.official_start <= self.central_date <= self.official_end:
            raise EventRegistryError("central_date must fall within the official holiday period")
        if self.restriction not in (0, 1):
            raise EventRegistryError("restriction must be either 0 or 1")

    @property
    def window_start(self) -> date:
        return self.official_start - timedelta(days=1)

    @property
    def window_end(self) -> date:
        return self.official_end + timedelta(days=1)


def _parse_row(row: dict[str, str | None], line_number: int) -> EventOccurrence:
    try:
        holiday_type = row["holiday_type"]
        restriction = row["restriction"]
        if holiday_type is None or restriction is None:
            raise ValueError("missing value")
        return EventOccurrence(
            occurrence_id=(row["occurrence_id"] or "").strip(),
            holiday_type=cast(HolidayType, holiday_type.strip()),
            central_date=date.fromisoformat((row["central_date"] or "").strip()),
            official_start=date.fromisoformat((row["official_start"] or "").strip()),
            official_end=date.fromisoformat((row["official_end"] or "").strip()),
            restriction=int(restriction.strip()),
        )
    except (TypeError, ValueError) as error:
        raise EventRegistryError(f"invalid event row {line_number}: {error}") from error


def _validate_events(events: tuple[EventOccurrence, ...]) -> tuple[EventOccurrence, ...]:
    ids = [event.occurrence_id for event in events]
    if len(ids) != len(set(ids)):
        raise EventRegistryError("duplicate occurrence_id")

    central_dates = [event.central_date for event in events]
    if len(central_dates) != len(set(central_dates)):
        raise EventRegistryError("duplicate central_date")

    type_years = [(event.holiday_type, event.central_date.year) for event in events]
    if len(type_years) != len(set(type_years)):
        raise EventRegistryError("each holiday type/year must occur exactly once")

    ordered = sorted(events, key=lambda event: event.window_start)
    for previous, current in zip(ordered, ordered[1:]):
        if current.window_start <= previous.window_end:
            raise EventRegistryError(
                f"event windows overlap: {previous.occurrence_id} and {current.occurrence_id}"
            )
    return events


def _load_events(path: Path) -> tuple[EventOccurrence, ...]:
    try:
        with path.open(newline="", encoding="utf-8") as registry_file:
            reader = csv.DictReader(registry_file)
            if reader.fieldnames is None or set(reader.fieldnames) != _REQUIRED_COLUMNS:
                raise EventRegistryError("registry columns do not match the required schema")
            events = tuple(
                _parse_row(row, line_number) for line_number, row in enumerate(reader, start=2)
            )
    except OSError as error:
        raise EventRegistryError(f"unable to load event registry {path}: {error}") from error
    return _validate_events(events)


def load_event_registry(path: Path) -> tuple[EventOccurrence, ...]:
    """Load correction events; this registry is intentionally limited to 2020-2024."""

    events = _load_events(path)
    years = {event.central_date.year for event in events}
    if years != {2020, 2021, 2022, 2023, 2024} or len(events) != 10:
        raise EventRegistryError(
            "correction registry must contain one event per type for 2020-2024"
        )
    return events


def load_holiday_calendar(path: Path) -> tuple[EventOccurrence, ...]:
    """Load the feature calendar, including the required 2019 lookback year."""

    events = _load_events(path)
    years = {event.central_date.year for event in events}
    if years != {2019, 2020, 2021, 2022, 2023, 2024} or len(events) != 12:
        raise EventRegistryError("feature calendar must contain one event per type for 2019-2024")
    return events
