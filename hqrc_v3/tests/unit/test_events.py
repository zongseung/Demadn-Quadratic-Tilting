from collections import Counter
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import pytest

from hqrc_v3.events import (
    EventOccurrence,
    EventRegistryError,
    load_event_registry,
    load_holiday_calendar,
    validate_feature_event_alignment,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EVENT_REGISTRY = PROJECT_ROOT / "configs/events.csv"
HOLIDAY_CALENDAR = PROJECT_ROOT / "configs/holiday_calendar.csv"


def test_registry_has_exactly_five_occurrences_per_type():
    events = load_event_registry(EVENT_REGISTRY)

    assert len(events) == 10
    assert Counter(event.holiday_type for event in events) == {
        "seollal": 5,
        "chuseok": 5,
    }
    assert {event.occurrence_id for event in events if event.restriction == 1} == {
        "chuseok-2020",
        "seollal-2021",
        "chuseok-2021",
        "seollal-2022",
    }


def test_feature_calendar_includes_only_declared_distance_support_outside_event_years():
    events = load_event_registry(EVENT_REGISTRY)
    calendar = load_holiday_calendar(HOLIDAY_CALENDAR)

    assert min(event.central_date.year for event in events) == 2020
    assert min(event.central_date.year for event in calendar) == 2018
    assert max(event.central_date.year for event in calendar) == 2025
    assert len(events) == 10
    assert len(calendar) == 14
    assert {event.occurrence_id for event in calendar} - {
        event.occurrence_id for event in events
    } == {"chuseok-2018", "seollal-2019", "chuseok-2019", "seollal-2025"}
    assert {event.occurrence_id for event in calendar if event.restriction == 1} == {
        "chuseok-2020",
        "seollal-2021",
        "chuseok-2021",
        "seollal-2022",
    }


@pytest.mark.parametrize("restriction", (False, True, 0.0, 1.0))
def test_event_occurrence_rejects_non_integer_restriction(restriction: object) -> None:
    with pytest.raises(EventRegistryError, match="restriction must be either 0 or 1"):
        EventOccurrence(
            occurrence_id="seollal-2024",
            holiday_type="seollal",
            central_date=date(2024, 2, 10),
            official_start=date(2024, 2, 9),
            official_end=date(2024, 2, 12),
            restriction=restriction,  # type: ignore[arg-type]
        )


def test_feature_and_correction_registries_must_align() -> None:
    calendar = load_holiday_calendar(HOLIDAY_CALENDAR)
    events = load_event_registry(EVENT_REGISTRY)
    validate_feature_event_alignment(calendar, events)
    changed = replace(events[0], official_end=events[0].official_end + timedelta(days=1))

    with pytest.raises(EventRegistryError, match="feature/correction window"):
        validate_feature_event_alignment(calendar, (changed, *events[1:]))


@pytest.mark.parametrize(
    ("source_name", "loader"),
    (("events.csv", load_event_registry), ("holiday_calendar.csv", load_holiday_calendar)),
)
def test_registry_rejects_wrong_but_binary_restriction_mapping(tmp_path, source_name, loader):
    source = PROJECT_ROOT / "configs" / source_name
    mutated_registry = tmp_path / source_name
    mutated_registry.write_text(
        source.read_text(encoding="utf-8")
        .replace(
            "seollal-2020,seollal,2020-01-25,2020-01-24,2020-01-27,0",
            "seollal-2020,seollal,2020-01-25,2020-01-24,2020-01-27,1",
        )
        .replace(
            "chuseok-2020,chuseok,2020-10-01,2020-09-30,2020-10-02,1",
            "chuseok-2020,chuseok,2020-10-01,2020-09-30,2020-10-02,0",
        ),
        encoding="utf-8",
    )

    with pytest.raises(EventRegistryError, match="restriction mapping"):
        loader(mutated_registry)
