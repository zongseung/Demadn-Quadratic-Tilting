from collections import Counter
from pathlib import Path

from hqrc_v3.events import load_event_registry, load_holiday_calendar

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_registry_has_exactly_five_occurrences_per_type():
    events = load_event_registry(PROJECT_ROOT / "configs/events.csv")

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


def test_feature_calendar_includes_2019_but_correction_registry_does_not():
    events = load_event_registry(PROJECT_ROOT / "configs/events.csv")
    calendar = load_holiday_calendar(PROJECT_ROOT / "configs/holiday_calendar.csv")

    assert min(event.central_date.year for event in events) == 2020
    assert min(event.central_date.year for event in calendar) == 2019
    assert len(events) == 10
    assert len(calendar) == 12
    assert {event.occurrence_id for event in calendar if event.restriction == 1} == {
        "chuseok-2020",
        "seollal-2021",
        "chuseok-2021",
        "seollal-2022",
    }
