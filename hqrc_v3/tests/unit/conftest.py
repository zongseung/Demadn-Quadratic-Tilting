from __future__ import annotations

import importlib.util
from pathlib import Path

_FIXTURE_PATH = Path(__file__).parents[1] / "fixtures" / "hourly.py"
_SPEC = importlib.util.spec_from_file_location("hourly_fixtures", _FIXTURE_PATH)
if _SPEC is None or _SPEC.loader is None:  # pragma: no cover - repository corruption guard
    raise RuntimeError(f"unable to load hourly fixtures from {_FIXTURE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
hourly_frame = _MODULE.hourly_frame
holiday_calendar = _MODULE.holiday_calendar
