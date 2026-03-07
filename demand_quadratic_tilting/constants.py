"""Korean holiday label constants for Chuseok and Seollal."""

from __future__ import annotations

CHUSEOK_CORE = {"Chuseok"}
CHUSEOK_ALIASES = {
    "The day preceding Chuseok",
    "The second day of Chuseok",
    "Alternative holiday for Chuseok",
}
SEOLLAL_CORE = {"Korean New Year"}
SEOLLAL_ALIASES = {
    "The day preceding Korean New Year",
    "The second day of Korean New Year",
    "Alternative holiday for Korean New Year",
}

CHUSEOK_LABELS = CHUSEOK_CORE | CHUSEOK_ALIASES
SEOLLAL_LABELS = SEOLLAL_CORE | SEOLLAL_ALIASES
