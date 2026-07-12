"""Configuration shared by the machine-learning and Seq2Seq baselines."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ForecastConfig:
    """Experiment definition for daily 24-hour load forecasts.

    Weather variables are observed only in the 168-hour encoder window.  The
    known covariates (the four season indicators by default) are also supplied
    for the 24-hour forecast horizon because they are known at forecast time.
    """

    datetime_col: str = "일시"
    target_col: str = "power demand(MW)"
    weather_cols: tuple[str, ...] = ("ta", "hm")
    known_covariate_cols: tuple[str, ...] = (
        "spring",
        "summer",
        "autoum",
        "winter",
    )
    metadata_cols: tuple[str, ...] = (
        "holiday_name",
        "weekday",
        "weekend",
        "spring",
        "summer",
        "autoum",
        "winter",
        "is_holiday_dummies",
    )
    history_hours: int = 168
    horizon_hours: int = 24
    stride_hours: int = 24
    forecast_origin_hour: int = 0
    train_end: str = "2022-12-31 23:00:00"
    validation_end: str = "2023-12-31 23:00:00"
    random_seed: int = 2025

    @property
    def past_feature_cols(self) -> tuple[str, ...]:
        return (
            self.target_col,
            *self.weather_cols,
            *self.known_covariate_cols,
        )

    def validate(self) -> None:
        if self.history_hours <= 0:
            raise ValueError("history_hours must be positive")
        if self.horizon_hours <= 0:
            raise ValueError("horizon_hours must be positive")
        if self.stride_hours <= 0:
            raise ValueError("stride_hours must be positive")
        if not 0 <= self.forecast_origin_hour <= 23:
            raise ValueError("forecast_origin_hour must be between 0 and 23")
        if not self.weather_cols:
            raise ValueError("at least one weather column is required")
        if not self.known_covariate_cols:
            raise ValueError("at least one known covariate is required")
        if len(set(self.past_feature_cols)) != len(self.past_feature_cols):
            raise ValueError("past feature columns must be unique")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
