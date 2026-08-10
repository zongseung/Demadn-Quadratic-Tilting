"""Immutable configuration contract for the HQRC v3 experiment."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the v3 contract."""


@dataclass(frozen=True)
class DataConfig:
    """Data-handling options which must be fixed for comparable runs."""

    gap_days: int

    def __post_init__(self) -> None:
        if self.gap_days != 0:
            raise ConfigError("HQRC v3 requires gap_days to be 0")


@dataclass(frozen=True)
class SplitConfig:
    """The fixed expanding-window evaluation horizon."""

    first_train_year: int
    oof_years: tuple[int, ...]
    final_year: int

    def __post_init__(self) -> None:
        if self.oof_years != (2020, 2021, 2022, 2023) or self.final_year != 2024:
            raise ConfigError("HQRC v3 requires OOF 2020-2023 and final year 2024")
        if self.first_train_year >= self.oof_years[0]:
            raise ConfigError("first_train_year must precede the first OOF year")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level typed configuration for a reproducible experiment."""

    data: DataConfig
    split: SplitConfig


def _mapping(value: object, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ConfigError(f"missing or invalid [{name}] table")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{field} must be an integer")
    return value


def load_config(path: Path) -> ExperimentConfig:
    """Load a TOML experiment config and enforce the fixed v3 protocol."""

    try:
        with path.open("rb") as config_file:
            document = tomllib.load(config_file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"unable to load config {path}: {error}") from error

    data = _mapping(document.get("data"), "data")
    split = _mapping(document.get("split"), "split")
    oof_years = split.get("oof_years")
    if not isinstance(oof_years, list) or any(
        isinstance(year, bool) or not isinstance(year, int) for year in oof_years
    ):
        raise ConfigError("split.oof_years must be an array of integers")

    return ExperimentConfig(
        data=DataConfig(gap_days=_integer(data.get("gap_days"), "data.gap_days")),
        split=SplitConfig(
            first_train_year=_integer(split.get("first_train_year"), "split.first_train_year"),
            oof_years=tuple(oof_years),
            final_year=_integer(split.get("final_year"), "split.final_year"),
        ),
    )
