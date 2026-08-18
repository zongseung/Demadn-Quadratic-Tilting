"""Immutable expanding-origin and final-holdout split contracts."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

import numpy as np

from hqrc_v3.contracts import DataContractError, ForecastMatrix

_FIRST_TRAIN_YEAR = 2019
_OOF_YEARS = (2020, 2021, 2022, 2023)
_FINAL_YEAR = 2024


@dataclass(frozen=True)
class AnnualFold:
    """A zero-gap, target-complete annual expanding-origin partition."""

    split_id: str
    train_start: date
    train_end: date
    eval_start: date
    eval_end: date
    eval_year: int
    kind: Literal["oof", "final"]

    def __post_init__(self) -> None:
        expected_kind = "final" if self.eval_year == _FINAL_YEAR else "oof"
        if self.kind != expected_kind or self.eval_year not in (*_OOF_YEARS, _FINAL_YEAR):
            raise DataContractError("fold must be one of the immutable OOF or final years")
        if self.train_start != date(_FIRST_TRAIN_YEAR, 1, 1):
            raise DataContractError("HQRC v3 folds always start training in 2019")
        if self.train_end != date(self.eval_year - 1, 12, 31):
            raise DataContractError("fold train end must be the year before evaluation")
        expected_eval_start = date(self.eval_year, 1, 1)
        expected_eval_end = date(self.eval_year, 12, 31)
        if self.eval_start != expected_eval_start or self.eval_end != expected_eval_end:
            raise DataContractError("fold evaluation must cover its whole calendar year")
        if self.split_id != f"{self.kind}-{self.eval_year}":
            raise DataContractError("fold split_id must match its kind and evaluation year")

    @classmethod
    def oof(cls, *, eval_year: int, first_train_year: int) -> AnnualFold:
        """Return one of the four fixed pre-2024 OOF folds."""

        if first_train_year != _FIRST_TRAIN_YEAR or eval_year not in _OOF_YEARS:
            raise DataContractError("HQRC v3 OOF folds are 2019 through 2020-2023 only")
        return cls._create(eval_year=eval_year, kind="oof")

    @classmethod
    def final(cls) -> AnnualFold:
        """Return the fixed 2019--2023 train, 2024 evaluation fold."""

        return cls._create(eval_year=_FINAL_YEAR, kind="final")

    @classmethod
    def _create(cls, *, eval_year: int, kind: Literal["oof", "final"]) -> AnnualFold:
        return cls(
            split_id=f"{kind}-{eval_year}",
            train_start=date(_FIRST_TRAIN_YEAR, 1, 1),
            train_end=date(eval_year - 1, 12, 31),
            eval_start=date(eval_year, 1, 1),
            eval_end=date(eval_year, 12, 31),
            eval_year=eval_year,
            kind=kind,
        )


def expanding_oof_folds() -> tuple[AnnualFold, ...]:
    """Return 2019→2020, 2019--20→2021, 2019--21→2022, 2019--22→2023."""

    return tuple(
        AnnualFold.oof(eval_year=year, first_train_year=_FIRST_TRAIN_YEAR) for year in _OOF_YEARS
    )


def final_fold() -> AnnualFold:
    """Return the fixed 2019--2023→2024 final refit split."""

    return AnnualFold.final()


def fold_for_split_id(split_id: str) -> AnnualFold:
    """Resolve one of the five immutable prediction split identities."""

    folds = (*expanding_oof_folds(), final_fold())
    for fold in folds:
        if fold.split_id == split_id:
            return fold
    raise DataContractError(f"unknown immutable split_id: {split_id!r}")


def is_oof_split_id(split_id: str) -> bool:
    """Return whether a split identity is one of the four allowed OOF folds."""

    return split_id in {fold.split_id for fold in expanding_oof_folds()}


def select_fold_samples(matrix: ForecastMatrix, fold: AnnualFold) -> tuple[np.ndarray, np.ndarray]:
    """Select only samples whose complete 24-hour targets belong to a fold side."""

    target_times = matrix.target_times
    if target_times.ndim != 2 or target_times.shape[0] != matrix.origins.shape[0]:
        raise DataContractError("forecast matrix target_times must align with sample origins")
    if target_times.shape[1] != 24:
        raise DataContractError("forecast matrix targets must contain exactly 24 hours")
    if not np.issubdtype(target_times.dtype, np.datetime64) or np.isnat(target_times).any():
        raise DataContractError("forecast matrix target times must be complete datetimes")

    train_start = np.datetime64(fold.train_start)
    eval_start = np.datetime64(fold.eval_start)
    eval_end_exclusive = np.datetime64(fold.eval_end + timedelta(days=1))
    train_membership = ((target_times >= train_start) & (target_times < eval_start)).all(axis=1)
    eval_membership = ((target_times >= eval_start) & (target_times < eval_end_exclusive)).all(
        axis=1
    )
    return np.flatnonzero(train_membership), np.flatnonzero(eval_membership)
