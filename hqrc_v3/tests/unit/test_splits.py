from __future__ import annotations

from datetime import datetime

import numpy as np
from hqrc_v3.contracts import ForecastMatrix
from hqrc_v3.splits import AnnualFold, expanding_oof_folds, final_fold, select_fold_samples


def _matrix_for_split_tests() -> ForecastMatrix:
    origins = np.array(
        [
            np.datetime64("2022-12-31T00:00"),
            np.datetime64("2022-12-31T12:00"),
            np.datetime64("2023-01-01T00:00"),
        ],
        dtype="datetime64[ns]",
    )
    target_times = np.stack([origin + np.arange(24).astype("timedelta64[h]") for origin in origins])
    values = np.zeros((len(origins), 24, 1), dtype=float)
    return ForecastMatrix(
        origins=origins,
        target_times=target_times,
        history=np.zeros((len(origins), 168, 1), dtype=float),
        future=values,
        target=values[:, :, 0],
        history_columns=("load_mw",),
        future_columns=("hour",),
    )


def test_expanding_folds_are_exact():
    assert [(fold.train_end.year, fold.eval_year) for fold in expanding_oof_folds()] == [
        (2019, 2020),
        (2020, 2021),
        (2021, 2022),
        (2022, 2023),
    ]
    assert (final_fold().train_end.year, final_fold().eval_year) == (2023, 2024)


def test_partition_requires_complete_target_membership():
    matrix = _matrix_for_split_tests()
    fold = AnnualFold.oof(eval_year=2023, first_train_year=2019)

    train, evaluation = select_fold_samples(matrix, fold)

    np.testing.assert_array_equal(train, np.array([0]))
    np.testing.assert_array_equal(evaluation, np.array([2]))
    assert matrix.target_times[train].max() < np.datetime64("2023-01-01")
    assert matrix.target_times[evaluation].min() >= np.datetime64("2023-01-01")


def test_fold_ranges_are_fixed_to_the_immutable_zero_gap_contract():
    fold = AnnualFold.oof(eval_year=2021, first_train_year=2019)

    assert fold.train_start == datetime(2019, 1, 1).date()
    assert fold.train_end == datetime(2020, 12, 31).date()
    assert fold.eval_start == datetime(2021, 1, 1).date()
    assert fold.eval_end == datetime(2021, 12, 31).date()
    assert fold.split_id == "oof-2021"
