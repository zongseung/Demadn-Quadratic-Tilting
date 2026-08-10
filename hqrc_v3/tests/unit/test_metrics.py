import numpy as np
import polars as pl
from hqrc_v3.evaluation.metrics import (
    event_metric_frame,
    point_metric_frame,
    probabilistic_metric_frame,
)


def test_point_metrics_guard_zero_mape_and_constant_r2():
    result = point_metric_frame("event-a", np.array([0.0, 0.0]), np.array([1.0, -1.0]))

    assert result["mape"].item() is None
    assert result["r2"].item() is None
    assert result["rmse"].item() == 1.0


def test_probabilistic_metrics_are_empirical_and_have_requested_intervals():
    draws = np.array([[0.0, 0.0], [2.0, 2.0], [1.0, 1.0]])
    result = probabilistic_metric_frame("event-a", np.array([1.0, 1.0]), draws)

    assert result["crps"].item() == 2.0 / 9.0
    assert set(result.columns) >= {"pinball_05", "pinball_95", "coverage_50", "coverage_90"}
    assert (
        len(
            [
                column
                for column in result.columns
                if column.startswith("pinball_") and column != "pinball_mean"
            ]
        )
        == 19
    )


def test_sorted_empirical_crps_matches_bruteforce_without_quadratic_tensor():
    rng = np.random.default_rng(4)
    draws = rng.normal(size=(500, 3))
    observed = np.array([0.1, -0.2, 0.3])
    from hqrc_v3.evaluation.metrics import empirical_crps

    expected = np.abs(draws - observed).mean(axis=0) - 0.5 * np.abs(
        draws[:, None] - draws[None, :]
    ).mean(axis=(0, 1))
    np.testing.assert_allclose(empirical_crps(observed, draws), expected)


def test_event_metric_frame_preserves_each_exact_timestamp():
    timestamps = pl.Series(
        "target_timestamp",
        [
            __import__("datetime").datetime(2024, 2, 9),
            __import__("datetime").datetime(2024, 2, 9, 1),
        ],
    )
    result = event_metric_frame(
        "event-a",
        timestamps,
        np.array([100.0, 102.0]),
        np.array([101.0, 101.0]),
        np.array([[100.0, 101.0], [102.0, 103.0]]),
    )

    assert result["target_timestamp"].to_list() == timestamps.to_list()
