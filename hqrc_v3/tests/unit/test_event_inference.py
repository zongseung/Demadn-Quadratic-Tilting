import numpy as np
from hqrc_v3.evaluation.inference import (
    bootstrap_event_median,
    hac_dm_test,
    holm_adjust,
    wilcoxon_event_test,
)


def test_event_bootstrap_resamples_whole_events():
    result = bootstrap_event_median(
        {"a": np.array([1.0, 1.0]), "b": np.array([9.0, 9.0])}, draws=100, seed=2
    )

    assert result.resampled_units == "event"
    assert result.seed == 2
    assert result.samples.shape == (100,)


def test_event_wilcoxon_and_holm_are_deterministic():
    result = wilcoxon_event_test(np.array([2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0]))
    adjusted = holm_adjust({"first": 0.02, "second": 0.03, "third": 0.8})

    assert result.n_events == 3
    assert adjusted == {"first": 0.06, "second": 0.06, "third": 0.8}


def test_hac_dm_reports_bandwidth_and_event_block_seed():
    result = hac_dm_test(
        np.array([1.0, 1.2, 0.9, 1.1]),
        np.array([0.8, 0.7, 0.8, 0.9]),
        event_ids=["a", "a", "b", "b"],
        bandwidth=1,
        event_block_seed=17,
    )

    assert result.bandwidth == 1
    assert result.event_block_seed == 17
