from datetime import datetime, timedelta

import numpy as np
import pytest
from hqrc_v3.corrections.variants import VariantContext, run_event_loeo, run_variant


@pytest.fixture
def tiny_variant_context():
    timestamps = [datetime(2024, 2, 9) + timedelta(hours=index) for index in range(4)]
    return VariantContext(
        timestamps=timestamps,
        baseline=np.full(4, 100.0),
        observed=np.array([90.0, 91.0, 92.0, 93.0]),
        sigma_n=10.0,
        holiday_type=0,
        tau_days=np.arange(4) / 24.0,
        hour=np.arange(4),
        posterior={
            "mu": np.zeros((1, 2, 3)),
            "phi": np.array([0.0]),
            "sigma_r": np.array([0.0]),
            "gamma": np.zeros((1, 2, 24)),
            "day_effect": np.zeros((1, 2, 1)),
        },
        non_event_blocks=np.array([[1.0, 1.0, 1.0, 1.0]]),
        training_profiles={"event-x": np.ones(4)},
        draws=3,
        seed=9,
    )


@pytest.mark.parametrize("variant", ["H0", "H1", "H2", "H3", "H4", "H5"])
def test_variant_returns_same_timestamps(variant, tiny_variant_context):
    result = run_variant(variant, tiny_variant_context)

    assert result["target_timestamp"].to_list() == tiny_variant_context.timestamps


class _RecordingBackend:
    def __init__(self):
        self.calls = []

    def fit_predict(self, held_out, fit_events, ar_events, *, causal):
        self.calls.append((held_out, tuple(fit_events), tuple(ar_events), causal))
        return {"target_timestamp": [datetime(2024, 1, 1)], "point_forecast_mw": [100.0]}


def test_loeo_excludes_heldout_event_from_fit_and_ar_calibration():
    backend = _RecordingBackend()
    results = run_event_loeo({"a": object(), "b": object(), "c": object()}, backend)

    assert results.height == 3
    for held_out, fit_events, ar_events, causal in backend.calls:
        assert held_out not in fit_events
        assert held_out not in ar_events
        assert set(fit_events) == set(ar_events)
        assert causal is False
