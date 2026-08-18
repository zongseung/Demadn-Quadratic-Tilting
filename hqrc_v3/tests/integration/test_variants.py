import copy
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hqrc_v3.corrections.similar_day import SimilarDayError, same_holiday_profile
from hqrc_v3.corrections.variants import (
    VariantContext,
    build_non_event_block_pool,
    run_event_loeo,
    run_variant,
)
from hqrc_v3.diagnostics.ar import (
    approve_calibration,
    calibrate_beta_prior,
    load_approved_calibration,
    write_ar_diagnostics,
)


def _timestamps(day=datetime(2024, 2, 9)):
    return [day + timedelta(hours=index) for index in range(24)]


@pytest.fixture
def tiny_variant_context():
    timestamps = _timestamps()
    return VariantContext(
        timestamps=timestamps,
        baseline=np.full(24, 100.0),
        observed=np.arange(90.0, 114.0),
        sigma_n=10.0,
        holiday_type=0,
        tau_days=np.arange(24) / 24.0,
        hour=np.arange(24),
        posterior={
            "mu": np.zeros((1, 2, 3)),
            "between_scale": np.zeros((1, 2, 3)),
            "phi": np.array([0.0]),
            "sigma_r": np.array([0.0]),
            "gamma": np.zeros((1, 2, 24)),
            "day_effect": np.zeros((1, 2, 1)),
        },
        non_event_pool=build_non_event_block_pool(
            pl.DataFrame(
                {
                    "origin": [timestamps[0]] * 24,
                    "target_timestamp": timestamps,
                    "horizon": np.arange(1, 25),
                    "residual_mw": np.ones(24),
                    "is_event": [False] * 24,
                }
            )
        ),
        similar_day_training=pl.DataFrame(
            {
                "occurrence_id": ["seollal-2023"] * 24,
                "holiday_type": [0] * 24,
                "relative_day": [0] * 24,
                "hour": np.arange(24),
                "standardized_residual": np.arange(1.0, 25.0),
            }
        ),
        held_out_occurrence_id="seollal-2024",
        day_positions=np.array([0]),
        draws=3,
        seed=9,
    )


@pytest.mark.parametrize("variant", ["H0", "H1", "H2", "H3", "H4", "H5"])
def test_variant_returns_same_timestamps(variant, tiny_variant_context):
    result = run_variant(variant, tiny_variant_context)
    assert result["target_timestamp"].to_list() == tiny_variant_context.timestamps


def test_h0_uses_only_complete_non_event_block_pool(tiny_variant_context):
    frame = pl.DataFrame(
        {
            "origin": _timestamps()[:1] * 24,
            "target_timestamp": _timestamps(),
            "horizon": np.arange(1, 25),
            "residual_mw": np.arange(24.0),
            "is_event": [False] * 24,
        }
    )
    pool = build_non_event_block_pool(frame)
    result = run_variant(
        "H0",
        tiny_variant_context.__class__(**{**tiny_variant_context.__dict__, "non_event_pool": pool}),
    )
    assert result["target_timestamp"].to_list() == _timestamps()


def test_h0_pool_rejects_clone_and_public_tamper_attempts(tiny_variant_context):
    pool = tiny_variant_context.non_event_pool
    assert not hasattr(pool, "digest") and not hasattr(pool, "shape")
    with pytest.raises(AttributeError):
        pool.digest = "forged"
    with pytest.raises(AttributeError):
        pool._NonEventBlockPool__blocks = np.zeros((1, 24))
    with pytest.raises(Exception, match="authentic"):
        run_variant("H0", replace(tiny_variant_context, non_event_pool=copy.copy(pool)))


def test_h4_requires_explicit_unshifted_training_day_positions(tiny_variant_context):
    with pytest.raises(ValueError, match="day_positions"):
        run_variant("H4", replace(tiny_variant_context, day_positions=None))
    with pytest.raises(ValueError, match="day_effect"):
        run_variant("H4", replace(tiny_variant_context, day_positions=np.array([1])))


def test_h5_rejects_heldout_occurrence_in_training(tiny_variant_context):
    training = tiny_variant_context.similar_day_training.with_columns(
        pl.lit("seollal-2024").alias("occurrence_id")
    )
    target = pl.DataFrame(
        {"holiday_type": [0] * 24, "relative_day": [0] * 24, "hour": np.arange(24)}
    )
    with pytest.raises(SimilarDayError, match="exclude"):
        same_holiday_profile(training, target, held_out_occurrence_id="seollal-2024")


def test_h5_rejects_numerically_integral_float_keys(tiny_variant_context):
    target = pl.DataFrame(
        {"holiday_type": [0] * 24, "relative_day": [0.0] * 24, "hour": np.arange(24, dtype=float)}
    )
    with pytest.raises(SimilarDayError, match="integer dtypes"):
        same_holiday_profile(
            tiny_variant_context.similar_day_training, target, held_out_occurrence_id="seollal-2024"
        )


@pytest.fixture
def approved_calibration(tmp_path):
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibrate_beta_prior([0.4, 0.5, 0.6], event_ids=("a", "b", "c")),
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="events",
        context={"model": "baseline", "feature_set": "B1", "seed": 1, "split_ids": ("oof-2020",)},
    )
    approved = approve_calibration(
        proposal,
        tmp_path / "approved.json",
        current_residual_sha256="residual",
        current_config_sha256="config",
        current_event_sha256="events",
    )
    return load_approved_calibration(
        approved,
        current_residual_sha256="residual",
        current_config_sha256="config",
        current_event_sha256="events",
    )


class _RecordingBackend:
    def __init__(self, tmp_path):
        self.tmp_path = tmp_path
        self.calls = []

    def load_approved_fold_calibration(self, held_out, ar_events):
        self.calls.append([held_out, (), tuple(ar_events), None])
        proposal = write_ar_diagnostics(
            self.tmp_path / f"{held_out}-proposal.json",
            (),
            calibrate_beta_prior(np.linspace(0.2, 0.6, len(ar_events)), event_ids=tuple(ar_events)),
            residual_sha256="residual",
            config_sha256="config",
            event_sha256="events",
            context={
                "model": "baseline",
                "feature_set": "B1",
                "seed": 1,
                "split_ids": ("oof-2020",),
            },
        )
        approved = approve_calibration(
            proposal,
            self.tmp_path / f"{held_out}-approved.json",
            current_residual_sha256="residual",
            current_config_sha256="config",
            current_event_sha256="events",
        )
        return load_approved_calibration(
            approved,
            current_residual_sha256="residual",
            current_config_sha256="config",
            current_event_sha256="events",
        )

    def fit_correction(self, fit_events, calibration):
        self.calls[-1][1] = tuple(fit_events)
        return calibration

    def predict_heldout(self, fitted, held_out, frame, *, causal):
        self.calls[-1][3] = causal
        return frame.select("target_timestamp").with_columns(
            pl.lit(100.0).alias("point_forecast_mw")
        )


def test_loeo_excludes_heldout_event_from_fit_and_ar_calibration(tmp_path):
    ids = [f"{holiday}-{year}" for holiday in ("seollal", "chuseok") for year in range(2020, 2025)]
    frames = {
        event_id: pl.DataFrame({"target_timestamp": [datetime(2024, 1, 1)]}) for event_id in ids
    }
    backend = _RecordingBackend(tmp_path)
    results = run_event_loeo(frames, backend)
    assert results.height == 10
    for held_out, fit_events, ar_events, causal in backend.calls:
        assert held_out not in fit_events
        assert held_out not in ar_events
        assert set(fit_events) == set(ar_events)
        assert causal is False


def test_loeo_rejects_valid_but_wrong_event_id_calibration(tmp_path, approved_calibration):
    ids = [f"{holiday}-{year}" for holiday in ("seollal", "chuseok") for year in range(2020, 2025)]
    frames = {
        event_id: pl.DataFrame({"target_timestamp": [datetime(2024, 1, 1)]}) for event_id in ids
    }

    class WrongArtifactBackend(_RecordingBackend):
        def load_approved_fold_calibration(self, held_out, ar_events):
            self.calls.append([held_out, (), tuple(ar_events), None])
            return approved_calibration

    with pytest.raises(ValueError, match="event ids"):
        run_event_loeo(frames, WrongArtifactBackend(tmp_path))
