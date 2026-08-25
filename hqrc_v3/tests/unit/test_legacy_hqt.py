from __future__ import annotations

import json
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pymc as pm
import pytest

from hqrc_v3.bayes.legacy_hqt import (
    LEGACY_HQT_MODEL_SPEC,
    build_legacy_hqt_model,
    new_event_correction_draws,
    sample_legacy_hqt,
)
from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.bayes.samplers import SamplingDiagnostics
from hqrc_v3.cli import build_parser
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.diagnostics.loeo import LOEOFold
from hqrc_v3.legacy_hqt_loeo import (
    LegacyHQTError,
    build_legacy_hqt_data,
    build_legacy_hqt_data_from_frame,
    cosine_boundary_taper,
    event_type_mean_shift_mw,
)


def _tiny_data() -> HQRCData:
    return HQRCData(
        observations=np.linspace(-0.3, 0.4, 16),
        occurrence_index=np.repeat(np.arange(4), 4),
        holiday_type_index=np.repeat(np.array([0, 1, 0, 1]), 4),
        tau_days=np.tile(np.arange(4, dtype=float) / 24.0, 4),
        hour=np.tile(np.arange(4), 4),
        restriction=np.zeros(16, dtype=np.int64),
        occurrence_ids=("seollal-2020", "chuseok-2020", "seollal-2021", "chuseok-2021"),
    )


def _physical_fold(tmp_path: Path) -> LOEOFold:
    event_ids = tuple(
        [f"seollal-{year}" for year in range(2020, 2025)]
        + [f"chuseok-{year}" for year in range(2020, 2024)]
    )
    rows = []
    for event_index, event_id in enumerate(event_ids):
        holiday = event_id.rsplit("-", 1)[0]
        for hour in range(2):
            residual = 10.0 + event_index if holiday == "seollal" else -20.0 - event_index
            rows.append(
                {
                    "causal": False,
                    "feature_set": "B1",
                    "holiday_type": holiday,
                    "hour": hour,
                    "model": "xgboost",
                    "occurrence_id": event_id,
                    "residual_mw": residual,
                    "restriction": 0,
                    "seed": 7,
                    "standardized_residual": residual / 100.0,
                    "tau_days": hour / 24.0,
                }
            )
    return LOEOFold(
        held_out_occurrence_id="chuseok-2024",
        occurrence_ids=event_ids,
        path=tmp_path / "fold.parquet",
        residual_sha256="fold-sha",
        frame=pl.DataFrame(rows),
        context=EventResidualContext(
            "xgboost", "B1", 7, ("oof-2020", "oof-2021", "oof-2022", "oof-2023")
        ),
        causal=False,
    )


def _legacy_frame(*, event_years: range, feature_set: str) -> pl.DataFrame:
    rows = []
    for year in event_years:
        for holiday_type in ("seollal", "chuseok"):
            occurrence_id = f"{holiday_type}-{year}"
            for hour in range(2):
                rows.append(
                    {
                        "feature_set": feature_set,
                        "holiday_type": holiday_type,
                        "hour": hour,
                        "occurrence_id": occurrence_id,
                        "restriction": 0,
                        "standardized_residual": float(hour),
                        "tau_days": hour / 24.0,
                    }
                )
    return pl.DataFrame(rows)


def test_legacy_hqt_graph_is_exact_iid_quadratic_partial_pooling() -> None:
    model = build_legacy_hqt_model(_tiny_data())
    names = set(model.named_vars)

    assert {"mu", "beta_offset", "between_cholesky", "between_scale", "sigma_r"} <= names
    assert {"phi", "u_phi", "delta", "gamma", "nu", "ar2_phi"}.isdisjoint(names)
    assert model.named_vars["event_log_likelihood"].type.shape == (4,)
    assert LEGACY_HQT_MODEL_SPEC["innovation"] == "iid-normal"


def test_legacy_hqt_sampler_never_accepts_or_serializes_ar_calibration(monkeypatch) -> None:
    captured = {}

    def fake_sample(**kwargs):
        captured.update(kwargs)
        rng = np.random.default_rng(9)
        return az.from_dict(
            posterior={"event_log_likelihood": rng.normal(size=(2, 5, 4))},
            sample_stats={"diverging": np.zeros((2, 5), dtype=np.int8)},
        )

    monkeypatch.setattr(pm, "sample", fake_sample)
    monkeypatch.setattr(
        "hqrc_v3.bayes.legacy_hqt.validate_inference_data",
        lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 500.0, 500.0, 0),
    )
    idata = sample_legacy_hqt(
        _tiny_data(), draws=5, tune=5, chains=2, cores=2, seed=31, target_accept=0.9
    )

    assert captured["cores"] == 2
    assert idata.attrs["legacy_hqt_ar"] == "none"
    assert "calibration" not in idata.attrs
    assert json.loads(idata.attrs["legacy_hqt_model_json"])["hour_profile"] is False


def test_new_event_draws_are_seeded_and_follow_quadratic_design() -> None:
    mu = np.zeros((1, 3, 2, 3))
    mu[:, :, 0, :] = np.array([1.0, 2.0, 3.0])
    cholesky = np.zeros((1, 3, 2, 3, 3))
    idata = az.from_dict(posterior={"mu": mu, "between_cholesky": cholesky})
    tau = np.array([-1.0, 0.0, 2.0])

    left = new_event_correction_draws(idata, holiday_type_index=0, tau_days=tau, seed=11)
    right = new_event_correction_draws(idata, holiday_type_index=0, tau_days=tau, seed=11)

    np.testing.assert_array_equal(left, right)
    np.testing.assert_allclose(left, np.broadcast_to([2.0, 1.0, 17.0], left.shape))


def test_physical_fold_excludes_held_out_and_h1_is_same_type_raw_mean(tmp_path: Path) -> None:
    fold = _physical_fold(tmp_path)
    data = build_legacy_hqt_data(fold)

    assert "chuseok-2024" not in data.occurrence_ids
    assert data.occurrence_ids == fold.occurrence_ids
    expected = float(fold.frame.filter(pl.col("holiday_type") == "chuseok")["residual_mw"].mean())
    assert event_type_mean_shift_mw(fold, "chuseok") == expected


def test_generic_legacy_builder_accepts_eight_causal_training_events() -> None:
    frame = _legacy_frame(event_years=range(2020, 2024), feature_set="B1W")
    ids = tuple(frame["occurrence_id"].unique(maintain_order=True))

    data = build_legacy_hqt_data_from_frame(frame, occurrence_ids=ids)

    assert len(data.occurrence_ids) == 8
    assert tuple(np.unique(data.holiday_type_index)) == (0, 1)
    assert data.observations.size == frame.height


def test_generic_legacy_builder_rejects_unordered_occurrence_rows() -> None:
    frame = _legacy_frame(event_years=range(2020, 2024), feature_set="B1W").sort(
        "occurrence_id", descending=True
    )
    ids = tuple(sorted(frame["occurrence_id"].unique().to_list()))

    with pytest.raises(LegacyHQTError, match="occurrence ordering"):
        build_legacy_hqt_data_from_frame(frame, occurrence_ids=ids)


def test_cosine_taper_is_zero_at_boundaries_and_one_in_official_center() -> None:
    weight = cosine_boundary_taper(120, edge_hours=24)

    assert weight[0] == 0.0
    assert weight[-1] == 0.0
    assert weight[24] == weight[-25] == 1.0
    assert np.all(np.diff(weight[:24]) > 0)
    assert np.all(np.diff(weight[-24:]) < 0)


def test_cli_exposes_ar_free_hqt_scope_without_ar_approval() -> None:
    arguments = build_parser().parse_args(
        [
            "run-hqt-loeo",
            "--source-run-dir",
            "source",
            "--config",
            "experiment.toml",
            "--output-root",
            "results",
            "--model",
            "xgboost",
            "--feature-set",
            "B1",
            "--held-out",
            "seollal-2020",
        ]
    )

    assert arguments.command == "run-hqt-loeo"
    assert arguments.model == "xgboost"
    assert arguments.feature_set == "B1"
    assert arguments.held_out == ["seollal-2020"]
    assert not hasattr(arguments, "approved_ar")
    assert not hasattr(arguments, "approve_derived_ar")
