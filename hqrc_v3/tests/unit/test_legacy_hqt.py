from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

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
from hqrc_v3.diagnostics.loeo import LOEOFold, LOEOHeldOut, LOEOPublication
from hqrc_v3.legacy_hqt_loeo import (
    LegacyHQTError,
    _select_contexts,
    build_legacy_hqt_data,
    build_legacy_hqt_data_from_frame,
    cosine_boundary_taper,
    event_type_mean_shift_mw,
    run_legacy_hqt_loeo,
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


def test_legacy_hqt_paper_profile_allows_more_than_four_chains(monkeypatch) -> None:
    captured = {}

    def fake_sample(**kwargs):
        captured.update(kwargs)
        return az.from_dict(
            posterior={"event_log_likelihood": np.zeros((5, 6, 4))},
            sample_stats={"diverging": np.zeros((5, 6), dtype=np.int8)},
        )

    monkeypatch.setattr(pm, "sample", fake_sample)
    monkeypatch.setattr(
        "hqrc_v3.bayes.legacy_hqt.validate_inference_data",
        lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 500.0, 500.0, 0),
    )

    sample_legacy_hqt(
        _tiny_data(),
        draws=1_000,
        tune=1_000,
        chains=5,
        cores=1,
        seed=31,
        target_accept=0.99,
        paper_profile=True,
    )

    assert captured["chains"] == 5


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


def test_cli_defaults_reviewer_hqt_to_b1w() -> None:
    arguments = build_parser().parse_args(
        [
            "run-hqt-loeo",
            "--source-run-dir",
            "source",
            "--config",
            "experiment.toml",
            "--output-root",
            "results",
        ]
    )

    assert arguments.feature_set == "B1W"


def test_legacy_hqt_selects_b1w_as_a_distinct_context() -> None:
    split_ids = tuple(f"oof-{year}" for year in range(2020, 2024))
    b1 = EventResidualContext("xgboost", "B1", 7, split_ids)
    b1w = EventResidualContext("xgboost", "B1W", 7, split_ids)
    source = SimpleNamespace(available_contexts=(b1, b1w))

    selected = _select_contexts(
        source,  # type: ignore[arg-type]
        models=("xgboost",),
        feature_sets=("B1W",),
    )

    assert selected == (b1w,)


def test_legacy_hqt_loeo_writes_products_then_reuses_physical_fold(
    tmp_path: Path, monkeypatch
) -> None:
    context = EventResidualContext(
        "xgboost", "B1", 7, ("oof-2020", "oof-2021", "oof-2022", "oof-2023")
    )
    fold = _physical_fold(tmp_path)
    start = datetime(2024, 9, 14)
    tau = np.arange(72, dtype=float) / 24.0
    held_out = LOEOHeldOut(
        occurrence_id="chuseok-2024",
        path=tmp_path / "held-out.parquet",
        universe_sha256="universe-sha",
        frame=pl.DataFrame(
            {
                "causal": [False] * tau.size,
                "holiday_type": ["chuseok"] * tau.size,
                "occurrence_id": ["chuseok-2024"] * tau.size,
                "observed_mw": 100.0 + np.arange(tau.size),
                "predicted_mw": 99.0 + np.arange(tau.size),
                "sigma_n_mw": [100.0] * tau.size,
                "target_timestamp": [start + timedelta(hours=hour) for hour in range(tau.size)],
                "tau_days": tau,
            }
        ),
        context=context,
        causal=False,
    )
    publication = LOEOPublication(
        output_dir=tmp_path / "publication",
        generation_dir=tmp_path / "publication" / "generation",
        manifest_path=tmp_path / "publication" / "manifest.json",
        universe_path=tmp_path / "publication" / "universe.parquet",
        universe_sha256="universe-sha",
        occurrence_ids=("chuseok-2024",),
        fold_paths={"chuseok-2024": tmp_path / "publication" / "fold.parquet"},
        fold_sha256={"chuseok-2024": "fold-sha"},
        context=context,
        causal=False,
    )
    baseline_manifest = tmp_path / "baseline_manifest.json"
    baseline_manifest.write_text("{}")
    source = SimpleNamespace(
        available_contexts=(context,),
        baseline_manifest_path=baseline_manifest,
        residual_sha256="residual-sha",
    )
    sampled: list[HQRCData] = []

    def fake_sample(data: HQRCData, **_kwargs) -> az.InferenceData:
        sampled.append(data)
        idata = az.from_dict(
            posterior={
                "mu": np.zeros((1, 2, 2, 3)),
                "between_cholesky": np.broadcast_to(np.eye(3), (1, 2, 2, 3, 3)),
            }
        )
        idata.attrs["legacy_hqt_model_json"] = json.dumps(LEGACY_HQT_MODEL_SPEC, sort_keys=True)
        return idata

    monkeypatch.setattr("hqrc_v3.legacy_hqt_loeo.validate_correction_source", lambda **_kw: source)
    monkeypatch.setattr(
        "hqrc_v3.legacy_hqt_loeo.publish_loeo_universe", lambda *_args, **_kw: publication
    )
    monkeypatch.setattr(
        "hqrc_v3.legacy_hqt_loeo.load_loeo_universe", lambda *_args, **_kw: publication
    )
    monkeypatch.setattr("hqrc_v3.legacy_hqt_loeo.load_loeo_fold", lambda *_args, **_kw: fold)
    monkeypatch.setattr("hqrc_v3.legacy_hqt_loeo.load_loeo_event", lambda *_args, **_kw: held_out)
    monkeypatch.setattr("hqrc_v3.legacy_hqt_loeo.sample_legacy_hqt", fake_sample)
    monkeypatch.setattr("hqrc_v3.legacy_hqt_loeo.validate_inference_data", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        "hqrc_v3.legacy_hqt_loeo._scale_stability",
        lambda *_args: pl.DataFrame({"scale_ratio": [1.0]}),
    )

    first = run_legacy_hqt_loeo(
        source_run_dir=tmp_path / "source",
        config_path=tmp_path / "experiment.toml",
        output_root=tmp_path / "output",
        models=("xgboost",),
        feature_sets=("B1",),
        held_out_occurrence_ids=None,
        root_seed=17,
        profile="smoke",
        draws=2,
        tune=2,
        chains=1,
        cores=1,
    )
    fold_dir = first[0].output_dir / "folds" / "chuseok-2024"

    assert len(sampled) == 1
    assert len(sampled[0].occurrence_ids) == 9
    assert "chuseok-2024" not in sampled[0].occurrence_ids
    assert first[0].sampler_fit_count == 1
    assert first[0].reused_fold_count == 0
    assert {path.name for path in fold_dir.iterdir()} == {
        "COMPLETE",
        "hourly_predictions.parquet",
        "input_identity.json",
        "manifest.json",
        "metrics.parquet",
        "posterior.nc",
    }
    assert pl.read_parquet(fold_dir / "metrics.parquet")["correction"].to_list() == [
        "H0",
        "H1",
        "H2",
        "H2-taper",
    ]
    assert (first[0].output_dir / "event_metrics.parquet").is_file()
    assert (first[0].output_dir / "pooled_metrics.parquet").is_file()
    assert (first[0].output_dir / "event_improvements.parquet").is_file()
    assert (first[0].output_dir / "improvement_summary.parquet").is_file()
    assert (first[0].output_dir / "scale_stability.parquet").is_file()

    second = run_legacy_hqt_loeo(
        source_run_dir=tmp_path / "source",
        config_path=tmp_path / "experiment.toml",
        output_root=tmp_path / "output",
        models=("xgboost",),
        feature_sets=("B1",),
        held_out_occurrence_ids=None,
        root_seed=17,
        profile="smoke",
        draws=2,
        tune=2,
        chains=1,
        cores=1,
    )

    assert len(sampled) == 1
    assert second[0].sampler_fit_count == 0
    assert second[0].reused_fold_count == 1
