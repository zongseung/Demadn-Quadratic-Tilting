from __future__ import annotations

import hashlib
import json
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import polars as pl
import pytest

import hqrc_v3.correction_stage as stage
from hqrc_v3.bayes.samplers import PYMC_INITIALIZATION, SAMPLER_GEOMETRY, SamplingError
from hqrc_v3.correction_stage import (
    CausalCorrectionError,
    CausalCorrectionInputs,
    build_causal_hqrc_data,
    build_causal_prediction_data,
    fit_causal_2024_correction,
    generate_causal_products,
    select_latest_oof_scale,
)
from hqrc_v3.diagnostics.ar import (
    EventResidualContext,
    approve_calibration,
    calibrate_beta_prior,
    load_approved_calibration,
    write_ar_diagnostics,
)
from hqrc_v3.events import load_event_registry

EVENTS = Path(__file__).parents[2] / "configs/events.csv"
SPLITS = tuple(f"oof-{year}" for year in range(2020, 2024))
GEOMETRY_NAMESPACE = f"init-{PYMC_INITIALIZATION}-geometry-{SAMPLER_GEOMETRY}"


def _symlink_or_skip(path: Path, target: Path) -> None:
    try:
        path.symlink_to(target)
    except OSError as error:
        if getattr(error, "winerror", None) == 1314:
            pytest.skip("host cannot create the requested test symlink")
        raise


def _approved(tmp_path: Path, event_ids: tuple[str, ...]):
    proposal = write_ar_diagnostics(
        tmp_path / "proposal.json",
        (),
        calibrate_beta_prior(np.linspace(0.2, 0.55, len(event_ids)), event_ids=event_ids),
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="events",
        context=EventResidualContext("lightgbm", "B1", 7, SPLITS),
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


def _training_frame() -> pl.DataFrame:
    rows = []
    for event in load_event_registry(EVENTS):
        if event.central_date.year == 2024:
            continue
        start = datetime.combine(event.window_start, time.min)
        hours = (event.window_end - event.window_start).days * 24 + 24
        center = datetime.combine(event.central_date, time.min)
        for offset in range(hours):
            timestamp = start + timedelta(hours=offset)
            rows.append(
                {
                    "target_timestamp": timestamp,
                    "standardized_residual": float(offset + 1) / 100.0,
                    "model": "lightgbm",
                    "feature_set": "B1",
                    "seed": 7,
                    "split_id": f"oof-{event.central_date.year}",
                    "occurrence_id": event.occurrence_id,
                    "holiday_type": event.holiday_type,
                    "tau_days": (timestamp - center).total_seconds() / 86_400.0,
                    "hour": timestamp.hour,
                    "restriction": event.restriction,
                }
            )
    return pl.DataFrame(rows).sort("target_timestamp")


def _final_stream() -> pl.DataFrame:
    rows = []
    start = datetime(2024, 1, 1)
    for day_offset in range(305):
        origin = start + timedelta(days=day_offset)
        for horizon in range(1, 25):
            timestamp = origin + timedelta(hours=horizon - 1)
            rows.append(
                {
                    "origin": origin,
                    "target_timestamp": timestamp,
                    "horizon": horizon,
                    "observed_mw": 50_000.0 + horizon,
                    "predicted_mw": 49_000.0 + horizon,
                    "model": "lightgbm",
                    "feature_set": "B1",
                    "seed": 7,
                    "split_id": "final-2024",
                }
            )
    return pl.DataFrame(rows)


def test_training_adapter_binds_exact_approval_context_and_registered_events(tmp_path):
    events = load_event_registry(EVENTS)
    frame = _training_frame()
    event_ids = tuple(sorted(frame["occurrence_id"].unique().to_list()))
    approved = _approved(tmp_path, event_ids)

    data = build_causal_hqrc_data(frame, approved, events=events)

    assert data.occurrence_ids == tuple(
        event.occurrence_id for event in events if event.central_date.year < 2024
    )
    assert data.observations.size == 1_032
    assert tuple(np.unique(data.holiday_type_index)) == (0, 1)
    assert np.array_equal(data.hour, frame.sort("target_timestamp")["hour"].to_numpy())


@pytest.mark.parametrize("mutation", ["missing", "extra", "final", "context", "nonhourly"])
def test_training_adapter_rejects_noncausal_or_incomplete_inputs(tmp_path, mutation):
    events = load_event_registry(EVENTS)
    frame = _training_frame()
    approved = _approved(tmp_path, tuple(sorted(frame["occurrence_id"].unique().to_list())))
    if mutation == "missing":
        frame = frame.slice(1)
    elif mutation == "extra":
        frame = pl.concat(
            [frame, frame.tail(1).with_columns(pl.lit("extra").alias("occurrence_id"))]
        )
    elif mutation == "final":
        frame = frame.with_columns(pl.lit("final-2024").alias("split_id"))
    elif mutation == "context":
        frame = frame.with_columns(pl.lit("B0").alias("feature_set"))
    else:
        frame = frame.with_columns(
            pl.when(pl.arange(0, pl.len()) == 1)
            .then(pl.col("target_timestamp") + pl.duration(hours=1))
            .otherwise(pl.col("target_timestamp"))
            .alias("target_timestamp")
        )

    with pytest.raises(CausalCorrectionError):
        build_causal_hqrc_data(frame, approved, events=events)


def test_latest_scale_is_only_finite_positive_oof_2023_context_value(tmp_path):
    frame = _training_frame()
    approved = _approved(tmp_path, tuple(sorted(frame["occurrence_id"].unique().to_list())))
    record = {
        "model": "lightgbm",
        "feature_set": "B1",
        "seed": 7,
        "latest_complete_oof_scale": {"split_id": "oof-2023", "sigma_n_mw": 123.5},
    }
    manifest = {"contexts": [record]}

    assert select_latest_oof_scale(manifest, approved) == 123.5
    record["latest_complete_oof_scale"]["split_id"] = "final-2024"
    with pytest.raises(CausalCorrectionError, match="oof-2023"):
        select_latest_oof_scale(manifest, approved)


def test_final_stream_selects_exact_two_2024_events_and_excludes_october_first():
    prediction = build_causal_prediction_data(
        _final_stream(),
        context=EventResidualContext("lightgbm", "B1", 7, SPLITS),
        events=load_event_registry(EVENTS),
    )

    assert prediction.full_frame.height == 7_320
    assert prediction.event_frame.height == 264
    assert prediction.occurrence_ids == ("seollal-2024", "chuseok-2024")
    counts = prediction.event_frame.group_by("occurrence_id").len().sort("occurrence_id")
    assert dict(counts.iter_rows()) == {"chuseok-2024": 120, "seollal-2024": 144}
    assert datetime(2024, 10, 1) not in prediction.event_frame["target_timestamp"].to_list()


def test_final_stream_rejects_context_or_coverage_change():
    frame = _final_stream().with_columns(pl.lit("B0").alias("feature_set"))
    with pytest.raises(CausalCorrectionError):
        build_causal_prediction_data(
            frame,
            context=EventResidualContext("lightgbm", "B1", 7, SPLITS),
            events=load_event_registry(EVENTS),
        )


def test_causal_input_adapter_uses_validated_source_without_changing_bound_values(
    tmp_path, monkeypatch
):
    events = load_event_registry(EVENTS)
    training = _training_frame()
    approved = _approved(tmp_path, tuple(sorted(training["occurrence_id"].unique().to_list())))
    final = _final_stream()
    residual_manifest = {
        "outputs": {"standardized_residuals": {"sha256": "residual"}},
        "contexts": [
            {
                "model": "lightgbm",
                "feature_set": "B1",
                "seed": 7,
                "latest_complete_oof_scale": {
                    "split_id": "oof-2023",
                    "sigma_n_mw": 123.5,
                },
            }
        ],
    }
    baseline_manifest = {"identity": "unchanged"}
    source_paths = {"data": tmp_path / "data.csv"}
    source_hashes = {"experiment_config": "config", "event_registry": "events"}
    residual_path = tmp_path / "residual.parquet"
    members_path = tmp_path / "members.parquet"
    point_path = tmp_path / "point.parquet"
    calls: list[tuple[object, ...]] = []

    class FakeSource:
        def __init__(self):
            self.source_profile = "smoke"
            self.residual_manifest = residual_manifest
            self.baseline_manifest = baseline_manifest
            self.source_paths = source_paths
            self.source_hashes = source_hashes
            self.residual_path = residual_path
            self.final_members_path = members_path
            self.final_point_path = point_path
            self.events = events

        def load_standardized_context(self, context, *, through):
            calls.append(("oof", context, through))
            return training

        def load_final_point_context(self, context):
            calls.append(("final", context))
            return final

    def fake_validate_source(**kwargs):
        calls.append(("preflight", kwargs))
        return FakeSource()

    monkeypatch.setattr(stage, "validate_correction_source", fake_validate_source)
    monkeypatch.setattr(stage, "load_approved_calibration", lambda *_args, **_kwargs: approved)

    prepared = stage.prepare_causal_correction_inputs(
        run_dir=tmp_path / "run",
        config_path=tmp_path / "experiment.toml",
        approved_ar_path=approved.artifact_path,
        profile="smoke",
    )

    assert calls == [
        (
            "preflight",
            {
                "run_dir": tmp_path / "run",
                "config_path": tmp_path / "experiment.toml",
                "profile": "smoke",
            },
        ),
        ("oof", approved.context, 2023),
        ("final", approved.context),
    ]
    assert prepared.source_profile == "smoke"
    assert prepared.sigma_n_mw == 123.5
    assert prepared.residual_manifest is residual_manifest
    assert prepared.baseline_manifest is baseline_manifest
    assert prepared.source_paths is source_paths
    assert prepared.source_hashes is source_hashes
    assert prepared.residual_path == residual_path
    assert prepared.final_members_path == members_path
    assert prepared.final_point_path == point_path
    assert prepared.training_frame.equals(training)
    assert prepared.prediction.full_frame.equals(final)


def test_causal_products_use_q_plus_event_reset_e_and_leave_other_hours_bitwise_equal(
    tmp_path,
):
    events = load_event_registry(EVENTS)
    training = _training_frame()
    approved = _approved(tmp_path, tuple(sorted(training["occurrence_id"].unique().to_list())))
    prediction = build_causal_prediction_data(
        _final_stream(), context=approved.context, events=events
    )
    inputs = CausalCorrectionInputs(
        run_dir=tmp_path,
        source_profile="smoke",
        approved=approved,
        hqrc_data=build_causal_hqrc_data(training, approved, events=events),
        training_frame=training,
        prediction=prediction,
        sigma_n_mw=10.0,
        residual_manifest={},
        baseline_manifest={},
        source_paths={},
        source_hashes={},
        residual_path=tmp_path / "residual.parquet",
        final_members_path=tmp_path / "members.parquet",
        final_point_path=tmp_path / "point.parquet",
    )
    mu = np.zeros((2, 3, 2, 3))
    mu[:, :, 0, 0] = 1.0
    mu[:, :, 1, 0] = 2.0
    posterior = az.from_dict(
        posterior={
            "mu": mu,
            "delta": np.zeros_like(mu),
            "between_cholesky": np.zeros((2, 3, 2, 3, 3)),
            "gamma": np.zeros((2, 3, 2, 24)),
            "phi": np.zeros((2, 3)),
            "sigma_r": np.zeros((2, 3)),
        }
    )

    products = generate_causal_products(inputs, posterior, seed=19, predictive_draws=6)

    event = products.event_predictions
    seollal = event.filter(pl.col("occurrence_id") == "seollal-2024")
    chuseok = event.filter(pl.col("occurrence_id") == "chuseok-2024")
    np.testing.assert_allclose(seollal["corrected_point_mw"] - seollal["baseline_mw"], 10.0)
    np.testing.assert_allclose(chuseok["corrected_point_mw"] - chuseok["baseline_mw"], 20.0)
    for row in event.iter_rows(named=True):
        np.testing.assert_allclose(row["predictive_draws_mw"], [row["corrected_point_mw"]] * 6)
    full = products.full_period_point_predictions
    outside = full.filter(~pl.col("is_hqrc_event"))
    assert np.array_equal(
        outside["baseline_mw"].to_numpy(), outside["corrected_point_mw"].to_numpy()
    )
    assert event.height == 264 and products.event_metrics.height == 2


def _stage_inputs(tmp_path: Path, *, source_profile: str = "smoke") -> CausalCorrectionInputs:
    events = load_event_registry(EVENTS)
    training = _training_frame()
    approved = _approved(tmp_path, tuple(sorted(training["occurrence_id"].unique().to_list())))
    run = tmp_path / "source-run"
    (run / "inputs").mkdir(parents=True)
    (run / "predictions").mkdir(parents=True)
    paths = {
        "residual_manifest": run / "inputs/standardized_residuals_manifest.json",
        "baseline_manifest": run / "predictions/baseline_manifest.json",
        "residual": run / "inputs/standardized_residuals.parquet",
        "members": run / "predictions/final_2024_members.parquet",
        "point": run / "predictions/final_2024.parquet",
    }
    for name, path in paths.items():
        path.write_text(name, encoding="utf-8")
    return CausalCorrectionInputs(
        run_dir=run,
        source_profile=source_profile,
        approved=approved,
        hqrc_data=build_causal_hqrc_data(training, approved, events=events),
        training_frame=training,
        prediction=build_causal_prediction_data(
            _final_stream(), context=approved.context, events=events
        ),
        sigma_n_mw=10.0,
        residual_manifest={},
        baseline_manifest={},
        source_paths={},
        source_hashes={},
        residual_path=paths["residual"],
        final_members_path=paths["members"],
        final_point_path=paths["point"],
    )


def _fake_sample(
    approved,
    *,
    draws: int,
    chains: int,
    cores: int = 1,
    tune: int,
    seed: int,
    paper_profile: bool = False,
):
    mu = np.zeros((chains, draws, 2, 3))
    mu[:, :, 0, 0] = 1.0
    mu[:, :, 1, 0] = 2.0
    idata = az.from_dict(
        posterior={
            "mu": mu,
            "delta": np.zeros_like(mu),
            "between_cholesky": np.zeros((chains, draws, 2, 3, 3)),
            "gamma": np.zeros((chains, draws, 2, 24)),
            "phi": np.zeros((chains, draws)),
            "sigma_r": np.zeros((chains, draws)),
            "event_log_likelihood": np.zeros((chains, draws, 8)),
        },
        sample_stats={"diverging": np.zeros((chains, draws), dtype=np.int8)},
    )
    idata.attrs.update(
        {
            "hqrc_backend": "pymc",
            "hqrc_calibration_json": json.dumps(
                {
                    "artifact_path": str(approved.artifact_path),
                    "artifact_digest": approved.artifact_digest,
                    "residual_sha256": approved.residual_sha256,
                    "config_sha256": approved.config_sha256,
                    "event_sha256": approved.event_sha256,
                    "context": {
                        "model": approved.context.model,
                        "feature_set": approved.context.feature_set,
                        "seed": approved.context.seed,
                        "split_ids": list(approved.context.split_ids),
                    },
                    "a": approved.a,
                    "b": approved.b,
                },
                sort_keys=True,
            ),
            "hqrc_model_json": json.dumps(
                {
                    "variant": "H3",
                    "pooling": "partial",
                    "options": {
                        "covariance": "full",
                        "include_restriction": True,
                        "lkj_eta": 2.0,
                        "between_scale_prior": 1.0,
                        "innovation": "normal_ar1",
                    },
                    "cyclic_hour_parameterization": "noncentered-rw1-v1",
                },
                sort_keys=True,
            ),
            "hqrc_sampler_json": json.dumps(
                {
                    "draws": draws,
                    "tune": tune,
                    "chains": chains,
                    "cores": cores,
                    "seed": seed,
                    "target_accept": 0.99 if paper_profile else 0.9,
                    "paper_profile": paper_profile,
                    "init": PYMC_INITIALIZATION,
                    "geometry": SAMPLER_GEOMETRY,
                },
                sort_keys=True,
            ),
        }
    )
    return idata


def _install_fake_stage(monkeypatch, inputs: CausalCorrectionInputs) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(stage, "prepare_causal_correction_inputs", lambda **_: inputs)

    def fake_sampler(_data, approved, **kwargs):
        calls.append(kwargs)
        return _fake_sample(
            approved,
            draws=kwargs["draws"],
            chains=kwargs["chains"],
            cores=kwargs["cores"],
            tune=kwargs["tune"],
            seed=kwargs["seed"],
            paper_profile=kwargs["paper_profile"],
        )

    monkeypatch.setattr(stage, "sample_hqrc", fake_sampler)
    return calls


def _stage_arguments(
    tmp_path: Path,
    inputs: CausalCorrectionInputs,
    *,
    sampler_seed: int = 19,
    profile: str = "smoke",
    draws: int = 3,
    tune: int = 3,
    chains: int = 2,
) -> dict[str, object]:
    return {
        "run_dir": inputs.run_dir,
        "config_path": tmp_path / "experiment.toml",
        "approved_ar_path": inputs.approved.artifact_path,
        "sampler_seed": sampler_seed,
        "profile": profile,
        "draws": draws,
        "tune": tune,
        "chains": chains,
        "cores": 1,
        "output_root": tmp_path / "output",
    }


def test_causal_sampler_contract_resolves_and_validates_cores():
    assert stage._sampler_contract("smoke", seed=19, draws=3, tune=3, chains=2)["cores"] == 1
    assert (
        stage._sampler_contract("smoke", seed=19, draws=3, tune=3, chains=4, cores=4)["cores"] == 4
    )
    with pytest.raises(CausalCorrectionError, match="cores"):
        stage._sampler_contract("smoke", seed=19, draws=3, tune=3, chains=2, cores=3)


def _canonical_json(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _write_rebound_manifest(result, manifest: dict[str, object]) -> None:
    unsigned = {
        key: manifest[key] for key in ("schema_version", "state", "identity", "outputs", "rows")
    }
    manifest["manifest_digest"] = hashlib.sha256(_canonical_json(unsigned)).hexdigest()
    result.manifest_path.write_bytes(_canonical_json(manifest) + b"\n")
    complete = {
        "manifest_sha256": hashlib.sha256(result.manifest_path.read_bytes()).hexdigest(),
        "state": "COMPLETE",
    }
    (result.output_dir / "COMPLETE").write_bytes(_canonical_json(complete) + b"\n")


def _rebind_completed_publication(result, output_name: str) -> None:
    manifest = json.loads(result.manifest_path.read_bytes())
    output_path = result.output_dir / manifest["outputs"][output_name]["path"]
    manifest["outputs"][output_name]["sha256"] = hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    _write_rebound_manifest(result, manifest)


def _tamper_product(result, output_name: str) -> None:
    manifest = json.loads(result.manifest_path.read_bytes())
    path = result.output_dir / manifest["outputs"][output_name]["path"]
    frame = pl.read_parquet(path)
    if output_name == "event_predictions":
        changed_draws = [
            [float(value) + 1.0 for value in values]
            for values in frame["predictive_draws_mw"].to_list()
        ]
        frame = frame.with_columns(
            (pl.col("corrected_point_mw") + 1.0).alias("corrected_point_mw"),
            pl.Series("predictive_draws_mw", changed_draws),
        )
    elif output_name == "full_period_point_predictions":
        frame = frame.with_columns(
            (pl.col("observed_mw") + 1.0).alias("observed_mw"),
            (pl.col("baseline_mw") + 1.0).alias("baseline_mw"),
            (pl.col("corrected_point_mw") + 1.0).alias("corrected_point_mw"),
            pl.lit(True).alias("is_hqrc_event"),
        )
    else:
        frame = frame.with_columns((pl.col("rmse") + 1.0).alias("rmse"))
    frame.write_parquet(path)
    _rebind_completed_publication(result, output_name)


def test_post_sampling_checkpoint_resumes_without_sampling_twice(tmp_path, monkeypatch):
    inputs = _stage_inputs(tmp_path)
    calls = _install_fake_stage(monkeypatch, inputs)

    def crash_after_checkpoint(boundary):
        if boundary == "posterior-checkpointed":
            raise RuntimeError("prediction crash")

    monkeypatch.setattr(stage, "_publication_boundary", crash_after_checkpoint)
    arguments = _stage_arguments(tmp_path, inputs)
    with pytest.raises(RuntimeError, match="prediction crash"):
        fit_causal_2024_correction(**arguments)
    assert len(calls) == 1

    monkeypatch.setattr(stage, "_publication_boundary", lambda _: None)
    resumed = fit_causal_2024_correction(**arguments)
    assert resumed.sampler_fit_count == 0 and not resumed.reused
    assert len(calls) == 1
    reused = fit_causal_2024_correction(**arguments)
    assert reused.reused and reused.sampler_fit_count == 0
    assert len(calls) == 1


@pytest.mark.parametrize(
    "boundary",
    [
        "event_predictions-published",
        "full_period_point_predictions-published",
        "event_metrics-published",
        "full_period_point_metrics-published",
        "manifest-published",
    ],
)
def test_each_downstream_crash_resumes_from_checkpoint_without_resampling(
    tmp_path, monkeypatch, boundary
):
    inputs = _stage_inputs(tmp_path)
    calls = _install_fake_stage(monkeypatch, inputs)
    arguments = _stage_arguments(tmp_path, inputs)

    monkeypatch.setattr(
        stage,
        "_publication_boundary",
        lambda current: (
            (_ for _ in ()).throw(RuntimeError(boundary)) if current == boundary else None
        ),
    )
    with pytest.raises(RuntimeError, match=boundary):
        fit_causal_2024_correction(**arguments)
    assert len(calls) == 1

    monkeypatch.setattr(stage, "_publication_boundary", lambda _: None)
    resumed = fit_causal_2024_correction(**arguments)

    assert resumed.sampler_fit_count == 0 and not resumed.reused
    assert len(calls) == 1
    assert (resumed.output_dir / "COMPLETE").is_file()


@pytest.mark.parametrize("mutation", ["unknown", "symlink", "hqrc", "posterior"])
def test_partial_resume_fails_closed_on_unsafe_or_invalid_checkpoint(
    tmp_path, monkeypatch, mutation
):
    inputs = _stage_inputs(tmp_path)
    calls = _install_fake_stage(monkeypatch, inputs)
    arguments = _stage_arguments(tmp_path, inputs)
    monkeypatch.setattr(
        stage,
        "_publication_boundary",
        lambda current: (
            (_ for _ in ()).throw(RuntimeError("product crash"))
            if current == "event_predictions-published"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="product crash"):
        fit_causal_2024_correction(**arguments)
    output_dir = (
        tmp_path
        / "output/corrections/causal-2024/lightgbm/B1/seed-7/smoke/sampler-seed-19"
        / GEOMETRY_NAMESPACE
        / "draws-3-tune-3-chains-2-cores-1"
    )
    if mutation == "unknown":
        (output_dir / "unknown.txt").write_text("unknown", encoding="utf-8")
    elif mutation == "symlink":
        product = output_dir / "event_predictions.parquet"
        product.unlink()
        _symlink_or_skip(product, output_dir / "posterior.nc")
    elif mutation == "hqrc":
        (output_dir / "hqrc_data.current.json").write_bytes(b"{}")
    else:
        (output_dir / "posterior.checkpoint.json").write_bytes(b"{}")

    monkeypatch.setattr(stage, "_publication_boundary", lambda _: None)
    with pytest.raises(CausalCorrectionError):
        fit_causal_2024_correction(**arguments)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "output_name",
    [
        "event_predictions",
        "full_period_point_predictions",
        "event_metrics",
        "full_period_point_metrics",
    ],
)
def test_complete_reuse_rejects_rehashed_semantic_parquet_tampering(
    tmp_path, monkeypatch, output_name
):
    inputs = _stage_inputs(tmp_path)
    calls = _install_fake_stage(monkeypatch, inputs)
    arguments = _stage_arguments(tmp_path, inputs)
    completed = fit_causal_2024_correction(**arguments)
    assert len(calls) == 1
    _tamper_product(completed, output_name)

    with pytest.raises(CausalCorrectionError, match="semantics"):
        fit_causal_2024_correction(**arguments)
    assert len(calls) == 1


def test_complete_reuse_rejects_rehashed_manifest_output_alias(tmp_path, monkeypatch):
    inputs = _stage_inputs(tmp_path)
    calls = _install_fake_stage(monkeypatch, inputs)
    arguments = _stage_arguments(tmp_path, inputs)
    completed = fit_causal_2024_correction(**arguments)
    manifest = json.loads(completed.manifest_path.read_bytes())
    alias = completed.output_dir / "full_period_point_predictions.parquet"
    manifest["outputs"]["event_predictions"] = {
        "path": alias.name,
        "sha256": hashlib.sha256(alias.read_bytes()).hexdigest(),
    }
    _write_rebound_manifest(completed, manifest)

    with pytest.raises(CausalCorrectionError):
        fit_causal_2024_correction(**arguments)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "mutation", ["npz-symlink", "metadata-symlink", "unknown", "unknown-symlink"]
)
def test_partial_resume_rejects_unsafe_current_generation_entries(tmp_path, monkeypatch, mutation):
    inputs = _stage_inputs(tmp_path)
    calls = _install_fake_stage(monkeypatch, inputs)
    arguments = _stage_arguments(tmp_path, inputs)
    monkeypatch.setattr(
        stage,
        "_publication_boundary",
        lambda current: (
            (_ for _ in ()).throw(RuntimeError("product crash"))
            if current == "event_predictions-published"
            else None
        ),
    )
    with pytest.raises(RuntimeError, match="product crash"):
        fit_causal_2024_correction(**arguments)
    output_dir = (
        tmp_path
        / "output/corrections/causal-2024/lightgbm/B1/seed-7/smoke/sampler-seed-19"
        / GEOMETRY_NAMESPACE
        / "draws-3-tune-3-chains-2-cores-1"
    )
    pointer = json.loads((output_dir / "hqrc_data.current.json").read_bytes())
    generation_dir = output_dir / ".hqrc_data.generations"
    if mutation in {"npz-symlink", "metadata-symlink"}:
        key = "npz" if mutation == "npz-symlink" else "metadata"
        current = output_dir / pointer[key]
        alternate = generation_dir / f"alternate-{current.name}"
        alternate.write_bytes(current.read_bytes())
        current.unlink()
        _symlink_or_skip(current, alternate.name)
    elif mutation == "unknown":
        (generation_dir / "unknown.bin").write_bytes(b"unknown")
    else:
        _symlink_or_skip(generation_dir / "unknown-link", (output_dir / pointer["npz"]).name)

    monkeypatch.setattr(stage, "_publication_boundary", lambda _: None)
    with pytest.raises(CausalCorrectionError):
        fit_causal_2024_correction(**arguments)
    assert len(calls) == 1
    assert not (output_dir / "COMPLETE").exists()


def test_sampler_profile_and_rng_seed_have_distinct_namespaces(tmp_path, monkeypatch):
    inputs = _stage_inputs(tmp_path, source_profile="paper")
    calls = _install_fake_stage(monkeypatch, inputs)
    monkeypatch.setattr(
        stage,
        "validate_inference_data",
        lambda *_args, **_kwargs: SimpleNamespace(
            max_rhat=1.0,
            min_bulk_ess=1_000.0,
            min_tail_ess=1_000.0,
            divergences=0,
        ),
    )

    smoke_19 = fit_causal_2024_correction(**_stage_arguments(tmp_path, inputs))
    smoke_23 = fit_causal_2024_correction(**_stage_arguments(tmp_path, inputs, sampler_seed=23))
    paper_29 = fit_causal_2024_correction(
        **_stage_arguments(
            tmp_path,
            inputs,
            sampler_seed=29,
            profile="paper",
            draws=1_000,
            tune=1_000,
            chains=4,
        )
    )

    assert len({smoke_19.output_dir, smoke_23.output_dir, paper_29.output_dir}) == 3
    assert all("seed-7" in result.output_dir.parts for result in (smoke_19, smoke_23, paper_29))
    assert "smoke" in smoke_19.output_dir.parts
    assert "sampler-seed-19" in smoke_19.output_dir.parts
    assert "sampler-seed-23" in smoke_23.output_dir.parts
    assert "paper" in paper_29.output_dir.parts
    assert "sampler-seed-29" in paper_29.output_dir.parts
    assert all(
        GEOMETRY_NAMESPACE in result.output_dir.parts for result in (smoke_19, smoke_23, paper_29)
    )
    assert smoke_19.output_dir.name == "draws-3-tune-3-chains-2-cores-1"
    assert smoke_23.output_dir.name == "draws-3-tune-3-chains-2-cores-1"
    assert paper_29.output_dir.name == "draws-1000-tune-1000-chains-4-cores-1"
    assert all(
        (result.output_dir / "COMPLETE").is_file() for result in (smoke_19, smoke_23, paper_29)
    )
    assert len(calls) == 3


@pytest.mark.parametrize(
    ("profile", "initial", "retry"),
    [
        ("smoke", (3, 3, 2), (4, 3, 2)),
        ("smoke", (3, 3, 2), (3, 4, 2)),
        ("smoke", (3, 3, 2), (3, 3, 3)),
        ("paper", (1_000, 1_000, 4), (2_000, 1_000, 4)),
    ],
)
def test_failed_sampler_contract_and_retry_sizes_have_independent_namespaces(
    tmp_path, monkeypatch, profile, initial, retry
):
    inputs = _stage_inputs(tmp_path, source_profile="paper")
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(stage, "prepare_causal_correction_inputs", lambda **_: inputs)
    monkeypatch.setattr(
        stage,
        "validate_inference_data",
        lambda *_args, **_kwargs: SimpleNamespace(
            max_rhat=1.0,
            min_bulk_ess=10_000.0,
            min_tail_ess=10_000.0,
            divergences=0,
        ),
    )

    def fail_then_sample(_data, approved, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise SamplingError("strict diagnostics rejected first attempt")
        return _fake_sample(
            approved,
            draws=kwargs["draws"],
            chains=kwargs["chains"],
            cores=kwargs["cores"],
            tune=kwargs["tune"],
            seed=kwargs["seed"],
            paper_profile=kwargs["paper_profile"],
        )

    monkeypatch.setattr(stage, "sample_hqrc", fail_then_sample)
    initial_arguments = _stage_arguments(
        tmp_path,
        inputs,
        profile=profile,
        draws=initial[0],
        tune=initial[1],
        chains=initial[2],
    )
    retry_arguments = _stage_arguments(
        tmp_path,
        inputs,
        profile=profile,
        draws=retry[0],
        tune=retry[1],
        chains=retry[2],
    )

    with pytest.raises(SamplingError, match="strict diagnostics"):
        fit_causal_2024_correction(**initial_arguments)
    completed = fit_causal_2024_correction(**retry_arguments)
    failed_directory = completed.output_dir.parent / (
        f"draws-{initial[0]}-tune-{initial[1]}-chains-{initial[2]}-cores-1"
    )

    assert failed_directory != completed.output_dir
    assert (failed_directory / "hqrc_data.current.json").is_file()
    assert not (failed_directory / "COMPLETE").exists()
    assert completed.output_dir.name == (
        f"draws-{retry[0]}-tune-{retry[1]}-chains-{retry[2]}-cores-1"
    )
    assert (completed.output_dir / "COMPLETE").is_file()
    assert len(calls) == 2
    reused = fit_causal_2024_correction(**retry_arguments)
    assert reused.reused and reused.output_dir == completed.output_dir
    assert len(calls) == 2
