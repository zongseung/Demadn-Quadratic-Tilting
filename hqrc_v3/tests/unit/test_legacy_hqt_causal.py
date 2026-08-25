from __future__ import annotations

import json
from datetime import datetime, time, timedelta
from pathlib import Path

import arviz as az
import numpy as np
import polars as pl
import pytest

import hqrc_v3.legacy_hqt_causal as module
from hqrc_v3.bayes.legacy_hqt import LEGACY_HQT_MODEL_SPEC
from hqrc_v3.bayes.model import HQRCData
from hqrc_v3.cli import build_parser
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.events import EventOccurrence, load_event_registry
from hqrc_v3.features import B1W_WINDOW_VERSION, feature_columns, history_columns
from hqrc_v3.legacy_hqt_causal import (
    LegacyHQTCausalError,
    build_causal_training_frame,
    run_legacy_hqt_causal_2024,
)
from hqrc_v3.provenance import file_sha256

PROJECT = Path(__file__).resolve().parents[2]
EXPERIMENT_CONFIG = PROJECT / "configs/experiment.toml"
EVENT_REGISTRY = PROJECT / "configs/events.csv"
SPLIT_IDS = tuple(f"oof-{year}" for year in range(2020, 2024))
B1W_CONTEXT = EventResidualContext("xgboost", "B1W", 7, SPLIT_IDS)
EVALUATION_IDS = ("seollal-2024", "chuseok-2024")


def _hours(event: EventOccurrence) -> list[datetime]:
    current = datetime.combine(event.window_start, time.min)
    end = datetime.combine(event.window_end + timedelta(days=1), time.min)
    values = []
    while current < end:
        values.append(current)
        current += timedelta(hours=1)
    return values


def _b1w_schema() -> dict[str, list[str]]:
    return {
        "history": list(history_columns("B1W")),
        "future": list(feature_columns("B1W")),
        "window_definition": [B1W_WINDOW_VERSION],
    }


class FakeSource:
    def __init__(self, tmp_path: Path) -> None:
        self.events = load_event_registry(EVENT_REGISTRY)
        self.available_contexts = (B1W_CONTEXT,)
        self.residual_sha256 = "a" * 64
        self.residual_manifest_path = tmp_path / "residual_manifest.json"
        self.residual_manifest_path.write_text("{}", encoding="utf-8")
        self.final_members_sha256 = "d" * 64
        self.final_point_sha256 = "b" * 64
        self.baseline_manifest_path = tmp_path / "baseline_manifest.json"
        self.baseline_manifest_path.write_text("{}", encoding="utf-8")
        self.baseline_manifest = {"feature_schemas": {"B1W": _b1w_schema()}}
        self.oof_2023_scale = 123.5
        self.fail_if_2024_non_event_scale_requested = False
        self.standardized_calls: list[int] = []
        self.final_calls = 0
        self._training = self._build_training()
        self._final = self._build_final()

    def _build_training(self) -> pl.DataFrame:
        rows = []
        for event in self.events:
            if event.central_date.year > 2023:
                continue
            scale = self.oof_2023_scale if event.central_date.year == 2023 else 100.0
            residual = 10.0 if event.holiday_type == "seollal" else -20.0
            center = datetime.combine(event.central_date, time.min)
            for timestamp in _hours(event):
                rows.append(
                    {
                        "model": B1W_CONTEXT.model,
                        "feature_set": B1W_CONTEXT.feature_set,
                        "seed": B1W_CONTEXT.seed,
                        "split_id": f"oof-{event.central_date.year}",
                        "occurrence_id": event.occurrence_id,
                        "holiday_type": event.holiday_type,
                        "target_timestamp": timestamp,
                        "hour": timestamp.hour,
                        "restriction": event.restriction,
                        "tau_days": (timestamp - center).total_seconds() / 86_400.0,
                        "residual_mw": residual,
                        "standardized_residual": residual / scale,
                        "sigma_n_mw": scale,
                    }
                )
        return pl.DataFrame(rows).sort("target_timestamp")

    def _build_final(self) -> pl.DataFrame:
        rows = [
            {
                "model": B1W_CONTEXT.model,
                "feature_set": B1W_CONTEXT.feature_set,
                "seed": B1W_CONTEXT.seed,
                "split_id": "final-2024",
                "target_timestamp": datetime(2024, 3, 1),
                "observed_mw": 200.0,
                "predicted_mw": 199.0,
            }
        ]
        for event in self.events:
            if event.central_date.year != 2024:
                continue
            for index, timestamp in enumerate(_hours(event)):
                rows.append(
                    {
                        "model": B1W_CONTEXT.model,
                        "feature_set": B1W_CONTEXT.feature_set,
                        "seed": B1W_CONTEXT.seed,
                        "split_id": "final-2024",
                        "target_timestamp": timestamp,
                        "observed_mw": 1_000.0 + index,
                        "predicted_mw": 990.0 + index,
                    }
                )
        return pl.DataFrame(rows).sort("target_timestamp")

    def load_standardized_context(
        self, context: EventResidualContext, *, through: int
    ) -> pl.DataFrame:
        assert context == B1W_CONTEXT
        self.standardized_calls.append(through)
        return self._training.clone()

    def load_final_point_context(self, context: EventResidualContext) -> pl.DataFrame:
        assert context == B1W_CONTEXT
        self.final_calls += 1
        return self._final.clone()

    def load_2024_non_event_scale(self, *_args, **_kwargs) -> float:
        if self.fail_if_2024_non_event_scale_requested:
            raise AssertionError("causal products requested the forbidden 2024 non-event scale")
        return 999.0


@pytest.fixture
def source(tmp_path: Path) -> FakeSource:
    return FakeSource(tmp_path)


def _fake_posterior() -> az.InferenceData:
    mu = np.zeros((1, 2, 2, 3))
    mu[:, :, 0, 0] = 1.0
    mu[:, :, 1, 0] = 2.0
    cholesky = np.zeros((1, 2, 2, 3, 3))
    idata = az.from_dict(posterior={"mu": mu, "between_cholesky": cholesky})
    idata.attrs["legacy_hqt_model_json"] = json.dumps(LEGACY_HQT_MODEL_SPEC, sort_keys=True)
    return idata


def _install_runner_fakes(
    monkeypatch: pytest.MonkeyPatch,
    source: FakeSource,
) -> tuple[list[dict[str, object]], list[HQRCData]]:
    validations: list[dict[str, object]] = []
    sampled: list[HQRCData] = []

    def fake_validate(**kwargs):
        validations.append(kwargs)
        return source

    def fake_sample(data: HQRCData, **_kwargs) -> az.InferenceData:
        sampled.append(data)
        return _fake_posterior()

    monkeypatch.setattr(module, "validate_correction_source", fake_validate)
    monkeypatch.setattr(module, "sample_legacy_hqt", fake_sample)
    monkeypatch.setattr(module, "validate_inference_data", lambda *_args, **_kwargs: None)
    return validations, sampled


def _run(tmp_path: Path):
    return run_legacy_hqt_causal_2024(
        source_run_dir=tmp_path / "source",
        config_path=EXPERIMENT_CONFIG,
        output_root=tmp_path / "result",
        models=("xgboost",),
        feature_set="B1W",
        root_seed=20260813,
        profile="smoke",
        draws=5,
        tune=5,
        chains=2,
        cores=2,
    )


def test_causal_training_uses_only_eight_pre_2024_events(source: FakeSource) -> None:
    frame, ids, sigma_fit = build_causal_training_frame(source, B1W_CONTEXT)

    assert ids == tuple(
        f"{holiday}-{year}" for holiday in ("chuseok", "seollal") for year in range(2020, 2024)
    )
    assert len(ids) == 8
    assert all(not event_id.endswith("-2024") for event_id in ids)
    assert set(frame["split_id"]) == set(SPLIT_IDS)
    assert sigma_fit == pytest.approx(source.oof_2023_scale)
    assert source.standardized_calls == [2023]


def test_causal_training_rejects_a_2024_residual(source: FakeSource) -> None:
    leaked = source._training.head(1).with_columns(
        pl.lit("seollal-2024").alias("occurrence_id"),
        pl.lit("oof-2024").alias("split_id"),
    )
    source._training = pl.concat((source._training, leaked))

    with pytest.raises(LegacyHQTCausalError, match="eight pre-2024 events"):
        build_causal_training_frame(source, B1W_CONTEXT)


def test_public_causal_runner_validates_source_fits_once_and_writes_two_events(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    source.fail_if_2024_non_event_scale_requested = True
    validations, sampled = _install_runner_fakes(monkeypatch, source)

    result = _run(tmp_path)

    assert validations == [
        {
            "run_dir": (tmp_path / "source").resolve(),
            "config_path": EXPERIMENT_CONFIG.resolve(),
            "profile": "smoke",
        }
    ]
    assert len(sampled) == 1
    assert len(sampled[0].occurrence_ids) == 8
    assert result[0].sampler_fit_count == 1
    assert result[0].reused is False
    assert source.final_calls == 1

    output = result[0].output_dir
    assert {path.name for path in output.iterdir()} == {
        "COMPLETE",
        "event_metrics.parquet",
        "hourly_predictions.parquet",
        "input_identity.json",
        "manifest.json",
        "pooled_metrics.parquet",
        "posterior.nc",
    }
    hourly = pl.read_parquet(output / "hourly_predictions.parquet")
    event_metrics = pl.read_parquet(output / "event_metrics.parquet")
    pooled = pl.read_parquet(output / "pooled_metrics.parquet")
    assert tuple(hourly["occurrence_id"].unique(maintain_order=True)) == EVALUATION_IDS
    assert hourly.height == sum(
        len(_hours(event)) for event in source.events if event.occurrence_id in EVALUATION_IDS
    )
    assert set(event_metrics["correction"]) == {"H0", "H1", "H2", "H2-taper"}
    assert event_metrics.height == 8
    assert set(pooled["scope"]) == {"all", "seollal", "chuseok"}
    assert pooled.height == 12
    assert hourly.filter(pl.col("holiday_type") == "seollal")["h1_shift_mw"].unique().item() == 10.0
    assert (
        hourly.filter(pl.col("holiday_type") == "chuseok")["h1_shift_mw"].unique().item() == -20.0
    )
    assert hourly["sigma_n_mw"].unique().to_list() == [source.oof_2023_scale]
    assert (
        hourly.filter(pl.col("holiday_type") == "seollal")["H2_mw"]
        - hourly.filter(pl.col("holiday_type") == "seollal")["H0_mw"]
    ).unique().item() == pytest.approx(source.oof_2023_scale)
    assert (
        hourly.filter(pl.col("holiday_type") == "chuseok")["H2_mw"]
        - hourly.filter(pl.col("holiday_type") == "chuseok")["H0_mw"]
    ).unique().item() == pytest.approx(2.0 * source.oof_2023_scale)

    identity = json.loads((output / "input_identity.json").read_text())
    assert identity["training_occurrence_ids"] == [
        f"{holiday}-{year}" for holiday in ("chuseok", "seollal") for year in range(2020, 2024)
    ]
    assert identity["evaluation_occurrence_ids"] == list(EVALUATION_IDS)
    assert identity["source"]["residual_sha256"] == source.residual_sha256
    assert identity["source"]["residual_manifest_sha256"] == file_sha256(
        source.residual_manifest_path
    )
    assert identity["source"]["final_members_sha256"] == source.final_members_sha256
    assert identity["source"]["final_point_sha256"] == source.final_point_sha256
    assert identity["feature_schema"] == _b1w_schema()
    assert identity["model_spec"] == LEGACY_HQT_MODEL_SPEC
    assert identity["sampler"]["chains"] == 2
    assert identity["sampler"]["root_seed"] == 20260813


def test_complete_causal_context_is_hash_validated_and_reused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    validations, sampled = _install_runner_fakes(monkeypatch, source)
    first = _run(tmp_path)[0]
    before = {path.name: file_sha256(path) for path in first.output_dir.iterdir()}

    second = _run(tmp_path)[0]

    after = {path.name: file_sha256(path) for path in second.output_dir.iterdir()}
    assert len(validations) == 2
    assert len(sampled) == 1
    assert second.sampler_fit_count == 0
    assert second.reused is True
    assert before == after


def test_incomplete_context_resumes_bound_posterior_without_claiming_full_reuse(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    _, sampled = _install_runner_fakes(monkeypatch, source)
    first = _run(tmp_path)[0]
    (first.output_dir / "COMPLETE").unlink()

    resumed = _run(tmp_path)[0]

    assert len(sampled) == 1
    assert resumed.sampler_fit_count == 0
    assert resumed.reused is False
    assert (resumed.output_dir / "COMPLETE").is_file()


def test_posterior_checkpoint_manifest_exists_before_product_generation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    _install_runner_fakes(monkeypatch, source)

    def interrupt_products(*_args, **_kwargs):
        raise RuntimeError("interrupted after posterior")

    monkeypatch.setattr(module, "_build_products", interrupt_products)

    with pytest.raises(RuntimeError, match="interrupted after posterior"):
        _run(tmp_path)

    output = tmp_path / "result/xgboost/B1W"
    manifest = json.loads((output / "manifest.json").read_text())
    assert set(manifest["files"]) == {"input_identity.json", "posterior.nc"}
    assert manifest["files"]["input_identity.json"] == file_sha256(output / "input_identity.json")
    assert manifest["files"]["posterior.nc"] == file_sha256(output / "posterior.nc")


def test_incomplete_context_rejects_manifestless_tampered_posterior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    _, sampled = _install_runner_fakes(monkeypatch, source)
    first = _run(tmp_path)[0]
    (first.output_dir / "COMPLETE").unlink()
    (first.output_dir / "manifest.json").unlink()
    with (first.output_dir / "posterior.nc").open("ab") as destination:
        destination.write(b"tampered")

    with pytest.raises(LegacyHQTCausalError, match="posterior manifest"):
        _run(tmp_path)

    assert len(sampled) == 1


def test_incomplete_context_rejects_missing_identity_beside_posterior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    _, sampled = _install_runner_fakes(monkeypatch, source)
    first = _run(tmp_path)[0]
    (first.output_dir / "COMPLETE").unlink()
    (first.output_dir / "input_identity.json").unlink()

    with pytest.raises(LegacyHQTCausalError, match="identity is missing"):
        _run(tmp_path)

    assert len(sampled) == 1


def test_incomplete_context_rejects_posterior_symlink(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    _, sampled = _install_runner_fakes(monkeypatch, source)
    first = _run(tmp_path)[0]
    (first.output_dir / "COMPLETE").unlink()
    posterior = first.output_dir / "posterior.nc"
    copied = tmp_path / "copied-posterior.nc"
    copied.write_bytes(posterior.read_bytes())
    posterior.unlink()
    posterior.symlink_to(copied)

    with pytest.raises(LegacyHQTCausalError, match="posterior.*unsafe"):
        _run(tmp_path)

    assert len(sampled) == 1


def test_incomplete_context_rejects_full_manifest_product_digest_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    _, sampled = _install_runner_fakes(monkeypatch, source)
    first = _run(tmp_path)[0]
    (first.output_dir / "COMPLETE").unlink()
    with (first.output_dir / "hourly_predictions.parquet").open("ab") as destination:
        destination.write(b"tampered")

    with pytest.raises(LegacyHQTCausalError, match="incomplete causal HQT artifact digest"):
        _run(tmp_path)

    assert len(sampled) == 1


def test_incomplete_context_rejects_unknown_checkpoint_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    _install_runner_fakes(monkeypatch, source)
    first = _run(tmp_path)[0]
    (first.output_dir / "COMPLETE").unlink()
    (first.output_dir / "unexpected.txt").write_text("unknown", encoding="utf-8")

    with pytest.raises(LegacyHQTCausalError, match="incomplete causal HQT namespace"):
        _run(tmp_path)


@pytest.mark.parametrize(
    "artifact",
    [
        "input_identity.json",
        "posterior.nc",
        "hourly_predictions.parquet",
        "event_metrics.parquet",
        "pooled_metrics.parquet",
        "manifest.json",
        "COMPLETE",
    ],
)
def test_complete_context_rejects_every_tampered_identity_or_product(
    artifact: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: FakeSource,
) -> None:
    _, sampled = _install_runner_fakes(monkeypatch, source)
    first = _run(tmp_path)[0]
    with (first.output_dir / artifact).open("ab") as destination:
        destination.write(b"tampered")

    with pytest.raises(LegacyHQTCausalError, match="completed causal HQT"):
        _run(tmp_path)

    assert len(sampled) == 1


def test_completed_context_rejects_changed_source_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    _, sampled = _install_runner_fakes(monkeypatch, source)
    _run(tmp_path)
    source.final_point_sha256 = "c" * 64

    with pytest.raises(LegacyHQTCausalError, match="identity differs"):
        _run(tmp_path)

    assert len(sampled) == 1


@pytest.mark.parametrize(
    "override",
    (
        {"draws": 999},
        {"tune": 999},
        {"chains": 2},
        {"target_accept": 0.98},
    ),
)
def test_paper_causal_sampler_never_weakens_the_minimum_contract(
    override: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    source: FakeSource,
) -> None:
    monkeypatch.setattr(module, "validate_correction_source", lambda **_kwargs: source)
    options: dict[str, object] = {
        "source_run_dir": tmp_path / "source",
        "config_path": EXPERIMENT_CONFIG,
        "output_root": tmp_path / "result",
        "models": ("xgboost",),
        "feature_set": "B1W",
        "root_seed": 20260813,
        "profile": "paper",
        "draws": 1_000,
        "tune": 1_000,
        "chains": 4,
        "cores": 4,
        "target_accept": 0.99,
    }
    options.update(override)

    with pytest.raises(LegacyHQTCausalError, match="paper causal HQT requires"):
        run_legacy_hqt_causal_2024(**options)  # type: ignore[arg-type]


def test_paper_causal_sampler_allows_more_than_four_chains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    monkeypatch.setattr(module, "validate_correction_source", lambda **_kwargs: source)
    reached: list[dict[str, object]] = []

    def fake_sample(_data: HQRCData, **kwargs) -> az.InferenceData:
        reached.append(kwargs)
        return _fake_posterior()

    monkeypatch.setattr(module, "sample_legacy_hqt", fake_sample)

    result = run_legacy_hqt_causal_2024(
        source_run_dir=tmp_path / "source",
        config_path=EXPERIMENT_CONFIG,
        output_root=tmp_path / "result",
        models=("xgboost",),
        feature_set="B1W",
        root_seed=20260813,
        profile="paper",
        draws=1_000,
        tune=1_000,
        chains=5,
        cores=5,
        target_accept=0.99,
    )

    assert result[0].sampler_fit_count == 1
    assert reached[0]["chains"] == 5


def test_cli_exposes_causal_b1w_runner_without_ar_options() -> None:
    arguments = build_parser().parse_args(
        [
            "run-hqt-causal-2024",
            "--source-run-dir",
            "source",
            "--config",
            "experiment.toml",
            "--output-root",
            "results/causal-2024",
            "--model",
            "xgboost",
            "--profile",
            "smoke",
            "--draws",
            "5",
            "--tune",
            "5",
            "--chains",
            "2",
            "--cores",
            "2",
        ]
    )

    assert arguments.command == "run-hqt-causal-2024"
    assert arguments.feature_set == "B1W"
    assert arguments.root_seed == 20260813
    assert not hasattr(arguments, "approved_ar")
    assert not hasattr(arguments, "approve_derived_ar")
