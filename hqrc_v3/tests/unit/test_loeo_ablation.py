"""H1/H2 LOEO ablation product and identity tests."""

from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import arviz as az
import numpy as np
import polars as pl
import pytest
from test_loeo_stage import _fake_idata, _tree_snapshot
from test_loeo_stage import approved_fold as approved_fold_fixture
from test_loeo_stage import base_source as base_source_fixture
from test_loeo_stage import source as source_fixture

import hqrc_v3._loeo_publication as fold_contract
import hqrc_v3.loeo_ablation as ablation_module
from hqrc_v3._loeo_types import LOEOFoldError
from hqrc_v3.bayes.samplers import SamplingDiagnostics, SamplingError
from hqrc_v3.loeo_ablation import _aggregate_products
from hqrc_v3.loeo_stage import prepare_loeo_fold_inputs
from hqrc_v3.provenance import file_sha256

approved_fold = approved_fold_fixture
base_source = base_source_fixture
source = source_fixture


def _occurrence_ids() -> tuple[str, ...]:
    return tuple(
        f"{holiday}-{year}" for year in range(2020, 2025) for holiday in ("seollal", "chuseok")
    )


@pytest.mark.parametrize("variant", ["H1", "H2"])
def test_ablation_aggregate_contains_h0_and_table_ready_variant_metrics(
    tmp_path: Path, variant: str
) -> None:
    occurrence_ids = _occurrence_ids()
    fold_dirs: list[Path] = []
    for index, occurrence_id in enumerate(occurrence_ids):
        directory = tmp_path / occurrence_id
        directory.mkdir()
        observed = np.linspace(50_000.0, 52_300.0, 24) + index
        timestamps = [datetime(2020, 1, 1) + timedelta(hours=hour) for hour in range(24)]
        pl.DataFrame(
            {
                "target_timestamp": timestamps,
                "occurrence_id": [occurrence_id] * 24,
                "holiday_type": [occurrence_id.rsplit("-", 1)[0]] * 24,
                "observed_mw": observed,
                "baseline_mw": observed + 1_000.0,
                "corrected_point_mw": observed + 500.0,
                "variant": [variant] * 24,
            }
        ).write_parquet(directory / "hourly_predictions.parquet")
        fold_dirs.append(directory)

    hourly, per_event, aggregate = _aggregate_products(
        tuple(fold_dirs), occurrence_ids, variant=variant, root_seed=71
    )

    assert hourly.height == 240
    assert per_event.height == 10
    assert set(aggregate["variant"].to_list()) == {"H0", variant}
    pooled = aggregate.filter(pl.col("group") == "pooled")
    h0 = pooled.filter(pl.col("variant") == "H0").row(0, named=True)
    corrected = pooled.filter(pl.col("variant") == variant).row(0, named=True)
    assert h0["rmse"] == pytest.approx(1_000.0)
    assert corrected["rmse"] == pytest.approx(500.0)
    assert corrected["delta_rmse_percent"] == pytest.approx(50.0)
    assert corrected["event_median_delta_rmse_percent"] == pytest.approx(50.0)
    assert corrected["wilcoxon_p_one_sided"] < 0.01


def test_variant_sampler_seeds_are_separate_and_h3_is_backward_compatible() -> None:
    common = {
        "profile": "smoke",
        "root_seed": 71,
        "held_out_occurrence_id": "seollal-2024",
        "draws": 4,
        "tune": 3,
        "chains": 2,
    }
    seeds = {
        variant: fold_contract.sampler_contract(**common, variant=variant)["seed"]
        for variant in ("H1", "H2", "H3")
    }
    default_h3 = fold_contract.sampler_contract(**common)["seed"]
    assert len(set(seeds.values())) == 3
    assert seeds["H3"] == default_h3


def test_parallel_ablation_core_count_creates_a_distinct_fold_identity(
    approved_fold, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    monkeypatch.setattr(
        fold_contract,
        "resolve_device",
        lambda _request: SimpleNamespace(kind="cpu", logical_device="cpu", physical_device=None),
    )
    common = {
        "profile": "smoke",
        "root_seed": 71,
        "held_out_occurrence_id": "seollal-2024",
        "variant": "H1",
        "draws": 4,
        "tune": 3,
        "chains": 4,
        "backend": "pyro",
        "device": "cpu",
    }
    sequential = fold_contract.sampler_contract(**common, cores=1)
    parallel = fold_contract.sampler_contract(**common, cores=4)
    sequential_identity = ablation_module._fold_identity(
        source,
        publication,
        approved,
        held_out="seollal-2024",
        variant="H1",
        sampler=sequential,
    )
    parallel_identity = ablation_module._fold_identity(
        source,
        publication,
        approved,
        held_out="seollal-2024",
        variant="H1",
        sampler=parallel,
    )

    assert fold_contract.sha_json(sequential_identity) != fold_contract.sha_json(parallel_identity)
    assert sequential_identity["sampler"]["cores"] == 1
    assert parallel_identity["sampler"]["cores"] == 4


def test_ablation_forwards_backend_and_device_to_each_fold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    occurrence_ids = _occurrence_ids()
    publication = SimpleNamespace(
        occurrence_ids=occurrence_ids,
        context=SimpleNamespace(model="xgboost", feature_set="B1", seed=7),
    )
    received: list[dict[str, object]] = []

    class FoldReached(RuntimeError):
        pass

    def fit_fold(*_args, **kwargs):
        received.append(kwargs)
        raise FoldReached

    monkeypatch.setattr(ablation_module, "validate_loeo_fold_sources", lambda *_args: None)
    monkeypatch.setattr(ablation_module, "_fit_fold", fit_fold)

    with pytest.raises(FoldReached):
        ablation_module.fit_loeo_ablation(
            SimpleNamespace(),
            publication,
            SimpleNamespace(),
            variant="H1",
            held_out_occurrence_ids=occurrence_ids,
            root_seed=71,
            profile="smoke",
            draws=4,
            tune=3,
            chains=4,
            cores=1,
            backend="pyro",
            device="cuda:0",
            output_root=tmp_path.resolve(),
        )

    assert received[0]["held_out"] == occurrence_ids[0]
    assert received[0]["backend"] == "pyro"
    assert received[0]["device"] == "cuda:0"


@pytest.mark.parametrize("variant", ["H1", "H2"])
def test_ablation_rejects_fresh_and_reused_posterior_metadata_without_rewriting_evidence(
    variant: str,
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, publication, approved = approved_fold
    held_out = "seollal-2024"
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id=held_out,
    )
    monkeypatch.setattr(
        fold_contract,
        "resolve_device",
        lambda _request: SimpleNamespace(kind="cuda", logical_device="cuda:0", physical_device="1"),
    )
    valid_runtime = False

    def fake_sample(_data, _calibration, **kwargs):
        sampler = {
            "draws": kwargs["draws"],
            "tune": kwargs["tune"],
            "chains": kwargs["chains"],
            "cores": 1,
            "seed": kwargs["seed"],
            "target_accept": 0.9,
            "paper_profile": False,
            "init": "pyro-default",
            "geometry": "noncentered-cyclic-hour-rw1-v1",
            "resolved_device_kind": "cuda",
            "logical_device": "cuda:0",
            "physical_device": "1",
            "dtype": "float64",
            "chain_execution": "sequential",
            "capability_probe": {
                "success": True,
                "detail": "float64-gradient-lkj-ar",
            },
            "fallback_reason": None,
        }
        return _fake_idata(
            inputs,
            chains=kwargs["chains"],
            draws=kwargs["draws"],
            sampler=sampler,
            backend="pyro",
            variant=variant,
            extra_attrs={
                "hqrc_arviz_version": "0.21.0",
                "hqrc_torch_version": "2.7.1",
                "hqrc_pyro_version": "1.9.1",
                "hqrc_device": "cuda:0",
                "hqrc_physical_device": "1" if valid_runtime else "0",
                "hqrc_dtype": "float64",
                "hqrc_device_probe": "float64-gradient-lkj-ar",
                "hqrc_device_fallback_reason": "",
                "hqrc_chain_execution": "sequential",
            },
        )

    monkeypatch.setattr(ablation_module, "sample_hqrc", fake_sample)
    output_root = tmp_path / f"ablation-{variant.lower()}"
    kwargs = {
        "held_out": held_out,
        "variant": variant,
        "root_seed": 71,
        "profile": "smoke",
        "draws": 4,
        "tune": 3,
        "chains": 4,
        "cores": 1,
        "init": None,
        "target_accept": None,
        "backend": "pyro",
        "device": "auto",
        "output_root": output_root,
    }

    with pytest.raises((LOEOFoldError, ablation_module.LOEOAblationError), match="metadata|device"):
        ablation_module._fit_fold(source, publication, approved, **kwargs)
    assert list(output_root.rglob("posterior.nc")) == []

    valid_runtime = True
    directory, fitted, reused = ablation_module._fit_fold(source, publication, approved, **kwargs)
    assert (fitted, reused) == (1, False)
    idata = az.from_netcdf(directory / "posterior.nc").load()
    try:
        idata.attrs["hqrc_physical_device"] = "0"
        replacement = tmp_path / f"{variant.lower()}-metadata-mismatch.nc"
        az.to_netcdf(idata, replacement)
    finally:
        idata.close()
    os.replace(replacement, directory / "posterior.nc")
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["outputs"]["posterior.nc"]["sha256"] = file_sha256(directory / "posterior.nc")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    manifest["manifest_digest"] = fold_contract.sha_json(unsigned)
    ablation_module._write_json(directory / "manifest.json", manifest)
    ablation_module._write_json(
        directory / "COMPLETE",
        {
            "manifest_sha256": file_sha256(directory / "manifest.json"),
            "state": "COMPLETE",
        },
    )
    before = _tree_snapshot(directory)

    with pytest.raises((LOEOFoldError, ablation_module.LOEOAblationError), match="metadata|device"):
        ablation_module._fit_fold(source, publication, approved, **kwargs)

    assert _tree_snapshot(directory) == before


def _ablation_products() -> SimpleNamespace:
    return SimpleNamespace(
        hourly_predictions=pl.DataFrame({"value": [1.0]}),
        metrics=pl.DataFrame({"metric": ["rmse"], "value": [1.0]}),
        posterior_summary={"summary": "test"},
    )


def _paper_ablation_kwargs(tmp_path: Path, variant: str) -> dict[str, object]:
    return {
        "held_out": "seollal-2024",
        "variant": variant,
        "root_seed": 71,
        "profile": "paper",
        "draws": 1_000,
        "tune": 1_000,
        "chains": 4,
        "cores": 1,
        "init": None,
        "target_accept": None,
        "backend": "pymc",
        "device": "cpu",
        "output_root": tmp_path / "ablation",
    }


def test_aggregate_publication_allows_primary_identity_without_sampler(tmp_path: Path) -> None:
    directory = tmp_path / "primary"
    identity = {
        "schema_version": 1,
        "evaluation": "retrospective-loeo-ablation-primary",
        "causal": False,
        "variant": "H1",
        "fold_identities": ["base-fold", "retry-fold"],
    }

    def write(temporary: Path) -> None:
        pl.DataFrame({"hour": [0]}).write_parquet(temporary / "hourly_predictions.parquet")
        pl.DataFrame({"event": ["seollal-2024"]}).write_parquet(
            temporary / "per_event_metrics.parquet"
        )
        pl.DataFrame({"group": ["pooled"]}).write_parquet(temporary / "aggregate_metrics.parquet")

    ablation_module._publish_directory(
        directory,
        identity=identity,
        files=ablation_module._PRIMARY_FILES,
        writer=write,
    )

    ablation_module._validate_complete(
        directory, identity=identity, files=ablation_module._PRIMARY_FILES
    )


@pytest.mark.parametrize("variant", ["H1", "H2"])
def test_ablation_retries_paper_divergence_once_with_signed_provenance_and_reuses_retry(
    variant: str,
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, publication, approved = approved_fold
    calls: list[dict[str, object]] = []
    diagnostics = SamplingDiagnostics(1.02, 500.0, 450.0, 3)

    def sample(*_args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise SamplingError("divergences", diagnostics=diagnostics)
        return az.InferenceData()

    monkeypatch.setattr(ablation_module, "sample_hqrc", sample)
    monkeypatch.setattr(
        ablation_module, "_validate_posterior_metadata", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        ablation_module,
        "generate_loeo_fold_products",
        lambda *_args, **_kwargs: _ablation_products(),
    )
    kwargs = _paper_ablation_kwargs(tmp_path, variant)

    directory, fitted, reused = ablation_module._fit_fold(source, publication, approved, **kwargs)

    assert (fitted, reused) == (2, False)
    assert len(calls) == 2
    assert calls[0]["target_accept"] == 0.99
    assert calls[0]["tune"] == 1_000
    assert calls[1]["target_accept"] == 0.999
    assert calls[1]["tune"] == 2_000
    assert calls[1]["seed"] == fold_contract.derive_loeo_seed(
        71, f"{variant}-fold-sampler-retry-1:seollal-2024"
    )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    base_sampler = fold_contract.sampler_contract(
        "paper",
        root_seed=71,
        held_out_occurrence_id="seollal-2024",
        variant=variant,
        draws=1_000,
        tune=1_000,
        chains=4,
        cores=1,
        init=None,
        target_accept=None,
        backend="pymc",
        device="cpu",
    )
    assert manifest["retry_failure"] == {
        "reason": "divergence-only",
        "base_sampler_sha256": fold_contract.sha_json(base_sampler),
        "diagnostics": {
            "max_rhat": 1.02,
            "min_bulk_ess": 500.0,
            "min_tail_ess": 450.0,
            "divergences": 3,
        },
    }

    reused_directory, reused_fitted, reused = ablation_module._fit_fold(
        source, publication, approved, **kwargs
    )

    assert reused_directory == directory
    assert (reused_fitted, reused) == (0, True)
    assert len(calls) == 2


@pytest.mark.parametrize("variant", ["H1", "H2"])
def test_ablation_prefers_a_valid_base_publication_over_retry(
    variant: str,
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, publication, approved = approved_fold
    monkeypatch.setattr(
        ablation_module, "_validate_posterior_metadata", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        ablation_module,
        "generate_loeo_fold_products",
        lambda *_args, **_kwargs: _ablation_products(),
    )
    monkeypatch.setattr(
        ablation_module, "sample_hqrc", lambda *_args, **_kwargs: az.InferenceData()
    )
    kwargs = _paper_ablation_kwargs(tmp_path, variant)
    directory, _, _ = ablation_module._fit_fold(source, publication, approved, **kwargs)
    base_sampler = fold_contract.sampler_contract(
        "paper",
        root_seed=71,
        held_out_occurrence_id="seollal-2024",
        variant=variant,
        draws=1_000,
        tune=1_000,
        chains=4,
        cores=1,
        init=None,
        target_accept=None,
        backend="pymc",
        device="cpu",
    )
    retry_sampler = fold_contract.retry_sampler_contract(
        base_sampler, variant=variant, held_out_occurrence_id="seollal-2024"
    )
    retry_identity = ablation_module._fold_identity(
        source,
        publication,
        approved,
        held_out="seollal-2024",
        variant=variant,
        sampler=retry_sampler,
    )
    retry_directory = directory.parent / f"identity-{fold_contract.sha_json(retry_identity)}"

    def write(temporary: Path) -> None:
        az.to_netcdf(az.InferenceData(), temporary / "posterior.nc")
        _ablation_products().hourly_predictions.write_parquet(
            temporary / "hourly_predictions.parquet"
        )
        _ablation_products().metrics.write_parquet(temporary / "metrics.parquet")
        ablation_module._write_json(temporary / "posterior_summary.json", {"summary": "test"})

    ablation_module._publish_directory(
        retry_directory,
        identity=retry_identity,
        files=ablation_module._FOLD_FILES,
        writer=write,
        retry_failure=fold_contract.retry_failure_payload(
            retry_sampler, SamplingDiagnostics(1.02, 500.0, 450.0, 1)
        ),
    )
    monkeypatch.setattr(
        ablation_module,
        "sample_hqrc",
        lambda *_args, **_kwargs: pytest.fail("sampled despite base reuse"),
    )

    reused_directory, fitted, reused = ablation_module._fit_fold(
        source, publication, approved, **kwargs
    )

    assert reused_directory == directory
    assert (fitted, reused) == (0, True)


@pytest.mark.parametrize(
    ("profile", "diagnostics"),
    [
        ("smoke", SamplingDiagnostics(1.02, 500.0, 450.0, 1)),
        ("paper", SamplingDiagnostics(1.02, 500.0, 450.0, 0)),
        ("paper", SamplingDiagnostics(1.0, 1.0, 1.0, 0)),
        ("paper", None),
    ],
)
def test_ablation_does_not_retry_ineligible_sampling_failure(
    profile: str,
    diagnostics: SamplingDiagnostics | None,
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, publication, approved = approved_fold
    calls = 0

    def sample(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise SamplingError("ineligible", diagnostics=diagnostics)

    monkeypatch.setattr(ablation_module, "sample_hqrc", sample)
    kwargs = _paper_ablation_kwargs(tmp_path, "H1")
    if profile == "smoke":
        kwargs.update({"profile": "smoke", "draws": 4, "tune": 3, "chains": 2})

    with pytest.raises(SamplingError, match="ineligible"):
        ablation_module._fit_fold(source, publication, approved, **kwargs)

    assert calls == 1


@pytest.mark.parametrize("variant", ["H1", "H2"])
def test_ablation_propagates_retry_failure_after_exactly_two_attempts(
    variant: str,
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source, publication, approved = approved_fold
    calls = 0

    def sample(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise SamplingError("divergences", diagnostics=SamplingDiagnostics(1.02, 500.0, 450.0, 1))

    monkeypatch.setattr(ablation_module, "sample_hqrc", sample)

    with pytest.raises(SamplingError, match="divergences"):
        ablation_module._fit_fold(
            source, publication, approved, **_paper_ablation_kwargs(tmp_path, variant)
        )

    assert calls == 2
