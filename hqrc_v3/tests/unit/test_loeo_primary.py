"""Task 15E immutable primary-H3 LOEO matrix contracts."""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from test_loeo_stage import _fake_idata
from test_loeo_stage import approved_fold as approved_fold_fixture
from test_loeo_stage import base_source as base_source_fixture
from test_loeo_stage import source as source_fixture

import hqrc_v3._loeo_primary_products as primary_products_module
import hqrc_v3._loeo_publication as fold_publication_module
import hqrc_v3.loeo_primary as primary_module
from hqrc_v3._loeo_contract import sha_json
from hqrc_v3._loeo_types import LOEOFoldMaterial, LOEOFoldResult
from hqrc_v3.bayes.samplers import SamplingDiagnostics
from hqrc_v3.loeo_primary import LOEOPrimaryError, fit_loeo_primary
from hqrc_v3.loeo_stage import generate_loeo_fold_products, prepare_loeo_fold_inputs

approved_fold = approved_fold_fixture
base_source = base_source_fixture
source = source_fixture


def _fake_material(
    approved_fold, held_out: str, root: Path, *, backend: str = "pymc"
) -> LOEOFoldMaterial:
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source, publication, approved, held_out_occurrence_id=held_out
    )
    sampler = fold_publication_module.sampler_contract(
        "smoke",
        root_seed=71,
        held_out_occurrence_id=held_out,
        draws=4,
        tune=3,
        chains=2,
        backend=backend,
    )
    identity = fold_publication_module.input_identity(inputs, sampler)
    idata = _fake_idata(
        inputs,
        chains=2,
        draws=4,
        sampler={
            "draws": 4,
            "tune": 3,
            "chains": 2,
            "cores": 1,
            "seed": sampler["seed"],
            "target_accept": 0.9,
            "paper_profile": False,
            "init": sampler["init"],
            "geometry": sampler["geometry"],
        },
    )
    products = generate_loeo_fold_products(
        inputs,
        idata,
        predictive_seed=int(identity["predictive_seed"]),
        predictive_draws=8,
    )
    directory = root / "fake-folds" / held_out
    directory.mkdir(parents=True, exist_ok=True)
    result = LOEOFoldResult(
        output_dir=directory,
        manifest_path=directory / "manifest.json",
        posterior_path=directory / "posterior.nc",
        hourly_predictions_path=directory / "hourly_predictions.parquet",
        metrics_path=directory / "metrics.parquet",
        posterior_summary_path=directory / "posterior_summary.json",
        sampler_fit_count=1,
        reused=False,
    )
    outputs = {
        name: {"path": f"{name}.artifact", "sha256": sha_json([held_out, name])}
        for name in (
            "hqrc_data_pointer",
            "hqrc_data_npz",
            "hqrc_data_metadata",
            "posterior",
            "posterior_checkpoint",
            "hourly_predictions",
            "metrics",
            "posterior_summary",
        )
    }
    unsigned = {
        "schema_version": 1,
        "state": "COMPLETE",
        "causal": False,
        "identity": identity,
        "outputs": outputs,
        "rows": {"hourly_predictions": products.hourly_predictions.height, "metrics": 1},
    }
    manifest = {**unsigned, "manifest_digest": sha_json(unsigned)}
    return LOEOFoldMaterial(result, inputs, identity, sampler, products, manifest, idata)


def _paper_material(
    approved_fold,
    held_out: str,
    root: Path,
    *,
    retry: bool = False,
) -> LOEOFoldMaterial:
    material = _fake_material(approved_fold, held_out, root)
    base = fold_publication_module.sampler_contract(
        "paper",
        root_seed=71,
        held_out_occurrence_id=held_out,
        draws=1_000,
        tune=1_000,
        chains=4,
        cores=1,
    )
    sampler = (
        fold_publication_module.retry_sampler_contract(
            base, variant="H3", held_out_occurrence_id=held_out
        )
        if retry
        else base
    )
    identity = fold_publication_module.input_identity(material.inputs, sampler)
    unsigned = {
        key: value
        for key, value in material.manifest.items()
        if key not in {"identity", "manifest_digest"}
    }
    unsigned["identity"] = identity
    if retry:
        unsigned["retry_failure"] = fold_publication_module.retry_failure_payload(
            sampler,
            SamplingDiagnostics(1.01, 500.0, 450.0, 1),
        )
    return replace(
        material,
        identity=identity,
        sampler=sampler,
        manifest={**unsigned, "manifest_digest": sha_json(unsigned)},
    )


def _install_fake_matrix_kernel(
    monkeypatch: pytest.MonkeyPatch,
    materials: dict[str, LOEOFoldMaterial],
) -> list[dict[str, object]]:
    calls: list[dict[str, object]] = []
    seen: dict[str, int] = {}

    def fake_fit(source, publication, approved, **kwargs):
        del source, publication, approved
        held = kwargs["held_out_occurrence_id"]
        calls.append(kwargs)
        count = seen.get(held, 0)
        seen[held] = count + 1
        return replace(
            materials[held].result,
            sampler_fit_count=1 if count == 0 else 0,
            reused=count > 0,
        )

    def fake_load(source, publication, approved, **kwargs):
        del source, publication, approved
        return materials[kwargs["held_out_occurrence_id"]]

    monkeypatch.setattr(primary_module, "fit_loeo_fold", fake_fit)
    monkeypatch.setattr(primary_module, "load_loeo_fold_material", fake_load)
    monkeypatch.setattr(
        primary_products_module,
        "psis_loo_summary",
        lambda _idata, *, var_name: SimpleNamespace(
            elpd_loo=-12.5,
            standard_error=1.25,
            pareto_k=np.linspace(0.1, 0.9, 9),
            var_name=var_name,
        ),
    )
    return calls


def _fit_kwargs(root: Path, selected: tuple[str, ...]) -> dict[str, object]:
    return {
        "held_out_occurrence_ids": selected,
        "root_seed": 71,
        "profile": "smoke",
        "draws": 4,
        "tune": 3,
        "chains": 2,
        "output_root": root,
    }


def _snapshot(root: Path) -> dict[str, tuple[int, bytes]]:
    return {
        path.relative_to(root).as_posix(): (path.lstat().st_ino, path.read_bytes())
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_matrix_identity_accepts_one_exact_retry_and_records_each_effective_sampler(
    approved_fold, tmp_path: Path
) -> None:
    _, publication, _ = approved_fold
    selected = publication.occurrence_ids
    retry_index = 3
    materials = tuple(
        _paper_material(
            approved_fold,
            held,
            tmp_path,
            retry=index == retry_index,
        )
        for index, held in enumerate(selected)
    )

    identity = primary_module._matrix_identity(
        materials,
        selected_occurrence_ids=selected,
        root_seed=71,
        profile="paper",
    )

    assert [fold["sampler"] for fold in identity["folds"]] == [
        dict(material.sampler) for material in materials
    ]
    assert identity["folds"][retry_index]["sampler"]["retry"] == {
        "attempt": 1,
        "reason": "diagnostic-rescue",
        "base_sampler_sha256": sha_json(
            fold_publication_module.sampler_contract(
                "paper",
                root_seed=71,
                held_out_occurrence_id=selected[retry_index],
                draws=1_000,
                tune=1_000,
                chains=4,
                cores=1,
            )
        ),
    }


def test_matrix_identity_accepts_all_retry_folds_from_requested_sampler_settings(
    approved_fold, tmp_path: Path
) -> None:
    _, publication, _ = approved_fold
    selected = publication.occurrence_ids
    materials = tuple(
        _paper_material(approved_fold, held, tmp_path, retry=True) for held in selected
    )

    identity = primary_module._matrix_identity(
        materials,
        selected_occurrence_ids=selected,
        root_seed=71,
        profile="paper",
    )

    assert [fold["sampler"] for fold in identity["folds"]] == [
        dict(material.sampler) for material in materials
    ]


def test_matrix_identity_rejects_each_mutated_retry_contract(
    approved_fold, tmp_path: Path
) -> None:
    _, publication, _ = approved_fold
    selected = publication.occurrence_ids
    retry_index = 3
    materials = tuple(
        _paper_material(
            approved_fold,
            held,
            tmp_path,
            retry=index == retry_index,
        )
        for index, held in enumerate(selected)
    )
    original = dict(materials[retry_index].sampler)
    mutations = {
        "seed": lambda sampler: sampler.__setitem__("seed", int(sampler["seed"]) + 1),
        "target_accept": lambda sampler: sampler.__setitem__("target_accept", 0.998),
        "tune": lambda sampler: sampler.__setitem__("tune", int(sampler["tune"]) + 1),
        "base_digest": lambda sampler: sampler["retry"].__setitem__(
            "base_sampler_sha256", "0" * 64
        ),
    }
    for mutate in mutations.values():
        sampler = deepcopy(original)
        mutate(sampler)
        material = materials[retry_index]
        changed = replace(
            material,
            sampler=sampler,
            identity=fold_publication_module.input_identity(material.inputs, sampler),
        )
        candidate = (*materials[:retry_index], changed, *materials[retry_index + 1 :])

        with pytest.raises(LOEOPrimaryError, match="sampler|identity|seeds"):
            primary_module._matrix_identity(
                candidate,
                selected_occurrence_ids=selected,
                root_seed=71,
                profile="paper",
            )


@pytest.mark.parametrize(
    "selected",
    [
        ("seollal-2024",),
        ("chuseok-2024", "seollal-2024"),
        ("seollal-2024", "seollal-2024"),
        (),
    ],
)
def test_paper_preflight_rejects_noncanonical_population_before_any_fold_call(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected: tuple[str, ...]
) -> None:
    source, publication, approved = approved_fold
    source = replace(source, source_profile="paper")
    calls: list[object] = []
    monkeypatch.setattr(primary_module, "fit_loeo_fold", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(LOEOPrimaryError, match="all ten canonical"):
        fit_loeo_primary(
            source,
            publication,
            approved,
            held_out_occurrence_ids=selected,
            root_seed=71,
            profile="paper",
            draws=1_000,
            tune=1_000,
            chains=4,
            output_root=tmp_path,
        )
    assert calls == []


def test_paper_preflight_rejects_nonpaper_source_and_smoke_sampler_before_fold_calls(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    calls: list[object] = []
    monkeypatch.setattr(primary_module, "fit_loeo_fold", lambda *args, **kwargs: calls.append(args))
    nonpaper_source = replace(source, source_profile="smoke")
    with pytest.raises(LOEOPrimaryError, match="all ten canonical"):
        fit_loeo_primary(
            nonpaper_source,
            publication,
            approved,
            held_out_occurrence_ids=publication.occurrence_ids,
            root_seed=71,
            profile="paper",
            draws=1_000,
            tune=1_000,
            chains=4,
            output_root=tmp_path,
        )
    paper_source = replace(source, source_profile="paper")
    with pytest.raises(LOEOPrimaryError, match="sampler contract"):
        fit_loeo_primary(
            paper_source,
            publication,
            approved,
            held_out_occurrence_ids=publication.occurrence_ids,
            root_seed=71,
            profile="paper",
            draws=4,
            tune=3,
            chains=2,
            output_root=tmp_path,
        )
    assert calls == []


def test_primary_translates_invalid_core_contract_to_primary_error(approved_fold, tmp_path: Path):
    source, publication, approved = approved_fold
    with pytest.raises(LOEOPrimaryError, match="sampler contract"):
        fit_loeo_primary(
            source,
            publication,
            approved,
            held_out_occurrence_ids=("seollal-2024",),
            root_seed=71,
            profile="smoke",
            draws=4,
            tune=3,
            chains=2,
            cores=3,
            output_root=tmp_path,
        )


def test_paper_preflight_revalidates_all_physical_folds_before_any_fold_call(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    source = replace(source, source_profile="paper")
    missing = publication.fold_paths[publication.occurrence_ids[-1]]
    missing.rename(missing.with_suffix(".missing"))
    calls: list[str] = []

    def unexpected_fold_call(*_args, **kwargs):
        calls.append(kwargs["held_out_occurrence_id"])
        raise AssertionError("fold kernel must not be called")

    monkeypatch.setattr(primary_module, "fit_loeo_fold", unexpected_fold_call)
    with pytest.raises(LOEOPrimaryError, match="complete revalidation"):
        fit_loeo_primary(
            source,
            publication,
            approved,
            held_out_occurrence_ids=publication.occurrence_ids,
            root_seed=71,
            profile="paper",
            draws=1_000,
            tune=1_000,
            chains=4,
            output_root=tmp_path / "matrix-paper",
        )
    assert calls == []


@pytest.mark.parametrize("error", [TypeError("bad source"), ValueError("bad approval")])
def test_paper_preflight_wraps_lower_level_reload_errors_before_fold_calls(
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    error: Exception,
) -> None:
    source, publication, approved = approved_fold
    source = replace(source, source_profile="paper")
    calls: list[object] = []
    monkeypatch.setattr(
        primary_module,
        "validate_loeo_fold_sources",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(error),
    )
    monkeypatch.setattr(primary_module, "fit_loeo_fold", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(LOEOPrimaryError, match="complete revalidation"):
        fit_loeo_primary(
            source,
            publication,
            approved,
            held_out_occurrence_ids=publication.occurrence_ids,
            root_seed=71,
            profile="paper",
            draws=1_000,
            tune=1_000,
            chains=4,
            output_root=tmp_path / "matrix-paper",
        )
    assert calls == []


@pytest.mark.parametrize(
    "selected",
    [
        (),
        ("seollal-2024", "seollal-2024"),
        ("chuseok-2024", "seollal-2024"),
        ("not-an-event",),
    ],
)
def test_smoke_preflight_rejects_noncanonical_subset_before_any_fold_call(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selected: tuple[str, ...]
) -> None:
    source, publication, approved = approved_fold
    calls: list[object] = []
    monkeypatch.setattr(primary_module, "fit_loeo_fold", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(LOEOPrimaryError, match="canonical subset"):
        fit_loeo_primary(source, publication, approved, **_fit_kwargs(tmp_path, selected))
    assert calls == []


def test_fake_ten_fold_matrix_calls_only_reviewed_kernel_and_aggregates_exactly(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = publication.occurrence_ids
    materials = {
        held: _fake_material(approved_fold, held, tmp_path, backend="nutpie")
        for held in selected
    }
    calls = _install_fake_matrix_kernel(monkeypatch, materials)
    result = fit_loeo_primary(
        source,
        publication,
        approved,
        **{
            **_fit_kwargs(tmp_path / "matrix", selected),
            "backend": "nutpie",
            "device": "cuda:0",
        },
    )
    assert tuple(call["held_out_occurrence_id"] for call in calls) == selected
    assert all(call["sampler_seed"] == 71 for call in calls)
    assert all(call["profile"] == "smoke" for call in calls)
    assert all(call["draws"] == 4 and call["tune"] == 3 and call["chains"] == 2 for call in calls)
    assert all(call["backend"] == "nutpie" and call["device"] == "cuda:0" for call in calls)
    hourly = pl.read_parquet(result.hourly_predictions_path)
    per_event = pl.read_parquet(result.per_event_metrics_path)
    aggregate = pl.read_parquet(result.aggregate_metrics_path)
    posterior = pl.read_parquet(result.posterior_summaries_path)
    psis = pl.read_parquet(result.training_psis_loo_path)
    assert hourly.height == 1_296
    assert per_event.height == posterior.height == psis.height == 10
    assert tuple(hourly["occurrence_id"].unique(maintain_order=True)) == selected
    assert aggregate["group"].to_list() == ["pooled", "seollal", "chuseok"]
    assert all(
        frame["causal"].unique().to_list() == [False]
        for frame in (hourly, per_event, aggregate, posterior, psis)
    )
    assert psis["diagnostic_label"].unique().to_list() == ["training_nine_event_psis_loo"]
    assert psis["training_event_count"].unique().to_list() == [9]
    assert {"fold_identity_sha256", "fold_manifest_sha256", "fold_output_dir"}.issubset(
        hourly.columns
    )
    assert not any(
        "draw" in column
        for frame in (hourly, per_event, posterior, psis)
        for column in frame.columns
    )
    pooled = aggregate.row(0, named=True)
    error = hourly["corrected_point_mw"].to_numpy() - hourly["observed_mw"].to_numpy()
    assert pooled["rmse"] == pytest.approx(float(np.sqrt(np.mean(error**2))))
    expected_crps = float(
        np.average(per_event["crps"].to_numpy(), weights=per_event["held_out_row_count"].to_numpy())
    )
    assert pooled["crps"] == pytest.approx(expected_crps)
    manifest = json.loads(result.manifest_path.read_bytes())
    seeds = [fold["sampler"]["seed"] for fold in manifest["identity"]["folds"]]
    predictive = [fold["predictive_seed"] for fold in manifest["identity"]["folds"]]
    assert len(set(seeds)) == len(set(predictive)) == 10
    assert manifest["identity"]["root_seed"] == 71
    assert manifest["identity"]["publication_scope"] == "smoke-subset"
    assert manifest["identity"]["aggregation"]["full_posterior_draws_duplicated"] is False


def test_manifest_persists_initial_fold_fit_status_across_complete_reuse(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    _install_fake_matrix_kernel(monkeypatch, materials)
    kwargs = _fit_kwargs(tmp_path / "matrix", selected)

    first = fit_loeo_primary(source, publication, approved, **kwargs)
    before = first.manifest_path.read_bytes()
    manifest = json.loads(before)
    assert manifest["fold_execution"] == [
        {
            "held_out_occurrence_id": held,
            "fit_status": "fit",
            "sampler_fit_count": 1,
        }
        for held in selected
    ]

    reused = fit_loeo_primary(source, publication, approved, **kwargs)
    assert reused.fold_sampler_fit_counts == {held: 0 for held in selected}
    assert reused.manifest_path.read_bytes() == before
    assert json.loads(before)["fold_execution"] == manifest["fold_execution"]


def test_fold_failure_leaves_no_aggregate_complete_and_retry_fits_only_missing_fold(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    calls = _install_fake_matrix_kernel(monkeypatch, materials)
    original = primary_module.fit_loeo_fold
    failed = False

    def fail_second(*args, **kwargs):
        nonlocal failed
        if kwargs["held_out_occurrence_id"] == selected[1] and not failed:
            failed = True
            raise LOEOPrimaryError("injected fold failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(primary_module, "fit_loeo_fold", fail_second)
    output = tmp_path / "matrix"
    with pytest.raises(LOEOPrimaryError, match="fold failure"):
        fit_loeo_primary(source, publication, approved, **_fit_kwargs(output, selected))
    assert list(output.rglob("COMPLETE")) == []
    recovered = fit_loeo_primary(source, publication, approved, **_fit_kwargs(output, selected))
    assert recovered.sampler_fit_count == 1
    assert recovered.fold_sampler_fit_counts == {selected[0]: 0, selected[1]: 1}
    assert [call["held_out_occurrence_id"] for call in calls] == [
        selected[0],
        selected[0],
        selected[1],
    ]


def test_complete_matrix_reuse_is_zero_fit_and_inode_byte_stable(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    calls = _install_fake_matrix_kernel(monkeypatch, materials)
    kwargs = _fit_kwargs(tmp_path / "matrix", selected)
    first = fit_loeo_primary(source, publication, approved, **kwargs)
    before = _snapshot(first.output_dir)
    reused = fit_loeo_primary(source, publication, approved, **kwargs)
    assert reused.reused is True and reused.sampler_fit_count == 0
    assert reused.fold_sampler_fit_counts == {held: 0 for held in selected}
    assert _snapshot(first.output_dir) == before
    assert len(calls) == 4


def test_complete_matrix_revalidates_current_fold_before_trusting_aggregate(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    _install_fake_matrix_kernel(monkeypatch, materials)
    kwargs = _fit_kwargs(tmp_path / "matrix", selected)
    completed = fit_loeo_primary(source, publication, approved, **kwargs)
    before = _snapshot(completed.output_dir)

    def reject_substitution(*_args, **_kwargs):
        raise LOEOPrimaryError("fold substitution")

    monkeypatch.setattr(primary_module, "load_loeo_fold_material", reject_substitution)
    with pytest.raises(LOEOPrimaryError, match="fold substitution"):
        fit_loeo_primary(source, publication, approved, **kwargs)
    assert _snapshot(completed.output_dir) == before


def test_fully_rehashed_aggregate_semantic_mutation_rejects_without_cleanup(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    _install_fake_matrix_kernel(monkeypatch, materials)
    kwargs = _fit_kwargs(tmp_path / "matrix", selected)
    completed = fit_loeo_primary(source, publication, approved, **kwargs)
    mutated = pl.read_parquet(completed.hourly_predictions_path).with_columns(
        (pl.col("corrected_point_mw") + 1.0).alias("corrected_point_mw")
    )
    mutated.write_parquet(completed.hourly_predictions_path)
    manifest = json.loads(completed.manifest_path.read_bytes())
    manifest["outputs"]["hourly_predictions"]["sha256"] = hashlib.sha256(
        completed.hourly_predictions_path.read_bytes()
    ).hexdigest()
    unsigned = {
        key: manifest[key]
        for key in (
            "schema_version",
            "state",
            "evaluation",
            "causal",
            "identity",
            "outputs",
            "rows",
            "fold_execution",
        )
    }
    manifest["manifest_digest"] = sha_json(unsigned)
    completed.manifest_path.write_bytes(fold_publication_module.canonical_json(manifest) + b"\n")
    (completed.output_dir / "COMPLETE").write_bytes(
        fold_publication_module.canonical_json(
            {
                "manifest_sha256": hashlib.sha256(completed.manifest_path.read_bytes()).hexdigest(),
                "state": "COMPLETE",
                "causal": False,
            }
        )
        + b"\n"
    )
    before = _snapshot(completed.output_dir)
    with pytest.raises(LOEOPrimaryError, match="semantics differ"):
        fit_loeo_primary(source, publication, approved, **kwargs)
    assert _snapshot(completed.output_dir) == before


@pytest.mark.parametrize(
    "boundary_name",
    [
        "matrix-manifest-prewrite",
        "matrix-complete-prewrite",
        "matrix-complete-published",
    ],
)
def test_publication_time_semantic_mutation_never_leaves_complete(
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary_name: str,
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    _install_fake_matrix_kernel(monkeypatch, materials)
    output_root = tmp_path / "matrix"
    mutated = False

    def mutate_at_boundary(name: str) -> None:
        nonlocal mutated
        if name != boundary_name or mutated:
            return
        hourly_path = next(output_root.rglob("hourly_predictions.parquet"))
        hourly = pl.read_parquet(hourly_path).with_columns(
            (pl.col("corrected_point_mw") + 123.0).alias("corrected_point_mw")
        )
        hourly.write_parquet(hourly_path)
        mutated = True

    monkeypatch.setattr(fold_publication_module, "publication_boundary", mutate_at_boundary)
    with pytest.raises(LOEOPrimaryError):
        fit_loeo_primary(source, publication, approved, **_fit_kwargs(output_root, selected))
    assert mutated
    assert list(output_root.rglob("COMPLETE")) == []


def test_postpublish_complete_substitution_is_preserved_and_rejected(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    _install_fake_matrix_kernel(monkeypatch, materials)
    output_root = tmp_path / "matrix"
    planted: list[tuple[Path, int]] = []

    def replace_complete(name: str) -> None:
        if name != "matrix-complete-published" or planted:
            return
        complete = next(output_root.rglob("COMPLETE"))
        complete.unlink()
        complete.write_bytes(b"foreign-complete")
        planted.append((complete, complete.lstat().st_ino))

    monkeypatch.setattr(fold_publication_module, "publication_boundary", replace_complete)
    with pytest.raises(LOEOPrimaryError):
        fit_loeo_primary(source, publication, approved, **_fit_kwargs(output_root, selected))
    complete, inode = planted[0]
    assert complete.lstat().st_ino == inode
    assert complete.read_bytes() == b"foreign-complete"


def test_interrupted_ordered_prefix_recovers_without_rewriting_verified_inode(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    _install_fake_matrix_kernel(monkeypatch, materials)
    interrupted = False

    def boundary(name: str) -> None:
        nonlocal interrupted
        if name == "matrix-hourly_predictions.parquet-published" and not interrupted:
            interrupted = True
            raise OSError("aggregate interruption")

    monkeypatch.setattr(fold_publication_module, "publication_boundary", boundary)
    kwargs = _fit_kwargs(tmp_path / "matrix", selected)
    with pytest.raises(OSError, match="aggregate interruption"):
        fit_loeo_primary(source, publication, approved, **kwargs)
    hourly_paths = list((tmp_path / "matrix").rglob("hourly_predictions.parquet"))
    assert len(hourly_paths) == 1
    inode = hourly_paths[0].lstat().st_ino
    monkeypatch.setattr(fold_publication_module, "publication_boundary", lambda _name: None)
    recovered = fit_loeo_primary(source, publication, approved, **kwargs)
    assert recovered.hourly_predictions_path.lstat().st_ino == inode
    assert recovered.sampler_fit_count == 0


@pytest.mark.parametrize("kind", ["unknown", "symlink", "fifo", "gap"])
def test_partial_matrix_rejects_and_preserves_foreign_or_gapped_evidence(
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    _install_fake_matrix_kernel(monkeypatch, materials)
    interrupted = False

    def boundary(name: str) -> None:
        nonlocal interrupted
        if name == "matrix-hourly_predictions.parquet-published" and not interrupted:
            interrupted = True
            raise OSError("leave prefix")

    monkeypatch.setattr(fold_publication_module, "publication_boundary", boundary)
    kwargs = _fit_kwargs(tmp_path / "matrix", selected)
    with pytest.raises(OSError, match="leave prefix"):
        fit_loeo_primary(source, publication, approved, **kwargs)
    directory = list((tmp_path / "matrix").rglob(".loeo-primary.lock"))[0].parent
    target = directory / ("aggregate_metrics.parquet" if kind == "gap" else "foreign")
    outside = tmp_path / "outside"
    outside.write_bytes(b"outside")
    if kind in {"unknown", "gap"}:
        target.write_bytes(b"foreign")
    elif kind == "symlink":
        try:
            target.symlink_to(outside)
        except OSError as error:
            pytest.skip(f"cannot create a file symlink on this platform: {error}")
    else:
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO entries are unavailable on this platform")
        os.mkfifo(target)
    before = target.lstat()
    payload = (
        os.readlink(target)
        if kind == "symlink"
        else (target.read_bytes() if kind != "fifo" else None)
    )
    monkeypatch.setattr(fold_publication_module, "publication_boundary", lambda _name: None)
    with pytest.raises(LOEOPrimaryError, match="ordered prefix"):
        fit_loeo_primary(source, publication, approved, **kwargs)
    after = target.lstat()
    assert (after.st_ino, after.st_mode) == (before.st_ino, before.st_mode)
    if kind == "symlink":
        assert os.readlink(target) == payload
    elif kind != "fifo":
        assert target.read_bytes() == payload
    assert outside.read_bytes() == b"outside"


def test_final_name_insertion_fails_without_replacing_foreign_evidence(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    selected = ("seollal-2024", "chuseok-2024")
    materials = {held: _fake_material(approved_fold, held, tmp_path) for held in selected}
    _install_fake_matrix_kernel(monkeypatch, materials)
    planted: list[tuple[Path, int]] = []

    def insert(name: str) -> None:
        if name != "matrix-aggregate_metrics.parquet-prewrite" or planted:
            return
        directory = list((tmp_path / "matrix").rglob(".loeo-primary.lock"))[0].parent
        target = directory / "aggregate_metrics.parquet"
        target.write_bytes(b"foreign-final")
        planted.append((target, target.lstat().st_ino))

    monkeypatch.setattr(fold_publication_module, "publication_boundary", insert)
    with pytest.raises(LOEOPrimaryError, match="publication failed safely"):
        fit_loeo_primary(
            source, publication, approved, **_fit_kwargs(tmp_path / "matrix", selected)
        )
    target, inode = planted[0]
    assert target.lstat().st_ino == inode
    assert target.read_bytes() == b"foreign-final"


@pytest.mark.parametrize(
    "summary",
    [
        SimpleNamespace(elpd_loo=np.nan, standard_error=1.0, pareto_k=np.zeros(9)),
        SimpleNamespace(elpd_loo=-1.0, standard_error=1.0, pareto_k=np.zeros(8)),
        SimpleNamespace(
            elpd_loo=-1.0, standard_error=1.0, pareto_k=np.asarray([0.0] * 8 + [np.inf])
        ),
    ],
)
def test_training_psis_rejects_nonfinite_or_wrong_event_population(
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    summary: SimpleNamespace,
) -> None:
    material = _fake_material(approved_fold, "seollal-2024", tmp_path)
    monkeypatch.setattr(
        primary_products_module, "psis_loo_summary", lambda *_args, **_kwargs: summary
    )
    with pytest.raises(LOEOPrimaryError, match="nine finite"):
        primary_module.generate_loeo_primary_products(
            (material,), selected_occurrence_ids=("seollal-2024",)
        )
