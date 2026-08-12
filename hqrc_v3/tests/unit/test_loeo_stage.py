"""Task 15D one-fold immutable H3 LOEO contracts."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import arviz as az
import hqrc_v3._loeo_contract as loeo_contract_module
import hqrc_v3._loeo_products as loeo_products_module
import hqrc_v3._loeo_publication as loeo_publication_module
import hqrc_v3.loeo_stage as loeo_stage_module
import numpy as np
import pytest
import xarray as xr
from hqrc_v3.bayes.artifacts import load_hqrc_data
from hqrc_v3.bayes.model import CYCLIC_HOUR_PARAMETERIZATION, HQRCModelOptions, build_hqrc_model
from hqrc_v3.bayes.samplers import SamplingDiagnostics, SamplingError
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.diagnostics.loeo import load_loeo_event, publish_loeo_universe
from hqrc_v3.diagnostics.loeo_ar import (
    approve_loeo_ar_proposal_set,
    load_approved_loeo_ar_set,
    prepare_loeo_ar_proposal_set,
)
from hqrc_v3.loeo_stage import (
    LOEOFoldError,
    LOEOFoldInputs,
    LOEOFoldProducts,
    LOEOFoldResult,
    derive_loeo_seed,
    fit_loeo_fold,
    generate_loeo_fold_products,
    load_loeo_fold_result,
    prepare_loeo_fold_inputs,
)
from hqrc_v3.provenance import file_sha256
from test_loeo_ar import _ar_source as ar_source_fixture
from test_loeo_diagnostics import CONTEXT
from test_loeo_diagnostics import source as source_fixture

base_source = source_fixture
source = ar_source_fixture


def _posterior_dataset(*, chains: int = 2, draws: int = 4) -> xr.Dataset:
    sample = (chains, draws)
    phi = np.linspace(0.2, 0.5, chains * draws).reshape(sample)
    return xr.Dataset(
        {
            "mu": (("chain", "draw", "holiday", "coefficient"), np.zeros((*sample, 2, 3))),
            "delta": (
                ("chain", "draw", "holiday", "coefficient"),
                np.zeros((*sample, 2, 3)),
            ),
            "between_cholesky": (
                ("chain", "draw", "holiday", "coefficient", "coefficient_aux"),
                np.broadcast_to(np.eye(3), (*sample, 2, 3, 3)).copy(),
            ),
            "gamma": (("chain", "draw", "holiday", "hour"), np.zeros((*sample, 2, 24))),
            "u_phi": (("chain", "draw"), (phi + 1.0) / 2.0),
            "phi": (("chain", "draw"), phi),
            "sigma_r": (("chain", "draw"), np.full(sample, 0.1)),
            "event_log_likelihood": (
                ("chain", "draw", "event"),
                np.zeros((*sample, 9)),
            ),
        }
    )


def _fake_idata(inputs, *, chains: int = 2, draws: int = 4, sampler=None) -> az.InferenceData:
    idata = az.InferenceData(
        posterior=_posterior_dataset(chains=chains, draws=draws),
        sample_stats=xr.Dataset(
            {"diverging": (("chain", "draw"), np.zeros((chains, draws), dtype=np.int8))}
        ),
    )
    if sampler is not None:
        idata.attrs.update(
            {
                "hqrc_backend": "pymc",
                "hqrc_calibration_json": json.dumps(
                    {
                        "artifact_path": str(inputs.approved.artifact_path),
                        "artifact_digest": inputs.approved.artifact_digest,
                        "residual_sha256": inputs.approved.residual_sha256,
                        "config_sha256": inputs.approved.config_sha256,
                        "event_sha256": inputs.approved.event_sha256,
                        "context": {
                            "model": inputs.approved.context.model,
                            "feature_set": inputs.approved.context.feature_set,
                            "seed": inputs.approved.context.seed,
                            "split_ids": list(inputs.approved.context.split_ids),
                        },
                        "a": inputs.approved.a,
                        "b": inputs.approved.b,
                    },
                    sort_keys=True,
                ),
                "hqrc_model_json": json.dumps(
                    {
                        "variant": "H3",
                        "pooling": "partial",
                        "options": asdict(
                            HQRCModelOptions(
                                covariance="full",
                                include_restriction=True,
                                innovation="normal_ar1",
                            )
                        ),
                        "cyclic_hour_parameterization": CYCLIC_HOUR_PARAMETERIZATION,
                    },
                    sort_keys=True,
                ),
                "hqrc_sampler_json": json.dumps(sampler, sort_keys=True),
            }
        )
    return idata


@pytest.fixture
def approved_fold(source: ValidatedCorrectionSource, tmp_path: Path):
    publication = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "loeo")
    proposal = prepare_loeo_ar_proposal_set(source, publication, output_dir=tmp_path / "loeo-ar")
    approve_loeo_ar_proposal_set(
        source,
        publication,
        output_dir=proposal.output_dir,
        confirm_proposal_set_sha256=proposal.proposal_set_sha256,
    )
    approved = load_approved_loeo_ar_set(source, publication, output_dir=proposal.output_dir)
    return source, publication, approved


def test_public_api_is_typed_and_seed_derivation_is_stable() -> None:
    assert issubclass(LOEOFoldError, ValueError)
    assert LOEOFoldInputs.__name__ == "LOEOFoldInputs"
    assert LOEOFoldProducts.__name__ == "LOEOFoldProducts"
    assert LOEOFoldResult.__name__ == "LOEOFoldResult"
    assert callable(prepare_loeo_fold_inputs)
    assert callable(generate_loeo_fold_products)
    assert callable(fit_loeo_fold)
    assert callable(load_loeo_fold_result)
    assert derive_loeo_seed(41, "sampler:seollal-2024") == derive_loeo_seed(
        41, "sampler:seollal-2024"
    )
    assert derive_loeo_seed(41, "sampler:seollal-2024") != derive_loeo_seed(
        41, "sampler:chuseok-2024"
    )


def test_safe_event_loader_and_preparation_bind_one_physical_fold(
    approved_fold, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    held_out = "seollal-2024"
    event = load_loeo_event(
        source,
        CONTEXT,
        output_dir=publication.output_dir,
        occurrence_id=held_out,
    )
    calls: list[str] = []
    original = approved.calibration_for

    def calibration_for(self, occurrence_id: str):
        assert self is approved
        calls.append(occurrence_id)
        return original(occurrence_id)

    monkeypatch.setattr(type(approved), "calibration_for", calibration_for)
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id=held_out,
    )

    assert event.occurrence_id == held_out
    assert event.frame["occurrence_id"].unique().to_list() == [held_out]
    assert event.frame["causal"].unique().to_list() == [False]
    assert calls == [held_out]
    assert len(inputs.hqrc_data.occurrence_ids) == 9
    assert held_out not in inputs.hqrc_data.occurrence_ids
    assert inputs.training_fold.occurrence_ids == inputs.hqrc_data.occurrence_ids
    assert inputs.held_out.frame.equals(event.frame, null_equal=True)
    assert inputs.sigma_eval == pytest.approx(23.0)
    assert inputs.causal is False
    assert inputs.approved.calibration.event_ids == tuple(sorted(inputs.hqrc_data.occurrence_ids))
    for index in range(9):
        positions = np.flatnonzero(inputs.hqrc_data.occurrence_index == index)
        assert positions.size > 1
        assert np.all(np.diff(positions) == 1)


@pytest.mark.parametrize(
    ("held_out", "expected_scale"),
    [("seollal-2020", 20.0), ("seollal-2024", 23.0), ("chuseok-2024", 23.0)],
)
def test_evaluation_scale_is_the_unique_source_frozen_fold_scale(
    approved_fold, held_out: str, expected_scale: float
) -> None:
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id=held_out,
    )
    assert inputs.sigma_eval == pytest.approx(expected_scale)
    assert inputs.held_out.frame["sigma_n_mw"].unique().to_list() == [expected_scale]


def test_preparation_rejects_raw_untrusted_and_incomplete_approval_wrappers(
    approved_fold,
) -> None:
    source, publication, approved = approved_fold
    with pytest.raises(TypeError, match="ApprovedLOEOARSet"):
        prepare_loeo_fold_inputs(
            source,
            publication,
            object(),
            held_out_occurrence_id="seollal-2024",
        )
    untrusted = replace(approved, _token=object())
    with pytest.raises(LOEOFoldError, match="revalidation"):
        prepare_loeo_fold_inputs(
            source,
            publication,
            untrusted,
            held_out_occurrence_id="seollal-2024",
        )
    incomplete = replace(approved, occurrence_ids=approved.occurrence_ids[:-1])
    with pytest.raises(LOEOFoldError, match="approved LOEO AR set"):
        prepare_loeo_fold_inputs(
            source,
            publication,
            incomplete,
            held_out_occurrence_id="seollal-2024",
        )


def test_h3_model_keeps_exact_geometry_options_and_posterior_phi_semantics(approved_fold) -> None:
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id="seollal-2024",
    )
    options = HQRCModelOptions(covariance="full", include_restriction=True, innovation="normal_ar1")
    model = build_hqrc_model(inputs.hqrc_data, inputs.approved, "H3", "partial", options)
    free_names = {variable.name for variable in model.free_RVs}
    assert "u_phi" in free_names
    assert "phi" in model.named_vars
    assert "gamma_innovation_raw" in free_names
    assert "delta" in free_names
    assert "between_cov_0" in free_names
    assert "between_cov_1" in free_names
    assert any(potential.name == "event_reset_ar1" for potential in model.potentials)


def test_products_use_known_restriction_one_ar_reset_and_exactly_one_noise_term(
    approved_fold, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id="seollal-2021",
    )
    assert inputs.restriction == 1
    idata = _fake_idata(inputs)
    captured: dict[str, object] = {}

    def fixed_beta(_posterior, **kwargs):
        captured["restriction"] = kwargs["restriction"]
        captured["coefficient_draws"] = kwargs["draws"]
        return np.broadcast_to(np.asarray([1.0, 0.5, 0.1]), (kwargs["draws"], 3)).copy()

    def fixed_ar(*, phi, sigma, horizon, seed):
        captured["ar_horizon"] = horizon
        captured["ar_calls"] = int(captured.get("ar_calls", 0)) + 1
        captured["ar_seed"] = seed
        return np.full((len(np.asarray(phi).reshape(-1)), horizon), 0.25)

    original_formula = loeo_products_module.corrected_predictive_draws

    def exact_formula(baseline, sigma_n, q, e):
        captured["formula_inputs"] = (baseline.copy(), sigma_n, q.copy(), e.copy())
        return original_formula(baseline, sigma_n, q, e)

    monkeypatch.setattr(loeo_products_module, "draw_new_event_correction", fixed_beta)
    monkeypatch.setattr(loeo_products_module, "simulate_stationary_ar1", fixed_ar)
    monkeypatch.setattr(loeo_products_module, "corrected_predictive_draws", exact_formula)
    products = generate_loeo_fold_products(inputs, idata, predictive_seed=991, predictive_draws=8)

    tau = inputs.held_out.frame["tau_days"].to_numpy()
    expected_q = 1.0 + 0.5 * tau + 0.1 * tau**2
    baseline = inputs.held_out.frame["predicted_mw"].to_numpy()
    expected_point = baseline + inputs.sigma_eval * expected_q
    expected_predictive_mean = baseline + inputs.sigma_eval * (expected_q + 0.25)
    assert captured["restriction"] == 1
    assert captured["ar_calls"] == 1
    assert captured["ar_horizon"] == inputs.held_out.frame.height
    assert products.hourly_predictions["occurrence_id"].unique().to_list() == ["seollal-2021"]
    assert products.hourly_predictions["causal"].unique().to_list() == [False]
    np.testing.assert_allclose(products.hourly_predictions["corrected_point_mw"], expected_point)
    np.testing.assert_allclose(
        products.hourly_predictions["predictive_mean_mw"], expected_predictive_mean
    )
    assert products.metrics["occurrence_id"].to_list() == ["seollal-2021"]
    assert products.metrics["causal"].to_list() == [False]
    assert products.posterior_summary["phi"]["parameterization"].startswith("phi=2*u_phi-1")
    formula_baseline, formula_scale, formula_q, formula_e = captured["formula_inputs"]
    np.testing.assert_array_equal(formula_baseline, baseline)
    assert formula_scale == inputs.sigma_eval
    np.testing.assert_allclose(formula_q, np.broadcast_to(expected_q, formula_q.shape))
    np.testing.assert_allclose(formula_e, 0.25)


def test_checkpoint_recovery_complete_reuse_namespaces_and_fail_closed_products(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    sample_calls: list[dict[str, object]] = []

    def fake_sample(data, calibration, **kwargs):
        held_out = (set(publication.occurrence_ids) - set(data.occurrence_ids)).pop()
        current_inputs = prepare_loeo_fold_inputs(
            source,
            publication,
            approved,
            held_out_occurrence_id=held_out,
        )
        assert data.occurrence_ids == current_inputs.hqrc_data.occurrence_ids
        assert calibration.artifact_digest == current_inputs.approved.artifact_digest
        sample_calls.append(kwargs)
        return _fake_idata(
            current_inputs,
            chains=kwargs["chains"],
            draws=kwargs["draws"],
            sampler={
                "draws": kwargs["draws"],
                "tune": kwargs["tune"],
                "chains": kwargs["chains"],
                "seed": kwargs["seed"],
                "target_accept": 0.9,
                "paper_profile": False,
                "init": "adapt_diag",
                "geometry": "noncentered-cyclic-hour-rw1-v1",
            },
        )

    monkeypatch.setattr(loeo_stage_module, "sample_hqrc", fake_sample)
    monkeypatch.setattr(
        loeo_publication_module,
        "validate_inference_data",
        lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 800.0, 700.0, 0),
    )
    interrupted = True

    def boundary(name: str) -> None:
        nonlocal interrupted
        if name == "posterior-checkpointed" and interrupted:
            interrupted = False
            raise OSError("injected downstream crash")

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", boundary)
    fit_kwargs = {
        "held_out_occurrence_id": "seollal-2024",
        "sampler_seed": 71,
        "profile": "smoke",
        "draws": 4,
        "tune": 3,
        "chains": 2,
        "output_root": tmp_path / "results",
    }
    with pytest.raises(OSError, match="downstream crash"):
        fit_loeo_fold(source, publication, approved, **fit_kwargs)
    assert len(sample_calls) == 1
    monkeypatch.setattr(loeo_publication_module, "publication_boundary", lambda _name: None)
    recovered = fit_loeo_fold(source, publication, approved, **fit_kwargs)
    assert recovered.sampler_fit_count == 0 and recovered.reused is False
    assert len(sample_calls) == 1
    reused = fit_loeo_fold(source, publication, approved, **fit_kwargs)
    loaded = load_loeo_fold_result(source, publication, approved, **fit_kwargs)
    assert reused.reused is True and reused.sampler_fit_count == 0
    assert loaded.reused is True and loaded.output_dir == recovered.output_dir
    assert len(sample_calls) == 1
    _, hqrc_settings = load_hqrc_data(recovered.output_dir / "hqrc_data.npz")
    assert hqrc_settings["causal"] is False
    published_idata = az.from_netcdf(recovered.posterior_path)
    try:
        assert published_idata.attrs["hqrc_causal"] == "false"
    finally:
        published_idata.close()
    assert (
        json.loads((recovered.output_dir / "posterior.checkpoint.json").read_bytes())["causal"]
        is False
    )
    assert json.loads(recovered.manifest_path.read_bytes())["causal"] is False
    assert json.loads((recovered.output_dir / "COMPLETE").read_bytes())["causal"] is False
    assert sample_calls[0]["variant"] == "H3"
    assert sample_calls[0]["pooling"] == "partial"
    assert sample_calls[0]["options"] == HQRCModelOptions(
        covariance="full", include_restriction=True, innovation="normal_ar1"
    )
    assert sample_calls[0]["backend"] == "pymc"
    assert sample_calls[0]["paper_profile"] is False
    assert sample_calls[0]["seed"] == derive_loeo_seed(71, "H3-fold-sampler:seollal-2024")

    other = fit_loeo_fold(
        source,
        publication,
        approved,
        **{**fit_kwargs, "held_out_occurrence_id": "chuseok-2024"},
    )
    other_seed = fit_loeo_fold(
        source,
        publication,
        approved,
        **{**fit_kwargs, "sampler_seed": 72},
    )
    assert len({recovered.output_dir, other.output_dir, other_seed.output_dir}) == 3
    assert len(sample_calls) == 3

    unknown = recovered.output_dir / "unknown"
    unknown.write_bytes(b"foreign evidence")
    with pytest.raises(LOEOFoldError, match="unknown|partial"):
        load_loeo_fold_result(source, publication, approved, **fit_kwargs)
    unknown.unlink()

    metrics_bytes = recovered.metrics_path.read_bytes()
    target = tmp_path / "metrics-target"
    target.write_bytes(metrics_bytes)
    recovered.metrics_path.unlink()
    recovered.metrics_path.symlink_to(target)
    with pytest.raises(LOEOFoldError, match="unsafe|manifest"):
        load_loeo_fold_result(source, publication, approved, **fit_kwargs)
    recovered.metrics_path.unlink()
    recovered.metrics_path.write_bytes(metrics_bytes)

    mutated = loeo_products_module.pl.read_parquet(recovered.hourly_predictions_path).with_columns(
        (loeo_products_module.pl.col("corrected_point_mw") + 1.0).alias("corrected_point_mw")
    )
    mutated.write_parquet(recovered.hourly_predictions_path)
    manifest = json.loads(recovered.manifest_path.read_bytes())
    manifest["outputs"]["hourly_predictions"]["sha256"] = file_sha256(
        recovered.hourly_predictions_path
    )
    unsigned = {
        key: manifest[key]
        for key in ("schema_version", "state", "causal", "identity", "outputs", "rows")
    }
    manifest["manifest_digest"] = loeo_contract_module.sha_json(unsigned)
    recovered.manifest_path.write_bytes(loeo_contract_module.canonical_json(manifest) + b"\n")
    (recovered.output_dir / "COMPLETE").write_bytes(
        loeo_contract_module.canonical_json(
            {
                "manifest_sha256": file_sha256(recovered.manifest_path),
                "state": "COMPLETE",
                "causal": False,
            }
        )
        + b"\n"
    )
    with pytest.raises(LOEOFoldError, match="semantics"):
        load_loeo_fold_result(source, publication, approved, **fit_kwargs)


def test_strict_diagnostic_failure_publishes_no_complete_result(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id="seollal-2024",
    )

    def fake_sample(_data, _calibration, **kwargs):
        return _fake_idata(
            inputs,
            chains=kwargs["chains"],
            draws=kwargs["draws"],
            sampler={
                "draws": kwargs["draws"],
                "tune": kwargs["tune"],
                "chains": kwargs["chains"],
                "seed": kwargs["seed"],
                "target_accept": 0.99,
                "paper_profile": True,
                "init": "adapt_diag",
                "geometry": "noncentered-cyclic-hour-rw1-v1",
            },
        )

    monkeypatch.setattr(loeo_stage_module, "sample_hqrc", fake_sample)
    monkeypatch.setattr(
        loeo_publication_module,
        "validate_inference_data",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(SamplingError("strict gate failed")),
    )
    output = tmp_path / "diagnostic-failure"
    with pytest.raises(SamplingError, match="strict gate failed"):
        fit_loeo_fold(
            source,
            publication,
            approved,
            held_out_occurrence_id="seollal-2024",
            sampler_seed=99,
            profile="paper",
            draws=1_000,
            tune=1_000,
            chains=4,
            output_root=output,
        )
    assert not list(output.rglob("COMPLETE"))
