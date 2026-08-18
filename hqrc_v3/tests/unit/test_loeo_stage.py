"""Task 15D one-fold immutable H3 LOEO contracts."""

import json
import os
import stat
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

import arviz as az
import numpy as np
import pytest
import xarray as xr
from test_loeo_ar import _ar_source as ar_source_fixture
from test_loeo_diagnostics import CONTEXT
from test_loeo_diagnostics import source as source_fixture

import hqrc_v3._loeo_contract as loeo_contract_module
import hqrc_v3._loeo_posterior as loeo_posterior_module
import hqrc_v3._loeo_products as loeo_products_module
import hqrc_v3._loeo_publication as loeo_publication_module
import hqrc_v3.diagnostics.ar as ar_diagnostics_module
import hqrc_v3.diagnostics.loeo as loeo_diagnostics_module
import hqrc_v3.diagnostics.loeo_ar as loeo_ar_module
import hqrc_v3.loeo_stage as loeo_stage_module
from hqrc_v3.bayes.artifacts import load_hqrc_data
from hqrc_v3.bayes.model import (
    CYCLIC_HOUR_PARAMETERIZATION,
    HQRCModelOptions,
    build_hqrc_model,
    event_reset_ar1_logp_numpy,
    stationary_ar1_logp_numpy,
)
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
from hqrc_v3.publication_fs import _fsync_directory as portable_fsync_directory
from hqrc_v3.publication_fs import _fsync_file as portable_fsync_file
from hqrc_v3.publication_fs import exclusive_lock as portable_exclusive_lock

base_source = source_fixture
source = ar_source_fixture


def _posterior_dataset(inputs, *, chains: int = 2, draws: int = 4) -> xr.Dataset:
    sample = (chains, draws)
    phi = np.linspace(0.2, 0.5, chains * draws).reshape(sample)
    cholesky = np.broadcast_to(np.eye(3), (*sample, 2, 3, 3)).copy()
    event_likelihood = event_reset_ar1_logp_numpy(
        [inputs.hqrc_data.observations[segment] for segment in inputs.hqrc_data.segments],
        phi,
        np.full(sample, 0.1),
    )
    data_vars = {
        "mu": (("chain", "draw", "mu_dim_0", "mu_dim_1"), np.zeros((*sample, 2, 3))),
        "delta": (
            ("chain", "draw", "delta_dim_0", "delta_dim_1"),
            np.zeros((*sample, 2, 3)),
        ),
        "between_cholesky": (
            (
                "chain",
                "draw",
                "between_cholesky_dim_0",
                "between_cholesky_dim_1",
                "between_cholesky_dim_2",
            ),
            cholesky,
        ),
        "between_scale": (
            ("chain", "draw", "between_scale_dim_0", "between_scale_dim_1"),
            np.ones((*sample, 2, 3)),
        ),
    }
    for holiday in range(2):
        data_vars[f"between_cov_{holiday}"] = (
            ("chain", "draw", f"between_cov_{holiday}_dim_0"),
            np.broadcast_to(np.asarray([1.0, 0.0, 1.0, 0.0, 0.0, 1.0]), (*sample, 6)),
        )
        data_vars[f"between_cov_{holiday}_corr"] = (
            (
                "chain",
                "draw",
                f"between_cov_{holiday}_corr_dim_0",
                f"between_cov_{holiday}_corr_dim_1",
            ),
            np.broadcast_to(np.eye(3), (*sample, 3, 3)),
        )
        data_vars[f"between_cov_{holiday}_stds"] = (
            ("chain", "draw", f"between_cov_{holiday}_stds_dim_0"),
            np.ones((*sample, 3)),
        )
    data_vars.update(
        {
            "beta_offset": (
                ("chain", "draw", "beta_offset_dim_0", "beta_offset_dim_1"),
                np.zeros((*sample, 9, 3)),
            ),
            "beta": (
                ("chain", "draw", "beta_dim_0", "beta_dim_1"),
                np.zeros((*sample, 9, 3)),
            ),
            "sigma_gamma": (("chain", "draw", "sigma_gamma_dim_0"), np.ones((*sample, 2))),
            "gamma_innovation_raw": (
                (
                    "chain",
                    "draw",
                    "gamma_innovation_raw_dim_0",
                    "gamma_innovation_raw_dim_1",
                ),
                np.zeros((*sample, 2, 23)),
            ),
            "gamma_innovation": (
                (
                    "chain",
                    "draw",
                    "gamma_innovation_dim_0",
                    "gamma_innovation_dim_1",
                ),
                np.zeros((*sample, 2, 23)),
            ),
            "gamma": (
                ("chain", "draw", "gamma_dim_0", "gamma_dim_1"),
                np.zeros((*sample, 2, 24)),
            ),
            "u_phi": (("chain", "draw"), (phi + 1.0) / 2.0),
            "phi": (("chain", "draw"), phi),
            "sigma_r": (("chain", "draw"), np.full(sample, 0.1)),
            "event_log_likelihood": (
                ("chain", "draw", "event_log_likelihood_dim_0"),
                event_likelihood,
            ),
        }
    )
    dataset = xr.Dataset(data_vars)
    return dataset.assign_coords({name: np.arange(size) for name, size in dataset.sizes.items()})


def _fake_idata(inputs, *, chains: int = 2, draws: int = 4, sampler=None) -> az.InferenceData:
    idata = az.InferenceData(
        posterior=_posterior_dataset(inputs, chains=chains, draws=draws),
        sample_stats=xr.Dataset(
            {"diverging": (("chain", "draw"), np.zeros((chains, draws), dtype=np.int8))}
        ),
    )
    idata.add_groups(
        {"log_likelihood": xr.Dataset({"event": idata.posterior["event_log_likelihood"]})}
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


def _tree_snapshot(root: Path) -> dict[str, tuple[object, ...]]:
    snapshot: dict[str, tuple[object, ...]] = {}
    if not root.exists() and not root.is_symlink():
        return snapshot
    for path in (root, *sorted(root.rglob("*"))):
        relative = path.relative_to(root).as_posix() if path != root else "."
        identity = path.lstat()
        if stat.S_ISDIR(identity.st_mode):
            snapshot[relative] = ("directory", identity.st_ino)
        elif stat.S_ISREG(identity.st_mode):
            snapshot[relative] = ("regular", identity.st_ino, path.read_bytes())
        elif stat.S_ISLNK(identity.st_mode):
            snapshot[relative] = ("symlink", os.readlink(path))
        else:
            snapshot[relative] = ("special", stat.S_IFMT(identity.st_mode), identity.st_ino)
    return snapshot


def _entry_snapshot(path: Path) -> tuple[object, ...]:
    identity = path.lstat()
    prefix = (stat.S_IFMT(identity.st_mode), identity.st_dev, identity.st_ino)
    if stat.S_ISREG(identity.st_mode):
        return (*prefix, path.read_bytes())
    if stat.S_ISLNK(identity.st_mode):
        return (*prefix, os.readlink(path))
    return prefix


def _plant_foreign_entry(target: Path, kind: str, outside: Path) -> tuple[object, ...]:
    if kind == "regular":
        target.write_bytes(b"foreign-publication-evidence")
    elif kind == "symlink":
        try:
            target.symlink_to(outside / "marker")
        except OSError as error:
            pytest.skip(f"cannot create a file symlink on this platform: {error}")
    elif kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO entries are unavailable on this platform")
        os.mkfifo(target)
    else:  # pragma: no cover - closed test parameter set
        raise AssertionError(kind)
    return _entry_snapshot(target)


def _make_directory_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"cannot create a directory link: {error}")
        completed = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            capture_output=True,
            check=False,
            text=True,
        )
        if completed.returncode:
            pytest.skip(f"cannot create a directory junction: {completed.stderr}")


def _make_file_link(link: Path, target: Path) -> None:
    try:
        link.symlink_to(target)
    except OSError as error:
        pytest.skip(f"cannot create a file symlink on this platform: {error}")


@contextmanager
def _persistent_windows_lock(path: Path):
    with portable_exclusive_lock(path):
        yield
    path.touch(exist_ok=True)


def _pending_publication_target(directory: Path, boundary: str) -> Path:
    generation_suffix = {
        "hqrc-generation-npz-prewrite": ".npz",
        "hqrc-generation-metadata-prewrite": ".json",
    }.get(boundary)
    if generation_suffix is not None:
        generation_dir = directory / ".hqrc_data.generations"
        candidates = [
            path
            for path in generation_dir.iterdir()
            if path.name.startswith(".")
            and path.name.endswith(".tmp")
            and f"{generation_suffix}." in path.name
        ]
        assert len(candidates) == 1
        encoded_target, _nonce, suffix = candidates[0].name.rsplit(".", 2)
        assert suffix == "tmp"
        return generation_dir / encoded_target.removeprefix(".")
    names = {
        "hqrc-data-prewrite": "hqrc_data.current.json",
        "posterior-prewrite": "posterior.nc",
        "posterior-checkpoint-prewrite": "posterior.checkpoint.json",
        "hourly_predictions-prewrite": "hourly_predictions.parquet",
        "metrics-prewrite": "metrics.parquet",
        "posterior-summary-prewrite": "posterior_summary.json",
        "manifest-prewrite": "manifest.json",
        "complete-prewrite": "COMPLETE",
    }
    return directory / names[boundary]


def _publication_temporaries(directory: Path) -> list[Path]:
    return [
        path
        for path in directory.rglob("*")
        if path.name.startswith(".") and path.name.endswith(".tmp")
    ]


def _attacker_rewrite_json(path: Path, value: dict[str, object]) -> None:
    """Rewrite and rehash evidence through the attacker-controlled pathname."""

    path.write_bytes(loeo_contract_module.canonical_json(value) + b"\n")


def _single_checkpoint(root: Path) -> Path:
    checkpoints = list(root.rglob("posterior.checkpoint.json"))
    assert len(checkpoints) == 1
    return checkpoints[0].parent


def _install_fake_sampler(monkeypatch: pytest.MonkeyPatch, source, publication, approved):
    calls: list[dict[str, object]] = []

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
        calls.append(kwargs)
        return _fake_idata(
            current_inputs,
            chains=kwargs["chains"],
            draws=kwargs["draws"],
            sampler={
                "draws": kwargs["draws"],
                "tune": kwargs["tune"],
                "chains": kwargs["chains"],
                "cores": kwargs["cores"],
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
    return calls


def _smoke_fit_kwargs(output_root: Path, *, seed: int = 71) -> dict[str, object]:
    return {
        "held_out_occurrence_id": "seollal-2024",
        "sampler_seed": seed,
        "profile": "smoke",
        "draws": 4,
        "tune": 3,
        "chains": 2,
        "cores": 1,
        "output_root": output_root,
    }


@pytest.fixture
def approved_fold(source: ValidatedCorrectionSource, tmp_path: Path, monkeypatch):
    if os.name == "nt":
        monkeypatch.setattr(
            loeo_diagnostics_module, "_fsync_directory", portable_fsync_directory
        )
        monkeypatch.setattr(loeo_diagnostics_module, "_fsync_file", portable_fsync_file)
        monkeypatch.setattr(loeo_ar_module, "_fsync_directory", portable_fsync_directory)
        monkeypatch.setattr(ar_diagnostics_module, "exclusive_lock", _persistent_windows_lock)
        monkeypatch.setattr(loeo_ar_module, "exclusive_lock", _persistent_windows_lock)
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
        calls.append(occurrence_id)
        assert self is not approved
        return original.__func__(self, occurrence_id)

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
    assert inputs.approved_set is not approved
    assert inputs.approved.calibration.event_ids == tuple(sorted(inputs.hqrc_data.occurrence_ids))
    for index in range(9):
        positions = np.flatnonzero(inputs.hqrc_data.occurrence_index == index)
        assert positions.size > 1
        assert np.all(np.diff(positions) == 1)


@pytest.mark.parametrize(
    ("held_out", "expected_scale", "evaluation_split", "scale_source_split"),
    [
        ("seollal-2020", 20.0, "oof-2020", "oof-2020"),
        ("seollal-2021", 21.0, "oof-2021", "oof-2021"),
        ("seollal-2022", 22.0, "oof-2022", "oof-2022"),
        ("seollal-2023", 23.0, "oof-2023", "oof-2023"),
        ("seollal-2024", 23.0, "final-2024", "oof-2023"),
        ("chuseok-2024", 23.0, "final-2024", "oof-2023"),
    ],
)
def test_evaluation_scale_is_the_unique_source_frozen_fold_scale(
    approved_fold,
    held_out: str,
    expected_scale: float,
    evaluation_split: str,
    scale_source_split: str,
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
    assert inputs.evaluation_split_id == evaluation_split
    assert inputs.scale_source_split_id == scale_source_split
    identity = loeo_publication_module.input_identity(inputs, {"root_seed": 17})
    assert identity["evaluation_scale"] == {
        "evaluation_split_id": evaluation_split,
        "scale_source_split_id": scale_source_split,
        "sigma_n_mw": expected_scale,
    }


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


def test_preparation_rejects_retained_token_approval_wrapper_provenance_substitution(
    approved_fold, tmp_path: Path
) -> None:
    source, publication, approved = approved_fold
    held_out = "seollal-2024"
    other = "seollal-2020"
    replacement_calibrations = dict(approved.calibrations)
    replacement_calibrations[other] = approved.calibrations["chuseok-2020"]
    substitutions = {
        "approved_paths": {
            **approved.approved_paths,
            held_out: tmp_path / "forged-approved.json",
        },
        "approved_sha256": {**approved.approved_sha256, held_out: "0" * 64},
        "training_sha256": {**approved.training_sha256, other: "1" * 64},
        "proposal_sha256": {**approved.proposal_sha256, other: "2" * 64},
        "proposal_digests": {**approved.proposal_digests, other: "3" * 64},
        "plot_sha256": {**approved.plot_sha256, other: "4" * 64},
        "calibrations": replacement_calibrations,
    }
    for field, value in substitutions.items():
        forged = replace(approved, **{field: value})
        assert forged._token is approved._token
        with pytest.raises(LOEOFoldError, match="approved LOEO AR set|revalidation"):
            prepare_loeo_fold_inputs(
                source,
                publication,
                forged,
                held_out_occurrence_id=held_out,
            )
    untrusted = replace(approved, _token=object())
    with pytest.raises(LOEOFoldError, match="approved LOEO AR set|revalidation"):
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


def _mutate_exact_posterior(idata: az.InferenceData, mutation: str) -> None:
    posterior = idata.posterior
    if mutation == "extra-variable":
        posterior["nu_minus_two"] = (("chain", "draw"), np.ones((2, 4)))
    elif mutation == "wrong-event-dimension":
        posterior["event_log_likelihood"] = posterior["event_log_likelihood"].rename(
            {"event_log_likelihood_dim_0": "wrong_event"}
        )
    elif mutation == "packed-covariance":
        posterior["between_cov_0"] = posterior["between_cov_0"] + 0.25
    elif mutation == "negative-covariance":
        changed = posterior["between_cholesky"].values.copy()
        changed[..., 0, 0, 0] *= -1.0
        posterior["between_cholesky"] = (posterior["between_cholesky"].dims, changed)
    elif mutation == "gamma-innovation":
        posterior["gamma_innovation"] = posterior["gamma_innovation"] + 0.25
    elif mutation == "gamma-profile":
        posterior["gamma"] = posterior["gamma"] + 0.25
    elif mutation == "beta":
        posterior["beta"] = posterior["beta"] + 0.25
    elif mutation == "event-log-likelihood":
        posterior["event_log_likelihood"] = posterior["event_log_likelihood"] + 0.25
    elif mutation == "log-likelihood-mirror":
        idata.log_likelihood["event"] = idata.log_likelihood["event"] + 0.25
    else:
        raise AssertionError(f"unknown posterior mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    [
        "extra-variable",
        "wrong-event-dimension",
        "negative-covariance",
        "packed-covariance",
        "gamma-innovation",
        "gamma-profile",
        "beta",
        "event-log-likelihood",
        "log-likelihood-mirror",
    ],
)
def test_h3_posterior_rejects_exact_schema_and_deterministic_mutations(
    approved_fold, mutation: str
) -> None:
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id="seollal-2024",
    )
    idata = _fake_idata(inputs)
    _mutate_exact_posterior(idata, mutation)
    with pytest.raises(LOEOFoldError, match="posterior|likelihood|covariance|gamma|beta"):
        loeo_posterior_module.validate_h3_posterior(idata, inputs)


def test_event_reset_ar1_event_terms_sum_to_existing_scalar_contract() -> None:
    segments = (np.asarray([0.2, -0.1, 0.3]), np.asarray([-0.5, 0.4]))
    terms = event_reset_ar1_logp_numpy(segments, 0.35, 0.8)
    assert terms.shape == (2,)
    assert terms.sum() == pytest.approx(stationary_ar1_logp_numpy(segments, 0.35, 0.8))


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


@pytest.mark.parametrize(
    ("variant", "expected"),
    (
        ("H1", lambda tau: np.full(tau.size, 1.0)),
        ("H2", lambda tau: 1.0 + 0.5 * tau + 0.1 * tau**2),
    ),
)
def test_h1_h2_products_apply_the_declared_shape_without_hour_profile(
    approved_fold,
    monkeypatch: pytest.MonkeyPatch,
    variant: str,
    expected,
) -> None:
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id="seollal-2024",
    )
    idata = _fake_idata(inputs)
    monkeypatch.setattr(
        loeo_products_module,
        "draw_new_event_correction",
        lambda _posterior, **kwargs: np.broadcast_to(
            np.asarray([1.0, 0.5, 0.1]), (kwargs["draws"], 3)
        ).copy(),
    )
    monkeypatch.setattr(
        loeo_products_module,
        "simulate_stationary_ar1",
        lambda *, phi, sigma, horizon, seed: np.zeros((np.asarray(phi).size, horizon)),
    )
    products = generate_loeo_fold_products(
        inputs,
        idata,
        variant=variant,
        predictive_seed=913,
        predictive_draws=8,
    )
    tau = inputs.held_out.frame["tau_days"].to_numpy()
    np.testing.assert_allclose(products.hourly_predictions["q_mean_standardized"], expected(tau))
    assert products.posterior_summary["variant"] == variant


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
                "cores": kwargs["cores"],
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
    _make_file_link(recovered.metrics_path, target)
    with pytest.raises(LOEOFoldError, match="unsafe|manifest"):
        load_loeo_fold_result(source, publication, approved, **fit_kwargs)
    recovered.metrics_path.unlink()
    recovered.metrics_path.write_bytes(metrics_bytes)

    tamper_kwargs = {**fit_kwargs, "sampler_seed": 73}
    tampered = fit_loeo_fold(source, publication, approved, **tamper_kwargs)
    tamper_inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id="seollal-2024",
    )
    tampered_idata = az.from_netcdf(tampered.posterior_path)
    tampered_idata.posterior["phi"] = tampered_idata.posterior["phi"] + 0.05
    replacement_posterior = tmp_path / "rehashed-mutated-posterior.nc"
    az.to_netcdf(tampered_idata, replacement_posterior)
    tampered_idata.close()
    os.replace(replacement_posterior, tampered.posterior_path)
    checkpoint = json.loads((tampered.output_dir / "posterior.checkpoint.json").read_bytes())
    checkpoint["posterior_sha256"] = file_sha256(tampered.posterior_path)
    _attacker_rewrite_json(tampered.output_dir / "posterior.checkpoint.json", checkpoint)
    mutated_idata = az.from_netcdf(tampered.posterior_path)
    try:
        original_manifest = json.loads(tampered.manifest_path.read_bytes())
        mutated_products = generate_loeo_fold_products(
            tamper_inputs,
            mutated_idata,
            predictive_seed=original_manifest["identity"]["predictive_seed"],
            predictive_draws=8,
        )
    finally:
        mutated_idata.close()
    mutated_products.hourly_predictions.write_parquet(tampered.hourly_predictions_path)
    mutated_products.metrics.write_parquet(tampered.metrics_path)
    _attacker_rewrite_json(tampered.posterior_summary_path, mutated_products.posterior_summary)
    with loeo_publication_module.fold_lock(tampered.output_dir) as tampered_handle:
        rehashed_manifest = loeo_publication_module.manifest_payload(
            tampered_handle,
            identity=original_manifest["identity"],
            products=mutated_products,
        )
    _attacker_rewrite_json(tampered.manifest_path, rehashed_manifest)
    _attacker_rewrite_json(
        tampered.output_dir / "COMPLETE",
        {
            "manifest_sha256": file_sha256(tampered.manifest_path),
            "state": "COMPLETE",
            "causal": False,
        },
    )
    with pytest.raises(LOEOFoldError, match="posterior|phi|semantics"):
        load_loeo_fold_result(source, publication, approved, **tamper_kwargs)

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


def test_recovery_resamples_from_verified_input_only_checkpoint(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash after input publication must resume sampling, not reject the cache."""

    source, publication, approved = approved_fold
    calls = _install_fake_sampler(monkeypatch, source, publication, approved)
    kwargs = _smoke_fit_kwargs(tmp_path / "input-only")
    interrupted = True

    def interrupt(name: str) -> None:
        nonlocal interrupted
        if name == "hqrc-data-published" and interrupted:
            interrupted = False
            raise OSError("leave input checkpoint")

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", interrupt)
    with pytest.raises(OSError, match="leave input checkpoint"):
        fit_loeo_fold(source, publication, approved, **kwargs)
    assert calls == []
    directory = next((tmp_path / "input-only").rglob("hqrc_data.current.json")).parent
    assert {entry.name for entry in directory.iterdir()} == {
        ".loeo-fold.lock",
        ".hqrc_data.generations",
        "hqrc_data.current.json",
    }
    cache_before = _tree_snapshot(directory)

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", lambda _name: None)
    recovered = fit_loeo_fold(source, publication, approved, **kwargs)
    assert recovered.sampler_fit_count == 1 and recovered.reused is False
    assert len(calls) == 1
    for name in (".hqrc_data.generations", "hqrc_data.current.json"):
        assert _tree_snapshot(directory)[name] == cache_before[name]


def test_fully_rehashed_checkpoint_rejects_every_exact_posterior_semantic_mutation(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    calls = _install_fake_sampler(monkeypatch, source, publication, approved)
    kwargs = _smoke_fit_kwargs(tmp_path / "fully-rehashed", seed=811)
    fitted = fit_loeo_fold(source, publication, approved, **kwargs)
    originals = {
        name: (fitted.output_dir / name).read_bytes()
        for name in ("posterior.nc", "posterior.checkpoint.json", "manifest.json", "COMPLETE")
    }
    mutations = (
        "extra-variable",
        "wrong-event-dimension",
        "negative-covariance",
        "packed-covariance",
        "gamma-innovation",
        "gamma-profile",
        "beta",
        "event-log-likelihood",
        "log-likelihood-mirror",
    )
    for mutation in mutations:
        for name, payload in originals.items():
            (fitted.output_dir / name).write_bytes(payload)
        idata = az.from_netcdf(fitted.posterior_path).load()
        try:
            _mutate_exact_posterior(idata, mutation)
            replacement = tmp_path / f"{mutation}.nc"
            az.to_netcdf(idata, replacement)
        finally:
            idata.close()
        os.replace(replacement, fitted.posterior_path)
        checkpoint = json.loads(originals["posterior.checkpoint.json"])
        checkpoint["posterior_sha256"] = file_sha256(fitted.posterior_path)
        _attacker_rewrite_json(fitted.output_dir / "posterior.checkpoint.json", checkpoint)
        manifest = json.loads(originals["manifest.json"])
        manifest["outputs"]["posterior"]["sha256"] = file_sha256(fitted.posterior_path)
        manifest["outputs"]["posterior_checkpoint"]["sha256"] = file_sha256(
            fitted.output_dir / "posterior.checkpoint.json"
        )
        unsigned = {
            key: manifest[key]
            for key in ("schema_version", "state", "causal", "identity", "outputs", "rows")
        }
        manifest["manifest_digest"] = loeo_contract_module.sha_json(unsigned)
        _attacker_rewrite_json(fitted.manifest_path, manifest)
        _attacker_rewrite_json(
            fitted.output_dir / "COMPLETE",
            {
                "manifest_sha256": file_sha256(fitted.manifest_path),
                "state": "COMPLETE",
                "causal": False,
            },
        )
        before = _tree_snapshot(fitted.output_dir)
        with pytest.raises(LOEOFoldError, match="posterior|likelihood|covariance|gamma|beta"):
            load_loeo_fold_result(source, publication, approved, **kwargs)
        assert _tree_snapshot(fitted.output_dir) == before
    assert len(calls) == 1


@pytest.mark.parametrize(
    ("boundary", "expected_existing"),
    [
        ("hourly_predictions-published", ("hourly_predictions.parquet",)),
        (
            "metrics-published",
            ("hourly_predictions.parquet", "metrics.parquet"),
        ),
        (
            "posterior-summary-published",
            ("hourly_predictions.parquet", "metrics.parquet", "posterior_summary.json"),
        ),
        (
            "manifest-published",
            (
                "hourly_predictions.parquet",
                "metrics.parquet",
                "posterior_summary.json",
                "manifest.json",
            ),
        ),
    ],
)
def test_recovery_preserves_verified_downstream_prefix_without_overwrite(
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_existing: tuple[str, ...],
) -> None:
    source, publication, approved = approved_fold
    calls = _install_fake_sampler(monkeypatch, source, publication, approved)
    interrupted = True

    def interrupt(name: str) -> None:
        nonlocal interrupted
        if name == boundary and interrupted:
            interrupted = False
            raise OSError("leave verified prefix")

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", interrupt)
    kwargs = _smoke_fit_kwargs(tmp_path / boundary)
    with pytest.raises(OSError, match="verified prefix"):
        fit_loeo_fold(source, publication, approved, **kwargs)
    directory = _single_checkpoint(tmp_path / boundary)
    before = {
        name: (directory / name).lstat().st_ino
        for name in expected_existing
        if (directory / name).exists()
    }
    assert tuple(before) == expected_existing
    monkeypatch.setattr(loeo_publication_module, "publication_boundary", lambda _name: None)
    recovered = fit_loeo_fold(source, publication, approved, **kwargs)
    assert recovered.sampler_fit_count == 0 and len(calls) == 1
    assert {name: (directory / name).lstat().st_ino for name in expected_existing} == before


@pytest.mark.parametrize(
    ("kind", "filename"),
    [
        ("foreign", "hourly_predictions.parquet"),
        ("gap", "metrics.parquet"),
        ("manifest_only", "manifest.json"),
    ],
)
def test_recovery_rejects_and_preserves_unverified_downstream_evidence(
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
    filename: str,
) -> None:
    source, publication, approved = approved_fold
    calls = _install_fake_sampler(monkeypatch, source, publication, approved)
    interrupted = True

    def interrupt(name: str) -> None:
        nonlocal interrupted
        if name == "posterior-checkpointed" and interrupted:
            interrupted = False
            raise OSError("leave checkpoint")

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", interrupt)
    kwargs = _smoke_fit_kwargs(tmp_path / kind)
    with pytest.raises(OSError, match="leave checkpoint"):
        fit_loeo_fold(source, publication, approved, **kwargs)
    directory = _single_checkpoint(tmp_path / kind)
    (directory / filename).write_bytes(f"foreign-{kind}".encode())
    before = _tree_snapshot(directory)
    monkeypatch.setattr(loeo_publication_module, "publication_boundary", lambda _name: None)
    with pytest.raises(LOEOFoldError, match="partial|prefix|downstream|product|manifest"):
        fit_loeo_fold(source, publication, approved, **kwargs)
    assert _tree_snapshot(directory) == before
    assert len(calls) == 1


def _namespace_material(approved_fold, root: Path):
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source,
        publication,
        approved,
        held_out_occurrence_id="seollal-2024",
    )
    sampler = loeo_publication_module.sampler_contract(
        "smoke",
        root_seed=71,
        held_out_occurrence_id="seollal-2024",
        draws=4,
        tune=3,
        chains=2,
    )
    identity = loeo_publication_module.input_identity(inputs, sampler)
    return publication, inputs, sampler, identity


def test_sampler_cores_are_resolved_and_bound_to_loeo_identity(approved_fold, tmp_path: Path):
    _, _, default_sampler, default_identity = _namespace_material(approved_fold, tmp_path)
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source, publication, approved, held_out_occurrence_id="seollal-2024"
    )
    multi_core_sampler = loeo_publication_module.sampler_contract(
        "smoke",
        root_seed=71,
        held_out_occurrence_id="seollal-2024",
        draws=4,
        tune=3,
        chains=4,
        cores=4,
    )
    assert default_sampler["cores"] == 1
    assert multi_core_sampler["cores"] == 4
    assert loeo_publication_module.input_identity(inputs, multi_core_sampler) != default_identity
    with pytest.raises(LOEOFoldError, match="cores"):
        loeo_publication_module.sampler_contract(
            "smoke",
            root_seed=71,
            held_out_occurrence_id="seollal-2024",
            draws=4,
            tune=3,
            chains=2,
            cores=4,
        )


def test_paper_v2_jitter_sampler_contract_is_identity_bound(approved_fold) -> None:
    source, publication, approved = approved_fold
    inputs = prepare_loeo_fold_inputs(
        source, publication, approved, held_out_occurrence_id="seollal-2024"
    )
    baseline = loeo_publication_module.sampler_contract(
        "paper",
        root_seed=71,
        held_out_occurrence_id="seollal-2024",
        draws=5000,
        tune=5000,
        chains=4,
        cores=4,
    )
    v2 = loeo_publication_module.sampler_contract(
        "paper",
        root_seed=71,
        held_out_occurrence_id="seollal-2024",
        draws=5000,
        tune=5000,
        chains=4,
        cores=4,
        init="jitter+adapt_diag",
        target_accept=0.99,
    )
    assert v2["init"] == "jitter+adapt_diag"
    assert v2["target_accept"] == 0.99
    assert loeo_publication_module.input_identity(inputs, v2) != (
        loeo_publication_module.input_identity(inputs, baseline)
    )
    with pytest.raises(LOEOFoldError, match="init"):
        loeo_publication_module.sampler_contract(
            "paper",
            root_seed=71,
            held_out_occurrence_id="seollal-2024",
            draws=5000,
            tune=5000,
            chains=4,
            cores=4,
            init="advi",
        )
    with pytest.raises(LOEOFoldError, match="target_accept"):
        loeo_publication_module.sampler_contract(
            "paper",
            root_seed=71,
            held_out_occurrence_id="seollal-2024",
            draws=5000,
            tune=5000,
            chains=4,
            cores=4,
            target_accept=0.95,
        )


def test_namespace_rejects_intermediate_symlink_without_touching_external_target(
    approved_fold, tmp_path: Path
) -> None:
    publication, inputs, sampler, identity = _namespace_material(approved_fold, tmp_path / "unused")
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_bytes(b"untouched")
    root = tmp_path / "symlink-root"
    root.mkdir()
    _make_directory_link(root / "loeo-h3", outside)
    with pytest.raises(LOEOFoldError, match="unsafe|symlink|namespace"):
        loeo_publication_module.namespace(root, inputs, sampler, identity)
    assert marker.read_bytes() == b"untouched"
    assert not (outside / publication.context.model).exists()


def test_fit_rejects_intermediate_swap_after_namespace_without_touching_outside(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    calls = _install_fake_sampler(monkeypatch, source, publication, approved)
    output_root = tmp_path / "results"
    outside = tmp_path / "outside"
    outside.mkdir()
    outside_before: list[dict[str, tuple[object, ...]]] = []
    original_namespace = loeo_publication_module.namespace

    def swap_after_namespace(*args, **kwargs):
        namespace = original_namespace(*args, **kwargs)
        directory = getattr(namespace, "path", namespace)
        original_component = output_root / "loeo-h3"
        moved_component = outside / "loeo-h3"
        original_component.rename(moved_component)
        _make_directory_link(original_component, moved_component)
        assert Path(directory).is_relative_to(original_component)
        outside_before.append(_tree_snapshot(outside))
        return namespace

    monkeypatch.setattr(loeo_publication_module, "namespace", swap_after_namespace)
    with pytest.raises(LOEOFoldError, match="namespace|changed|unsafe"):
        fit_loeo_fold(source, publication, approved, **_smoke_fit_kwargs(output_root))
    assert len(outside_before) == 1
    assert _tree_snapshot(outside) == outside_before[0]
    assert calls == []


def test_fit_rejects_intermediate_swap_at_boundary_before_sampler_or_outside_write(
    approved_fold, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, publication, approved = approved_fold
    calls = _install_fake_sampler(monkeypatch, source, publication, approved)
    output_root = tmp_path / "results"
    outside = tmp_path / "outside"
    outside.mkdir()
    marker = outside / "marker"
    marker.write_bytes(b"untouched")
    outside_before = _tree_snapshot(outside)
    swapped = False

    def swap_at_boundary(name: str) -> None:
        nonlocal swapped
        if name != "hqrc-data-published" or swapped:
            return
        swapped = True
        component = output_root / "loeo-h3"
        component.rename(output_root / "quarantined-loeo-h3")
        _make_directory_link(component, outside)

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", swap_at_boundary)
    with pytest.raises(LOEOFoldError, match="namespace|changed|unsafe"):
        fit_loeo_fold(source, publication, approved, **_smoke_fit_kwargs(output_root))
    assert swapped is True
    assert _tree_snapshot(outside) == outside_before
    assert len(calls) == 0


@pytest.mark.parametrize(
    ("boundary", "expected_sampler_calls"),
    [
        ("hqrc-data-prewrite", 0),
        ("posterior-prewrite", 1),
        ("hourly_predictions-prewrite", 1),
        ("manifest-prewrite", 1),
        ("complete-prewrite", 1),
    ],
)
def test_every_publication_class_uses_held_directory_after_prewrite_namespace_swap(
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    expected_sampler_calls: int,
) -> None:
    source, publication, approved = approved_fold
    calls = _install_fake_sampler(monkeypatch, source, publication, approved)
    output_root = tmp_path / boundary
    outside = tmp_path / f"outside-{boundary}"
    outside.mkdir()
    (outside / "marker").write_bytes(b"untouched")
    outside_before = _tree_snapshot(outside)
    swapped = False

    def swap_before_write(name: str) -> None:
        nonlocal swapped
        if name != boundary or swapped:
            return
        swapped = True
        component = output_root / "loeo-h3"
        component.rename(output_root / "quarantined-loeo-h3")
        _make_directory_link(component, outside)

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", swap_before_write)
    with pytest.raises(LOEOFoldError, match="namespace|changed|unsafe"):
        fit_loeo_fold(source, publication, approved, **_smoke_fit_kwargs(output_root))
    assert swapped is True
    assert _tree_snapshot(outside) == outside_before
    assert len(calls) == expected_sampler_calls


@pytest.mark.parametrize("kind", ["regular", "symlink", "fifo"])
def test_publication_primitive_never_replaces_foreign_target_inserted_at_prewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    directory = tmp_path / kind
    directory.mkdir()
    outside = tmp_path / f"outside-{kind}"
    outside.mkdir()
    (outside / "marker").write_bytes(b"outside-unchanged")
    outside_before = _tree_snapshot(outside)
    target = directory / "artifact.bin"
    planted: list[tuple[object, ...]] = []

    def insert_target(name: str) -> None:
        assert name == "primitive-prewrite"
        planted.append(_plant_foreign_entry(target, kind, outside))

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", insert_target)
    with loeo_publication_module.fold_lock(directory) as publication_handle:
        with pytest.raises(LOEOFoldError, match="publication failed safely"):
            loeo_publication_module._publish_bytes(
                publication_handle,
                target.name,
                b"implementation-payload",
                boundary="primitive-prewrite",
            )
    assert len(planted) == 1
    assert _entry_snapshot(target) == planted[0]
    assert _tree_snapshot(outside) == outside_before
    assert _publication_temporaries(directory) == []


def test_windows_backend_fold_publication_writes_loads_validates_and_reuses(tmp_path):
    namespace = loeo_publication_module.secure_namespace(
        (tmp_path / "results").resolve(), ("fold",)
    )
    payload = {"schema_version": 1, "state": "COMPLETE"}

    with loeo_publication_module.fold_lock(namespace, backend="windows") as publication:
        loeo_publication_module.publish_json(
            publication, "manifest.json", payload, boundary="manifest-prewrite"
        )
        loeo_publication_module.guard_namespace(publication)
        assert loeo_publication_module._relative_json(
            publication.directory, "manifest.json", "manifest"
        ) == payload
        directory_identity = publication.directory.identity
        artifact_identity = (publication.path / "manifest.json").stat().st_ino

    with loeo_publication_module.fold_lock(namespace, backend="windows") as publication:
        loeo_publication_module.guard_namespace(publication)
        assert publication.directory.identity == directory_identity
        assert loeo_publication_module.publication_has(publication, "manifest.json")
        assert loeo_publication_module._relative_json(
            publication.directory, "manifest.json", "manifest"
        ) == payload
        assert (publication.path / "manifest.json").stat().st_ino == artifact_identity
    assert _publication_temporaries(namespace.path) == []


@pytest.mark.parametrize(
    ("boundary", "kind", "expected_sampler_calls"),
    [
        ("hqrc-generation-npz-prewrite", "regular", 0),
        ("hqrc-generation-metadata-prewrite", "symlink", 0),
        ("hqrc-data-prewrite", "fifo", 0),
        ("posterior-prewrite", "regular", 1),
        ("posterior-checkpoint-prewrite", "symlink", 1),
        ("hourly_predictions-prewrite", "fifo", 1),
        ("metrics-prewrite", "regular", 1),
        ("posterior-summary-prewrite", "symlink", 1),
        ("manifest-prewrite", "fifo", 1),
        ("complete-prewrite", "regular", 1),
    ],
)
def test_every_publication_boundary_preserves_foreign_target_inserted_at_prewrite(
    approved_fold,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
    kind: str,
    expected_sampler_calls: int,
) -> None:
    source, publication, approved = approved_fold
    calls = _install_fake_sampler(monkeypatch, source, publication, approved)
    output_root = tmp_path / boundary
    outside = tmp_path / f"outside-{boundary}"
    outside.mkdir()
    (outside / "marker").write_bytes(b"outside-unchanged")
    outside_before = _tree_snapshot(outside)
    target: list[Path] = []
    planted: list[tuple[object, ...]] = []

    def insert_target(name: str) -> None:
        if name != boundary or planted:
            return
        locks = list(output_root.rglob(".loeo-fold.lock"))
        assert len(locks) == 1
        target.append(_pending_publication_target(locks[0].parent, boundary))
        planted.append(_plant_foreign_entry(target[0], kind, outside))

    monkeypatch.setattr(loeo_publication_module, "publication_boundary", insert_target)
    with pytest.raises(LOEOFoldError, match="publication failed safely"):
        fit_loeo_fold(source, publication, approved, **_smoke_fit_kwargs(output_root))
    assert len(target) == len(planted) == 1
    assert _entry_snapshot(target[0]) == planted[0]
    assert _tree_snapshot(outside) == outside_before
    assert _publication_temporaries(target[0].parents[1]) == []
    assert len(calls) == expected_sampler_calls


@pytest.mark.parametrize("kind", ["dangling", "fifo", "special"])
def test_fold_lock_rejects_dangling_fifo_and_special_without_escape_or_block(
    approved_fold, tmp_path: Path, kind: str
) -> None:
    del approved_fold
    directory = Path(tempfile.mkdtemp(prefix="hqrc-lock-", dir=tmp_path))
    try:
        lock_path = directory / ".loeo-fold.lock"
        if kind == "dangling":
            external_lock = tmp_path / "external-lock"
            _make_file_link(lock_path, external_lock)
            with pytest.raises(LOEOFoldError, match="lock|unsafe"):
                with loeo_publication_module.fold_lock(directory):
                    pass
            assert not external_lock.exists()
            assert lock_path.is_symlink()
            return
        if kind == "fifo":
            if not hasattr(os, "mkfifo"):
                pytest.skip("FIFO entries are unavailable on this platform")
            os.mkfifo(lock_path)
            before = lock_path.lstat()
            command = (
                "from pathlib import Path\n"
                "from time import monotonic\n"
                "from hqrc_v3._loeo_publication import fold_lock\n"
                "from hqrc_v3._loeo_types import LOEOFoldError\n"
                f"p = Path({str(directory)!r})\n"
                "started = monotonic()\n"
                "try:\n"
                "    with fold_lock(p):\n"
                "        pass\n"
                "except LOEOFoldError:\n"
                "    print(f'LOEO_FOLD_ERROR {monotonic() - started:.9f}')\n"
                "    raise SystemExit(23)\n"
                "raise SystemExit(0)\n"
            )
            completed = subprocess.run(
                [sys.executable, "-c", command],
                cwd=Path(__file__).parents[2],
                env={**os.environ, "PYTHONPATH": str(Path(__file__).parents[2] / "src")},
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            assert completed.returncode == 23
            marker, elapsed_text = completed.stdout.strip().split()
            assert marker == "LOEO_FOLD_ERROR"
            assert float(elapsed_text) < 1.0
            after = lock_path.lstat()
            assert stat.S_ISFIFO(after.st_mode) and after.st_ino == before.st_ino
            return
        lock_path.mkdir()
        before = lock_path.lstat()
        with pytest.raises(LOEOFoldError, match="lock|unsafe"):
            with loeo_publication_module.fold_lock(directory):
                pass
        after = lock_path.lstat()
        assert stat.S_ISDIR(after.st_mode) and after.st_ino == before.st_ino
    finally:
        for child in directory.iterdir():
            if child.is_dir() and not child.is_symlink():
                child.rmdir()
            else:
                child.unlink()
        directory.rmdir()


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
                "cores": kwargs["cores"],
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
