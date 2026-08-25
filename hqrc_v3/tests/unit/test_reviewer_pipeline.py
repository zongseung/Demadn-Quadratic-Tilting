"""One-command reviewer-pipeline orchestration contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hqrc_v3.baselines.config import MODEL_NAMES
from hqrc_v3.cli import build_parser
from hqrc_v3.reviewer_pipeline import (
    REVIEWER_FEATURE_SUITE,
    ReviewerPipelineError,
    ReviewerPipelineOptions,
    run_hqt_reviewer_pipeline,
)


def _options(tmp_path: Path, **overrides: object) -> ReviewerPipelineOptions:
    values: dict[str, object] = {
        "matrices": {"B0": object(), "B1W": object()},
        "baseline_config": object(),
        "artifact_hashes": {"data_sha256": "a" * 64},
        "data_path": tmp_path / "power.csv",
        "config_path": tmp_path / "experiment.toml",
        "model_config_path": tmp_path / "models.toml",
        "event_registry_path": tmp_path / "events.csv",
        "holiday_calendar_path": tmp_path / "holidays.csv",
        "temporary_holiday_availability_path": tmp_path / "availability.csv",
        "source_run_dir": tmp_path / "source",
        "cache_dir": tmp_path / "cache",
        "output_root": tmp_path / "results",
        "baseline_seed": 7,
        "root_seed": 20260813,
        "models": ("xgboost",),
        "profile": "smoke",
        "draws": 5,
        "tune": 5,
        "chains": 2,
        "cores": 2,
        "init": "adapt_diag",
        "target_accept": 0.9,
        "smoke_boosting_rounds": 2,
    }
    values.update(overrides)
    return ReviewerPipelineOptions(**values)  # type: ignore[arg-type]


def _install_stage_recorders(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> list[tuple[str, dict[str, object]]]:
    from hqrc_v3 import reviewer_pipeline as module

    calls: list[tuple[str, dict[str, object]]] = []

    def oof(**kwargs: object) -> object:
        calls.append(("oof", kwargs))
        return SimpleNamespace(fit_count=4, cache_hit_count=3)

    def final(**kwargs: object) -> object:
        calls.append(("final", kwargs))
        return SimpleNamespace(
            fit_count=1,
            cache_hit_count=2,
            point_path=tmp_path / "source/predictions/final_2024.parquet",
        )

    def residual(**kwargs: object) -> object:
        calls.append(("residual", kwargs))
        return SimpleNamespace(reused=False)

    def loeo(**kwargs: object) -> object:
        calls.append(("loeo", kwargs))
        return (
            SimpleNamespace(
                context=SimpleNamespace(model="xgboost", feature_set="B1W"),
                output_dir=tmp_path / "results/retrospective-loeo/xgboost/B1W",
                sampler_fit_count=10,
                reused_fold_count=0,
            ),
        )

    def causal(**kwargs: object) -> object:
        calls.append(("causal", kwargs))
        return (
            SimpleNamespace(
                context=SimpleNamespace(model="xgboost", feature_set="B1W"),
                output_dir=tmp_path / "results/causal-2024/xgboost/B1W",
                sampler_fit_count=1,
                reused=False,
            ),
        )

    def report(inputs: object, *, output_root: Path) -> object:
        calls.append(("report", {"inputs": inputs, "output_root": output_root}))
        return SimpleNamespace(output_root=output_root)

    monkeypatch.setattr(module, "run_paper_oof_stage", oof)
    monkeypatch.setattr(module, "run_paper_final_stage", final)
    monkeypatch.setattr(module, "prepare_standardized_residual_artifact", residual)
    monkeypatch.setattr(module, "run_legacy_hqt_loeo", loeo)
    monkeypatch.setattr(module, "run_legacy_hqt_causal_2024", causal)
    monkeypatch.setattr(module, "build_reviewer_report", report)
    return calls


def test_reviewer_pipeline_runs_exact_stages_in_order(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stage_recorders(monkeypatch, tmp_path)

    result = run_hqt_reviewer_pipeline(_options(tmp_path))

    assert [name for name, _ in calls] == [
        "oof",
        "final",
        "residual",
        "loeo",
        "causal",
        "report",
    ]
    assert result.baseline_fit_count == 5
    assert result.baseline_cache_hit_count == 5
    assert result.hqt_fit_count == 11
    assert result.hqt_reuse_count == 0


def test_reviewer_pipeline_maps_scopes_and_separate_output_roots(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stage_recorders(monkeypatch, tmp_path)

    result = run_hqt_reviewer_pipeline(_options(tmp_path))

    by_name = dict(calls)
    for stage in ("oof", "final"):
        assert by_name[stage]["models"] == ("xgboost",)
        assert by_name[stage]["feature_sets"] == REVIEWER_FEATURE_SUITE
        assert by_name[stage]["run_dir"] == tmp_path / "source"
        assert by_name[stage]["cache_dir"] == tmp_path / "cache"
    assert by_name["loeo"]["feature_sets"] == ("B1W",)
    assert by_name["loeo"]["held_out_occurrence_ids"] is None
    assert by_name["loeo"]["output_root"] == tmp_path / "results/retrospective-loeo"
    assert by_name["causal"]["feature_set"] == "B1W"
    assert by_name["causal"]["output_root"] == tmp_path / "results/causal-2024"
    report_inputs = by_name["report"]["inputs"]
    assert report_inputs.final_predictions_path.name == "final_2024.parquet"
    assert report_inputs.loeo_context_dirs == (tmp_path / "results/retrospective-loeo/xgboost/B1W",)
    assert report_inputs.causal_context_dirs == (tmp_path / "results/causal-2024/xgboost/B1W",)
    assert result.source_run_dir == tmp_path / "source"
    assert result.loeo_root == tmp_path / "results/retrospective-loeo"
    assert result.causal_root == tmp_path / "results/causal-2024"
    assert result.paper_root == tmp_path / "results/paper"


def _reviewer_cli_args(*, profile: str = "smoke", model: str = "xgboost") -> list[str]:
    return [
        "run-hqt-reviewer",
        "--data",
        "power.csv",
        "--config",
        "experiment.toml",
        "--frozen-model-config",
        "models.toml",
        "--frozen-model-hash",
        "a" * 64,
        "--event-registry",
        "events.csv",
        "--holiday-calendar",
        "holidays.csv",
        "--temporary-holiday-availability",
        "availability.csv",
        "--run-dir",
        "source",
        "--cache-dir",
        "cache",
        "--output-root",
        "results",
        "--profile",
        profile,
        "--model",
        model,
        "--draws",
        "5" if profile == "smoke" else "1000",
        "--tune",
        "5" if profile == "smoke" else "1000",
        "--chains",
        "2" if profile == "smoke" else "4",
        "--cores",
        "2" if profile == "smoke" else "4",
        "--smoke-boosting-rounds",
        "2",
    ]


def test_reviewer_pipeline_has_no_ar_approval_option() -> None:
    arguments = build_parser().parse_args(_reviewer_cli_args())

    assert arguments.command == "run-hqt-reviewer"
    assert not hasattr(arguments, "approved_ar")
    assert not hasattr(arguments, "approve_derived_ar")
    assert not hasattr(arguments, "diagnostic_attempts")


def test_paper_profile_requires_all_models_before_any_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stage_recorders(monkeypatch, tmp_path)

    with pytest.raises(ReviewerPipelineError, match="all five"):
        run_hqt_reviewer_pipeline(
            _options(
                tmp_path,
                profile="paper",
                models=("xgboost",),
                draws=1_000,
                tune=1_000,
                chains=4,
                cores=4,
                target_accept=0.99,
                smoke_boosting_rounds=None,
            )
        )

    assert calls == []


def test_paper_profile_forwards_all_models_and_minimum_sampler(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stage_recorders(monkeypatch, tmp_path)

    run_hqt_reviewer_pipeline(
        _options(
            tmp_path,
            profile="paper",
            models=MODEL_NAMES,
            draws=1_000,
            tune=1_000,
            chains=4,
            cores=4,
            target_accept=0.99,
            smoke_boosting_rounds=None,
        )
    )

    by_name = dict(calls)
    assert by_name["oof"]["models"] == MODEL_NAMES
    assert by_name["oof"]["oof_years"] is None
    for stage in ("loeo", "causal"):
        assert by_name[stage]["models"] == MODEL_NAMES
        assert by_name[stage]["draws"] == 1_000
        assert by_name[stage]["tune"] == 1_000
        assert by_name[stage]["chains"] == 4
        assert by_name[stage]["target_accept"] == 0.99


def test_paper_profile_accepts_more_than_four_chains(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stage_recorders(monkeypatch, tmp_path)

    run_hqt_reviewer_pipeline(
        _options(
            tmp_path,
            profile="paper",
            models=MODEL_NAMES,
            draws=1_000,
            tune=1_000,
            chains=5,
            cores=5,
            target_accept=0.99,
            smoke_boosting_rounds=None,
        )
    )

    by_name = dict(calls)
    assert by_name["loeo"]["chains"] == 5
    assert by_name["causal"]["chains"] == 5


def test_paper_default_chains_rejects_oversized_cores_before_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stage_recorders(monkeypatch, tmp_path)

    with pytest.raises(ReviewerPipelineError, match="cores.*chains"):
        run_hqt_reviewer_pipeline(
            _options(
                tmp_path,
                profile="paper",
                models=MODEL_NAMES,
                draws=None,
                tune=None,
                chains=None,
                cores=5,
                target_accept=None,
                smoke_boosting_rounds=None,
            )
        )

    assert calls == []


@pytest.mark.parametrize(
    "overrides",
    [
        {"profile": "smoke", "draws": None},
        {"profile": "smoke", "smoke_boosting_rounds": None},
        {
            "profile": "paper",
            "models": MODEL_NAMES,
            "draws": 999,
            "tune": 1_000,
            "chains": 4,
            "target_accept": 0.99,
            "smoke_boosting_rounds": None,
        },
        {
            "profile": "paper",
            "models": MODEL_NAMES,
            "draws": 1_000,
            "tune": 1_000,
            "chains": 4.5,
            "cores": 4,
            "target_accept": 0.99,
            "smoke_boosting_rounds": None,
        },
    ],
)
def test_invalid_profile_contract_fails_before_stages(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    overrides: dict[str, object],
) -> None:
    calls = _install_stage_recorders(monkeypatch, tmp_path)

    with pytest.raises(ReviewerPipelineError):
        run_hqt_reviewer_pipeline(_options(tmp_path, **overrides))

    assert calls == []


def test_negative_root_seed_fails_before_any_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls = _install_stage_recorders(monkeypatch, tmp_path)

    with pytest.raises(ReviewerPipelineError, match="root seed.*non-negative"):
        run_hqt_reviewer_pipeline(_options(tmp_path, root_seed=-1))

    assert calls == []


def test_pipeline_propagates_stage_failure_and_does_not_run_later_stages(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hqrc_v3 import reviewer_pipeline as module

    calls = _install_stage_recorders(monkeypatch, tmp_path)

    def fail(**kwargs: object) -> object:
        calls.append(("residual-failed", kwargs))
        raise RuntimeError("interrupted publication")

    monkeypatch.setattr(module, "prepare_standardized_residual_artifact", fail)

    with pytest.raises(RuntimeError, match="interrupted"):
        run_hqt_reviewer_pipeline(_options(tmp_path))

    assert [name for name, _ in calls] == ["oof", "final", "residual-failed"]


def test_completed_rerun_aggregates_reuse_across_multiple_contexts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from hqrc_v3 import reviewer_pipeline as module

    _install_stage_recorders(monkeypatch, tmp_path)
    monkeypatch.setattr(
        module,
        "run_paper_oof_stage",
        lambda **kwargs: SimpleNamespace(fit_count=0, cache_hit_count=8),
    )
    monkeypatch.setattr(
        module,
        "run_paper_final_stage",
        lambda **kwargs: SimpleNamespace(
            fit_count=0,
            cache_hit_count=2,
            point_path=tmp_path / "source/predictions/final_2024.parquet",
        ),
    )
    monkeypatch.setattr(
        module,
        "run_legacy_hqt_loeo",
        lambda **kwargs: (
            SimpleNamespace(
                context=SimpleNamespace(model="xgboost", feature_set="B1W"),
                output_dir=tmp_path / "results/retrospective-loeo/xgboost/B1W",
                sampler_fit_count=0,
                reused_fold_count=10,
            ),
            SimpleNamespace(
                context=SimpleNamespace(model="lightgbm", feature_set="B1W"),
                output_dir=tmp_path / "results/retrospective-loeo/lightgbm/B1W",
                sampler_fit_count=0,
                reused_fold_count=10,
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "run_legacy_hqt_causal_2024",
        lambda **kwargs: (
            SimpleNamespace(
                context=SimpleNamespace(model="xgboost", feature_set="B1W"),
                output_dir=tmp_path / "results/causal-2024/xgboost/B1W",
                sampler_fit_count=0,
                reused=True,
            ),
            SimpleNamespace(
                context=SimpleNamespace(model="lightgbm", feature_set="B1W"),
                output_dir=tmp_path / "results/causal-2024/lightgbm/B1W",
                sampler_fit_count=0,
                reused=False,
            ),
        ),
    )
    messages: list[str] = []

    result = run_hqt_reviewer_pipeline(
        _options(tmp_path, models=("xgboost", "lightgbm")), progress=messages.append
    )

    assert result.baseline_fit_count == 0
    assert result.baseline_cache_hit_count == 10
    assert result.hqt_fit_count == 0
    assert result.hqt_reuse_count == 21
    assert "[HQT] xgboost/B1W retrospective LOEO complete fits=0 reused=10" in messages
    assert "[HQT] xgboost/B1W causal-2024 complete fits=0 reused=1" in messages
