"""Unit tests for the portable, model-sequential manuscript runner."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import hqrc_v3.paper_pipeline as pipeline
from hqrc_v3.diagnostics.ar import EventResidualContext


@pytest.fixture
def source() -> SimpleNamespace:
    return SimpleNamespace(
        available_contexts=(
            EventResidualContext("xgboost", "B1", 7, ("oof-2020", "oof-2021")),
            EventResidualContext("lightgbm", "B1", 7, ("oof-2020", "oof-2021")),
        )
    )


def _run(
    source: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    approve_derived_ar: bool,
) -> tuple[tuple[pipeline.PipelineContextResult, ...], list[str]]:
    calls: list[str] = []
    monkeypatch.setattr(pipeline, "validate_correction_source", lambda **_: source)

    def universe(_source, context, *, output_dir):
        calls.append(f"universe:{context.model}")
        return SimpleNamespace(occurrence_ids=("seollal-2024",), output_dir=output_dir)

    def proposal(_source, publication, *, output_dir):
        model = publication.output_dir.parent.parent.parent.name
        calls.append(f"proposal:{model}")
        return SimpleNamespace(proposal_set_sha256="a" * 64)

    def approve(_source, publication, *, output_dir, confirm_proposal_set_sha256):
        model = publication.output_dir.parent.parent.parent.name
        assert confirm_proposal_set_sha256 == "a" * 64
        calls.append(f"approve:{model}")
        return output_dir / "generation/approval/approved-set.json"

    def load(_source, publication, *, output_dir):
        model = publication.output_dir.parent.parent.parent.name
        calls.append(f"load:{model}")
        return SimpleNamespace()

    def primary(_source, publication, approved, **kwargs):
        model = publication.output_dir.parent.parent.parent.name
        assert approved is not None
        assert kwargs["held_out_occurrence_ids"] == publication.occurrence_ids
        calls.append(f"primary:{model}")
        return SimpleNamespace(
            output_dir=tmp_path / model / "primary", sampler_fit_count=1, reused=False
        )

    def ablation(_source, publication, approved, **kwargs):
        model = publication.output_dir.parent.parent.parent.name
        variant = kwargs["variant"]
        assert approved is not None
        assert kwargs["held_out_occurrence_ids"] == publication.occurrence_ids
        calls.append(f"{variant.lower()}:{model}")
        return SimpleNamespace(
            output_dir=tmp_path / model / variant.lower(), sampler_fit_count=1, reused=False
        )

    monkeypatch.setattr(pipeline, "publish_loeo_universe", universe)
    monkeypatch.setattr(pipeline, "prepare_loeo_ar_proposal_set", proposal)
    monkeypatch.setattr(pipeline, "approve_loeo_ar_proposal_set", approve)
    monkeypatch.setattr(pipeline, "load_approved_loeo_ar_set", load)
    monkeypatch.setattr(pipeline, "fit_loeo_ablation", ablation)
    monkeypatch.setattr(pipeline, "fit_loeo_primary", primary)
    return (
        pipeline.run_paper_loeo_pipeline(
            source_run_dir=tmp_path / "source",
            config_path=tmp_path / "experiment.toml",
            output_root=tmp_path / "output",
            models=("lightgbm", "xgboost"),
            feature_sets=("B1",),
            root_seed=71,
            profile="paper",
            draws=1_000,
            tune=1_000,
            chains=4,
            cores=4,
            approve_derived_ar=approve_derived_ar,
        ),
        calls,
    )


def test_pipeline_completes_each_context_before_advancing(
    source: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, calls = _run(source, monkeypatch, tmp_path, approve_derived_ar=True)

    assert [item.context.model for item in result] == ["xgboost", "lightgbm"]
    assert [item.status for item in result] == ["COMPLETE", "COMPLETE"]
    assert calls == [
        "universe:xgboost",
        "proposal:xgboost",
        "approve:xgboost",
        "load:xgboost",
        "h1:xgboost",
        "h2:xgboost",
        "primary:xgboost",
        "universe:lightgbm",
        "proposal:lightgbm",
        "approve:lightgbm",
        "load:lightgbm",
        "h1:lightgbm",
        "h2:lightgbm",
        "primary:lightgbm",
    ]


def test_pipeline_stops_after_acf_proposals_without_explicit_approval(
    source: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result, calls = _run(source, monkeypatch, tmp_path, approve_derived_ar=False)

    assert [item.status for item in result] == ["AR_REVIEW_REQUIRED", "AR_REVIEW_REQUIRED"]
    assert calls == [
        "universe:xgboost",
        "proposal:xgboost",
        "universe:lightgbm",
        "proposal:lightgbm",
    ]
