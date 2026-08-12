"""Opt-in two-fold reduced PyMC smoke for Task 15E primary aggregation."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import polars as pl
import pytest
from hqrc_v3.correction_source import validate_correction_source
from hqrc_v3.diagnostics.loeo import load_loeo_universe
from hqrc_v3.diagnostics.loeo_ar import load_approved_loeo_ar_set
from hqrc_v3.loeo_primary import fit_loeo_primary
from hqrc_v3.provenance import file_sha256


def _tree_hashes(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for candidate in sorted(root.rglob("*")):
        mode = candidate.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise AssertionError(f"protected tree contains symlink: {candidate}")
        if stat.S_ISREG(mode):
            result[candidate.relative_to(root).as_posix()] = file_sha256(candidate)
    return result


@pytest.mark.slow
def test_real_xgboost_b1_two_fold_primary_smoke_does_not_mutate_inputs(tmp_path: Path):
    names = (
        "HQRC_V3_REAL_RUN_DIR",
        "HQRC_V3_REAL_LOEO_DIR",
        "HQRC_V3_REAL_LOEO_AR_DIR",
        "HQRC_V3_REAL_PAPER_ARTIFACT_DIR",
    )
    values = tuple(os.environ.get(name) for name in names)
    if not any(values):
        pytest.skip(f"set {', '.join(names)}")
    if not all(values):
        pytest.fail("all four real LOEO environment variables are required")
    run, loeo_dir, ar_dir, paper_dir = (Path(value).resolve() for value in values)
    protected_before = {
        "source": _tree_hashes(run),
        "loeo": _tree_hashes(loeo_dir),
        "ar": _tree_hashes(ar_dir),
        "paper": _tree_hashes(paper_dir),
    }
    residual_manifest = json.loads(
        (run / "inputs/standardized_residuals_manifest.json").read_bytes()
    )
    source = validate_correction_source(
        run_dir=run,
        config_path=Path(residual_manifest["inputs"]["experiment_config"]["path"]),
        profile="smoke",
    )
    contexts = [
        context
        for context in source.available_contexts
        if (context.model, context.feature_set, context.seed) == ("xgboost", "B1", 7)
    ]
    assert len(contexts) == 1
    publication = load_loeo_universe(source, contexts[0], output_dir=loeo_dir)
    approved = load_approved_loeo_ar_set(source, publication, output_dir=ar_dir)
    selected = ("seollal-2024", "chuseok-2024")
    result = fit_loeo_primary(
        source,
        publication,
        approved,
        held_out_occurrence_ids=selected,
        root_seed=20260812,
        profile="smoke",
        draws=50,
        tune=20,
        chains=2,
        output_root=tmp_path,
    )
    assert result.sampler_fit_count == 2 and result.reused is False
    hourly = pl.read_parquet(result.hourly_predictions_path)
    per_event = pl.read_parquet(result.per_event_metrics_path)
    aggregate = pl.read_parquet(result.aggregate_metrics_path)
    psis = pl.read_parquet(result.training_psis_loo_path)
    assert hourly.height == 264
    assert tuple(hourly["occurrence_id"].unique(maintain_order=True)) == selected
    assert per_event.height == psis.height == 2
    assert per_event["held_out_row_count"].to_list() == [144, 120]
    assert aggregate["group"].to_list() == ["pooled", "seollal", "chuseok"]
    assert psis["diagnostic_label"].unique().to_list() == ["training_nine_event_psis_loo"]
    assert all(
        frame["causal"].unique().to_list() == [False]
        for frame in (hourly, per_event, aggregate, psis)
    )
    before_reuse = _tree_hashes(result.output_dir)
    reused = fit_loeo_primary(
        source,
        publication,
        approved,
        held_out_occurrence_ids=selected,
        root_seed=20260812,
        profile="smoke",
        draws=50,
        tune=20,
        chains=2,
        output_root=tmp_path,
    )
    assert reused.reused is True and reused.sampler_fit_count == 0
    assert _tree_hashes(result.output_dir) == before_reuse
    assert {
        "source": _tree_hashes(run),
        "loeo": _tree_hashes(loeo_dir),
        "ar": _tree_hashes(ar_dir),
        "paper": _tree_hashes(paper_dir),
    } == protected_before
