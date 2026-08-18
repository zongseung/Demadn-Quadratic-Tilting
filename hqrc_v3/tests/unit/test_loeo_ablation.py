"""H1/H2 LOEO ablation product and identity tests."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

import hqrc_v3._loeo_publication as fold_contract
import hqrc_v3.loeo_ablation as ablation_module
from hqrc_v3.loeo_ablation import _aggregate_products


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
