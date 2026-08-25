import hashlib
import json
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from hqrc_v3.evaluation.reviewer import (
    ReviewerReportError,
    ReviewerReportInputs,
    build_reviewer_report,
)

MODELS = ("xgboost", "transformer")
EVENT_IDS = tuple(
    f"{holiday}-{year}" for year in range(2020, 2025) for holiday in ("seollal", "chuseok")
)
METRICS = ("rmse", "mae", "mape", "smape", "r2")


def _metric_row(
    model: str,
    event_id: str,
    correction: str,
    rmse: float,
    *,
    seed: int = 17,
) -> dict[str, object]:
    return {
        "model": model,
        "feature_set": "B1W",
        "seed": seed,
        "event_id": event_id,
        "holiday_type": event_id.split("-")[0],
        "correction": correction,
        "n_timestamps": 120,
        "rmse": rmse,
        "mae": rmse * 0.8,
        "mape": rmse * 0.01,
        "smape": rmse * 0.02,
        "r2": 0.9,
    }


def _write_contexts(root: Path) -> tuple[tuple[Path, ...], tuple[Path, ...]]:
    loeo_dirs = []
    causal_dirs = []
    for model_index, model in enumerate(MODELS):
        loeo = root / "loeo" / model / "B1W"
        loeo.mkdir(parents=True)
        rows = []
        for event_index, event_id in enumerate(EVENT_IDS):
            h0 = 10.0 + model_index + event_index / 10
            improvement = -5.0 if event_index == 0 else 2.0 + event_index
            h2 = h0 * (1.0 - improvement / 100.0)
            rows.extend(
                (
                    _metric_row(model, event_id, "H0", h0),
                    _metric_row(model, event_id, "H1", h0 * 0.99),
                    _metric_row(model, event_id, "H2", h2),
                    _metric_row(model, event_id, "H2-taper", h2 * 0.99),
                )
            )
        event = pl.DataFrame(rows)
        event.write_parquet(loeo / "event_metrics.parquet")
        pooled = pl.concat(
            [
                event.filter(pl.lit(True) if scope == "all" else pl.col("holiday_type") == scope)
                .group_by("correction")
                .agg(
                    pl.lit(model).alias("model"),
                    pl.lit("B1W").alias("feature_set"),
                    pl.lit(17).alias("seed"),
                    pl.lit(scope).alias("scope"),
                    pl.col("n_timestamps").sum(),
                    *(pl.col(metric).mean().alias(metric) for metric in METRICS),
                )
                .select(
                    "model",
                    "feature_set",
                    "seed",
                    "scope",
                    "correction",
                    "n_timestamps",
                    *METRICS,
                )
                for scope in ("all", "seollal", "chuseok")
            ]
        )
        pooled.write_parquet(loeo / "pooled_metrics.parquet")
        pl.DataFrame(
            {
                "model": [model],
                "feature_set": ["B1W"],
                "seed": [17],
                "sigma_fit_oof_2023_mw": [100.0],
                "sigma_2024_non_event_mw": [102.0],
                "scale_ratio": [100 / 102],
                "n_2024_non_event_hours": [8_000],
            }
        ).write_parquet(loeo / "scale_stability.parquet")
        loeo_dirs.append(loeo)

        causal = root / "causal" / model / "B1W"
        causal.mkdir(parents=True)
        hourly_rows = []
        causal_metrics = []
        start = datetime(2024, 2, 9)
        for event_index, event_id in enumerate(("seollal-2024", "chuseok-2024")):
            event_start = start + timedelta(days=event_index * 220)
            observed = []
            h0_values = []
            h2_values = []
            for hour in range(12):
                value = 100.0 + hour + model_index
                observed.append(value)
                h0_values.append(value + 5.0)
                h2_values.append(value + 1.0)
                hourly_rows.append(
                    {
                        "model": model,
                        "feature_set": "B1W",
                        "seed": 17,
                        "occurrence_id": event_id,
                        "holiday_type": event_id.split("-")[0],
                        "target_timestamp": event_start + timedelta(hours=hour),
                        "observed_mw": value,
                        "H0_mw": value + 5.0,
                        "H1_mw": value + 3.0,
                        "H2_mw": value + 1.0,
                        "H2_taper_mw": value + 2.0,
                    }
                )
            causal_metrics.extend(
                (
                    _metric_row(model, event_id, "H0", 5.0),
                    _metric_row(model, event_id, "H1", 3.0),
                    _metric_row(model, event_id, "H2", 1.0),
                    _metric_row(model, event_id, "H2-taper", 2.0),
                )
            )
        pl.DataFrame(hourly_rows).write_parquet(causal / "hourly_predictions.parquet")
        event_frame = pl.DataFrame(causal_metrics)
        event_frame.write_parquet(causal / "event_metrics.parquet")
        pl.concat(
            [
                event_frame.filter(
                    pl.lit(True) if scope == "all" else pl.col("holiday_type") == scope
                )
                .group_by("correction")
                .agg(
                    pl.lit(model).alias("model"),
                    pl.lit("B1W").alias("feature_set"),
                    pl.lit(17).alias("seed"),
                    pl.lit(scope).alias("scope"),
                    pl.col("n_timestamps").sum(),
                    *(pl.col(metric).mean().alias(metric) for metric in METRICS),
                )
                .select(
                    "model",
                    "feature_set",
                    "seed",
                    "scope",
                    "correction",
                    "n_timestamps",
                    *METRICS,
                )
                for scope in ("all", "seollal", "chuseok")
            ]
        ).write_parquet(causal / "pooled_metrics.parquet")
        causal_dirs.append(causal)
    return tuple(loeo_dirs), tuple(causal_dirs)


def _reviewer_fixture(tmp_path: Path) -> ReviewerReportInputs:
    rows = []
    start = datetime(2024, 1, 1)
    for model_index, model in enumerate(MODELS):
        for feature_set, error in (("B0", 10.0), ("B1W", 8.0)):
            for hour in range(24):
                observed = 1_000.0 + hour
                rows.append(
                    {
                        "origin": start,
                        "target_timestamp": start + timedelta(hours=hour),
                        "horizon": hour + 1,
                        "observed_mw": observed,
                        "predicted_mw": observed + error + model_index,
                        "model": model,
                        "feature_set": feature_set,
                        "seed": 17,
                        "split_id": "final-2024",
                    }
                )
    final_path = tmp_path / "final_2024.parquet"
    pl.DataFrame(rows).write_parquet(final_path)
    loeo_dirs, causal_dirs = _write_contexts(tmp_path)
    return ReviewerReportInputs(
        final_predictions_path=final_path,
        loeo_context_dirs=loeo_dirs,
        causal_context_dirs=causal_dirs,
        root_seed=20260813,
    )


def test_reviewer_report_separates_feature_and_hqt_gains(tmp_path: Path):
    result = build_reviewer_report(_reviewer_fixture(tmp_path), output_root=tmp_path / "paper")
    overall = pl.read_parquet(result.tables_dir / "overall_2024.parquet")
    inference = pl.read_parquet(result.tables_dir / "loeo_inference.parquet")

    assert set(overall["comparison"]) == {"B0-H0_to_B1W-H0"}
    assert set(inference["comparison"]) == {"B1W-H0_to_B1W-H2"}
    assert inference["n_events"].to_list() == [10, 10]
    assert inference["alternative"].to_list() == ["greater", "greater"]
    assert result.figures_dir.joinpath("fig_per_event.png").is_file()
    assert result.figures_dir.joinpath("fig_correction_2024.png").is_file()


def test_reviewer_report_writes_exact_normalized_tables_and_manifest(tmp_path: Path):
    result = build_reviewer_report(_reviewer_fixture(tmp_path), output_root=tmp_path / "paper")

    expected = {
        "overall_2024",
        "loeo_event",
        "loeo_pooled",
        "loeo_inference",
        "causal_2024",
        "scale_stability",
    }
    assert {path.stem for path in result.tables_dir.iterdir()} == expected
    assert {path.suffix for path in result.tables_dir.iterdir()} == {".csv", ".parquet"}
    for name in expected:
        assert (
            pl.read_csv(result.tables_dir / f"{name}.csv").height
            == pl.read_parquet(result.tables_dir / f"{name}.parquet").height
        )

    event = pl.read_parquet(result.tables_dir / "loeo_event.parquet")
    causal = pl.read_parquet(result.tables_dir / "causal_2024.parquet")
    assert event.filter(pl.col("correction") == "H2")[
        "rmse_improvement_pct"
    ].min() == pytest.approx(-5.0)
    assert set(event["correction"]) == {"H0", "H1", "H2"}
    assert set(causal["correction"]) == {"H0", "H1", "H2"}
    assert not any(
        token in column.lower()
        for column in causal.columns
        for token in ("crps", "pinball", "coverage")
    )

    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["inference"]["bootstrap_draws"] == 10_000
    assert manifest["inference"]["resampled_units"] == "event"
    assert manifest["inference"]["wilcoxon_alternative"] == "greater"
    assert set(manifest["inference"]["bootstrap_seeds"]) == set(MODELS)
    for relative, digest in manifest["outputs"].items():
        assert hashlib.sha256((result.output_root / relative).read_bytes()).hexdigest() == digest


def test_reviewer_report_rejects_non_ten_event_inference(tmp_path: Path):
    inputs = _reviewer_fixture(tmp_path)
    path = inputs.loeo_context_dirs[0] / "event_metrics.parquet"
    pl.read_parquet(path).filter(pl.col("event_id") != EVENT_IDS[-1]).write_parquet(path)

    with pytest.raises(ReviewerReportError, match="exactly 10 physical events"):
        build_reviewer_report(inputs, output_root=tmp_path / "paper")
