"""Task 15B immutable LOEO universe and physical-fold contracts."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, time, timedelta
from hashlib import sha256
from pathlib import Path

import polars as pl
import pytest
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.diagnostics.ar import EventResidualContext
from hqrc_v3.diagnostics.loeo import (
    LOEOError,
    load_loeo_fold,
    load_loeo_universe,
    publish_loeo_universe,
)
from hqrc_v3.events import EventOccurrence
from hqrc_v3.provenance import file_sha256
from hqrc_v3.residual_stage import STANDARDIZED_RESIDUAL_COLUMNS

CONTEXT = EventResidualContext(
    "lightgbm", "B1", 7, tuple(f"oof-{year}" for year in range(2020, 2024))
)


def _events() -> tuple[EventOccurrence, ...]:
    return (
        EventOccurrence(
            "seollal-2020",
            "seollal",
            datetime(2020, 1, 25).date(),
            datetime(2020, 1, 24).date(),
            datetime(2020, 1, 27).date(),
            0,
        ),
        EventOccurrence(
            "chuseok-2020",
            "chuseok",
            datetime(2020, 10, 1).date(),
            datetime(2020, 9, 30).date(),
            datetime(2020, 10, 2).date(),
            1,
        ),
        EventOccurrence(
            "seollal-2021",
            "seollal",
            datetime(2021, 2, 12).date(),
            datetime(2021, 2, 11).date(),
            datetime(2021, 2, 14).date(),
            1,
        ),
        EventOccurrence(
            "chuseok-2021",
            "chuseok",
            datetime(2021, 9, 21).date(),
            datetime(2021, 9, 20).date(),
            datetime(2021, 9, 22).date(),
            1,
        ),
        EventOccurrence(
            "seollal-2022",
            "seollal",
            datetime(2022, 2, 1).date(),
            datetime(2022, 1, 31).date(),
            datetime(2022, 2, 2).date(),
            1,
        ),
        EventOccurrence(
            "chuseok-2022",
            "chuseok",
            datetime(2022, 9, 10).date(),
            datetime(2022, 9, 9).date(),
            datetime(2022, 9, 11).date(),
            0,
        ),
        EventOccurrence(
            "seollal-2023",
            "seollal",
            datetime(2023, 1, 22).date(),
            datetime(2023, 1, 21).date(),
            datetime(2023, 1, 24).date(),
            0,
        ),
        EventOccurrence(
            "chuseok-2023",
            "chuseok",
            datetime(2023, 9, 29).date(),
            datetime(2023, 9, 28).date(),
            datetime(2023, 9, 30).date(),
            0,
        ),
        EventOccurrence(
            "seollal-2024",
            "seollal",
            datetime(2024, 2, 10).date(),
            datetime(2024, 2, 9).date(),
            datetime(2024, 2, 12).date(),
            0,
        ),
        EventOccurrence(
            "chuseok-2024",
            "chuseok",
            datetime(2024, 9, 17).date(),
            datetime(2024, 9, 16).date(),
            datetime(2024, 9, 18).date(),
            0,
        ),
    )


def _event_rows(event: EventOccurrence, *, split_id: str, scale: float) -> list[dict[str, object]]:
    start = datetime.combine(event.window_start, time.min)
    end = datetime.combine(event.window_end + timedelta(days=1), time.min)
    return [
        {
            "origin": timestamp - timedelta(hours=1),
            "target_timestamp": timestamp,
            "horizon": 1,
            "observed_mw": 100.0 + index,
            "predicted_mw": 90.0,
            "residual_mw": 10.0 + index,
            "standardized_residual": (10.0 + index) / scale,
            "sigma_n_mw": scale,
            "model": CONTEXT.model,
            "feature_set": CONTEXT.feature_set,
            "seed": CONTEXT.seed,
            "split_id": split_id,
            "occurrence_id": event.occurrence_id,
            "holiday_type": event.holiday_type,
            "tau_days": (timestamp - datetime.combine(event.central_date, time.min)).total_seconds()
            / 86400,
            "hour": timestamp.hour,
            "restriction": event.restriction,
        }
        for index, timestamp in enumerate(
            start + timedelta(hours=offset)
            for offset in range(int((end - start).total_seconds() // 3600))
        )
    ]


@pytest.fixture
def source(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> ValidatedCorrectionSource:
    run = tmp_path / "source"
    (run / "inputs").mkdir(parents=True)
    (run / "predictions").mkdir()
    (run / "inputs/.standardized-residuals.lock").touch()
    (run / "predictions/.baseline-publication.lock").touch()
    events = _events()
    standardized = (
        pl.DataFrame(
            [
                row
                for event in events[:8]
                for row in _event_rows(
                    event,
                    split_id=f"oof-{event.central_date.year}",
                    scale=20.0 + event.central_date.year - 2020,
                )
            ]
        )
        .select(STANDARDIZED_RESIDUAL_COLUMNS)
        .with_columns(pl.col("origin", "target_timestamp").cast(pl.Datetime("ns")))
        .sort("target_timestamp")
    )
    final = (
        pl.DataFrame(
            [
                {
                    **row,
                    "observed_mw": row["observed_mw"],
                    "predicted_mw": row["predicted_mw"],
                    "split_id": "final-2024",
                }
                for event in events[8:]
                for row in _event_rows(event, split_id="final-2024", scale=23.0)
            ]
        )
        .select(
            [
                "origin",
                "target_timestamp",
                "horizon",
                "observed_mw",
                "predicted_mw",
                "model",
                "feature_set",
                "seed",
                "split_id",
            ]
        )
        .with_columns(pl.col("origin", "target_timestamp").cast(pl.Datetime("ns")))
    )
    for path, frame in (
        (run / "inputs/standardized_residuals.parquet", standardized),
        (run / "predictions/oof.parquet", pl.DataFrame(schema=final.schema)),
        (run / "predictions/final_2024.parquet", final),
        (run / "predictions/oof_members.parquet", pl.DataFrame(schema=final.schema)),
        (run / "predictions/final_2024_members.parquet", pl.DataFrame(schema=final.schema)),
    ):
        frame.write_parquet(path)
    source = ValidatedCorrectionSource(
        run_dir=run,
        source_profile="paper",
        events=events,
        source_paths={},
        source_hashes={},
        available_contexts=(CONTEXT,),
        residual_manifest_path=run / "inputs/standardized_residuals_manifest.json",
        residual_path=run / "inputs/standardized_residuals.parquet",
        baseline_manifest_path=run / "predictions/baseline_manifest.json",
        oof_members_path=run / "predictions/oof_members.parquet",
        oof_point_path=run / "predictions/oof.parquet",
        final_members_path=run / "predictions/final_2024_members.parquet",
        final_point_path=run / "predictions/final_2024.parquet",
        residual_sha256=file_sha256(run / "inputs/standardized_residuals.parquet"),
        oof_members_sha256=file_sha256(run / "predictions/oof_members.parquet"),
        oof_point_sha256=file_sha256(run / "predictions/oof.parquet"),
        final_members_sha256=file_sha256(run / "predictions/final_2024_members.parquet"),
        final_point_sha256=file_sha256(run / "predictions/final_2024.parquet"),
        _residual_manifest_json=b'{"profile":"paper"}',
        _baseline_manifest_json=b"{}",
    )
    monkeypatch.setattr(
        ValidatedCorrectionSource,
        "load_standardized_context",
        lambda _self, context, *, through: (
            standardized.clone()
            if context == CONTEXT and through == 2023
            else (_ for _ in ()).throw(AssertionError("wrong context"))
        ),
    )
    monkeypatch.setattr(
        ValidatedCorrectionSource,
        "load_final_point_context",
        lambda _self, context: (
            final.clone()
            if context == CONTEXT
            else (_ for _ in ()).throw(AssertionError("wrong context"))
        ),
    )
    return source


def test_publish_exact_universe_and_physical_folds(
    source: ValidatedCorrectionSource, tmp_path: Path
):
    published = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "loeo")
    universe = pl.read_parquet(published.universe_path)
    assert published.occurrence_ids == tuple(event.occurrence_id for event in _events())
    assert universe.height == 1_296
    assert universe["causal"].unique().to_list() == [False]
    assert universe.select("occurrence_id").unique(maintain_order=True)[
        "occurrence_id"
    ].to_list() == list(published.occurrence_ids)
    assert (
        universe.filter(pl.col("occurrence_id") == "seollal-2023")[
            "standardized_residual"
        ].to_list()
        == source.load_standardized_context(CONTEXT, through=2023)
        .filter(pl.col("occurrence_id") == "seollal-2023")["standardized_residual"]
        .to_list()
    )
    for held_out in published.occurrence_ids:
        fold = load_loeo_fold(
            source, CONTEXT, output_dir=tmp_path / "loeo", held_out_occurrence_id=held_out
        )
        assert fold.path.is_file() and not fold.path.is_symlink()
        assert (
            fold.frame.height == 1_296 - universe.filter(pl.col("occurrence_id") == held_out).height
        )
        assert fold.frame["occurrence_id"].unique().to_list().count(held_out) == 0
        assert fold.occurrence_ids == tuple(
            value for value in published.occurrence_ids if value != held_out
        )


def test_2024_uses_only_oof_2023_scale(source: ValidatedCorrectionSource, tmp_path: Path):
    published = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "loeo")
    universe = pl.read_parquet(published.universe_path)
    actual = universe.filter(pl.col("occurrence_id") == "seollal-2024").row(0, named=True)
    assert actual["residual_mw"] == actual["observed_mw"] - actual["predicted_mw"]
    assert actual["sigma_n_mw"] == 23.0
    assert actual["standardized_residual"] == actual["residual_mw"] / 23.0


def test_held_out_mutation_cannot_change_its_training_fold(
    source: ValidatedCorrectionSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "original"
    first = publish_loeo_universe(source, CONTEXT, output_dir=output)
    held_out = "seollal-2024"
    before = first.fold_sha256[held_out]
    original = source.load_final_point_context(CONTEXT)
    changed = original.with_columns(
        pl.when(pl.col("target_timestamp") == original["target_timestamp"].item(0))
        .then(pl.lit(9999.0))
        .otherwise(pl.col("observed_mw"))
        .alias("observed_mw")
    )
    monkeypatch.setattr(
        ValidatedCorrectionSource,
        "load_final_point_context",
        lambda _self, _context: changed.clone(),
    )
    second = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "changed")
    assert second.fold_sha256[held_out] == before
    assert second.universe_sha256 != first.universe_sha256


def test_held_out_oof_mutation_cannot_change_its_training_fold(
    source: ValidatedCorrectionSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    first = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "original")
    held_out = "seollal-2023"
    original = source.load_standardized_context(CONTEXT, through=2023)
    changed = original.with_columns(
        pl.when(pl.col("occurrence_id") == held_out)
        .then(pl.col("observed_mw") + 50.0)
        .otherwise(pl.col("observed_mw"))
        .alias("observed_mw")
    )
    monkeypatch.setattr(
        ValidatedCorrectionSource,
        "load_standardized_context",
        lambda _self, _context, *, through: changed.clone() if through == 2023 else None,
    )
    second = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "changed")
    assert second.fold_sha256[held_out] == first.fold_sha256[held_out]
    assert second.universe_sha256 != first.universe_sha256


def test_loader_rechecks_physical_fold_and_rejects_held_out_reinsertion(
    source: ValidatedCorrectionSource, tmp_path: Path
):
    published = publish_loeo_universe(source, CONTEXT, output_dir=tmp_path / "loeo")
    held_out = "chuseok-2024"
    path = published.fold_paths[held_out]
    row = (
        pl.read_parquet(published.universe_path).filter(pl.col("occurrence_id") == held_out).head(1)
    )
    pl.concat([pl.read_parquet(path), row], how="vertical").write_parquet(path)
    with pytest.raises(LOEOError, match="hash|held-out"):
        load_loeo_fold(
            source, CONTEXT, output_dir=tmp_path / "loeo", held_out_occurrence_id=held_out
        )


def _canonical_write(path: Path, value: object) -> None:
    path.write_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _rehash_manifest(published, output: Path, manifest: dict[str, object]) -> None:
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    manifest["manifest_sha256"] = sha256(
        json.dumps(unsigned, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _canonical_write(published.manifest_path, manifest)
    _canonical_write(
        published.generation_dir / "COMPLETE",
        {"manifest_sha256": manifest["manifest_sha256"]},
    )
    current = json.loads((output / "current.json").read_bytes())
    current["manifest_sha256"] = manifest["manifest_sha256"]
    _canonical_write(output / "current.json", current)


def test_loader_rejects_rehashed_semantic_universe_mutation(
    source: ValidatedCorrectionSource, tmp_path: Path
):
    output = tmp_path / "loeo"
    published = publish_loeo_universe(source, CONTEXT, output_dir=output)
    mutated = pl.read_parquet(published.universe_path).with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("observed_mw") + 1.0)
        .otherwise(pl.col("observed_mw"))
        .alias("observed_mw")
    )
    mutated.write_parquet(published.universe_path)
    manifest = json.loads(published.manifest_path.read_bytes())
    manifest["universe"]["sha256"] = file_sha256(published.universe_path)
    _rehash_manifest(published, output, manifest)

    with pytest.raises(LOEOError, match="semantics"):
        load_loeo_universe(source, CONTEXT, output_dir=output)


def test_incompatible_or_crashed_publication_cannot_be_reused(
    source: ValidatedCorrectionSource, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "loeo"
    publish_loeo_universe(source, CONTEXT, output_dir=output)
    changed = source.load_final_point_context(CONTEXT).with_columns(
        (pl.col("predicted_mw") + 1.0).alias("predicted_mw")
    )
    monkeypatch.setattr(
        ValidatedCorrectionSource,
        "load_final_point_context",
        lambda _self, _context: changed.clone(),
    )
    with pytest.raises(LOEOError, match="semantics|incompatible"):
        publish_loeo_universe(source, CONTEXT, output_dir=output)

    crashed = tmp_path / "crashed"
    crashed.mkdir()
    (crashed / "generations").mkdir()
    with pytest.raises(LOEOError, match="partial"):
        publish_loeo_universe(source, CONTEXT, output_dir=crashed)


@pytest.mark.parametrize("mutation", ["unknown", "symlink", "partial", "context", "order"])
def test_publication_boundaries_fail_closed(
    source: ValidatedCorrectionSource, tmp_path: Path, mutation: str
):
    output = tmp_path / "loeo"
    published = publish_loeo_universe(source, CONTEXT, output_dir=output)
    if mutation == "unknown":
        (output / "unknown").write_text("bad")
    elif mutation == "symlink":
        target = output / "copy"
        target.write_bytes(published.universe_path.read_bytes())
        published.universe_path.unlink()
        published.universe_path.symlink_to(target)
    elif mutation == "partial":
        (output / "current.json").unlink()
    elif mutation == "context":
        with pytest.raises(LOEOError):
            load_loeo_universe(
                source, replace(CONTEXT, split_ids=CONTEXT.split_ids[:-1]), output_dir=output
            )
        return
    else:
        frame = pl.read_parquet(published.universe_path)
        frame.reverse().write_parquet(published.universe_path)
    with pytest.raises(LOEOError):
        load_loeo_universe(source, CONTEXT, output_dir=output)
