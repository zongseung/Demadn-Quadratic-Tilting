from __future__ import annotations

import multiprocessing
import os
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hqrc_v3.contracts import DataContractError
from hqrc_v3.provenance import ArtifactMismatch
from hqrc_v3.residuals import PredictionCache


def _prediction_frame(
    origin: datetime = datetime(2023, 1, 1), split_id: str = "oof-2023"
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "origin": [origin] * 24,
            "target_timestamp": [origin + timedelta(hours=hour) for hour in range(24)],
            "horizon": list(range(1, 25)),
            "observed_mw": [100.0] * 24,
            "predicted_mw": [99.0] * 24,
            "model": ["xgb"] * 24,
            "feature_set": ["B0"] * 24,
            "seed": [7] * 24,
            "split_id": [split_id] * 24,
        }
    )


def _concurrent_write(cache_root: str, barrier, hashes: dict[str, str], result_queue) -> None:
    cache = PredictionCache(Path(cache_root))
    barrier.wait(timeout=5)
    try:
        cache.write("concurrent", _prediction_frame(), hashes=hashes)
    except ArtifactMismatch:
        result_queue.put(("mismatch", hashes["config"]))
    else:
        result_queue.put(("written", hashes["config"]))


def test_cache_refuses_different_config_hash(tmp_path):
    cache = PredictionCache(tmp_path)
    cache.write("xgb-B0-2020", _prediction_frame(), hashes={"config": "a", "data": "d"})

    with pytest.raises(ArtifactMismatch, match="config"):
        cache.read("xgb-B0-2020", expected_hashes={"config": "b", "data": "d"})


def test_cache_returns_the_existing_frame_only_for_identical_hashes(tmp_path):
    cache = PredictionCache(tmp_path)
    written = cache.write("xgb-B0-2020", _prediction_frame(), hashes={"config": "a", "data": "d"})

    read = cache.read("xgb-B0-2020", expected_hashes={"config": "a", "data": "d"})
    repeated = cache.write("xgb-B0-2020", _prediction_frame(), hashes={"config": "a", "data": "d"})

    assert read is not None
    assert read.equals(written)
    assert repeated.equals(written)
    assert (tmp_path / "xgb-B0-2020.parquet").is_file()
    assert (tmp_path / "xgb-B0-2020.json").is_file()


@pytest.mark.parametrize("key", ["../escape", "nested/key", "..", ""])
def test_cache_rejects_path_traversal_and_invalid_keys(tmp_path, key):
    with pytest.raises(ValueError, match="key"):
        PredictionCache(tmp_path).read(key, expected_hashes={"config": "a"})


def test_cache_rejects_partial_artifacts(tmp_path):
    (tmp_path / "partial.parquet").touch()

    with pytest.raises(ArtifactMismatch, match="partial"):
        PredictionCache(tmp_path).read("partial", expected_hashes={"config": "a"})


def test_write_recovers_a_stale_partial_pair_while_holding_the_exclusive_lock(tmp_path):
    (tmp_path / "retry-after-crash.parquet").touch()

    written = PredictionCache(tmp_path).write(
        "retry-after-crash", _prediction_frame(), hashes={"config": "a", "data": "d"}
    )

    assert written.height == 24
    assert (tmp_path / "retry-after-crash.parquet").is_file()
    assert (tmp_path / "retry-after-crash.json").is_file()


def test_cache_uses_prediction_validation_for_writes(tmp_path):
    invalid = _prediction_frame(datetime(2024, 1, 1), "oof-2023")

    with pytest.raises(DataContractError, match="evaluation range"):
        PredictionCache(tmp_path).write("invalid", invalid, hashes={"config": "a"})


def test_concurrent_writes_with_different_hashes_leave_one_immutable_winner(tmp_path):
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    result_queue = context.Queue()
    first_hashes = {"config": "a", "data": "d"}
    second_hashes = {"config": "b", "data": "d"}
    processes = [
        context.Process(
            target=_concurrent_write, args=(str(tmp_path), barrier, hashes, result_queue)
        )
        for hashes in (first_hashes, second_hashes)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = {result_queue.get(timeout=2), result_queue.get(timeout=2)}

    assert {status for status, _ in outcomes} == {"written", "mismatch"}
    winning_config = next(config for status, config in outcomes if status == "written")
    winning_hashes = first_hashes if winning_config == "a" else second_hashes
    assert PredictionCache(tmp_path).read("concurrent", expected_hashes=winning_hashes) is not None


def test_failed_second_publication_removes_its_partial_artifact_and_can_retry(
    tmp_path, monkeypatch
):
    cache = PredictionCache(tmp_path)
    original_replace = os.replace
    calls = 0

    def fail_second_publication(source, destination):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected metadata publication failure")
        original_replace(source, destination)

    monkeypatch.setattr("hqrc_v3.residuals.os.replace", fail_second_publication)
    with pytest.raises(OSError, match="injected"):
        cache.write("retry", _prediction_frame(), hashes={"config": "a", "data": "d"})
    assert not (tmp_path / "retry.parquet").exists()
    assert not (tmp_path / "retry.json").exists()

    monkeypatch.setattr("hqrc_v3.residuals.os.replace", original_replace)
    assert (
        cache.write("retry", _prediction_frame(), hashes={"config": "a", "data": "d"}).height == 24
    )
