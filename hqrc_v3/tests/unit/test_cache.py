from __future__ import annotations

import multiprocessing
import os
from contextlib import contextmanager
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


class _GatedPredictionCache(PredictionCache):
    """Test-only cache that stops the first writer after its absent-cache observation."""

    def __init__(
        self,
        root: Path,
        *,
        writer: str,
        first_absent,
        second_absent,
        release_first,
    ) -> None:
        super().__init__(root)
        self.writer = writer
        self.first_absent = first_absent
        self.second_absent = second_absent
        self.release_first = release_first

    def _read_unlocked(self, key: str, expected_hashes):
        frame = super()._read_unlocked(key, expected_hashes)
        if frame is not None:
            return frame
        if self.writer == "first":
            self.first_absent.set()
            if not self.release_first.wait(timeout=5):
                raise RuntimeError("test did not release the first writer")
        else:
            self.second_absent.set()
        return frame


class _UnserializedGatedPredictionCache(_GatedPredictionCache):
    """Test-only control that removes the production per-key serialized boundary."""

    @contextmanager
    def _lock(self, key: str, *, exclusive: bool):
        yield


def _concurrent_write(
    cache_type,
    cache_root: str,
    writer: str,
    first_absent,
    second_absent,
    release_first,
    hashes: dict[str, str],
    result_queue,
) -> None:
    cache = cache_type(
        Path(cache_root),
        writer=writer,
        first_absent=first_absent,
        second_absent=second_absent,
        release_first=release_first,
    )
    try:
        cache.write("concurrent", _prediction_frame(), hashes=hashes)
    except ArtifactMismatch:
        result_queue.put(("mismatch", hashes["config"]))
    else:
        result_queue.put(("written", hashes["config"]))


def _force_publication_race(tmp_path, cache_type, *, second_must_observe_absence: bool):
    context = multiprocessing.get_context("spawn")
    first_absent = context.Event()
    second_absent = context.Event()
    release_first = context.Event()
    result_queue = context.Queue()
    first_hashes = {"config": "a", "data": "d"}
    second_hashes = {"config": "b", "data": "d"}
    first = context.Process(
        target=_concurrent_write,
        args=(
            cache_type,
            str(tmp_path),
            "first",
            first_absent,
            second_absent,
            release_first,
            first_hashes,
            result_queue,
        ),
    )
    second = context.Process(
        target=_concurrent_write,
        args=(
            cache_type,
            str(tmp_path),
            "second",
            first_absent,
            second_absent,
            release_first,
            second_hashes,
            result_queue,
        ),
    )
    first.start()
    assert first_absent.wait(timeout=5)
    second.start()
    if second_must_observe_absence:
        assert second_absent.wait(timeout=5)
    else:
        assert not second_absent.wait(timeout=0.2)
    release_first.set()
    for process in (first, second):
        process.join(timeout=10)
        assert process.exitcode == 0
    outcomes = {result_queue.get(timeout=2), result_queue.get(timeout=2)}
    return outcomes, first_hashes, second_hashes


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
    outcomes, first_hashes, second_hashes = _force_publication_race(
        tmp_path, _GatedPredictionCache, second_must_observe_absence=False
    )

    assert {status for status, _ in outcomes} == {"written", "mismatch"}
    winning_config = next(config for status, config in outcomes if status == "written")
    winning_hashes = first_hashes if winning_config == "a" else second_hashes
    assert PredictionCache(tmp_path).read("concurrent", expected_hashes=winning_hashes) is not None


def test_controlled_gate_exposes_the_former_check_then_act_race(tmp_path):
    outcomes, _, _ = _force_publication_race(
        tmp_path, _UnserializedGatedPredictionCache, second_must_observe_absence=True
    )

    assert outcomes == {("written", "a"), ("written", "b")}


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
