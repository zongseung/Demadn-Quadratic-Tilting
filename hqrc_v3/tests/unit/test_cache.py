from __future__ import annotations

from datetime import datetime, timedelta

import polars as pl
import pytest
from hqrc_v3.provenance import ArtifactMismatch
from hqrc_v3.residuals import PredictionCache


def _prediction_frame() -> pl.DataFrame:
    origin = datetime(2023, 1, 1)
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
            "split_id": ["oof-2023"] * 24,
        }
    )


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
