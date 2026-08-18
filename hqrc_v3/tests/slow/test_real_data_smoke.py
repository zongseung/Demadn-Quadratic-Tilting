from datetime import datetime
from pathlib import Path

import pytest

from hqrc_v3.baselines.classical import make_classical_baseline, predictions_to_frame
from hqrc_v3.data import audit_hourly_data, read_hourly_data
from hqrc_v3.features import attach_calendar_features, build_daily_forecast_matrix
from hqrc_v3.residuals import PredictionCache
from hqrc_v3.splits import expanding_oof_folds, select_fold_samples


@pytest.mark.slow
def test_real_source_audits_and_caches_tiny_classical_fold(tmp_path):
    source = Path(__file__).resolve().parents[3] / "power_demand_final.csv"
    if not source.is_file():
        pytest.skip("repository power_demand_final.csv source is absent")
    audited = audit_hourly_data(
        read_hourly_data(source),
        expected_start=datetime(2019, 1, 1),
        expected_end=datetime(2024, 10, 31, 23),
        expected_rows=51_144,
    )
    assert audited.height == 51_144

    matrix = build_daily_forecast_matrix(attach_calendar_features(audited, ()), feature_set="B0")
    fold = expanding_oof_folds()[0]
    train_indices, evaluation_indices = select_fold_samples(matrix, fold)
    fitted = make_classical_baseline("svr", {"C": 1.0, "epsilon": 0.1}).fit(
        matrix.take(train_indices), validation=None, seed=7
    )
    evaluation = matrix.take(evaluation_indices[:1])
    frame = predictions_to_frame(
        evaluation,
        fitted.predict(evaluation),
        model="svr",
        feature_set="B0",
        seed=7,
        fold=fold,
    )
    cache = PredictionCache(tmp_path / "prediction-cache")
    cached = cache.write("real-svr-oof-2020", frame, hashes={"config": "c", "data": "d"})
    assert cached.height == 24
    assert cache.read("real-svr-oof-2020", expected_hashes={"config": "c", "data": "d"}) is not None
