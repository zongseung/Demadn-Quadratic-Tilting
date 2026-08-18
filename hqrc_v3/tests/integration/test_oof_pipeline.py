"""Integration contracts for immutable expanding-origin baseline forecasts."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pytest
from hqrc_v3.contracts import DataContractError, ForecastMatrix
from hqrc_v3.features import feature_columns
from hqrc_v3.oof import fit_final_baseline, generate_expanding_oof
from hqrc_v3.provenance import ArtifactMismatch
from hqrc_v3.residuals import PredictionCache


def _matrix(
    years: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023, 2024),
    *,
    feature_set: str = "B0",
) -> ForecastMatrix:
    origins = np.array([np.datetime64(f"{year}-01-01T00:00", "ns") for year in years])
    target_times = np.array(
        [
            np.datetime64(f"{year}-01-01T00:00", "ns")
            + np.arange(24).astype("timedelta64[h]")
            for year in years
        ]
    )
    count = len(years)
    future_columns = feature_columns(feature_set)  # type: ignore[arg-type]
    return ForecastMatrix(
        origins=origins,
        target_times=target_times,
        history=np.zeros((count, 168, 1)),
        future=np.zeros((count, 24, len(future_columns))),
        target=np.tile(np.arange(24, dtype=float), (count, 1)),
        history_columns=("load_lag",),
        future_columns=future_columns,
    )


def _population(times: np.ndarray, *, unit: str) -> dict[str, object]:
    unique = np.unique(np.asarray(times).reshape(-1).astype("datetime64[ns]"))
    return {
        "count": int(unique.size),
        "start": np.datetime_as_string(unique[0], unit="s"),
        "end": np.datetime_as_string(unique[-1], unit="s"),
        "unit": unit,
    }
@dataclass
class _Fitted:
    factory: RecordingFactory
    train_years: tuple[int, ...]
    population: dict[str, dict[str, object]]
    model_name: str = "recording"

    def population_contract(self) -> dict[str, dict[str, object]]:
        return self.population

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        self.factory.calls.append((self.train_years, int(str(batch.origins[0])[:4])))
        return batch.target + 1.0


@dataclass
class RecordingFactory:
    name: str = "recording"
    calls: list[tuple[tuple[int, ...], int]] = field(default_factory=list)
    tune_calls: int = 0

    def fit(self, train: ForecastMatrix, validation: ForecastMatrix | None, seed: int) -> _Fitted:
        assert validation is None
        train_years = tuple(sorted({int(str(value)[:4]) for value in train.origins}))
        return _Fitted(
            self,
            train_years,
            {
                "x": _population(train.origins, unit="daily-sample"),
                "target": _population(train.target_times, unit="unique-hour"),
            },
        )


@pytest.fixture
def synthetic_year_matrix() -> ForecastMatrix:
    return _matrix()


def test_oof_calls_exact_train_eval_years(synthetic_year_matrix, tmp_path):
    factory = RecordingFactory()
    summary = generate_expanding_oof(
        matrix=synthetic_year_matrix,
        factory=factory,
        feature_set="B0",
        cache=PredictionCache(tmp_path),
        artifact_hashes={"config": "config-hash", "data": "data-hash", "events": "event-hash"},
        seed=9,
    )
    assert factory.calls == [
        ((2019,), 2020),
        ((2019, 2020), 2021),
        ((2019, 2020, 2021), 2022),
        ((2019, 2020, 2021, 2022), 2023),
    ]
    assert summary.eval_years == (2020, 2021, 2022, 2023)
    assert summary.fit_count == 4
    assert summary.cache_hit_count == 0
    assert summary.combined_frame.height == 4 * 24


def test_oof_cache_hit_skips_refit_and_predict(synthetic_year_matrix, tmp_path):
    first = RecordingFactory()
    kwargs = dict(
        matrix=synthetic_year_matrix,
        feature_set="B0",
        cache=PredictionCache(tmp_path),
        artifact_hashes={"config": "c", "data": "d", "events": "e"},
        seed=9,
    )
    generate_expanding_oof(factory=first, **kwargs)
    second = RecordingFactory()
    summary = generate_expanding_oof(factory=second, **kwargs)
    assert second.calls == []
    assert summary.fit_count == 0
    assert summary.cache_hit_count == 4


def test_oof_cache_hash_mismatch_is_immutable(synthetic_year_matrix, tmp_path):
    cache = PredictionCache(tmp_path)
    generate_expanding_oof(
        synthetic_year_matrix,
        RecordingFactory(),
        "B0",
        cache,
        {"config": "c", "data": "d", "events": "e"},
        9,
    )
    with pytest.raises(ArtifactMismatch, match="config"):
        generate_expanding_oof(
            synthetic_year_matrix,
            RecordingFactory(),
            "B0",
            cache,
            {"config": "changed", "data": "d", "events": "e"},
            9,
        )


def test_oof_never_invokes_tuning(synthetic_year_matrix, tmp_path):
    factory = RecordingFactory()
    generate_expanding_oof(
        synthetic_year_matrix,
        factory,
        "B0",
        PredictionCache(tmp_path),
        {"config": "c", "data": "d", "events": "e"},
        9,
    )
    assert factory.tune_calls == 0


def test_final_fit_uses_2019_through_2023_only(synthetic_year_matrix):
    factory = RecordingFactory()
    result = fit_final_baseline(_matrix(feature_set="B1"), factory, "B1", seed=9)
    assert factory.calls == [((2019, 2020, 2021, 2022, 2023), 2024)]
    assert result.frame["split_id"].unique().to_list() == ["final-2024"]
    assert result.frame.height == 24


def test_oof_rejects_matrix_labeled_b0_when_it_has_b1_features(tmp_path):
    with pytest.raises(DataContractError, match="feature_set.*ForecastMatrix"):
        generate_expanding_oof(
            _matrix(feature_set="B1"),
            RecordingFactory(),
            "B0",
            PredictionCache(tmp_path),
            {"config": "c", "data": "d", "events": "e"},
            9,
        )


def test_oof_requires_config_data_and_event_hashes(synthetic_year_matrix, tmp_path):
    with pytest.raises(DataContractError, match="config.*data.*events"):
        generate_expanding_oof(
            synthetic_year_matrix,
            RecordingFactory(),
            "B0",
            PredictionCache(tmp_path),
            {"config": "complete-frozen-config"},
            9,
        )


def test_summary_rejects_inconsistent_fit_and_cache_counts(synthetic_year_matrix, tmp_path):
    summary = generate_expanding_oof(
        synthetic_year_matrix,
        RecordingFactory(),
        "B0",
        PredictionCache(tmp_path),
        {"config": "c", "data": "d", "events": "e"},
        9,
    )
    with pytest.raises(DataContractError, match="fit_count.*cache_hit_count"):
        replace(summary, fit_count=3)


@pytest.mark.parametrize("feature_set", ["B2", "", None])
def test_oof_rejects_unknown_feature_set(synthetic_year_matrix, tmp_path, feature_set):
    with pytest.raises((DataContractError, TypeError), match="feature_set"):
        generate_expanding_oof(
            synthetic_year_matrix,
            RecordingFactory(),
            feature_set,
            PredictionCache(tmp_path),
            {"config": "c", "data": "d", "events": "e"},
            9,
        )


@pytest.mark.parametrize("seed", [True, 1.5, "9"])
def test_oof_rejects_non_integer_seed(synthetic_year_matrix, tmp_path, seed):
    with pytest.raises((DataContractError, TypeError), match="seed"):
        generate_expanding_oof(
            synthetic_year_matrix,
            RecordingFactory(),
            "B0",
            PredictionCache(tmp_path),
            {"config": "c", "data": "d", "events": "e"},
            seed,
        )
