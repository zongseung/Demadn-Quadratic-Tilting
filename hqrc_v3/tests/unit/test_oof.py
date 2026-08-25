"""Actual fitted-preprocessing provenance across OOF/final cache boundaries."""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pytest

from hqrc_v3.contracts import ForecastMatrix
from hqrc_v3.features import feature_columns, history_columns
from hqrc_v3.oof import cache_key, generate_cached_final, generate_oof_stream
from hqrc_v3.provenance import ArtifactMismatch
from hqrc_v3.residuals import PredictionCache
from hqrc_v3.splits import expanding_oof_folds, final_fold

POPULATION = {
    "x": {
        "count": 1,
        "start": "2019-01-01T00:00:00",
        "end": "2019-01-01T00:00:00",
        "unit": "daily-sample",
    },
    "target": {
        "count": 24,
        "start": "2019-01-01T00:00:00",
        "end": "2019-01-01T23:00:00",
        "unit": "unique-hour",
    },
}
HASHES = {"config": "c", "data": "d", "events": "e"}


def _matrix(feature_set: str = "B0") -> ForecastMatrix:
    origins = np.array(
        [np.datetime64(f"{year}-01-01", "ns") for year in range(2019, 2025)]
    )
    target_times = origins[:, None] + np.arange(24).astype("timedelta64[h]")
    future = feature_columns(feature_set)  # type: ignore[arg-type]
    history = history_columns(feature_set)  # type: ignore[arg-type]
    return ForecastMatrix(
        origins=origins,
        target_times=target_times,
        history=np.zeros((len(origins), 168, len(history)), dtype=float),
        future=np.zeros((len(origins), 24, len(future)), dtype=float),
        target=np.full((len(origins), 24), 50_000.0),
        history_columns=history,
        future_columns=future,
    )


def test_oof_accepts_b1w_as_a_distinct_prediction_context(tmp_path) -> None:
    factory = PopulationFactory()

    result = generate_oof_stream(
        _matrix("B1W"),
        factory,
        "B1W",
        PredictionCache(tmp_path),
        HASHES,
        7,
        folds=(expanding_oof_folds()[0],),
    )

    assert factory.fit_calls == 1
    assert result.combined_frame["feature_set"].unique().to_list() == ["B1W"]


@dataclass(frozen=True)
class PopulationFitted:
    contract: dict[str, dict[str, object]]
    model_name: str = "population"

    def population_contract(self) -> dict[str, dict[str, object]]:
        return self.contract

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        return np.full(batch.target.shape, 49_000.0)


@dataclass
class PopulationFactory:
    contract: dict[str, dict[str, object]] = field(
        default_factory=lambda: json.loads(json.dumps(POPULATION))
    )
    name: str = "population"
    fit_calls: int = 0

    def fit(
        self,
        train: ForecastMatrix,
        validation: ForecastMatrix | None,
        seed: int,
    ) -> PopulationFitted:
        del train, validation, seed
        self.fit_calls += 1
        return PopulationFitted(self.contract)


def test_oof_actual_population_survives_cache_hit_and_process_restart(tmp_path) -> None:
    cache = PredictionCache(tmp_path)
    fold = expanding_oof_folds()[0]
    first_factory = PopulationFactory()

    first = generate_oof_stream(
        _matrix(),
        first_factory,
        "B0",
        cache,
        HASHES,
        7,
        folds=(fold,),
    )
    restarted_factory = PopulationFactory(
        contract={
            **POPULATION,
            "x": {**POPULATION["x"], "count": 999},
        }
    )
    restarted = generate_oof_stream(
        _matrix(),
        restarted_factory,
        "B0",
        PredictionCache(tmp_path),
        HASHES,
        7,
        folds=(fold,),
    )

    assert first_factory.fit_calls == 1
    assert restarted_factory.fit_calls == 0
    assert first.population_contracts == (POPULATION,)
    assert restarted.population_contracts == (POPULATION,)


def test_oof_cache_rejects_tampered_actual_population_metadata(tmp_path) -> None:
    cache = PredictionCache(tmp_path)
    fold = expanding_oof_folds()[0]
    generate_oof_stream(
        _matrix(),
        PopulationFactory(),
        "B0",
        cache,
        HASHES,
        7,
        folds=(fold,),
    )
    metadata_path = tmp_path / f"{cache_key('population', 'B0', 7, fold)}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["population_contract"]["x"]["count"] = 2
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises(ArtifactMismatch, match="metadata|population|integrity"):
        generate_oof_stream(
            _matrix(),
            PopulationFactory(),
            "B0",
            PredictionCache(tmp_path),
            HASHES,
            7,
            folds=(fold,),
        )


def test_final_actual_population_survives_cache_hit(tmp_path) -> None:
    first_factory = PopulationFactory()
    first = generate_cached_final(
        _matrix(),
        first_factory,
        "B0",
        PredictionCache(tmp_path),
        HASHES,
        7,
    )
    restarted_factory = PopulationFactory()
    restarted = generate_cached_final(
        _matrix(),
        restarted_factory,
        "B0",
        PredictionCache(tmp_path),
        HASHES,
        7,
    )

    assert first_factory.fit_calls == 1
    assert restarted_factory.fit_calls == 0
    assert first.population_contract == POPULATION
    assert restarted.population_contract == POPULATION
    assert cache_key("population", "B0", 7, final_fold())
