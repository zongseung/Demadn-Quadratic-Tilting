"""Cache-aware, fixed-configuration expanding-origin baseline forecasts.

Each invocation owns exactly one fitted seed stream.  Neural five-seed runs
therefore invoke this module once per seed and retain each seed's cache.  The
subsequent averaging of those streams into an HQRC residual stream belongs to
downstream evaluation/correction work, not to this orchestration layer.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import numpy as np
import polars as pl

from hqrc_v3.baselines.classical import predictions_to_frame
from hqrc_v3.baselines.preprocessing import (
    PopulationContract,
    validate_population_contract,
)
from hqrc_v3.baselines.protocol import BaselineFactory, FittedBaseline
from hqrc_v3.contracts import DataContractError, ForecastMatrix, validate_prediction_frame
from hqrc_v3.features import FeatureSet, feature_columns
from hqrc_v3.provenance import ArtifactMismatch
from hqrc_v3.residuals import PredictionCache
from hqrc_v3.splits import AnnualFold, expanding_oof_folds, final_fold, select_fold_samples

_FEATURE_SETS = frozenset(("B0", "B1", "B1W"))


def chronological_validation_tail(
    outer_train: ForecastMatrix,
    *,
    days: int,
) -> tuple[ForecastMatrix, ForecastMatrix]:
    """Split the final consecutive complete days from an outer training range.

    The caller first applies an immutable annual fold, so this function cannot
    admit evaluation-year rows.  It only partitions that already selected
    training matrix for fixed-configuration early stopping.
    """

    if not isinstance(outer_train, ForecastMatrix):
        raise TypeError("outer_train must be a ForecastMatrix")
    if isinstance(days, bool) or not isinstance(days, int) or days <= 0:
        raise ValueError("validation days must be a positive integer")
    count = outer_train.origins.shape[0]
    if count <= days:
        raise ValueError(f"training matrix must contain more than {days} complete days")
    if outer_train.target_times.shape != (count, 24):
        raise ValueError("training samples must contain 24 complete target hours")
    origins = np.asarray(outer_train.origins)
    targets = np.asarray(outer_train.target_times)
    if (
        not np.issubdtype(origins.dtype, np.datetime64)
        or not np.issubdtype(targets.dtype, np.datetime64)
        or np.isnat(origins).any()
        or np.isnat(targets).any()
    ):
        raise ValueError("training sample timestamps must be complete datetimes")
    expected_targets = origins[:, None] + np.arange(24).astype("timedelta64[h]")
    if not np.array_equal(targets.astype("datetime64[ns]"), expected_targets):
        raise ValueError("training samples must contain complete midnight-origin days")
    validation_block = origins[count - days - 1 :].astype("datetime64[D]")
    if not np.all(np.diff(validation_block) == np.timedelta64(1, "D")):
        raise ValueError("training samples must be consecutive daily samples")
    boundary = count - days
    return (
        outer_train.take(np.arange(boundary, dtype=np.int64)),
        outer_train.take(np.arange(boundary, count, dtype=np.int64)),
    )


def _require_feature_set(feature_set: object) -> FeatureSet:
    if feature_set not in _FEATURE_SETS:
        raise DataContractError("feature_set must be exactly 'B0', 'B1', or 'B1W'")
    return feature_set  # type: ignore[return-value]


def _require_seed(seed: object) -> int:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise DataContractError("seed must be an integer, not a boolean")
    return int(seed)


def _factory_name(factory: BaselineFactory) -> str:
    name = getattr(factory, "name", None)
    if not isinstance(name, str) or not name.strip():
        raise DataContractError("BaselineFactory.name must be a non-blank string")
    return name


def _fitted_population_contract(fitted: FittedBaseline) -> PopulationContract:
    reporter = getattr(fitted, "population_contract", None)
    if not callable(reporter):
        raise DataContractError(
            "fitted baseline must report its actual preprocessing population"
        )
    try:
        reported = reporter()
    except (TypeError, ValueError) as error:
        raise DataContractError(
            "fitted baseline returned an invalid preprocessing population"
        ) from error
    return validate_population_contract(reported)


def _cached_population_contract(value: object) -> PopulationContract:
    if value is None:
        raise ArtifactMismatch(
            "cached prediction artifact is missing fitted preprocessing population"
        )
    return validate_population_contract(value)


def _require_hashes(artifact_hashes: Mapping[str, str]) -> dict[str, str]:
    """Bind every cache artifact to immutable data, events, and frozen model config.

    ``config`` is the hash of the complete frozen experiment and selected
    model-config payload, rather than just the top-level experiment TOML.
    This prevents a different fixed model configuration sharing cached OOF
    predictions.  ``data`` and ``events`` are the corresponding source hashes.
    """

    if not isinstance(artifact_hashes, Mapping) or not artifact_hashes:
        raise DataContractError("artifact_hashes must be a non-empty mapping")
    unnormalized = dict(artifact_hashes)
    if any(
        not isinstance(key, str)
        or not key.strip()
        or not isinstance(value, str)
        or not value.strip()
        for key, value in unnormalized.items()
    ):
        raise DataContractError(
            "artifact hashes must have stable non-empty string names and values"
        )
    normalized = dict(sorted(unnormalized.items()))
    required = {"config", "data", "events"}
    if missing := sorted(required - set(normalized)):
        raise DataContractError(
            "artifact_hashes must include config, data, and events hashes; "
            f"missing {missing}"
        )
    return normalized


def _require_matrix_feature_set(matrix: ForecastMatrix, feature_set: FeatureSet) -> None:
    if not isinstance(matrix, ForecastMatrix):
        raise TypeError("matrix must be a ForecastMatrix")
    expected = feature_columns(feature_set)
    if matrix.future_columns != expected:
        raise DataContractError(
            "declared feature_set does not match ForecastMatrix future feature columns"
        )


def cache_key(model: str, feature_set: FeatureSet, seed: int, fold: AnnualFold) -> str:
    """Return a filesystem-safe key binding the complete immutable fit identity."""

    payload = json.dumps(
        {"feature_set": feature_set, "model": model, "seed": seed, "split_id": fold.split_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"prediction-{hashlib.sha256(payload).hexdigest()}"


def _validate_context(
    frame: pl.DataFrame,
    *,
    fold: AnnualFold,
    model: str,
    feature_set: FeatureSet,
    seed: int,
) -> pl.DataFrame:
    validated = validate_prediction_frame(frame, fold=fold)
    expected = {"model": model, "feature_set": feature_set, "seed": seed, "split_id": fold.split_id}
    actual = {name: validated[name].item(0) for name in expected}
    if actual != expected:
        raise DataContractError("prediction frame context does not match its requested OOF fit")
    return validated


def _select_nonempty(
    matrix: ForecastMatrix, fold: AnnualFold
) -> tuple[ForecastMatrix, ForecastMatrix]:
    train_indices, eval_indices = select_fold_samples(matrix, fold)
    if train_indices.size == 0 or eval_indices.size == 0:
        raise DataContractError(f"{fold.split_id} requires non-empty train and evaluation samples")
    return matrix.take(train_indices), matrix.take(eval_indices)


def _training_partitions(
    outer_train: ForecastMatrix,
    factory: BaselineFactory,
    validation_days: int | None,
) -> tuple[ForecastMatrix, ForecastMatrix | None]:
    if validation_days is None:
        return outer_train, None
    if not getattr(factory, "uses_validation_tail", False):
        return outer_train, None
    return chronological_validation_tail(outer_train, days=validation_days)


@dataclass(frozen=True)
class OOFRunSummary:
    """Validated prediction artifacts for all four immutable OOF folds."""

    frames: tuple[pl.DataFrame, ...]
    combined_frame: pl.DataFrame
    population_contracts: tuple[PopulationContract, ...]
    eval_years: tuple[int, ...]
    fit_count: int
    cache_hit_count: int
    model: str
    feature_set: FeatureSet
    seed: int

    def __post_init__(self) -> None:
        expected_folds = expanding_oof_folds()
        if self.eval_years != tuple(fold.eval_year for fold in expected_folds):
            raise DataContractError("OOF summary eval_years must be exactly 2020 through 2023")
        if len(self.frames) != len(expected_folds):
            raise DataContractError("OOF summary requires exactly four prediction frames")
        if len(self.population_contracts) != len(expected_folds):
            raise DataContractError("OOF summary requires one population per fold")
        for population in self.population_contracts:
            validate_population_contract(population)
        if (
            isinstance(self.fit_count, bool)
            or isinstance(self.cache_hit_count, bool)
            or not isinstance(self.fit_count, int)
            or not isinstance(self.cache_hit_count, int)
            or self.fit_count < 0
            or self.cache_hit_count < 0
            or self.fit_count + self.cache_hit_count != len(expected_folds)
        ):
            raise DataContractError("fit_count and cache_hit_count must sum to the four OOF folds")
        selected_feature_set = _require_feature_set(self.feature_set)
        selected_seed = _require_seed(self.seed)
        if not isinstance(self.model, str) or not self.model.strip():
            raise DataContractError("OOF summary model must be a non-blank string")
        if not isinstance(self.combined_frame, pl.DataFrame):
            raise DataContractError("OOF summary combined_frame must be a Polars DataFrame")
        validated_frames = tuple(
            _validate_context(
                frame,
                fold=fold,
                model=self.model,
                feature_set=selected_feature_set,
                seed=selected_seed,
            )
            for frame, fold in zip(self.frames, expected_folds, strict=True)
        )
        expected_combined = pl.concat(validated_frames, how="vertical")
        if not self.combined_frame.equals(expected_combined):
            raise DataContractError(
                "OOF summary combined_frame must equal the ordered validated frames"
            )

    @property
    def combined(self) -> pl.DataFrame:
        """Compatibility-friendly shorthand for the validated concatenated frame."""

        return self.combined_frame

    @classmethod
    def from_frames(
        cls,
        frames: list[pl.DataFrame],
        *,
        model: str,
        feature_set: FeatureSet,
        seed: int,
        fit_count: int,
        cache_hit_count: int,
        population_contracts: list[PopulationContract],
    ) -> OOFRunSummary:
        expected_folds = expanding_oof_folds()
        if len(frames) != len(expected_folds):
            raise DataContractError("OOF summary requires exactly four prediction frames")
        if len(population_contracts) != len(expected_folds):
            raise DataContractError("OOF summary requires one population per fold")
        by_split: dict[str, pl.DataFrame] = {}
        for frame in frames:
            validated = validate_prediction_frame(frame)
            split_id = str(validated["split_id"].item(0))
            if split_id in by_split:
                raise DataContractError("OOF summary rejects duplicate folds")
            by_split[split_id] = validated
        ordered: list[pl.DataFrame] = []
        for fold in expected_folds:
            frame = by_split.get(fold.split_id)
            if frame is None:
                raise DataContractError("OOF summary requires every immutable OOF fold")
            ordered.append(
                _validate_context(
                    frame,
                    fold=fold,
                    model=model,
                    feature_set=feature_set,
                    seed=seed,
                )
            )
        combined = pl.concat(ordered, how="vertical")
        duplicate_targets = combined.group_by(
            "model", "feature_set", "seed", "split_id", "target_timestamp"
        ).len()
        if not duplicate_targets.filter(pl.col("len") > 1).is_empty():
            raise DataContractError("OOF summary rejects duplicate prediction targets")
        return cls(
            frames=tuple(ordered),
            combined_frame=combined,
            population_contracts=tuple(
                validate_population_contract(population)
                for population in population_contracts
            ),
            eval_years=tuple(fold.eval_year for fold in expected_folds),
            fit_count=fit_count,
            cache_hit_count=cache_hit_count,
            model=model,
            feature_set=feature_set,
            seed=seed,
        )


@dataclass(frozen=True)
class OOFStreamResult:
    """A cache-aware OOF stream over an explicit immutable fold subset."""

    frames: tuple[pl.DataFrame, ...]
    combined_frame: pl.DataFrame
    population_contracts: tuple[PopulationContract, ...]
    folds: tuple[AnnualFold, ...]
    fit_count: int
    cache_hit_count: int
    model: str
    feature_set: FeatureSet
    seed: int


@dataclass(frozen=True)
class FinalBaselineResult:
    """The one final fitted baseline and its validated 2024 prediction frame."""

    model: FittedBaseline
    frame: pl.DataFrame
    population_contract: PopulationContract

    @property
    def fitted_model(self) -> FittedBaseline:
        """Explicit alias for callers that prefer the protocol terminology."""

        return self.model


def generate_expanding_oof(
    matrix: ForecastMatrix,
    factory: BaselineFactory,
    feature_set: FeatureSet,
    cache: PredictionCache,
    artifact_hashes: Mapping[str, str],
    seed: int,
    *,
    validation_days: int | None = None,
) -> OOFRunSummary:
    """Generate exactly the four frozen OOF folds without selection or tuning."""

    stream = generate_oof_stream(
        matrix,
        factory,
        feature_set,
        cache,
        artifact_hashes,
        seed,
        folds=expanding_oof_folds(),
        validation_days=validation_days,
    )
    return OOFRunSummary.from_frames(
        list(stream.frames),
        model=stream.model,
        feature_set=stream.feature_set,
        seed=stream.seed,
        fit_count=stream.fit_count,
        cache_hit_count=stream.cache_hit_count,
        population_contracts=list(stream.population_contracts),
    )


def generate_oof_stream(
    matrix: ForecastMatrix,
    factory: BaselineFactory,
    feature_set: FeatureSet,
    cache: PredictionCache,
    artifact_hashes: Mapping[str, str],
    seed: int,
    *,
    folds: tuple[AnnualFold, ...],
    validation_days: int | None = None,
) -> OOFStreamResult:
    """Generate a declared subset of the four immutable OOF folds."""

    selected_feature_set = _require_feature_set(feature_set)
    selected_seed = _require_seed(seed)
    _require_matrix_feature_set(matrix, selected_feature_set)
    model = _factory_name(factory)
    hashes = _require_hashes(artifact_hashes)
    if not isinstance(cache, PredictionCache):
        raise TypeError("cache must be a PredictionCache")
    allowed = {fold.split_id: fold for fold in expanding_oof_folds()}
    if (
        not isinstance(folds, tuple)
        or not folds
        or len({fold.split_id for fold in folds}) != len(folds)
        or any(allowed.get(fold.split_id) != fold for fold in folds)
    ):
        raise DataContractError("OOF stream folds must be a unique immutable OOF subset")

    frames: list[pl.DataFrame] = []
    population_contracts: list[PopulationContract] = []
    fit_count = 0
    cache_hit_count = 0
    for fold in folds:
        key = cache_key(model, selected_feature_set, selected_seed, fold)
        cached = cache.read_entry(key, expected_hashes=hashes)
        if cached is not None:
            frames.append(
                _validate_context(
                    cached.frame,
                    fold=fold,
                    model=model,
                    feature_set=selected_feature_set,
                    seed=selected_seed,
                )
            )
            population_contracts.append(
                _cached_population_contract(cached.population_contract)
            )
            cache_hit_count += 1
            continue
        outer_train, evaluation = _select_nonempty(matrix, fold)
        train, validation = _training_partitions(
            outer_train, factory, validation_days
        )
        fitted = factory.fit(train, validation=validation, seed=selected_seed)
        population = _fitted_population_contract(fitted)
        prediction = fitted.predict(evaluation)
        frame = predictions_to_frame(
            evaluation,
            prediction,
            model=model,
            feature_set=selected_feature_set,
            seed=selected_seed,
            fold=fold,
        )
        cached_entry = cache.write_entry(
            key,
            frame,
            hashes=hashes,
            population_contract=population,
        )
        frames.append(
            _validate_context(
                cached_entry.frame,
                fold=fold,
                model=model,
                feature_set=selected_feature_set,
                seed=selected_seed,
            )
        )
        population_contracts.append(
            _cached_population_contract(cached_entry.population_contract)
        )
        fit_count += 1
    return OOFStreamResult(
        frames=tuple(frames),
        combined_frame=pl.concat(frames, how="vertical"),
        population_contracts=tuple(population_contracts),
        folds=folds,
        model=model,
        feature_set=selected_feature_set,
        seed=selected_seed,
        fit_count=fit_count,
        cache_hit_count=cache_hit_count,
    )


def fit_final_baseline(
    matrix: ForecastMatrix,
    factory: BaselineFactory,
    feature_set: FeatureSet,
    *,
    seed: int,
    validation_days: int | None = None,
) -> FinalBaselineResult:
    """Fit only 2019--2023 and predict available complete 2024 targets."""

    selected_feature_set = _require_feature_set(feature_set)
    selected_seed = _require_seed(seed)
    _require_matrix_feature_set(matrix, selected_feature_set)
    model = _factory_name(factory)
    fold = final_fold()
    outer_train, evaluation = _select_nonempty(matrix, fold)
    train, validation = _training_partitions(outer_train, factory, validation_days)
    fitted = factory.fit(train, validation=validation, seed=selected_seed)
    population = _fitted_population_contract(fitted)
    frame = predictions_to_frame(
        evaluation,
        fitted.predict(evaluation),
        model=model,
        feature_set=selected_feature_set,
        seed=selected_seed,
        fold=fold,
    )
    return FinalBaselineResult(
        model=fitted,
        frame=_validate_context(
            frame,
            fold=fold,
            model=model,
            feature_set=selected_feature_set,
            seed=selected_seed,
        ),
        population_contract=population,
    )


@dataclass(frozen=True)
class CachedFinalResult:
    """One final prediction frame with explicit fit/cache accounting."""

    frame: pl.DataFrame
    population_contract: PopulationContract
    fit_count: int
    cache_hit_count: int
    model: str
    feature_set: FeatureSet
    seed: int

    def __post_init__(self) -> None:
        if self.fit_count + self.cache_hit_count != 1:
            raise DataContractError("final fit_count and cache_hit_count must sum to one")
        _validate_context(
            self.frame,
            fold=final_fold(),
            model=self.model,
            feature_set=_require_feature_set(self.feature_set),
            seed=_require_seed(self.seed),
        )
        validate_population_contract(self.population_contract)


def generate_cached_final(
    matrix: ForecastMatrix,
    factory: BaselineFactory,
    feature_set: FeatureSet,
    cache: PredictionCache,
    artifact_hashes: Mapping[str, str],
    seed: int,
    *,
    validation_days: int | None = None,
) -> CachedFinalResult:
    """Generate or reuse the one immutable 2019--2023 to 2024 stream."""

    selected_feature_set = _require_feature_set(feature_set)
    selected_seed = _require_seed(seed)
    _require_matrix_feature_set(matrix, selected_feature_set)
    model = _factory_name(factory)
    hashes = _require_hashes(artifact_hashes)
    if not isinstance(cache, PredictionCache):
        raise TypeError("cache must be a PredictionCache")
    fold = final_fold()
    key = cache_key(model, selected_feature_set, selected_seed, fold)
    cached = cache.read_entry(key, expected_hashes=hashes)
    if cached is not None:
        return CachedFinalResult(
            frame=_validate_context(
                cached.frame,
                fold=fold,
                model=model,
                feature_set=selected_feature_set,
                seed=selected_seed,
            ),
            population_contract=_cached_population_contract(
                cached.population_contract
            ),
            fit_count=0,
            cache_hit_count=1,
            model=model,
            feature_set=selected_feature_set,
            seed=selected_seed,
        )
    result = fit_final_baseline(
        matrix,
        factory,
        selected_feature_set,
        seed=selected_seed,
        validation_days=validation_days,
    )
    cached_entry = cache.write_entry(
        key,
        result.frame,
        hashes=hashes,
        population_contract=result.population_contract,
    )
    return CachedFinalResult(
        frame=_validate_context(
            cached_entry.frame,
            fold=fold,
            model=model,
            feature_set=selected_feature_set,
            seed=selected_seed,
        ),
        population_contract=_cached_population_contract(
            cached_entry.population_contract
        ),
        fit_count=1,
        cache_hit_count=0,
        model=model,
        feature_set=selected_feature_set,
        seed=selected_seed,
    )
