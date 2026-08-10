# HQRC v3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an isolated, reproducible HQRC v3 pipeline that generates expanding-window OOF residuals for 2020--2023, calibrates an AR(1) prior from residual diagnostics, fits the paper's hierarchical correction, and evaluates a final 2019--2023-to-2024 forecast.

**Architecture:** A standalone `hqrc_v3/` src-layout project keeps Polars/Arrow data contracts separate from NumPy/Torch model boundaries. Fixed baseline configurations are fitted once per expanding fold and once on all pre-2024 data; cached long-form predictions feed residual diagnostics, Bayesian correction, ablations, and evaluation without retraining. The PyMC HQRC model uses event-reset stationary AR(1) likelihood terms and reads a hash-bound Beta prior calibration artifact.

**Tech Stack:** Python 3.10+, Polars 1.38+, NumPy 2.2+, scikit-learn 1.7+, XGBoost 3.2+, LightGBM 4.6+, PyTorch 2.8+, PyMC 5.25+, ArviZ 0.22+, statsmodels 0.14+, SciPy 1.15+, Matplotlib 3.10+; optional nutpie for Rust NUTS benchmarking.

## Global Constraints

- Create or modify implementation files only under `hqrc_v3/`; existing research code and artifacts outside that directory are read-only.
- Use `power_demand_final.csv` through a configured relative or absolute path; never copy or rewrite the source data.
- The final baseline fit uses 2019--2023 and predicts 2024; 2024 observations must never influence configuration, AR calibration, HQRC fitting, or model selection.
- Generate HQRC training residuals with exactly four expanding folds: `2019 -> 2020`, `2019--2020 -> 2021`, `2019--2021 -> 2022`, and `2019--2022 -> 2023`.
- Hyperparameters are fixed across OOF folds and the final refit; no per-fold search is allowed.
- Use `gap_days=0`; enforce chronology by requiring every 24-hour target interval to remain inside its partition.
- B0 contains ordinary temporal/weather predictors but no holiday-derived predictor; B1 is B0 plus the six versioned holiday feature families from the design spec.
- Treat an occurrence, not an hour, as the independent unit for pooling and inferential resampling.
- AR lag pairs must reset at every event boundary; calibrate `(phi + 1) / 2 ~ Beta(a, b)` from pre-evaluation residuals and keep `phi` as a posterior parameter.
- A corrected predictive draw is `y_hat + sigma_N * (q + e)`; never add a second baseline residual draw to it.
- Keep Polars/Arrow data until an estimator boundary and cache predictions as Parquet; do not add a custom Rust crate unless a measured profile meets the design spec's 20% bottleneck gate.
- All random behavior accepts an explicit seed; tests use CPU and small synthetic data.
- Follow test-driven development: observe each named test fail before implementing its production code.

---

## File Map

- `hqrc_v3/pyproject.toml`: standalone package metadata, pytest/ruff configuration, slow-test marker.
- `hqrc_v3/configs/experiment.toml`: input mapping, fixed folds, model/sampler defaults.
- `hqrc_v3/configs/events.csv`: versioned ten-occurrence event registry.
- `hqrc_v3/configs/holiday_calendar.csv`: 2019--2024 Seollal/Chuseok calendar used only for B1 feature distances.
- `hqrc_v3/configs/model_spaces.toml`: fixed paper baseline parameters and optional one-shot candidate sets.
- `hqrc_v3/src/hqrc_v3/config.py`: typed TOML loading and cross-field validation.
- `hqrc_v3/src/hqrc_v3/contracts.py`: forecast arrays and long-form prediction schema.
- `hqrc_v3/src/hqrc_v3/provenance.py`: hashes, run manifest, immutable artifact checks.
- `hqrc_v3/src/hqrc_v3/data.py`: hourly input loading/audit and daily 168-to-24 sample construction.
- `hqrc_v3/src/hqrc_v3/events.py`: event registry validation and event-relative columns.
- `hqrc_v3/src/hqrc_v3/features.py`: B0/B1 feature derivation and leakage guard.
- `hqrc_v3/src/hqrc_v3/splits.py`: exact expanding/final fold contracts and sample selection.
- `hqrc_v3/src/hqrc_v3/residuals.py`: residual calculation, non-event scale, standardized events, cache I/O.
- `hqrc_v3/src/hqrc_v3/baselines/protocol.py`: common fitted-baseline interfaces and registry.
- `hqrc_v3/src/hqrc_v3/baselines/classical.py`: 24 horizon-specific XGBoost, LightGBM, and SVR adapters.
- `hqrc_v3/src/hqrc_v3/baselines/sequence.py`: Seq2Seq-LSTM, time-series Transformer, trainer, seed ensemble.
- `hqrc_v3/src/hqrc_v3/oof.py`: expanding OOF and final-refit orchestration.
- `hqrc_v3/src/hqrc_v3/diagnostics/ar.py`: event-reset ACF/PACF, phi estimates, Beta calibration, approval artifact.
- `hqrc_v3/src/hqrc_v3/bayes/model.py`: H1--H4 PyMC model construction with H3 as full HQRC.
- `hqrc_v3/src/hqrc_v3/bayes/samplers.py`: PyMC/nutpie sampling and diagnostics gate.
- `hqrc_v3/src/hqrc_v3/bayes/predictive.py`: new-event shape and joint event trajectory draws.
- `hqrc_v3/src/hqrc_v3/corrections/variants.py`: H0--H4 orchestration and pooling structures.
- `hqrc_v3/src/hqrc_v3/corrections/similar_day.py`: H5 profile-scaling competitor.
- `hqrc_v3/src/hqrc_v3/evaluation/metrics.py`: point/probabilistic metrics.
- `hqrc_v3/src/hqrc_v3/evaluation/inference.py`: event bootstrap, Wilcoxon, HAC-DM diagnostics, Holm correction.
- `hqrc_v3/src/hqrc_v3/evaluation/reports.py`: normalized tables/figures and sampler benchmark report.
- `hqrc_v3/src/hqrc_v3/cli.py`: staged commands and artifact dependency checks.
- `hqrc_v3/tests/`: unit, integration, synthetic fixture, and opt-in slow tests.

### Task 1: Project scaffold, typed configuration, event registry, and provenance

**Files:**
- Create: `hqrc_v3/pyproject.toml`
- Create: `hqrc_v3/src/hqrc_v3/__init__.py`
- Create: `hqrc_v3/src/hqrc_v3/config.py`
- Create: `hqrc_v3/src/hqrc_v3/events.py`
- Create: `hqrc_v3/src/hqrc_v3/provenance.py`
- Create: `hqrc_v3/configs/experiment.toml`
- Create: `hqrc_v3/configs/events.csv`
- Create: `hqrc_v3/configs/holiday_calendar.csv`
- Create: `hqrc_v3/configs/model_spaces.toml`
- Create: `hqrc_v3/tests/unit/test_config.py`
- Create: `hqrc_v3/tests/unit/test_events.py`
- Create: `hqrc_v3/tests/unit/test_provenance.py`

**Interfaces:**
- Consumes: only standard-library TOML/CSV/path/hash APIs.
- Produces: `ExperimentConfig`, `load_config(path: Path) -> ExperimentConfig`, `EventOccurrence`, `load_event_registry(path: Path) -> tuple[EventOccurrence, ...]`, `load_holiday_calendar(path: Path) -> tuple[EventOccurrence, ...]`, `file_sha256(path: Path) -> str`, `RunManifest`, and `assert_artifact_compatible(manifest: RunManifest, *, data_sha256: str, config_sha256: str, event_sha256: str) -> None`.

- [ ] **Step 1: Write configuration, registry, and hash tests**

```python
def test_default_config_fixes_expanding_years_and_zero_gap(tmp_path):
    config = load_config(PROJECT_ROOT / "configs/experiment.toml")
    assert config.data.gap_days == 0
    assert config.split.oof_years == (2020, 2021, 2022, 2023)
    assert config.split.final_year == 2024

def test_registry_has_exactly_five_occurrences_per_type():
    events = load_event_registry(PROJECT_ROOT / "configs/events.csv")
    assert len(events) == 10
    assert Counter(event.holiday_type for event in events) == {
        "seollal": 5,
        "chuseok": 5,
    }
    assert next(e for e in events if e.occurrence_id == "seollal-2022").restriction == 1

def test_feature_calendar_includes_2019_but_correction_registry_does_not():
    events = load_event_registry(PROJECT_ROOT / "configs/events.csv")
    calendar = load_holiday_calendar(PROJECT_ROOT / "configs/holiday_calendar.csv")
    assert min(event.central_date.year for event in events) == 2020
    assert min(event.central_date.year for event in calendar) == 2019
    assert len(events) == 10
    assert len(calendar) == 12

def test_manifest_rejects_changed_data_hash(tmp_path):
    manifest = RunManifest(data_sha256="aaa", config_sha256="bbb", event_sha256="ccc")
    with pytest.raises(ArtifactMismatch, match="data_sha256"):
        assert_artifact_compatible(manifest, data_sha256="changed", config_sha256="bbb", event_sha256="ccc")
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_config.py hqrc_v3/tests/unit/test_events.py hqrc_v3/tests/unit/test_provenance.py -q`

Expected: collection fails because `hqrc_v3.config`, `hqrc_v3.events`, and `hqrc_v3.provenance` do not exist.

- [ ] **Step 3: Implement immutable dataclasses and exact registry validation**

```python
@dataclass(frozen=True)
class SplitConfig:
    first_train_year: int
    oof_years: tuple[int, ...]
    final_year: int

    def __post_init__(self) -> None:
        if self.oof_years != (2020, 2021, 2022, 2023) or self.final_year != 2024:
            raise ConfigError("HQRC v3 requires OOF 2020-2023 and final year 2024")

@dataclass(frozen=True)
class EventOccurrence:
    occurrence_id: str
    holiday_type: Literal["seollal", "chuseok"]
    central_date: date
    official_start: date
    official_end: date
    restriction: int

    @property
    def window_start(self) -> date:
        return self.official_start - timedelta(days=1)

    @property
    def window_end(self) -> date:
        return self.official_end + timedelta(days=1)
```

Write all ten correction rows and all twelve feature-calendar rows exactly as specified in the approved design. Reject duplicate ids/dates, invalid ordering, overlapping event windows, non-binary restriction values, or a type/year count other than one. The 2019 calendar rows must never be returned by `load_event_registry`.

- [ ] **Step 4: Run focused tests and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_config.py hqrc_v3/tests/unit/test_events.py hqrc_v3/tests/unit/test_provenance.py -q`

Expected: all tests pass.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add hqrc_v3
git commit -m "feat(hqrc-v3): add configuration and event contracts"
```

### Task 2: Hourly data audit, B0/B1 features, and 168-to-24 samples

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/contracts.py`
- Create: `hqrc_v3/src/hqrc_v3/data.py`
- Create: `hqrc_v3/src/hqrc_v3/features.py`
- Create: `hqrc_v3/tests/fixtures/hourly.py`
- Create: `hqrc_v3/tests/unit/test_data.py`
- Create: `hqrc_v3/tests/unit/test_features.py`
- Create: `hqrc_v3/tests/unit/test_samples.py`

**Interfaces:**
- Consumes: `ExperimentConfig`, the 12-row feature calendar, and the 10-row correction registry from Task 1.
- Produces: `ForecastMatrix`, `read_hourly_data`, `audit_hourly_data`, `attach_calendar_features`, `feature_columns(feature_set)`, `assert_no_holiday_leakage`, and `build_daily_forecast_matrix`.

- [ ] **Step 1: Write audit, leakage, and shape tests**

```python
def test_audit_rejects_one_missing_hour(hourly_frame):
    broken = hourly_frame.filter(pl.col("timestamp") != datetime(2023, 1, 2, 5))
    with pytest.raises(DataContractError, match="one-hour continuity"):
        audit_hourly_data(broken, expected_start=None, expected_end=None, expected_rows=None)

def test_b0_has_no_holiday_derived_columns(hourly_frame, holiday_calendar):
    featured = attach_calendar_features(hourly_frame, holiday_calendar)
    b0 = feature_columns("B0")
    assert_no_holiday_leakage(b0)
    assert not ({"is_holiday", "holiday_type", "event_relative_day"} & set(b0))
    assert set(b0) <= set(featured.columns)

def test_daily_matrix_is_168_to_24(hourly_frame, holiday_calendar):
    matrix = build_daily_forecast_matrix(
        attach_calendar_features(hourly_frame, holiday_calendar), feature_set="B1"
    )
    assert matrix.history.shape[1] == 168
    assert matrix.future.shape[1] == 24
    assert matrix.target.shape[1] == 24
    assert np.all(np.diff(matrix.target_times.astype("datetime64[h]"), axis=1) == 1)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_data.py hqrc_v3/tests/unit/test_features.py hqrc_v3/tests/unit/test_samples.py -q`

Expected: imports fail for the new contracts/data/features modules.

- [ ] **Step 3: Implement Polars-first feature and sample construction**

```python
@dataclass(frozen=True)
class ForecastMatrix:
    origins: np.ndarray
    target_times: np.ndarray
    history: np.ndarray
    future: np.ndarray
    target: np.ndarray
    history_columns: tuple[str, ...]
    future_columns: tuple[str, ...]

    def take(self, indices: np.ndarray) -> "ForecastMatrix":
        return replace(
            self,
            origins=self.origins[indices],
            target_times=self.target_times[indices],
            history=self.history[indices],
            future=self.future[indices],
            target=self.target[indices],
        )
```

Map `일시`, `hm`, `ta`, and `power demand(MW)` explicitly. Add hour/day-of-week/weekend, annual and weekly sine/cosine pairs, heating/cooling degree hours, and the configured B1 holiday features. Keep Polars expressions lazy until collection; perform one NumPy conversion when constructing `ForecastMatrix`.

- [ ] **Step 4: Run focused tests and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_data.py hqrc_v3/tests/unit/test_features.py hqrc_v3/tests/unit/test_samples.py -q`

Expected: all tests pass.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add hqrc_v3/src/hqrc_v3 hqrc_v3/tests
git commit -m "feat(hqrc-v3): build audited forecast samples"
```

### Task 3: Expanding/final split contracts, prediction schema, residual scale, and cache

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/splits.py`
- Create: `hqrc_v3/src/hqrc_v3/residuals.py`
- Modify: `hqrc_v3/src/hqrc_v3/contracts.py`
- Create: `hqrc_v3/tests/unit/test_splits.py`
- Create: `hqrc_v3/tests/unit/test_predictions.py`
- Create: `hqrc_v3/tests/unit/test_residuals.py`
- Create: `hqrc_v3/tests/unit/test_cache.py`

**Interfaces:**
- Consumes: `ForecastMatrix`, event columns, and provenance hashes.
- Produces: `AnnualFold`, `expanding_oof_folds()`, `final_fold()`, `select_fold_samples`, `validate_prediction_frame`, `compute_fold_scale`, `standardize_event_residuals`, `PredictionCache.write`, and `PredictionCache.read`.

- [ ] **Step 1: Write exact fold, leakage, scale, and immutable-cache tests**

```python
def test_expanding_folds_are_exact():
    assert [(f.train_end.year, f.eval_year) for f in expanding_oof_folds()] == [
        (2019, 2020), (2020, 2021), (2021, 2022), (2022, 2023)
    ]
    assert (final_fold().train_end.year, final_fold().eval_year) == (2023, 2024)

def test_partition_requires_complete_target(matrix):
    fold = AnnualFold.oof(eval_year=2023, first_train_year=2019)
    train, evaluation = select_fold_samples(matrix, fold)
    assert matrix.target_times[train].max() < np.datetime64("2023-01-01")
    assert matrix.target_times[evaluation].min() >= np.datetime64("2023-01-01")

def test_scale_uses_only_non_event_rows(prediction_frame):
    frame = prediction_frame.with_columns(
        pl.Series("is_event", [False, False, True]),
        pl.Series("residual_mw", [3.0, 4.0, 1000.0]),
    )
    assert compute_fold_scale(frame) == pytest.approx(np.sqrt(12.5))

def test_cache_refuses_different_config_hash(tmp_path, prediction_frame):
    cache = PredictionCache(tmp_path)
    cache.write("xgb-B0-2020", prediction_frame, hashes={"config": "a", "data": "d"})
    with pytest.raises(ArtifactMismatch, match="config"):
        cache.read("xgb-B0-2020", expected_hashes={"config": "b", "data": "d"})
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_splits.py hqrc_v3/tests/unit/test_predictions.py hqrc_v3/tests/unit/test_residuals.py hqrc_v3/tests/unit/test_cache.py -q`

Expected: split and residual modules are missing.

- [ ] **Step 3: Implement exact long-form schema and atomic Parquet cache**

```python
PREDICTION_COLUMNS = (
    "origin", "target_timestamp", "horizon", "observed_mw", "predicted_mw",
    "model", "feature_set", "seed", "split_id",
)

@dataclass(frozen=True)
class AnnualFold:
    split_id: str
    train_start: date
    train_end: date
    eval_start: date
    eval_end: date
    eval_year: int
    kind: Literal["oof", "final"]
```

Validate 24 horizons per origin, target uniqueness, finite MW values, and no evaluation target at or before `train_end`. Cache by writing Parquet and JSON metadata to temporary sibling paths, fsyncing, then replacing final paths. If an existing artifact has matching hashes return it; if hashes differ raise `ArtifactMismatch` without overwriting.

Use this exact cache interface:

```python
class PredictionCache:
    def read(self, key: str, expected_hashes: Mapping[str, str]) -> pl.DataFrame | None:
        """Return None when absent; return a compatible frame; raise on hash mismatch."""

    def write(
        self, key: str, frame: pl.DataFrame, hashes: Mapping[str, str]
    ) -> pl.DataFrame:
        """Atomically persist a validated frame and return it."""
```

- [ ] **Step 4: Run focused tests and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_splits.py hqrc_v3/tests/unit/test_predictions.py hqrc_v3/tests/unit/test_residuals.py hqrc_v3/tests/unit/test_cache.py -q`

Expected: all tests pass.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add hqrc_v3/src/hqrc_v3 hqrc_v3/tests
git commit -m "feat(hqrc-v3): add expanding split and residual cache"
```

### Task 4: Common baseline protocol and classical 24-horizon models

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/baselines/__init__.py`
- Create: `hqrc_v3/src/hqrc_v3/baselines/protocol.py`
- Create: `hqrc_v3/src/hqrc_v3/baselines/classical.py`
- Create: `hqrc_v3/tests/unit/test_baseline_protocol.py`
- Create: `hqrc_v3/tests/integration/test_classical_baselines.py`

**Interfaces:**
- Consumes: train/evaluation `ForecastMatrix` subsets and fixed model dictionaries.
- Produces: `BaselineFactory`, `FittedBaseline`, `HorizonRegressor`, `make_classical_baseline(name, params)`, `select_fixed_baseline_config`, and `predictions_to_frame`.

- [ ] **Step 1: Write protocol and three-model synthetic tests**

```python
@pytest.mark.parametrize("name", ["xgboost", "lightgbm", "svr"])
def test_classical_baseline_returns_24_horizons(name, tiny_forecast_matrix):
    baseline = make_classical_baseline(name, tiny_params(name))
    fitted = baseline.fit(tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=7)
    prediction = fitted.predict(tiny_forecast_matrix.take(np.arange(20, 24)))
    assert prediction.shape == (4, 24)
    assert np.isfinite(prediction).all()

def test_horizon_regressor_builds_distinct_estimators(tiny_forecast_matrix):
    fitted = make_classical_baseline("svr", {"C": 1.0, "epsilon": 0.1, "gamma": "scale"}).fit(
        tiny_forecast_matrix.take(np.arange(20)), validation=None, seed=7
    )
    assert len({id(model) for model in fitted.estimators}) == 24

def test_one_shot_selection_returns_lowest_validation_rmse(fake_candidate_factory, tiny_forecast_matrix):
    selected = select_fixed_baseline_config(
        candidates=({"quality": 3.0}, {"quality": 1.0}),
        factory_builder=fake_candidate_factory,
        train=tiny_forecast_matrix.take(np.arange(16)),
        validation=tiny_forecast_matrix.take(np.arange(16, 24)),
        seed=7,
    )
    assert selected.params == {"quality": 1.0}
    assert selected.metric == "rmse"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_baseline_protocol.py hqrc_v3/tests/integration/test_classical_baselines.py -q`

Expected: baseline modules are missing.

- [ ] **Step 3: Implement flattened 168-hour adapters with fixed params**

```python
class BaselineFactory(Protocol):
    name: str
    def fit(
        self, train: ForecastMatrix, validation: ForecastMatrix | None, seed: int
    ) -> "FittedBaseline":
        raise NotImplementedError

class FittedBaseline(Protocol):
    model_name: str
    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        raise NotImplementedError
```

Flatten history and concatenate future-known covariates separately for each horizon. Clone 24 estimators, set all available random-state/thread parameters explicitly, and never run an internal search. Support small test parameters as well as versioned paper defaults from `model_spaces.toml`.

`select_fixed_baseline_config` is an explicit one-shot validation helper: it fits each declared candidate on 2019--2022, evaluates 2023 RMSE, applies deterministic first-in-order tie breaking, and writes the selected dictionary plus candidate scores. OOF orchestration consumes only that frozen dictionary and never calls this helper.

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_baseline_protocol.py hqrc_v3/tests/integration/test_classical_baselines.py -q`

Expected: all tests pass.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add hqrc_v3/src/hqrc_v3/baselines hqrc_v3/tests
git commit -m "feat(hqrc-v3): add classical baseline adapters"
```

### Task 5: Seq2Seq-LSTM and time-series Transformer baselines

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/baselines/sequence.py`
- Create: `hqrc_v3/tests/unit/test_sequence_models.py`
- Create: `hqrc_v3/tests/integration/test_sequence_training.py`

**Interfaces:**
- Consumes: baseline protocol and `ForecastMatrix`.
- Produces: `SequenceTrainingConfig`, `Seq2SeqLSTM`, `TimeSeriesTransformer`, `TorchBaselineFactory`, and `fit_seed_ensemble`.

- [ ] **Step 1: Write forward-shape, future-covariate, and deterministic smoke tests**

```python
@pytest.mark.parametrize("model_cls", [Seq2SeqLSTM, TimeSeriesTransformer])
def test_sequence_forward_shape(model_cls):
    model = model_cls(history_features=5, future_features=3, hidden_size=8, dropout=0.0)
    result = model(torch.zeros(2, 168, 5), torch.zeros(2, 24, 3))
    assert result.shape == (2, 24)

def test_seed_ensemble_is_reproducible(tiny_forecast_matrix):
    config = SequenceTrainingConfig(epochs=2, patience=2, batch_size=8, seeds=(3, 5))
    first = fit_seed_ensemble("lstm", tiny_forecast_matrix, None, config).predict(tiny_forecast_matrix)
    second = fit_seed_ensemble("lstm", tiny_forecast_matrix, None, config).predict(tiny_forecast_matrix)
    np.testing.assert_allclose(first, second, rtol=0, atol=1e-6)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_sequence_models.py hqrc_v3/tests/integration/test_sequence_training.py -q`

Expected: `baselines.sequence` is missing.

- [ ] **Step 3: Implement compact CPU-safe sequence learners**

```python
@dataclass(frozen=True)
class SequenceTrainingConfig:
    hidden_size: int = 64
    layers: int = 2
    heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3
    batch_size: int = 64
    epochs: int = 60
    patience: int = 8
    seeds: tuple[int, ...] = (11, 23, 37, 41, 53)
```

The LSTM encodes 168 history steps and decodes 24 known-future covariate steps from the final state. The Transformer projects history and future streams to the same hidden size, adds learned positional embeddings, and uses a decoder without target-demand leakage. Restore the best validation state; when validation is absent, train the fixed epoch count. Average seed predictions but retain seed-level prediction frames.

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_sequence_models.py hqrc_v3/tests/integration/test_sequence_training.py -q`

Expected: all tests pass on CPU.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add hqrc_v3/src/hqrc_v3/baselines/sequence.py hqrc_v3/tests
git commit -m "feat(hqrc-v3): add sequence baseline adapters"
```

### Task 6: Expanding OOF orchestration, final refit, and staged CLI

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/oof.py`
- Create: `hqrc_v3/src/hqrc_v3/cli.py`
- Modify: `hqrc_v3/pyproject.toml`
- Create: `hqrc_v3/tests/integration/test_oof_pipeline.py`
- Create: `hqrc_v3/tests/integration/test_cli_oof.py`

**Interfaces:**
- Consumes: folds, feature matrices, baseline registry, prediction schema, and cache.
- Produces: `generate_expanding_oof`, `fit_final_baseline`, `OOFRunSummary`, and CLI commands `audit-data`, `tune-baselines`, `generate-oof`, and `fit-final-baselines`.

- [ ] **Step 1: Write fake-baseline orchestration tests**

```python
def test_oof_calls_exact_train_eval_years(synthetic_year_matrix, recording_factory, tmp_path):
    summary = generate_expanding_oof(
        matrix=synthetic_year_matrix,
        factory=recording_factory,
        feature_set="B0",
        cache=PredictionCache(tmp_path),
        artifact_hashes={"config": "config-hash", "data": "data-hash", "events": "event-hash"},
        seed=9,
    )
    assert recording_factory.calls == [
        ((2019,), 2020),
        ((2019, 2020), 2021),
        ((2019, 2020, 2021), 2022),
        ((2019, 2020, 2021, 2022), 2023),
    ]
    assert summary.eval_years == (2020, 2021, 2022, 2023)

def test_final_fit_uses_2019_through_2023_only(synthetic_year_matrix, recording_factory):
    fit_final_baseline(synthetic_year_matrix, recording_factory, "B1", seed=9)
    assert recording_factory.final_train_years == (2019, 2020, 2021, 2022, 2023)
    assert recording_factory.final_eval_year == 2024

def test_oof_never_invokes_tuning(synthetic_year_matrix, recording_factory, tmp_path):
    generate_expanding_oof(
        matrix=synthetic_year_matrix,
        factory=recording_factory,
        feature_set="B0",
        cache=PredictionCache(tmp_path),
        artifact_hashes={"config": "c", "data": "d", "events": "e"},
        seed=9,
    )
    assert recording_factory.tune_calls == 0
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/integration/test_oof_pipeline.py hqrc_v3/tests/integration/test_cli_oof.py -q`

Expected: orchestration and CLI modules are missing.

- [ ] **Step 3: Implement cache-aware orchestration without per-fold tuning**

```python
def generate_expanding_oof(
    matrix: ForecastMatrix,
    factory: BaselineFactory,
    feature_set: Literal["B0", "B1"],
    cache: PredictionCache,
    artifact_hashes: Mapping[str, str],
    seed: int,
) -> OOFRunSummary:
    frames: list[pl.DataFrame] = []
    for fold in expanding_oof_folds():
        key = cache_key(factory.name, feature_set, seed, fold)
        cached = cache.read(key, expected_hashes=artifact_hashes)
        if cached is not None:
            frames.append(cached)
            continue
        train_idx, eval_idx = select_fold_samples(matrix, fold)
        fitted = factory.fit(matrix.take(train_idx), validation=None, seed=seed)
        frame = predictions_to_frame(fitted.predict(matrix.take(eval_idx)), matrix.take(eval_idx), factory.name, feature_set, seed, fold.split_id)
        frames.append(cache.write(key, frame, hashes=artifact_hashes))
    return OOFRunSummary.from_frames(frames)
```

Add console entry point `hqrc = "hqrc_v3.cli:main"` and argparse subcommands with nonzero exit status on contract failures.

- [ ] **Step 4: Run orchestration tests and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/integration/test_oof_pipeline.py hqrc_v3/tests/integration/test_cli_oof.py -q`

Expected: all tests pass.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add hqrc_v3
git commit -m "feat(hqrc-v3): orchestrate expanding OOF forecasts"
```

### Task 7: Event-reset AR diagnostics and data-derived Beta prior approval

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/diagnostics/__init__.py`
- Create: `hqrc_v3/src/hqrc_v3/diagnostics/ar.py`
- Create: `hqrc_v3/tests/unit/test_ar_diagnostics.py`
- Create: `hqrc_v3/tests/integration/test_ar_artifact.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`

**Interfaces:**
- Consumes: standardized event residual frame and provenance hashes.
- Produces: `EventARDiagnostic`, `ARCalibration`, `detrend_event_residuals`, `estimate_event_phi`, `calibrate_beta_prior`, `write_ar_diagnostics`, `approve_calibration`, and CLI commands `diagnose-ar`/`approve-ar-calibration`.

- [ ] **Step 1: Write boundary-reset, formula, concentration, and hash tests**

```python
def test_phi_never_pairs_two_events():
    residuals = {
        "event-a": np.array([1.0, 0.5]),
        "event-b": np.array([100.0, 50.0]),
    }
    result = estimate_event_phis(residuals)
    assert result == {"event-a": pytest.approx(0.5), "event-b": pytest.approx(0.5)}

def test_beta_calibration_is_finite_and_capped():
    calibration = calibrate_beta_prior(np.array([0.84, 0.88, 0.91, 0.86]))
    assert calibration.a > 1.0
    assert calibration.b > 1.0
    assert 8.0 <= calibration.a + calibration.b <= 40.0
    assert calibration.phi_center > 0.8

def test_approval_rejects_changed_residual_hash(tmp_path, calibration):
    proposed = write_calibration(tmp_path / "proposed.json", calibration, residual_sha256="abc")
    with pytest.raises(ArtifactMismatch, match="residual"):
        approve_calibration(proposed, tmp_path / "approved.json", current_residual_sha256="def")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_ar_diagnostics.py hqrc_v3/tests/integration/test_ar_artifact.py -q`

Expected: AR diagnostics module is missing.

- [ ] **Step 3: Implement the approved robust calibration rule**

```python
def estimate_event_phi(residual: np.ndarray) -> float:
    denominator = float(np.dot(residual[:-1], residual[:-1]))
    if denominator <= np.finfo(float).eps:
        raise ARCalibrationError("event residual has zero lag variance")
    return float(np.clip(np.dot(residual[:-1], residual[1:]) / denominator, -0.98, 0.98))

def calibrate_beta_prior(phi: np.ndarray) -> ARCalibration:
    u = (np.asarray(phi, dtype=float) + 1.0) / 2.0
    center = float(np.clip(np.median(u), 0.10, 0.975))
    robust_sd = max(1.4826 * float(np.median(np.abs(u - np.median(u)))), 0.025)
    kappa = float(np.clip(center * (1.0 - center) / robust_sd**2 - 1.0, 8.0, 40.0))
    a = max(1.05, center * kappa)
    b = max(1.05, (1.0 - center) * kappa)
    if a + b > 40.0:
        scale = 40.0 / (a + b)
        a, b = a * scale, b * scale
    return ARCalibration(a=a, b=b, phi_center=2.0 * center - 1.0, event_count=len(u))
```

Before phi estimation, remove occurrence-specific quadratic time and three Fourier hour harmonics. Save raw/detrended ACF and PACF lags 1--48, Bartlett bounds, AR innovation diagnostics, Ljung--Box lag 24, occurrence ids, and hashes. Approval copies the exact suggested values plus `approved=true`; fitting rejects a proposal-only artifact.

- [ ] **Step 4: Run tests and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_ar_diagnostics.py hqrc_v3/tests/integration/test_ar_artifact.py -q`

Expected: all tests pass.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 5: Commit**

```bash
git add hqrc_v3
git commit -m "feat(hqrc-v3): calibrate AR prior from OOF residuals"
```

### Task 8: Hierarchical HQRC model, event-reset AR likelihood, and sampler gate

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/bayes/__init__.py`
- Create: `hqrc_v3/src/hqrc_v3/bayes/model.py`
- Create: `hqrc_v3/src/hqrc_v3/bayes/samplers.py`
- Create: `hqrc_v3/tests/unit/test_ar_likelihood.py`
- Create: `hqrc_v3/tests/unit/test_hqrc_model.py`
- Create: `hqrc_v3/tests/integration/test_hqrc_sampling.py`

**Interfaces:**
- Consumes: standardized event rows and an approved `ARCalibration`.
- Produces: `HQRCData`, `HQRCModelOptions`, `build_hqrc_model(data, calibration, variant, pooling, options)`, `stationary_ar1_logp_numpy`, `sample_hqrc`, and `validate_inference_data`.

- [ ] **Step 1: Write numerical likelihood, model-structure, and tiny-sampling tests**

```python
def test_stationary_ar1_logp_resets_at_segments():
    segments = [np.array([1.0, 0.5]), np.array([2.0, 1.0])]
    got = stationary_ar1_logp_numpy(segments, phi=0.5, sigma=1.0)
    stationary_sd = 1.0 / np.sqrt(1.0 - 0.25)
    expected = sum(
        stats.norm.logpdf(segment[0], 0.0, stationary_sd)
        + stats.norm.logpdf(segment[1] - 0.5 * segment[0], 0.0, 1.0)
        for segment in segments
    )
    assert got == pytest.approx(expected)

def test_h3_contains_required_random_variables(tiny_hqrc_data, approved_calibration):
    model = build_hqrc_model(
        tiny_hqrc_data,
        approved_calibration,
        variant="H3",
        pooling="partial",
        options=HQRCModelOptions(),
    )
    assert {"mu", "delta", "between_scale", "gamma", "phi", "sigma_r"} <= set(model.named_vars)

def test_diagonal_no_restriction_sensitivity_omits_correlation_and_delta(tiny_hqrc_data, approved_calibration):
    model = build_hqrc_model(
        tiny_hqrc_data,
        approved_calibration,
        variant="H3",
        pooling="partial",
        options=HQRCModelOptions(covariance="diagonal", include_restriction=False),
    )
    assert "delta" not in model.named_vars
    assert not any("corr" in name for name in model.named_vars)

@pytest.mark.slow
def test_tiny_hqrc_sampling_returns_finite_posterior(tiny_hqrc_data, approved_calibration):
    idata = sample_hqrc(tiny_hqrc_data, approved_calibration, draws=30, tune=30, chains=2, seed=5)
    assert np.isfinite(idata.posterior["phi"]).all()
```

- [ ] **Step 2: Run fast tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_ar_likelihood.py hqrc_v3/tests/unit/test_hqrc_model.py -q`

Expected: Bayesian modules are missing.

- [ ] **Step 3: Implement non-centered partial pooling and AR potentials**

```python
def stationary_ar1_logp_numpy(segments: Sequence[np.ndarray], phi: float, sigma: float) -> float:
    stationary_sd = sigma / np.sqrt(1.0 - phi**2)
    total = 0.0
    for segment in segments:
        total += float(stats.norm.logpdf(segment[0], 0.0, stationary_sd))
        total += float(stats.norm.logpdf(segment[1:] - phi * segment[:-1], 0.0, sigma).sum())
    return total
```

In PyMC, loop over the two holiday types for `LKJCholeskyCov(eta=2)`, construct occurrence coefficients non-centrally, center each 24-hour gamma profile to zero, and add cyclic first-difference log-density. Define `u_phi ~ Beta(a,b)` and deterministic `phi = 2*u_phi-1`. For each occurrence slice, add a stationary initial Normal logp and conditional innovation Normal logp; never concatenate slices. Support `variant` H1/H2/H3/H4 and `pooling` complete/partial/none with the parameter omissions defined in the design.

Use this options contract so sensitivity fits are predeclared rather than ad hoc:

```python
@dataclass(frozen=True)
class HQRCModelOptions:
    covariance: Literal["full", "diagonal"] = "full"
    include_restriction: bool = True
    lkj_eta: float = 2.0
    between_scale_prior: float = 1.0
    innovation: Literal["normal_ar1", "student_t_ar1", "normal_ar2"] = "normal_ar1"
```

Reject non-positive `lkj_eta`/scale values. `student_t_ar1` estimates `nu_minus_two ~ Exponential(1/10)` and uses `nu=nu_minus_two+2`; `normal_ar2` uses two stable partial-autocorrelation parameters and resets both initial observations at each event. These are sensitivity fits only; H3 paper results use the defaults.

- [ ] **Step 4: Implement sampler and diagnostic gate**

```python
def validate_inference_data(idata: az.InferenceData, *, paper_profile: bool) -> SamplingDiagnostics:
    summary = az.summary(idata, kind="diagnostics")
    limit = 400 if paper_profile else 20
    diagnostics = SamplingDiagnostics(
        max_rhat=float(summary["r_hat"].max()),
        min_bulk_ess=float(summary["ess_bulk"].min()),
        min_tail_ess=float(summary["ess_tail"].min()),
        divergences=int(idata.sample_stats["diverging"].sum()),
    )
    if paper_profile and (diagnostics.max_rhat > 1.01 or diagnostics.min_bulk_ess < limit or diagnostics.min_tail_ess < limit or diagnostics.divergences):
        raise SamplingError(str(diagnostics))
    return diagnostics
```

Use PyMC as default and import nutpie only when requested. Preserve backend, versions, elapsed time, and diagnostics in the returned result.

- [ ] **Step 5: Run tests, including opt-in sampler smoke, and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_ar_likelihood.py hqrc_v3/tests/unit/test_hqrc_model.py -q`

Expected: fast tests pass.

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/integration/test_hqrc_sampling.py -m slow -q`

Expected: tiny sampler test passes with finite posterior values; it is not labeled a paper run.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 6: Commit**

```bash
git add hqrc_v3
git commit -m "feat(hqrc-v3): implement hierarchical AR correction model"
```

### Task 9: Predictive draws, H0--H5, pooling ablation, and event-level evaluation

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/bayes/predictive.py`
- Create: `hqrc_v3/src/hqrc_v3/corrections/__init__.py`
- Create: `hqrc_v3/src/hqrc_v3/corrections/variants.py`
- Create: `hqrc_v3/src/hqrc_v3/corrections/similar_day.py`
- Create: `hqrc_v3/src/hqrc_v3/evaluation/__init__.py`
- Create: `hqrc_v3/src/hqrc_v3/evaluation/metrics.py`
- Create: `hqrc_v3/src/hqrc_v3/evaluation/inference.py`
- Create: `hqrc_v3/tests/unit/test_predictive.py`
- Create: `hqrc_v3/tests/unit/test_metrics.py`
- Create: `hqrc_v3/tests/unit/test_event_inference.py`
- Create: `hqrc_v3/tests/integration/test_variants.py`

**Interfaces:**
- Consumes: posterior samples, baseline prediction frames, scales, event registry, and observed values.
- Produces: `draw_new_event_correction`, `corrected_predictive_draws`, `baseline_bootstrap_draws`, H0--H5 runners, `run_event_loeo`, point/probabilistic metric frames, PSIS-LOO summaries, and event-level inference results.

- [ ] **Step 1: Write no-double-count, metric, bootstrap, and variant tests**

```python
def test_corrected_predictive_has_no_second_baseline_noise():
    draws = corrected_predictive_draws(
        baseline=np.array([100.0, 100.0]),
        sigma_n=10.0,
        q=np.array([[1.0, 2.0]]),
        e=np.array([[0.5, -0.5]]),
    )
    np.testing.assert_allclose(draws, np.array([[115.0, 115.0]]))

def test_event_bootstrap_resamples_whole_events():
    result = bootstrap_event_median(
        {"a": np.array([1.0, 1.0]), "b": np.array([9.0, 9.0])},
        draws=100,
        seed=2,
    )
    assert result.resampled_units == "event"

def test_loeo_excludes_heldout_event_from_fit_and_ar_calibration(tiny_event_frames, recording_loeo_backend):
    run_event_loeo(tiny_event_frames, recording_loeo_backend)
    for held_out, fit_events, ar_events in recording_loeo_backend.calls:
        assert held_out not in fit_events
        assert held_out not in ar_events
        assert set(fit_events) == set(ar_events)

@pytest.mark.parametrize("variant", ["H0", "H1", "H2", "H3", "H4", "H5"])
def test_variant_returns_same_timestamps(variant, tiny_variant_context):
    result = run_variant(variant, tiny_variant_context)
    assert result["target_timestamp"].to_list() == tiny_variant_context.timestamps
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_predictive.py hqrc_v3/tests/unit/test_metrics.py hqrc_v3/tests/unit/test_event_inference.py hqrc_v3/tests/integration/test_variants.py -q`

Expected: predictive, correction, and evaluation modules are missing.

- [ ] **Step 3: Implement joint event trajectories and all correction contracts**

```python
def corrected_predictive_draws(
    baseline: np.ndarray,
    sigma_n: float,
    q: np.ndarray,
    e: np.ndarray,
) -> np.ndarray:
    baseline = np.asarray(baseline, dtype=float)
    if q.shape != e.shape or q.shape[1] != baseline.size:
        raise PredictiveShapeError("q/e draws must match baseline timestamps")
    return baseline[None, :] + sigma_n * (q + e)
```

Generate `beta_new` from holiday population draws, simulate AR trajectories jointly with a stationary reset, and use posterior means for point correction. H0 uses horizon-preserving non-event residual blocks. H1/H2/H3/H4 call the matching model variant. H5 selects prior occurrences of the same holiday type only, normalizes each daily profile by its daily mean, averages profiles equally by occurrence, and estimates one training-only least-squares scale per event-relative day.

- [ ] **Step 4: Implement event-level metrics and inference**

Compute RMSE, MAE, guarded MAPE, SMAPE, R2, empirical CRPS from draws, pinball losses for 0.05--0.95, and 50%/90% coverage. Wilcoxon receives ten paired event summaries; bootstrap samples occurrence ids with replacement; Holm adjusts within the declared comparison family. HAC-DM remains a secondary diagnostic and reports the bandwidth and event-block bootstrap seed.

`run_event_loeo` receives all 2020--2024 occurrence frames, removes one occurrence, recalibrates AR from the remaining ids, fits the requested correction, and predicts only the held-out occurrence. Mark the result `causal=False` because later occurrences may train an earlier holdout. Use `az.loo` on pointwise log likelihood for H1--H4 and preserve ELPD, standard error, and Pareto-k diagnostics.

- [ ] **Step 5: Run tests and lint**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_predictive.py hqrc_v3/tests/unit/test_metrics.py hqrc_v3/tests/unit/test_event_inference.py hqrc_v3/tests/integration/test_variants.py -q`

Expected: all tests pass.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] **Step 6: Commit**

```bash
git add hqrc_v3
git commit -m "feat(hqrc-v3): add corrections and event evaluation"
```

### Task 10: Reports, Rust-sampler benchmark, end-to-end smoke, and operator documentation

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/evaluation/reports.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Create: `hqrc_v3/tests/integration/test_report_pipeline.py`
- Create: `hqrc_v3/tests/integration/test_end_to_end.py`
- Create: `hqrc_v3/tests/slow/test_real_data_smoke.py`
- Create: `hqrc_v3/README.md`

**Interfaces:**
- Consumes: all versioned artifacts from Tasks 1--9.
- Produces: CLI commands `fit-corrections`, `run-ablations`, `benchmark-samplers`, and `report`; normalized CSV/Parquet tables, figures, benchmark JSON, and reproducible run instructions.

- [ ] **Step 1: Write report completeness and synthetic end-to-end tests**

```python
def test_report_refuses_missing_ar_approval(synthetic_run_dir):
    with pytest.raises(ReportContractError, match="approved AR calibration"):
        build_report(synthetic_run_dir)

def test_synthetic_end_to_end_writes_required_artifacts(tmp_path, synthetic_config):
    run = run_synthetic_pipeline(synthetic_config, output_dir=tmp_path)
    assert (run / "manifest.json").is_file()
    assert (run / "predictions/oof.parquet").is_file()
    assert (run / "ar_diagnostics/approved.json").is_file()
    assert (run / "metrics/event_metrics.parquet").is_file()
    assert (run / "COMPLETE").is_file()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/integration/test_report_pipeline.py hqrc_v3/tests/integration/test_end_to_end.py -q`

Expected: report builder and final commands are missing.

- [ ] **Step 3: Implement normalized reporting and sampler benchmark**

```python
@dataclass(frozen=True)
class SamplerBenchmark:
    backend: str
    wall_seconds: float
    peak_rss_mb: float
    min_bulk_ess_per_second: float
    min_tail_ess_per_second: float
    max_rhat: float
    divergences: int
```

Run PyMC and nutpie against the same model/data/seed/draw profile when nutpie is installed. Mark nutpie `eligible_default=true` only when posterior means agree within 0.1 pooled posterior SD, divergences do not increase, R-hat remains within 0.01, and either wall time improves by at least 20% or minimum bulk ESS/s improves by at least 20%. An unavailable optional backend is reported as `status="not-installed"`, not treated as a pipeline failure.

- [ ] **Step 4: Document the exact operator workflow**

The README must list the approved command sequence, explain that expanding OOF generates HQRC targets while the 2019--2023 refit produces the sole 2024 baseline, identify slow/full profiles, map each result table to the manuscript, and state that only a `paper` run satisfying posterior diagnostics may populate paper numbers.

- [ ] **Step 5: Run integration, full fast suite, lint, and real-data audit smoke**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/integration/test_report_pipeline.py hqrc_v3/tests/integration/test_end_to_end.py -q`

Expected: integration tests pass.

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m "not slow" -q`

Expected: all fast tests pass.

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/slow/test_real_data_smoke.py -m slow -q`

Expected: the real 51,144-row file passes audit and a tiny classical-baseline fold produces a schema-valid cached prediction; full five-baseline/NUTS paper execution is not performed by this smoke test.

- [ ] **Step 6: Commit**

```bash
git add hqrc_v3
git commit -m "feat(hqrc-v3): complete reproducible experiment workflow"
```

### Task 11: Frozen paper baselines and concrete OOF/final execution

**Files:**
- Modify: `hqrc_v3/configs/model_spaces.toml`
- Create: `hqrc_v3/src/hqrc_v3/baselines/config.py`
- Modify: `hqrc_v3/src/hqrc_v3/baselines/classical.py`
- Modify: `hqrc_v3/src/hqrc_v3/baselines/sequence.py`
- Modify: `hqrc_v3/src/hqrc_v3/oof.py`
- Create: `hqrc_v3/src/hqrc_v3/baselines/paper.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Modify: `hqrc_v3/README.md`
- Create: `hqrc_v3/tests/unit/test_paper_baseline_config.py`
- Create: `hqrc_v3/tests/integration/test_paper_baseline_stages.py`
- Create: `hqrc_v3/tests/slow/test_real_paper_baseline_stage.py`

**Interfaces:**
- Consumes: the immutable experiment/event/calendar files, the real hourly source, and a hash-verified `model_spaces.toml`.
- Produces: `PaperBaselineConfig`, `load_paper_baselines`, `make_paper_factory`, `run_paper_oof_stage`, `run_paper_final_stage`, concrete `generate-oof`/`fit-final-baselines` handlers, member-level neural predictions, and five-seed ensemble predictions.

- [ ] **Step 1: Write exact-config, validation-tail, ensemble, artifact, and CLI tests**

The frozen registry must contain only the five manuscript baselines:

```python
assert tuple(config.models) == (
    "xgboost", "lightgbm", "svr", "seq2seq_lstm", "transformer"
)
assert config.seq2seq_lstm.layers == 2
assert config.transformer.layers == 1
assert config.seq2seq_lstm.seeds == config.transformer.seeds == (11, 23, 37, 41, 53)
assert config.validation_days == 61
```

XGBoost, LightGBM, and SVR must remain 24 independent direct-horizon estimators. Seq2Seq-LSTM and Transformer must jointly output 24 hours and receive no target-demand decoder input. The last 61 complete daily samples inside each pre-evaluation training range form an internal, chronological early-stopping partition; evaluation-year samples must never enter it. This is training control, not per-fold model selection. All declared parameters remain identical across four OOF fits and the final fit.

Tests must prove that:

- the versioned config and its SHA-256 select the same five immutable factories;
- XGBoost/LightGBM use their internal validation tail for early stopping while never accepting an evaluation-year row;
- SVR keeps the declared RBF `C`, `epsilon`, and `gamma` values;
- LSTM has two recurrent layers, Transformer has one encoder and one decoder layer, both have hidden size 64/dropout 0.1/learning rate `1e-3`/batch 64/max 60 epochs;
- each neural stage retains all five member frames and creates a pointwise arithmetic-mean ensemble frame under one unambiguous ensemble identity;
- B0 and B1 are independently constructed and cached;
- OOF output contains exactly 2020--2023 and final output exactly 2024;
- a hash mismatch, partial output, wrong model name, changed seed set, or changed feature schema fails closed;
- rerunning a completed stream is a cache hit and does not fit again.

- [ ] **Step 2: Run focused tests and verify RED**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_paper_baseline_config.py hqrc_v3/tests/integration/test_paper_baseline_stages.py -q`

Expected: imports or concrete CLI execution fail because the frozen registry and stages do not exist.

- [ ] **Step 3: Freeze the manuscript model definitions**

Use the manuscript/previously versioned baseline values, without a search stage:

- XGBoost: squared-error objective, RMSE evaluation, `n_estimators=600`, learning rate `0.03`, depth 6, minimum child weight 3, row subsample 0.9, column subsample 0.8, L2 1, histogram trees, early stopping 35 rounds.
- LightGBM: regression objective, `n_estimators=800`, learning rate `0.025`, 31 leaves, minimum child samples 20, row subsample 0.9 with frequency 1, column subsample 0.8, L2 1, early stopping 35 rounds.
- SVR: RBF kernel, `C=10`, `epsilon=0.05`, `gamma="scale"`, cache size 512 MB.
- Seq2Seq-LSTM: two layers, hidden size 64, dropout 0.1, learning rate `1e-3`, batch 64, at most 60 epochs, patience 8, seeds 11/23/37/41/53.
- Transformer: hidden size 64, four heads, one encoder and one decoder layer, dropout 0.1, learning rate `1e-3`, batch 64, at most 60 epochs, patience 8, the same five seeds.

Outer orchestration may parallelize independent model/feature/seed streams, but every estimator/torch member must use one internal CPU thread so worker count is explicit and oversubscription is prevented. Do not introduce a Rust baseline implementation; Rust remains eligible only after the measured 20% bottleneck gate.

- [ ] **Step 4: Implement immutable stage publication and real CLI handlers**

`generate-oof` and `fit-final-baselines` must load and audit the source, load the holiday calendar, build the selected B0/B1 matrix, verify the frozen config hash, instantiate the requested manuscript model, execute the exact folds, and atomically publish validated Parquet plus JSON provenance. Add an all-model operator that resumes stream by stream, preserves individual neural seed predictions, and publishes ensemble OOF/final files only after all required streams validate.

The final products are:

```text
predictions/oof_members.parquet
predictions/oof.parquet
predictions/final_2024_members.parquet
predictions/final_2024.parquet
predictions/baseline_manifest.json
```

`oof.parquet` and `final_2024.parquet` contain one prediction per model/B0-or-B1/timestamp: classical streams directly and neural streams as the exact five-seed mean. Member files retain every seed. Publication must bind raw-data, experiment, model-config, event-registry, and holiday-calendar hashes. No command may invoke tuning.

- [ ] **Step 5: Verify focused, full-fast, lint, and real one-stream smoke**

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_paper_baseline_config.py hqrc_v3/tests/integration/test_paper_baseline_stages.py -q`

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m "not slow" -q`

Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/slow/test_real_paper_baseline_stage.py -m slow -q`

The slow test may use one real LightGBM-B1 OOF fold with reduced rounds under an explicitly non-paper smoke profile; it must traverse the same loader/factory/publication code. It must not silently substitute another model.

- [ ] **Step 6: Commit**

```bash
git add hqrc_v3 docs/superpowers/plans/2026-08-10-hqrc-v3-implementation.md
git commit -m "feat(hqrc-v3): execute frozen paper baselines"
```

## Final Verification

- [ ] Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m "not slow" -q`

Expected: every fast test passes.

- [ ] Run: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/slow/test_real_data_smoke.py -m slow -q`

Expected: real-data smoke passes.

- [ ] Run: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`

Expected: no diagnostics.

- [ ] Run: `git status --short`

Expected: only intentionally ignored run/workspace artifacts are untracked; all `hqrc_v3` source, config, tests, and documentation are committed.
