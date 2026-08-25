# HQT Reviewer B1W 실험 구현 계획

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** HQT event window와 동일한 휴일 더미를 사용하는 B1W 베이스라인, 10-event LOEO, 2024 causal holdout, 논문용 표·그림을 하나의 재현 가능한 실행 경로로 구현한다.

**Architecture:** 기존 B0/B1과 frozen model registry는 보존하고 B1W를 새로운 feature schema로 추가한다. Paper baseline publisher는 legacy suite `(B0, B1)`와 reviewer suite `(B0, B1W)`를 구분하며, shared prediction cache를 통해 B0 fit을 재사용한다. Reviewer 경로는 AR 없는 conference HQT만 호출하고 retrospective LOEO와 causal-2024를 서로 다른 artifact namespace에 기록한다.

**Tech Stack:** Python 3.12, Polars/Arrow/Parquet, NumPy, SciPy, PyMC, ArviZ, XGBoost, LightGBM, scikit-learn, PyTorch, Matplotlib, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-26-hqt-reviewer-experiment-design.md`

## Global Constraints

- 코드 feature-set 식별자는 `B1W`, 논문 표시명은 holiday-aware baseline `HA`다.
- `is_seollal_window`와 `is_chuseok_window`는 `official_start - 1일 <= date <= official_end + 1일`에서만 1이다.
- 중앙일 ±2일을 하드코딩하지 않는다. 대체공휴일이 있는 occurrence는 중앙일 +3일까지 포함될 수 있다.
- 기존 `is_seollal`/`is_chuseok`은 공식 연휴 구간에서만 1이고 `is_public_holiday`는 buffer 날짜 때문에 변경하지 않는다.
- B1W history/future width는 정확히 19/16이며 모든 베이스라인이 같은 열 순서를 사용한다.
- 기존 `configs/model_spaces.toml`의 내용과 SHA-256은 변경하지 않는다. B1W semantics는 feature schema manifest에 versioning한다.
- Paper baseline suite는 legacy `(B0, B1)` 또는 reviewer `(B0, B1W)` 중 정확히 하나여야 한다.
- B0/B1 residual이나 posterior를 B1W correction에 재사용하지 않는다.
- H2는 iid Gaussian quadratic partial-pooling conference HQT이며 AR, hour profile, pandemic covariate를 포함하지 않는다.
- Paper sampler는 4 chains, chain당 1,000 warm-up, 1,000 retained draws, `target_accept=0.99`를 최소 계약으로 사용한다.
- Reviewer 주 결과는 B1W-H0 대 B1W-H2다. H2-taper와 scale stability는 보조 민감도 결과로 분리한다.
- 10-event LOEO는 retrospective 분석이고, 2024 causal holdout은 2020--2023 OOF 잔차만 학습에 사용한다.
- 결과가 나오기 전에 개선 방향이나 통계적 유의성을 코드·문서에 고정하지 않는다.
- 기존 dirty worktree의 legacy HQT 초안은 사용자 작업으로 취급하여 삭제하거나 덮어쓰지 않고 Task 1에서 검증·통합한다.

---

### Task 1: AR 없는 conference HQT 초안 안정화

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/bayes/legacy_hqt.py`
- Modify: `hqrc_v3/src/hqrc_v3/legacy_hqt_loeo.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Modify: `hqrc_v3/tests/unit/test_legacy_hqt.py`
- Modify: `hqrc_v3/README.md`
- Inspect and include only if dependency resolution requires it: `hqrc_v3/uv.lock`

**Interfaces:**
- Consumes: `HQRCData`, `LOEOFold`, `EventResidualContext`, existing standardized residual artifact schema.
- Produces: `build_legacy_hqt_data_from_frame(frame: pl.DataFrame, occurrence_ids: tuple[str, ...]) -> HQRCData`, `sample_legacy_hqt(...) -> az.InferenceData`, `run_legacy_hqt_loeo(...) -> tuple[LegacyHQTContextResult, ...]`.

- [ ] **Step 1: 현재 dirty 초안의 범위를 기록하고 generic frame builder의 실패 테스트를 추가한다**

```python
def test_generic_legacy_builder_accepts_eight_causal_training_events() -> None:
    frame = _legacy_frame(event_years=range(2020, 2024), feature_set="B1W")
    ids = tuple(frame["occurrence_id"].unique(maintain_order=True))

    data = build_legacy_hqt_data_from_frame(frame, occurrence_ids=ids)

    assert len(data.occurrence_ids) == 8
    assert tuple(np.unique(data.holiday_type_index)) == (0, 1)
    assert data.observations.size == frame.height


def test_generic_legacy_builder_rejects_unordered_occurrence_rows() -> None:
    frame = _legacy_frame(event_years=range(2020, 2024), feature_set="B1W").sort(
        "occurrence_id", descending=True
    )
    ids = tuple(sorted(frame["occurrence_id"].unique().to_list()))

    with pytest.raises(LegacyHQTError, match="occurrence ordering"):
        build_legacy_hqt_data_from_frame(frame, occurrence_ids=ids)
```

- [ ] **Step 2: 추가한 테스트가 현재 초안에서 실패하는지 확인한다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_legacy_hqt.py -q`

Expected: FAIL because `build_legacy_hqt_data_from_frame` is not defined.

- [ ] **Step 3: LOEO와 causal evaluator가 공유할 generic builder를 구현한다**

```python
def build_legacy_hqt_data_from_frame(
    frame: pl.DataFrame, *, occurrence_ids: tuple[str, ...]
) -> HQRCData:
    required = {
        "holiday_type",
        "hour",
        "occurrence_id",
        "restriction",
        "standardized_residual",
        "tau_days",
    }
    if not required.issubset(frame.columns) or not occurrence_ids:
        raise LegacyHQTError("legacy HQT training frame schema differs")
    observed_order = tuple(frame["occurrence_id"].unique(maintain_order=True).to_list())
    if observed_order != occurrence_ids:
        raise LegacyHQTError("legacy HQT occurrence ordering differs")
    event_index = {event_id: index for index, event_id in enumerate(occurrence_ids)}
    return HQRCData(
        observations=frame["standardized_residual"].to_numpy(),
        occurrence_index=np.asarray(
            [event_index[value] for value in frame["occurrence_id"].to_list()], dtype=np.int64
        ),
        holiday_type_index=np.asarray(
            [_HOLIDAY_INDEX[value] for value in frame["holiday_type"].to_list()], dtype=np.int64
        ),
        tau_days=frame["tau_days"].to_numpy(),
        hour=frame["hour"].to_numpy(),
        restriction=frame["restriction"].to_numpy(),
        occurrence_ids=occurrence_ids,
    )
```

`build_legacy_hqt_data(fold)`는 physical 9-event LOEO 검증을 유지한 뒤 이 함수를 호출한다. 기존 model graph test는 `phi`, `gamma`, `delta`가 없고 `innovation == "iid-normal"`임을 계속 검증한다.

- [ ] **Step 4: legacy HQT unit tests와 정적 검사를 통과시킨다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_legacy_hqt.py -q`

Expected: PASS.

Run: `uv run --project hqrc_v3 --locked ruff check hqrc_v3/src/hqrc_v3/bayes/legacy_hqt.py hqrc_v3/src/hqrc_v3/legacy_hqt_loeo.py hqrc_v3/tests/unit/test_legacy_hqt.py`

Expected: PASS with no findings.

- [ ] **Step 5: 기존 초안을 하나의 검토 가능한 커밋으로 만든다**

```bash
git add hqrc_v3/src/hqrc_v3/bayes/legacy_hqt.py \
  hqrc_v3/src/hqrc_v3/legacy_hqt_loeo.py \
  hqrc_v3/src/hqrc_v3/cli.py \
  hqrc_v3/tests/unit/test_legacy_hqt.py \
  hqrc_v3/README.md
git commit -m "feat(hqrc-v3): add AR-free conference HQT runner"
```

`hqrc_v3/uv.lock`은 `uv lock --check` 또는 `uv sync --locked`가 해당 파일을 요구하고 현재 `pyproject.toml`과 일치할 때만 같은 커밋에 포함한다.

---

### Task 2: B1W calendar feature와 schema contract

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/features.py`
- Modify: `hqrc_v3/src/hqrc_v3/events.py`
- Modify: `hqrc_v3/src/hqrc_v3/baselines/config.py`
- Modify: `hqrc_v3/tests/unit/test_features.py`
- Modify: `hqrc_v3/tests/unit/test_events.py`
- Modify: `hqrc_v3/tests/unit/test_paper_baseline_config.py`

**Interfaces:**
- Consumes: `EventOccurrence.window_start`, `EventOccurrence.window_end`, feature용 holiday calendar, correction event registry.
- Produces: `FeatureSet = Literal["B0", "B1", "B1W"]`, `B1W_WINDOW_COLUMNS`, `B1W_WINDOW_VERSION`, `feature_columns("B1W")`, `history_columns("B1W")`, `validate_feature_event_alignment(...)`.

- [ ] **Step 1: B1W 경계와 폭을 고정하는 실패 테스트를 작성한다**

```python
def test_b1w_uses_registry_windows_not_fixed_central_offsets(
    holiday_calendar: tuple[EventOccurrence, ...]
) -> None:
    frame = _feature_frame("2023-01-06", "2023-01-12")
    featured = attach_calendar_features(frame, holiday_calendar)
    days = featured.select(
        pl.col("timestamp").dt.date().alias("date"),
        "is_seollal",
        "is_seollal_window",
    ).unique().sort("date")

    assert _value(days, "2023-01-06", "is_seollal_window") == 0
    assert _value(days, "2023-01-07", "is_seollal_window") == 1
    assert _value(days, "2023-01-08", "is_seollal") == 1
    assert _value(days, "2023-01-10", "is_seollal") == 1
    assert _value(days, "2023-01-11", "is_seollal") == 0
    assert _value(days, "2023-01-11", "is_seollal_window") == 1
    assert _value(days, "2023-01-12", "is_seollal_window") == 0


def test_b1w_allows_plus_three_when_official_sequence_is_extended() -> None:
    calendar = load_holiday_calendar(HOLIDAY_CALENDAR)
    frame = _feature_frame("2024-02-07", "2024-02-14")
    featured = attach_calendar_features(frame, calendar)

    assert _value(featured, "2024-02-13", "is_seollal_window") == 1
    assert _value(featured, "2024-02-14", "is_seollal_window") == 0


def test_b1w_schema_has_exact_history_and_future_widths() -> None:
    assert feature_columns("B1W")[-2:] == (
        "is_seollal_window",
        "is_chuseok_window",
    )
    assert len(feature_columns("B1W")) == 16
    assert len(history_columns("B1W")) == 19
```

```python
def test_feature_and_correction_registries_must_align() -> None:
    calendar = load_holiday_calendar(HOLIDAY_CALENDAR)
    events = load_event_registry(EVENT_REGISTRY)
    changed = replace(events[0], official_end=events[0].official_end + timedelta(days=1))

    with pytest.raises(EventRegistryError, match="feature/correction window"):
        validate_feature_event_alignment(calendar, (changed, *events[1:]))
```

- [ ] **Step 2: B1W 관련 테스트가 feature set 부재로 실패하는지 확인한다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_features.py hqrc_v3/tests/unit/test_events.py -q`

Expected: FAIL because `B1W` and its two columns are not implemented.

- [ ] **Step 3: B1W 열과 window semantics를 구현한다**

```python
FeatureSet = Literal["B0", "B1", "B1W"]
B1W_WINDOW_VERSION = "official-sequence-buffer-v1"
B1W_WINDOW_COLUMNS = ("is_seollal_window", "is_chuseok_window")


def feature_columns(feature_set: FeatureSet) -> tuple[str, ...]:
    if feature_set == "B0":
        return _B0_COLUMNS
    if feature_set == "B1":
        return _B0_COLUMNS + _B1_ONLY_COLUMNS
    if feature_set == "B1W":
        return _B0_COLUMNS + _B1_ONLY_COLUMNS + B1W_WINDOW_COLUMNS
    raise DataContractError(f"unknown feature set: {feature_set}")
```

`attach_calendar_features`의 holiday-calendar loop에서 official core와 window를 별도로 계산한다.

```python
in_official_period = day.is_between(occurrence.official_start, occurrence.official_end)
in_event_window = day.is_between(occurrence.window_start, occurrence.window_end)
if occurrence.holiday_type == "seollal":
    is_seollal = pl.when(in_official_period).then(1).otherwise(is_seollal)
    is_seollal_window = pl.when(in_event_window).then(1).otherwise(is_seollal_window)
else:
    is_chuseok = pl.when(in_official_period).then(1).otherwise(is_chuseok)
    is_chuseok_window = pl.when(in_event_window).then(1).otherwise(is_chuseok_window)
```

`baselines/config.py`에는 TOML 필드를 추가하지 않고 `B1W_WINDOW_COLUMNS`와 `B1W_WINDOW_VERSION`을 immutable reviewer schema constant로 다시 검증한다. 기존 `model_spaces.toml`은 수정하지 않는다.

- [ ] **Step 4: feature/event/config tests를 통과시킨다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_features.py hqrc_v3/tests/unit/test_events.py hqrc_v3/tests/unit/test_paper_baseline_config.py -q`

Expected: PASS.

- [ ] **Step 5: B1W feature contract를 커밋한다**

```bash
git add hqrc_v3/src/hqrc_v3/features.py \
  hqrc_v3/src/hqrc_v3/events.py \
  hqrc_v3/src/hqrc_v3/baselines/config.py \
  hqrc_v3/tests/unit/test_features.py \
  hqrc_v3/tests/unit/test_events.py \
  hqrc_v3/tests/unit/test_paper_baseline_config.py
git commit -m "feat(hqrc-v3): add event-window holiday features"
```

---

### Task 3: B1W baseline publication과 residual source 전파

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/oof.py`
- Modify: `hqrc_v3/src/hqrc_v3/baselines/paper.py`
- Modify: `hqrc_v3/src/hqrc_v3/residual_stage.py`
- Modify: `hqrc_v3/src/hqrc_v3/correction_source.py`
- Modify: `hqrc_v3/src/hqrc_v3/legacy_hqt_loeo.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Modify: `hqrc_v3/tests/unit/test_oof.py`
- Modify: `hqrc_v3/tests/integration/test_paper_baseline_stages.py`
- Modify: `hqrc_v3/tests/unit/test_residual_stage.py`
- Modify: `hqrc_v3/tests/unit/test_correction_source.py`
- Modify: `hqrc_v3/tests/unit/test_legacy_hqt.py`

**Interfaces:**
- Consumes: Task 2의 `FeatureSet`, `B1W_WINDOW_VERSION`, `feature_columns`, `history_columns`.
- Produces: `LEGACY_PAPER_FEATURE_SUITE = ("B0", "B1")`, `REVIEWER_PAPER_FEATURE_SUITE = ("B0", "B1W")`, paper publication과 correction source가 두 suite 중 하나를 검증하는 계약, CLI `--feature-set reviewer`.

- [ ] **Step 1: 두 paper suite와 B0 cache 재사용을 검증하는 실패 테스트를 작성한다**

```python
def test_paper_accepts_exact_reviewer_feature_suite(tmp_path: Path) -> None:
    builder, _ = _recording_builder()
    result = run_paper_oof_stage(
        matrices={"B0": _matrix("B0"), "B1W": _matrix("B1W")},
        config=load_paper_baselines(MODEL_CONFIG),
        run_dir=tmp_path / "reviewer",
        cache_dir=tmp_path / "shared-cache",
        artifact_hashes=HASHES,
        classical_seed=7,
        feature_sets=("B0", "B1W"),
        profile="paper",
        factory_builder=builder,
    )
    manifest = json.loads(result.manifest_path.read_text())
    assert manifest["feature_sets"] == ["B0", "B1W"]
    assert manifest["feature_schemas"]["B1W"]["window_definition"] == [
        "official-sequence-buffer-v1"
    ]


def test_paper_rejects_mixed_three_feature_suite(tmp_path: Path) -> None:
    builder, _ = _recording_builder()
    with pytest.raises(DataContractError, match="paper feature suite"):
        run_paper_oof_stage(
            matrices={name: _matrix(name) for name in ("B0", "B1", "B1W")},
            config=load_paper_baselines(MODEL_CONFIG),
            run_dir=tmp_path,
            cache_dir=tmp_path / "cache",
            artifact_hashes=HASHES,
            classical_seed=7,
            feature_sets=("B0", "B1", "B1W"),
            profile="paper",
            factory_builder=builder,
        )
```

같은 `shared-cache`를 사용하되 서로 다른 run directory에서 legacy suite와 reviewer suite를 순서대로 실행하는 integration test를 추가한다. 첫 실행은 `(B0, B1)`, 두 번째 실행은 `(B0, B1W)`이며, 두 번째 결과의 `fit_count`는 B1W fit unit 수와 같고 `cache_hit_count`는 B0 fit unit 수와 같아야 한다. B0 schema와 frozen model config hash가 그대로이므로 B0 factory는 두 번째 실행에서 호출되지 않아야 한다.

- [ ] **Step 2: suite propagation tests가 현재 B0/B1 제한으로 실패하는지 확인한다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_oof.py hqrc_v3/tests/integration/test_paper_baseline_stages.py hqrc_v3/tests/unit/test_residual_stage.py hqrc_v3/tests/unit/test_correction_source.py -q`

Expected: FAIL with B1W/feature-suite validation errors.

- [ ] **Step 3: B1W를 baseline과 residual contract 전 구간에 전파한다**

```python
LEGACY_PAPER_FEATURE_SUITE: tuple[FeatureSet, ...] = ("B0", "B1")
REVIEWER_PAPER_FEATURE_SUITE: tuple[FeatureSet, ...] = ("B0", "B1W")
PAPER_FEATURE_SUITES = frozenset(
    (LEGACY_PAPER_FEATURE_SUITE, REVIEWER_PAPER_FEATURE_SUITE)
)
```

`baselines.paper._require_stage_selection`, `residual_stage`, `correction_source`의 paper 검증은 정확히 이 두 suite만 허용한다. 기본값은 legacy suite를 유지한다. `_feature_schema`는 B1W에만 다음 metadata를 추가하여 기존 B0 cache hash를 바꾸지 않는다.

```python
schema = {
    "history": list(matrix.history_columns),
    "future": list(matrix.future_columns),
}
if feature_set == "B1W":
    schema["window_definition"] = [B1W_WINDOW_VERSION]
return schema
```

`oof.py`, prediction context validation, residual manifest와 legacy HQT `_select_contexts`는 B1W를 허용한다. AR 전용 기존 HQRC 명령은 B0/B1만 유지한다.

- [ ] **Step 4: CLI에서 baseline reviewer suite와 HQT B1W를 분리한다**

```python
def _baseline_feature_sets(value: str) -> tuple[str, ...]:
    if value == "all":
        return ("B0", "B1")
    if value == "reviewer":
        return ("B0", "B1W")
    return (value,)
```

`generate-oof`와 `fit-final-baselines`는 `--feature-set reviewer`를 허용한다. `run-hqt-loeo`는 `B1W`를 허용하고 reviewer command에서 기본값을 B1W로 둔다. `_pipeline_feature_sets("all")`의 legacy 의미는 변경하지 않는다.

- [ ] **Step 5: baseline/residual/source/CLI tests를 통과시킨다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_oof.py hqrc_v3/tests/integration/test_paper_baseline_stages.py hqrc_v3/tests/unit/test_residual_stage.py hqrc_v3/tests/unit/test_correction_source.py hqrc_v3/tests/unit/test_legacy_hqt.py -q`

Expected: PASS.

- [ ] **Step 6: B1W publication propagation을 커밋한다**

```bash
git add hqrc_v3/src/hqrc_v3/oof.py \
  hqrc_v3/src/hqrc_v3/baselines/paper.py \
  hqrc_v3/src/hqrc_v3/residual_stage.py \
  hqrc_v3/src/hqrc_v3/correction_source.py \
  hqrc_v3/src/hqrc_v3/legacy_hqt_loeo.py \
  hqrc_v3/src/hqrc_v3/cli.py \
  hqrc_v3/tests/unit/test_oof.py \
  hqrc_v3/tests/integration/test_paper_baseline_stages.py \
  hqrc_v3/tests/unit/test_residual_stage.py \
  hqrc_v3/tests/unit/test_correction_source.py \
  hqrc_v3/tests/unit/test_legacy_hqt.py
git commit -m "feat(hqrc-v3): publish reviewer B1W residual sources"
```

---

### Task 4: AR 없는 2024 causal HQT evaluator

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/legacy_hqt_causal.py`
- Create: `hqrc_v3/tests/unit/test_legacy_hqt_causal.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`

**Interfaces:**
- Consumes: `ValidatedCorrectionSource`, Task 1의 `build_legacy_hqt_data_from_frame`, `sample_legacy_hqt`, `new_event_correction_draws`, B1W OOF 2020--2023과 final-2024 streams.
- Produces: `run_legacy_hqt_causal_2024(...) -> tuple[LegacyHQTCausalContextResult, ...]`, `LegacyHQTCausalContextResult(context, output_dir, sampler_fit_count, reused)`.

- [ ] **Step 1: 2024 leakage와 scale 사용을 고정하는 실패 테스트를 작성한다**

```python
def test_causal_training_uses_only_eight_pre_2024_events(source: FakeSource) -> None:
    frame, ids, sigma_fit = build_causal_training_frame(source, B1W_CONTEXT)

    assert len(ids) == 8
    assert all(not event_id.endswith("-2024") for event_id in ids)
    assert set(frame["split_id"]) == {"oof-2020", "oof-2021", "oof-2022", "oof-2023"}
    assert sigma_fit == pytest.approx(source.oof_2023_scale)


def test_causal_products_never_use_2024_non_event_scale(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, source: FakeSource
) -> None:
    source.fail_if_2024_non_event_scale_requested = True
    monkeypatch.setattr(module, "validate_correction_source", lambda **_: source)

    result = run_legacy_hqt_causal_2024(
        source_run_dir=tmp_path / "source",
        config_path=EXPERIMENT_CONFIG,
        output_root=tmp_path / "result",
        models=("xgboost",),
        feature_set="B1W",
        root_seed=20260813,
        profile="smoke",
        draws=5,
        tune=5,
        chains=2,
        cores=2,
    )

    assert result[0].sampler_fit_count == 1
```

- [ ] **Step 2: 새 causal module이 없어 테스트가 실패하는지 확인한다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_legacy_hqt_causal.py -q`

Expected: FAIL with module import error.

- [ ] **Step 3: 8-event training과 두 2024 event 평가를 구현한다**

```python
@dataclass(frozen=True, slots=True)
class LegacyHQTCausalContextResult:
    context: EventResidualContext
    output_dir: Path
    sampler_fit_count: int
    reused: bool


def build_causal_training_frame(
    source: ValidatedCorrectionSource, context: EventResidualContext
) -> tuple[pl.DataFrame, tuple[str, ...], float]:
    frame = source.load_standardized_context(context, through=2023).sort(
        "occurrence_id", "target_timestamp"
    )
    ids = tuple(frame["occurrence_id"].unique(maintain_order=True).to_list())
    if len(ids) != 8 or any(event_id.endswith("-2024") for event_id in ids):
        raise LegacyHQTCausalError("causal HQT requires exactly eight pre-2024 events")
    scales = frame.filter(pl.col("split_id") == "oof-2023")["sigma_n_mw"].unique()
    if scales.len() != 1:
        raise LegacyHQTCausalError("pre-2024 evaluation scale is not unique")
    return frame, ids, float(scales.item())
```

공개 API `run_legacy_hqt_causal_2024`는 기존 LOEO runner처럼 `source_run_dir`와 `config_path`를 받아 내부에서 `validate_correction_source`를 호출한다. 한 context당 posterior를 한 번 적합하고 `seollal-2024`, `chuseok-2024`에 별도 new-event draw를 생성한다. H1은 8-event training frame의 동일 holiday raw-MW 평균만 사용한다. H2와 H2-taper는 `sigma_fit`을 사용한다.

- [ ] **Step 4: causal artifact identity와 자동 재개를 구현한다**

`input_identity.json`에는 source residual/final hashes, B1W schema, 8개 training ids, 두 evaluation ids, HQT model spec, sampler contract를 기록한다. `posterior.nc`, `hourly_predictions.parquet`, `event_metrics.parquet`, `pooled_metrics.parquet`, `manifest.json`, `COMPLETE`가 모두 hash-valid할 때만 재사용한다.

- [ ] **Step 5: causal unit tests와 legacy HQT tests를 통과시킨다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_legacy_hqt_causal.py hqrc_v3/tests/unit/test_legacy_hqt.py -q`

Expected: PASS.

- [ ] **Step 6: causal evaluator를 커밋한다**

```bash
git add hqrc_v3/src/hqrc_v3/legacy_hqt_causal.py \
  hqrc_v3/src/hqrc_v3/cli.py \
  hqrc_v3/tests/unit/test_legacy_hqt_causal.py
git commit -m "feat(hqrc-v3): add causal 2024 conference HQT evaluation"
```

---

### Task 5: Event-level 통계와 reviewer 표·그림 생성

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/evaluation/inference.py`
- Create: `hqrc_v3/src/hqrc_v3/evaluation/reviewer.py`
- Modify: `hqrc_v3/src/hqrc_v3/evaluation/__init__.py`
- Create: `hqrc_v3/tests/unit/test_reviewer_report.py`
- Modify: `hqrc_v3/tests/unit/test_event_inference.py`

**Interfaces:**
- Consumes: B0/B1W final predictions, retrospective LOEO event/pooled/scale artifacts, causal-2024 event/pooled/hourly artifacts.
- Produces: `build_reviewer_report(...) -> ReviewerReportResult`, one-sided event-level Wilcoxon, deterministic event bootstrap CI, normalized CSV/Parquet tables and PNG figures.

- [ ] **Step 1: one-sided Wilcoxon과 report schema 실패 테스트를 작성한다**

```python
def test_wilcoxon_supports_reference_greater_than_candidate() -> None:
    reference = np.array([10.0, 9.0, 8.0, 7.0])
    candidate = np.array([8.0, 7.0, 6.0, 5.0])

    result = wilcoxon_event_test(reference, candidate, alternative="greater")

    assert result.n_events == 4
    assert 0.0 <= result.p_value <= 1.0


def test_reviewer_report_separates_feature_and_hqt_gains(tmp_path: Path) -> None:
    result = build_reviewer_report(_reviewer_fixture(tmp_path), output_root=tmp_path / "paper")
    overall = pl.read_parquet(result.tables_dir / "overall_2024.parquet")
    inference = pl.read_parquet(result.tables_dir / "loeo_inference.parquet")

    assert set(overall["comparison"]) == {"B0-H0_to_B1W-H0"}
    assert set(inference["comparison"]) == {"B1W-H0_to_B1W-H2"}
    assert result.figures_dir.joinpath("fig_per_event.png").is_file()
    assert result.figures_dir.joinpath("fig_correction_2024.png").is_file()
```

- [ ] **Step 2: 새 API와 report module 부재로 테스트가 실패하는지 확인한다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_event_inference.py hqrc_v3/tests/unit/test_reviewer_report.py -q`

Expected: FAIL because `alternative` and `evaluation.reviewer` do not exist.

- [ ] **Step 3: event-level inference API를 확장한다**

```python
Alternative = Literal["two-sided", "greater", "less"]


def wilcoxon_event_test(
    reference: np.ndarray,
    candidate: np.ndarray,
    *,
    alternative: Alternative = "two-sided",
) -> WilcoxonEventResult:
    left = _finite_vector(reference, name="reference")
    right = _finite_vector(candidate, name="candidate")
    if alternative not in {"two-sided", "greater", "less"}:
        raise InferenceContractError("unsupported Wilcoxon alternative")
    result = stats.wilcoxon(left, right, alternative=alternative, method="auto")
    return WilcoxonEventResult(float(result.statistic), float(result.pvalue), int(left.size))
```

H0 event RMSE를 reference, H2 event RMSE를 candidate로 전달하고 `alternative="greater"`를 사용한다. Bootstrap은 10개 event improvement scalar를 event 단위로 10,000회 재표집하고 root seed에서 파생한 seed를 manifest에 기록한다.

기존 `bootstrap_event_median(event_values: Mapping[str, np.ndarray], draws, seed)`에는 occurrence id별 한 원소 improvement 배열을 전달한다. 기존 Wilcoxon 구현의 동일 벡터 특수 처리(`p=1.0`)는 유지하면서 `alternative`만 확장한다.

- [ ] **Step 4: 정규화된 표와 두 그림을 구현한다**

```text
paper/tables/overall_2024.{parquet,csv}
paper/tables/loeo_event.{parquet,csv}
paper/tables/loeo_pooled.{parquet,csv}
paper/tables/loeo_inference.{parquet,csv}
paper/tables/causal_2024.{parquet,csv}
paper/tables/scale_stability.{parquet,csv}
paper/figures/fig_per_event.png
paper/figures/fig_correction_2024.png
paper/manifest.json
```

`overall_2024`는 B0-H0→B1W-H0만, `loeo_inference`는 B1W-H0→B1W-H2만 기록한다. Per-event figure는 음수 개선율을 삭제하거나 0으로 자르지 않는다. Correction figure는 2024 causal hourly observed, H0, H2를 명절별 panel로 출력한다.

- [ ] **Step 5: inference/report tests와 정적 검사를 통과시킨다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_event_inference.py hqrc_v3/tests/unit/test_reviewer_report.py -q`

Expected: PASS.

Run: `uv run --project hqrc_v3 --locked ruff check hqrc_v3/src/hqrc_v3/evaluation/inference.py hqrc_v3/src/hqrc_v3/evaluation/reviewer.py hqrc_v3/tests/unit/test_reviewer_report.py`

Expected: PASS.

- [ ] **Step 6: reviewer reporting을 커밋한다**

```bash
git add hqrc_v3/src/hqrc_v3/evaluation/inference.py \
  hqrc_v3/src/hqrc_v3/evaluation/reviewer.py \
  hqrc_v3/src/hqrc_v3/evaluation/__init__.py \
  hqrc_v3/tests/unit/test_event_inference.py \
  hqrc_v3/tests/unit/test_reviewer_report.py
git commit -m "feat(hqrc-v3): generate reviewer event-level results"
```

---

### Task 6: 한 명령 reviewer pipeline, 자동 재개, operator 문서

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/reviewer_pipeline.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Create: `hqrc_v3/tests/unit/test_reviewer_pipeline.py`
- Modify: `hqrc_v3/tests/integration/test_cli_oof.py`
- Modify: `hqrc_v3/README.md`

**Interfaces:**
- Consumes: Task 3 reviewer baseline suite, `prepare_standardized_residual_artifact`, Task 1 LOEO runner, Task 4 causal runner, Task 5 reviewer report builder.
- Produces: `run_hqt_reviewer_pipeline(...) -> ReviewerPipelineResult`, CLI `hqrc run-hqt-reviewer`.

- [ ] **Step 1: 실행 순서와 AR 미호출을 고정하는 실패 테스트를 작성한다**

```python
def test_reviewer_pipeline_runs_exact_stages_in_order(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []
    monkeypatch.setattr(module, "run_paper_oof_stage", _record(calls, "oof"))
    monkeypatch.setattr(module, "run_paper_final_stage", _record(calls, "final"))
    monkeypatch.setattr(module, "prepare_standardized_residual_artifact", _record(calls, "residual"))
    monkeypatch.setattr(module, "run_legacy_hqt_loeo", _record(calls, "loeo"))
    monkeypatch.setattr(module, "run_legacy_hqt_causal_2024", _record(calls, "causal"))
    monkeypatch.setattr(module, "build_reviewer_report", _record(calls, "report"))

    run_hqt_reviewer_pipeline(_options(tmp_path))

    assert calls == ["oof", "final", "residual", "loeo", "causal", "report"]


def test_reviewer_pipeline_has_no_ar_approval_option() -> None:
    args = build_parser().parse_args(_reviewer_cli_args())
    assert args.command == "run-hqt-reviewer"
    assert not hasattr(args, "approved_ar")
    assert not hasattr(args, "approve_derived_ar")
```

- [ ] **Step 2: reviewer pipeline 부재로 테스트가 실패하는지 확인한다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_reviewer_pipeline.py -q`

Expected: FAIL with module/command not found.

- [ ] **Step 3: 순차·재개 가능한 orchestration을 구현한다**

```python
@dataclass(frozen=True, slots=True)
class ReviewerPipelineResult:
    source_run_dir: Path
    loeo_root: Path
    causal_root: Path
    paper_root: Path
    baseline_fit_count: int
    baseline_cache_hit_count: int
    hqt_fit_count: int


REVIEWER_FEATURE_SUITE = ("B0", "B1W")
```

Pipeline은 다음 순서를 고정한다.

1. reviewer suite B0/B1W expanding OOF
2. reviewer suite B0/B1W final-2024
3. standardized OOF residual publication
4. 다섯 모델 B1W retrospective 10-event H0/H1/H2
5. 다섯 모델 B1W causal-2024 H0/H1/H2
6. reviewer tables/figures

각 stage는 기존 manifest/COMPLETE를 검증하여 재사용하고, 완료된 context 다음부터 자동으로 진행한다. `--cache-dir`에 기존 baseline prediction cache를 지정하면 B0는 cache hit여야 한다. `--model all`은 paper profile에서 다섯 모델 전체를 의미하고, `--model xgboost` 같은 부분 선택은 smoke profile에서만 허용한다.

- [ ] **Step 4: CLI와 실행 예제를 연결한다**

```bash
MODEL_SHA256="$(openssl dgst -sha256 hqrc_v3/configs/model_spaces.toml | awk '{print $NF}')"

uv run --project hqrc_v3 --locked hqrc run-hqt-reviewer \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --frozen-model-hash "$MODEL_SHA256" \
  --event-registry hqrc_v3/configs/events.csv \
  --holiday-calendar hqrc_v3/configs/holiday_calendar.csv \
  --temporary-holiday-availability hqrc_v3/configs/temporary_holiday_availability.csv \
  --run-dir artifacts/hqt-reviewer-source \
  --cache-dir artifacts/hqrc-v3-paper-20260811/prediction-stream-cache \
  --output-root artifacts/hqt-reviewer-results \
  --baseline-seed 7 --root-seed 20260813 \
  --model all \
  --draws 1000 --tune 1000 --chains 4 --cores 4 \
  --init adapt_diag --target-accept 0.99
```

README는 B1W 정의, 기존 B1과의 비호환성, B0 cache 재사용 조건, 예상 stage 순서, 재개 방식과 산출물 위치를 설명한다.

- [ ] **Step 5: reviewer pipeline unit/integration tests를 통과시킨다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_reviewer_pipeline.py hqrc_v3/tests/integration/test_cli_oof.py -q`

Expected: PASS.

- [ ] **Step 6: 전체 unit/integration suite와 package lint를 실행한다**

Run: `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit hqrc_v3/tests/integration -q`

Expected: PASS. Slow tests are excluded from this command.

Run: `uv run --project hqrc_v3 --locked ruff check hqrc_v3/src hqrc_v3/tests`

Expected: PASS with no findings.

- [ ] **Step 7: 작은 smoke profile로 end-to-end checkpoint를 검증한다**

Run the reviewer command with `--profile smoke --model xgboost --draws 5 --tune 5 --chains 2 --cores 2 --smoke-boosting-rounds 2` in a temporary run/output root.

Expected:

```text
[baseline] reviewer OOF: B0 then B1W
[baseline] reviewer final-2024
[residuals] reviewer B0/B1W publication
[HQT] xgboost/B1W retrospective LOEO complete
[HQT] xgboost/B1W causal-2024 complete
[report] reviewer tables and figures complete
```

Run the identical command a second time.

Expected: baseline and HQT fit counts are zero, cache/reuse counts are positive, and every output digest is unchanged.

- [ ] **Step 8: reviewer pipeline과 문서를 커밋한다**

```bash
git add hqrc_v3/src/hqrc_v3/reviewer_pipeline.py \
  hqrc_v3/src/hqrc_v3/cli.py \
  hqrc_v3/tests/unit/test_reviewer_pipeline.py \
  hqrc_v3/tests/integration/test_cli_oof.py \
  hqrc_v3/README.md
git commit -m "feat(hqrc-v3): automate reviewer B1W experiment"
```
