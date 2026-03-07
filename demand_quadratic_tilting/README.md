# demand-quadratic-tilting

한국 명절(추석/설날) 기간 전력 수요 예측 보정을 위한 **Hierarchical Quadratic Tilting (HQT)** 패키지.

`workalendar` 기반 자동 공휴일 전처리 + 베이지안 계층 이차 틸트 모델을 Polars DataFrame 위에서 구동합니다.

---

## 설치

```bash
# uv (권장)
uv add demand-quadratic-tilting

# pip
pip install -e .
```

### 의존성

| 패키지 | 버전 | 용도 |
|---|---|---|
| `polars` | >=1.38 | DataFrame 처리 (pandas 대체) |
| `numpy` | >=2.2 | 수치 연산 |
| `pymc` | >=5.25 | 베이지안 MCMC (LKJ prior) |
| `pytensor` | >=2.31 | PyMC 텐서 연산 |
| `scipy` | >=1.15 | 통계 함수 (`norm.ppf`) |
| `workalendar` | >=17.0 | 한국 공휴일 자동 생성 |

선택적 의존성:
- `numpyro` + `jax`: GPU/TPU 가속 MCMC (`sampler="numpyro"`)
- `torch`: Seq2Seq LSTM 베이스라인 모델 (별도 학습)

---

## 빠른 시작

### 1. End-to-End 파이프라인

```python
import polars as pl
from demand_quadratic_tilting import annotate_holidays, run_hqt_pipeline

# 데이터 로드 (datetime 컬럼 필수)
df = pl.read_csv("power_demand.csv", try_parse_dates=True)

# workalendar로 공휴일 자동 부여
df = annotate_holidays(df, datetime_col="datetime")

# Train / Val / Test 분할
train = df.filter(pl.col("datetime") <= "2022-12-31")
val   = df.filter(
    (pl.col("datetime") > "2022-12-31") & (pl.col("datetime") <= "2023-06-30")
)
test  = df.filter(pl.col("datetime") > "2023-06-30")

# HQT 실행
results = run_hqt_pipeline(
    train, val, test,
    y_col="power_demand_MW",
    pred_col="hybrid_baseline",
    sampler="numpyro",       # "nuts" | "numpyro" | "advi"
    tilt_mode="hybrid",      # "hybrid" | "event" | "type"
)

# 결과 확인
print(results["test"]["metrics"])
# {'MAE_all': 1234.5, 'RMSE_all': 1567.8, 'MAE_holiday': 890.1,
#  'RMSE_holiday': 1023.4, 'PICP_95': 94.5, 'AIW': 25000.0, ...}

test_preds = results["test"]["preds"]  # pl.DataFrame
```

### 2. 적응형 게이팅 (Adaptive Gating)

베이스라인이 이미 잘 맞추는 시점에는 보정을 억제하여 과보정을 방지합니다.

```python
from demand_quadratic_tilting import tilt_from_posterior, apply_tilt

# gate=True로 적응형 게이팅 활성화
tilt_df, hw_const = tilt_from_posterior(
    hqt_result,
    sigma_resid,
    test_dates,
    gate=True,             # 적응형 게이팅 ON
    threshold_k=0.5,       # |e_tilt| < 0.5*sigma → 억제
    gate_scale_k=0.3,      # sigmoid 경사도
)

result = apply_tilt(test_df, tilt_df, "hybrid_baseline")
```

게이팅 수식:

```
w(e_tilt) = sigmoid( (|e_tilt| - threshold_k * sigma) / (gate_scale_k * sigma) )
e_tilt_gated = w * e_tilt
```

- `|e_tilt|`이 작으면 (`<< threshold`) → `w ≈ 0`, 보정 억제
- `|e_tilt|`이 크면 (`>> threshold`) → `w ≈ 1`, 전체 보정 적용

### 3. 개별 모듈 사용

```python
from demand_quadratic_tilting import (
    annotate_holidays,
    build_holiday_windows_and_tau,
    compute_sigma_and_residuals,
    fit_hqt_pymc_lkj,
    tilt_from_posterior,
    apply_tilt,
    evaluate_split,
)

# Step 1: 전처리 — datetime만 있으면 공휴일 자동 부여
df = annotate_holidays(raw_df, datetime_col="timestamp")

# Step 2: 명절 윈도우 + 시간 오프셋(τ) 구성
windows, tau_map, eid_map, type_map, tau_hours = build_holiday_windows_and_tau(
    df,
    datetime_col="timestamp",
    pre_pad_days=1,          # 명절 전 1일 패딩
    post_pad_days=1,         # 명절 후 1일 패딩
)

# Step 3: 비명절 잔차에서 σ 추정
sigma, residuals = compute_sigma_and_residuals(train_df, "y", "pred")

# Step 4: PyMC LKJ 계층 모델 적합
hqt = fit_hqt_pymc_lkj(
    train_df, sigma, residuals,
    tau_map, eid_map, type_map,
    datetime_col="timestamp",
    sampler="numpyro",
    chains=4, draws=1000, tune=1000,
)

# Step 5: 틸트 계산 및 적용
tilt_df, hw_const = tilt_from_posterior(hqt, sigma, test_dates)
result = apply_tilt(test_df, tilt_df, "pred", datetime_col="timestamp")

# Step 6: 평가
metrics = evaluate_split(result, "y", eid_map, datetime_col="timestamp")
print(metrics)
```

---

## 패키지 구조

```
demand_quadratic_tilting/
├── __init__.py           # Public API
├── constants.py          # 추석/설날 라벨 상수
├── preprocessing.py      # workalendar 기반 공휴일 전처리
├── windows.py            # 명절 윈도우 / τ(시간 오프셋) 생성
├── model.py              # HQTResult + PyMC LKJ 적합
├── tilt.py               # 틸트 계산 / 적용 / 적응형 게이팅
├── metrics.py            # MAE, RMSE, PICP, AIW 평가
└── pipeline.py           # End-to-End 파이프라인
```

---

## API Reference

### `preprocessing.py`

#### `annotate_holidays(df, datetime_col, start_year?, end_year?) -> pl.DataFrame`

datetime 컬럼이 있는 DataFrame에 공휴일 정보를 자동 부여합니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `df` | `pl.DataFrame` | (필수) | 입력 데이터 |
| `datetime_col` | `str` | `"datetime"` | datetime 컬럼명 |
| `start_year` | `int?` | `None` | 공휴일 조회 시작 연도 (None → 데이터에서 추출) |
| `end_year` | `int?` | `None` | 공휴일 조회 종료 연도 (None → 데이터에서 추출) |

추가되는 컬럼:
- `holiday_name`: 원본 공휴일 명칭 (예: `"Chuseok"`, `"The day preceding Chuseok"`, `"non-event"`)
- `holiday_type`: 분류 (`"Chuseok"` | `"Seollal"` | `"other"` | `"non-event"`)

#### `filter_major_holidays(df) -> pl.DataFrame`

추석/설날 행만 필터링합니다.

---

### `windows.py`

#### `build_holiday_windows_and_tau(df, ...) -> (windows, tau_map, eid_map, type_map, tau_hours)`

명절 윈도우 및 시간 오프셋(τ)을 구성합니다.

| 파라미터 | 타입 | 기본값 | 설명 |
|---|---|---|---|
| `holiday_name_col` | `str` | `"holiday_name"` | 공휴일 이름 컬럼 |
| `datetime_col` | `str` | `"datetime"` | datetime 컬럼명 |
| `center_hour` | `int` | `0` | 명절 중심 시각 |
| `tau_unit` | `str` | `"1h"` | τ 단위 (`"1h"`, `"30m"`, `"1d"`) |
| `pre_pad_days` | `int` | `0` | 명절 전 패딩 일수 |
| `post_pad_days` | `int` | `0` | 명절 후 패딩 일수 |

반환값:
- `windows`: `dict[event_id, list[datetime]]` — 이벤트별 시점 목록
- `tau_map`: `dict[(event_id, datetime), int]` — 시간 오프셋
- `event_id_of_date`: `dict[datetime, event_id]` — 날짜 → 이벤트 ID
- `type_of_date`: `dict[datetime, str]` — 날짜 → 명절 유형
- `tau_unit_hours`: `float` — τ 단위(시간)

---

### `model.py`

#### `HQTResult` (dataclass)

MCMC 사후 분포 결과를 담는 데이터 클래스.

| 필드 | Shape | 설명 |
|---|---|---|
| `draws_beta` | `[S, I, 3]` | 이벤트별 β 사후 샘플 |
| `draws_mu` | `[S, H, 3]` | 유형별 μ 사후 샘플 (H=2: Chuseok/Seollal) |
| `draws_L` | `[H]×(S,3,3)` | Cholesky factor L_h (Σ_h = L_h L_hᵀ) |
| `draws_sigma_r` | `[S]` | 잔차 노이즈 σ_r |
| `tau_unit_hours` | `float` | τ 기본 단위(시간) |
| `tau_scale_hours` | `float` | τ 스케일(시간) |

#### `compute_sigma_and_residuals(df, y_col, pred_col, holiday_name_col?) -> (sigma, residuals)`

비명절 잔차 표준편차 σ 추정.

#### `fit_hqt_pymc_lkj(df_train, sigma_resid, residuals, ...) -> HQTResult`

PyMC LKJ prior 기반 계층 이차 틸트 모델 적합.

| 주요 파라미터 | 기본값 | 설명 |
|---|---|---|
| `sampler` | `"nuts"` | `"nuts"` / `"numpyro"` / `"advi"` |
| `chains` | `4` | MCMC 체인 수 |
| `draws` | `1000` | 사후 드로우 수 |
| `tune` | `1000` | 번인(burn-in) 수 |
| `target_accept` | `0.95` | NUTS 목표 수용률 |

---

### `tilt.py`

#### `tilt_from_posterior(hqt, sigma_resid, dates, ...) -> (tilt_df, half_width_const)`

사후 분포에서 틸트를 계산합니다.

| 파라미터 | 기본값 | 설명 |
|---|---|---|
| `tilt_mode` | `"hybrid"` | 기존 이벤트는 event별, 새 이벤트는 type별 |
| `ci` | `0.95` | 예측 구간 신뢰수준 |
| `gate` | `False` | 적응형 게이팅 활성화 |
| `threshold_k` | `0.5` | 게이팅 임계값 (σ 배수) |
| `gate_scale_k` | `0.3` | sigmoid 경사도 (σ 배수) |

`tilt_mode` 옵션:
- `"hybrid"` (기본): 학습에 포함된 이벤트는 개별 β_i 사용, 새 이벤트는 유형 μ_h에서 샘플링
- `"event"`: 모든 이벤트에 개별 β_i 사용
- `"type"`: 유형 평균 μ_h만 사용

#### `apply_tilt(df, tilt_df, baseline_col, datetime_col?) -> pl.DataFrame`

베이스라인 예측에 틸트를 적용. 추가되는 컬럼: `e_tilt`, `tilted_pred`, `lower_t`, `upper_t`, `half_width_t`.

---

### `metrics.py`

| 함수 | 설명 |
|---|---|
| `mae(y, yhat)` | Mean Absolute Error |
| `rmse(y, yhat)` | Root Mean Squared Error |
| `picp(y, lower, upper)` | Prediction Interval Coverage Probability (%) |
| `aiw(lower, upper)` | Average Interval Width |
| `trough_bias_per_event(df, ...)` | 이벤트별 최저점 편향 평균 |
| `evaluate_split(df, y_col, eid_map)` | 전체 + 명절 지표 일괄 계산 |

---

### `pipeline.py`

#### `run_hqt_pipeline(train_df, val_df, test_df, ...) -> dict`

학습/검증/테스트 분할에 대해 HQT 적합 + 틸트 적용 + 평가를 한 번에 실행합니다.

반환값:

```python
{
    "hqt": HQTResult,          # 사후 분포 결과
    "sigma_resid": float,       # 비명절 잔차 표준편차
    "windows": dict,            # 이벤트별 윈도우
    "train": {"preds": pl.DataFrame, "metrics": dict},
    "val":   {"preds": pl.DataFrame, "metrics": dict},
    "test":  {"preds": pl.DataFrame, "metrics": dict},
}
```

`metrics` dict 키:
- `MAE_all`, `RMSE_all`: 전체 기간
- `MAE_holiday`, `RMSE_holiday`: 명절 기간만
- `PICP_95`: 95% 예측 구간 커버리지 (%)
- `AIW`: 평균 예측 구간 폭
- `half_width_const`: 고정 예측 구간 반폭

---

## 알고리즘 개요

```
데이터 (datetime + y + baseline_pred)
  │
  ├─ annotate_holidays()         ← workalendar 자동 전처리
  │
  ├─ build_holiday_windows_and_tau()  ← 명절 윈도우 + τ
  │
  ├─ compute_sigma_and_residuals()    ← σ 추정
  │
  ├─ fit_hqt_pymc_lkj()              ← MCMC 적합
  │     z_t ~ N(β₀ + β₁τ + β₂τ², σ_r²)
  │     β_i ~ N(μ_h, Σ_h),  Σ_h ~ LKJ(η=2)
  │
  ├─ tilt_from_posterior()            ← 틸트 계산
  │     e_tilt = σ * E[z_hat]
  │     (gate=True → sigmoid 게이팅)
  │
  └─ apply_tilt()                     ← 보정 적용
        ŷ_tilted = ŷ_baseline + e_tilt
```

자세한 수식과 알고리즘 플로우는 `docs/algorithm.md`, `docs/adaptive_gating.md`를 참조하세요.

---

## 라이선스

MIT
