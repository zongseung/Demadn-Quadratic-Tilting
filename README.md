# Event-Aware Load Forecasting via Hybrid Modeling and Hierarchical Bayesian Tilting

> **논문**: *Event-Aware Load Forecasting via Hybrid Modeling and Hierarchical Bayesian Tilting*
> **저자**: Jong-Seung Lee, Hyung-Tae Ha (Gachon University)

---

## 1. 프로젝트의 중요성

전력 계통 운영에서 **일 최대 부하(peak load) 예측**은 경제급전·예비력 확보·수요반응 설계의 핵심 입력값이다. 최근 딥러닝 기반 혼합 모형이 일반 일자의 예측 정확도를 크게 높였음에도, **추석·설날과 같은 다일 연속 명절 구간에서는 여전히 심각한 체계적 과대예측(over-prediction)** 이 발생한다.

| 문제 | 영향 |
|---|---|
| 명절 기간 수요 급감을 모형이 과소 반영 | 예비력 계획 오류 → 운영 비용 증가 |
| 수요 troughs가 모형 예측보다 훨씬 깊음 | 경제급전 왜곡 |
| 표준 예측구간이 명절 구간에서 교정 불량 | 불확실성 정보 신뢰 불가 |

이 문제는 다음 세 가지 구조적 어려움 때문에 순수 데이터 기반 모형으로는 해결하기 어렵다.

- **Data sparsity**: 명절은 연 1회, 유효 윈도우는 3~6일에 불과해 학습 표본이 극도로 적다.
- **Nonlinear, time-varying pattern**: 수요 감소-회복이 날짜 인덱스에 따라 비선형·비대칭적으로 진행된다.
- **Miscalibrated uncertainty**: 기존 예측구간이 명절 구간을 84% 수준에서만 커버(목표 95%)한다.

---

## 2. 연구 방향

본 프로젝트는 **두 단계 이벤트 인지(event-aware) 예측 프레임워크**를 제안한다.

```
[Stage 1]  Trend-Fourier-Seq2Seq LSTM  →  baseline forecast  ŷ_t^base
[Stage 2]  Hierarchical Quadratic Tilt  →  holiday correction  e_t^tilt
                                        →  adjusted forecast   ŷ_t^tilt
```

### 핵심 아이디어: Hierarchical Quadratic Tilting (HQT)

베이스라인 잔차를 표준화한 뒤, 명절 윈도우 내 날짜 인덱스(τ)에 대한 **이차(quadratic) 편향 곡선**을 Bayesian 계층 모형으로 추정한다.

- **같은 명절 유형**(추석·설날)의 과거 5년치를 pooling → 희소 데이터 문제 완화
- **이차 곡선** → 수요 감소-회복의 비선형·비대칭 패턴을 포착
- **Posterior predictive sampling** → 미래 신규 명절에도 외삽 가능
- 명절 예측구간 커버리지: 84% → **94.2%** (목표 95% 거의 달성)

---

## 3. 접근 방법

### 3.1 베이스라인: Trend-Fourier-Seq2Seq LSTM

> **주의**: 베이스라인은 SARIMAX-LSTM이 **아니다**.
> 통계적 선형 모형(SARIMAX) 대신, 명시적 추세·계절성 분해 + 신경망 잔차 보정 구조를 사용한다.

**Step A1. 추세 추출 (log-linear OLS)**

```
log(D_t) = c + β·t + ε_t   →   T̂_t = exp(ĉ + β̂·t)
```

학습 구간(~2022)만으로 추세 모형을 적합하고, 전체 구간에 외삽.

**Step A2. 다중 주기 Fourier 계절성**

```
S_t^(F) = Σ_k [ a_k sin(2πkt/s) + b_k cos(2πkt/s) ]
```

일간(s=24h), 주간(s=168h), 연간(s=8766h) 세 가지 주기를 탐색.
검증 MSE 최소화 기준으로 각 주기의 조화수(K) 자동 선택.

**Step A3. 잔차 시리즈**

```
r_t^(0) = D_t - T̂_t - Ŝ_t^(F)
```

**Step A4. Seq2Seq LSTM (encoder-decoder)**

- 입력: 과거 7일(168h) 잔차 + 외생 공변량(기온, 습도, 계절 더미, 명절 더미)
- 출력: 다음 1일(24h) 잔차 예측 r̂_{t+1}^(S2S)
- Teacher forcing + Masked MSE loss (패딩 구간 제외)

**Step A5. 하이브리드 베이스라인 예측**

```
ŷ_t^base = T̂_t + Ŝ_t^(F) + r̂_t^(S2S)
```

---

### 3.2 Hierarchical Quadratic Tilt (HQT)

**Step B. 명절 윈도우 & τ 인덱싱**

각 명절 이벤트 i의 핵심일 t_{0,i}로부터 상대 날짜 인덱스를 정의한다.

```
τ_{i,t} = (t - t_{0,i}) / Δτ       (Δτ = 1일)
```

τ < 0 : 명절 전, τ = 0 : 핵심일, τ > 0 : 명절 후.
윈도우는 공식 명절 라벨 + 앞뒤 각 1일 패딩을 포함한다.

**Step C. 잔차 표준화**

```
σ = Std{ r_t : t ∉ 명절 윈도우 }        (비명절 일에서만 추정)
z_t = r_t / σ
```

**Step D. 계층 이차 틸트 모형**

```
z_{i,t} | β_i, σ_r²  ~  N( β_{i0} + β_{i1}·τ + β_{i2}·τ²,  σ_r² )

β_i | h(i)  ~  N( μ_{h(i)},  Σ_{h(i)} )      h ∈ {Chuseok, Seollal}

μ_h  ~  N(0, 10²I)
Σ_h  ~  LKJ(η=2)   →   Σ_h = L_h L_h^T    (Cholesky 인수분해)
σ_r  ~  HalfNormal(1)
```

Non-centered parameterization으로 수치 안정성 확보:

```
β_i = μ_{h(i)} + L_{h(i)} ε_i,    ε_i ~ N(0, I)
```

사후 추론: Hamiltonian Monte Carlo (NUTS), 4 chains × 3,000 draws.

**Step E. 사후 예측 틸트 적용**

학습에서 본 이벤트 i:
```
ẑ_{i,t} = E_post[ β_{i0} + β_{i1}·τ + β_{i2}·τ² ]
```

새 이벤트(미래 명절):
```
β_new,s = μ_{h,s} + L_{h,s} @ ε_s,    ε_s ~ N(0,I)    (사후 L_h 직접 사용)
```

보정 예측:
```
ŷ_t^tilt = ŷ_t^base + σ · E[z_{i,t} | posterior]
```

적응형 예측구간:
```
PI_t = ŷ_t^tilt ± z_{α/2} · σ · sqrt( 1 + E[σ_r²] + Var(z_{i,t}) )
```

---

## 실험 결과 요약

| Metric | Baseline | Tilted | 개선 |
|---|---|---|---|
| MAE (holiday, MW) | 178.6 | 97.3 | **-45.6%** |
| RMSE (holiday, MW) | 243.1 | 132.5 | **-45.5%** |
| Trough bias (MW) | +24.5 | -2.1 | 거의 제거 |
| 95% PICP (holiday) | 84.0% | 94.2% | +10.2%p |

---

## 프로젝트 구조

```
Demadn-Quadratic-Tilting/
├── src/
│   ├── trend-fourier-Seq2Seq_LSTM.ipynb   # 베이스라인: Trend-Fourier-Seq2Seq LSTM
│   └── #2_trend_seasonality_DL.ipynb      # HQT 통합 파이프라인
├── seq2seq_tilting.ipynb                  # HQT 구현체
├── Hierachical_Quadratic_Tilting.pdf      # 논문 원문
├── pyproject.toml
└── README.md
```

---

## 환경 설정

```bash
# 가상환경 + 의존성 자동 설치
uv sync
```

### Device 설정

```python
# NVIDIA GPU → Apple MPS → CPU 자동 감지
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

DEVICE = get_device()
```

### HQT 파이프라인 실행

```python
results = run_hqt_pipeline_LKJ(
    train_df=train1, val_df=val1, test_df=test1,
    y_col="power demand(MW)",
    pred_col="hybrid",
    holiday_name_col="holiday_name",
    sampler="numpyro",          # "nuts" | "numpyro" | "advi"
    chains=4, draws=3000, tune=2000,
    target_accept=0.99,
    tilt_mode="hybrid",         # "event" | "type" | "hybrid"
    pre_pad_days=1,
    post_pad_days=1,
)
```

---

## 주요 설계 결정

| 항목 | 선택 | 이유 |
|---|---|---|
| 베이스라인 | Trend-Fourier-Seq2Seq LSTM | 명시적 분해로 해석 가능성 확보; SARIMAX 선형 가정 없이 비선형 잔차 학습 |
| 틸트 형태 | 이차(quadratic) | 수요 감소-회복의 오목(concave) 곡선 포착 |
| 계층 prior | LKJ Cholesky | β 계수 간 공분산 구조 유연하게 모형화, 수치 안정적 |
| Non-centered reparam | β_i = μ_h + L_h ε_i | 적은 이벤트 수에서 NUTS mixing 개선 |
| 새 이벤트 β_new | μ_{h,s} + L_{h,s} @ ε_s | 사후 Cholesky를 직접 사용 → 논문 수식과 정확히 일치 |
