# Event-Aware Load Forecasting via Multi-Model Baselines and Hierarchical Bayesian Tilting

> **논문 방향**: *Model-Agnostic Event-Aware Load Forecasting via Hierarchical Quadratic Tilting*

---

## 1. 프로젝트의 중요성

전력 계통 운영에서 **일 최대 부하(peak load) 예측**은 경제급전·예비력 확보·수요반응 설계의 핵심 입력값이다. ML·DL 모형이 일반 일자의 예측 정확도를 높였음에도, **추석·설날과 같은 다일 연속 명절 구간에서는 여전히 체계적 과대예측(over-prediction)** 이 발생할 수 있다.

| 문제 | 영향 |
|---|---|
| 명절 기간 수요 급감을 모형이 과소 반영 | 예비력 계획 오류 → 운영 비용 증가 |
| 수요 troughs가 모형 예측보다 훨씬 깊음 | 경제급전 왜곡 |
| 표준 예측구간이 명절 구간에서 교정 불량 | 불확실성 정보 신뢰 불가 |

이 문제는 다음 세 가지 구조적 어려움 때문에 순수 데이터 기반 모형으로는 해결하기 어렵다.

- **Data sparsity**: 명절은 연 1회, 유효 윈도우는 3~6일에 불과해 학습 표본이 극도로 적다.
- **Nonlinear, time-varying pattern**: 수요 감소-회복이 날짜 인덱스에 따라 비선형·비대칭적으로 진행된다.
- **Miscalibrated uncertainty**: 일반 구간에서 얻은 예측구간이 명절 구간에서 동일한 보정 수준을 유지하지 못한다.

---

## 2. 연구 방향

본 프로젝트는 **두 단계 이벤트 인지(event-aware) 예측 프레임워크**를 제안한다.

```
[Stage 1]  XGBoost / LightGBM / Random Forest / SVR / Seq2Seq-LSTM / Seq2Seq-GRU
           → model-specific baseline forecast  ŷ_t,m^base
[Stage 2]  Hierarchical Quadratic Tilt
           → common holiday correction e_t,m^tilt
           → adjusted forecast          ŷ_t,m^tilt
```

### 핵심 아이디어: Hierarchical Quadratic Tilting (HQT)

베이스라인 잔차를 표준화한 뒤, 명절 윈도우 내 날짜 인덱스(τ)에 대한 **이차(quadratic) 편향 곡선**을 Bayesian 계층 모형으로 추정한다.

- **같은 명절 유형**(추석·설날)의 과거 5년치를 pooling → 희소 데이터 문제 완화
- **이차 곡선** → 수요 감소-회복의 비선형·비대칭 패턴을 포착
- **Posterior predictive sampling** → 미래 신규 명절에도 외삽 가능
- 서로 다른 ML·DL 베이스라인에서 동일한 보정 절차를 적용해 model-agnostic 효과를 검증

---

## 3. 접근 방법

### 3.1 베이스라인: 동일 정보집합의 ML·DL 모형

HP filter와 Fourier 분해를 제거하고 다음 여섯 모형을 독립적인 베이스라인으로
사용한다.

- XGBoost, LightGBM, RBF-SVR: horizon별 24개 direct regressor
- Random Forest: 하나의 native multi-output regressor
- Seq2Seq-LSTM, Seq2Seq-GRU: encoder-decoder 기반 24시간 autoregressive forecast

모든 모형은 동일한 정보를 사용한다.

- encoder 입력: 과거 168시간의 전력수요·기온·습도·계절 더미
- decoder/direct 입력: 예측 시점에 알려진 향후 24시간 계절 더미
- 목표: 다음 24시간 전력수요
- 예측 원점: 매일 00시, stride 24시간
- 분할: 2019–2022 train / 2023 validation / 2024 test

기온·습도는 과거 관측값만 사용하며 scaler는 train에만 적합한다. validation과
test의 첫 윈도는 직전 구간의 과거 168시간을 사용할 수 있지만 목표 구간이
분할 경계를 넘지는 않는다. 상세 구현과 실행법은
[`ver2/README.md`](ver2/README.md)에 정리되어 있다.

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

사후 추론은 Hamiltonian Monte Carlo (NUTS)를 사용한다. 기본 설정은
4 chains × 1,000 posterior draws이며, mixing이 느린 SVR은
4 chains × 2,000 posterior draws로 재적합했다.

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
PI_t = ŷ_t^tilt ± z_{α/2} · σ · sqrt( E[σ_r²] + Var(z_{i,t}) )
```

---

## 베이스라인 실험 결과

2024년 1월 1일~10월 31일 7,320시간 hold-out의 HQT 적용 전 결과다.

| 모델 | MAE (MW) | RMSE (MW) | WAPE (%) |
|---|---:|---:|---:|
| LightGBM | 2,254.3 | 3,575.3 | 3.488 |
| XGBoost | 2,256.2 | 3,571.7 | 3.491 |
| SVR | 2,329.0 | 3,585.1 | 3.604 |
| Seq2Seq-LSTM | 2,418.3 | 3,751.5 | 3.742 |
| Random Forest | 2,476.0 | 3,801.1 | 3.831 |
| Seq2Seq-GRU | 2,689.5 | 3,958.6 | 4.162 |

같은 test에서 설날·추석 공식 7일(168시간)만 분리하면 모든 모형이 큰 양의
bias를 보여 HQT의 보정 대상을 명확히 확인할 수 있다. 아래 값 역시 HQT 적용
전 진단 결과다.

| 모델 | 명절 MAE (MW) | 명절 RMSE (MW) | 명절 bias (MW) |
|---|---:|---:|---:|
| Seq2Seq-LSTM | 4,846.8 | 6,750.0 | +4,139.8 |
| Seq2Seq-GRU | 6,065.7 | 8,161.9 | +5,839.0 |
| SVR | 6,319.5 | 8,684.9 | +5,954.9 |
| XGBoost | 7,095.5 | 9,337.1 | +7,029.1 |
| LightGBM | 7,508.3 | 9,986.5 | +7,434.1 |
| Random Forest | 7,820.7 | 10,236.9 | +7,630.3 |

### HQT 적용 결과

HQT는 2019–2023년의 설날·추석 이벤트로 적합하고 2024년 이벤트를 새로운
이벤트로 예측했다. 공식 설날·추석 7일(168시간)의 결과는 다음과 같다.

현재 HQT 적합 잔차 중 2019–2022년 베이스라인 예측은 in-sample이고 2023년은
hold-out validation이다. 논문 최종 추정에서는 2019–2022년도 rolling-origin
OOF 예측으로 교체하는 것이 권장된다.

| 모델 | Baseline MAE (MW) | Tilted MAE (MW) | MAE 개선 | Tilted bias (MW) |
|---|---:|---:|---:|---:|
| Random Forest | 7,820.7 | 6,029.2 | 22.9% | +4,239.7 |
| SVR | 6,319.5 | 4,941.0 | 21.8% | +2,591.4 |
| Seq2Seq-GRU | 6,065.7 | 4,839.4 | 20.2% | +2,932.9 |
| LightGBM | 7,508.3 | 6,398.9 | 14.8% | +5,777.5 |
| XGBoost | 7,095.5 | 6,300.1 | 11.2% | +5,945.0 |
| Seq2Seq-LSTM | 4,846.8 | 4,401.8 | 9.2% | +2,899.3 |

모든 모형에서 공식 명절 집계 MAE와 RMSE가 감소했다. 다만 보정 후에도 양의
bias가 남고, 이벤트 유형별로는 효과가 이질적이다. 예를 들어 GRU의 추석
윈도우 MAE는 23.9% 개선됐지만 설날 윈도우는 22.0% 악화됐다. 따라서 HQT의
효과를 단일 평균만으로 해석하지 않고 명절 유형별 결과도 함께 보고한다.

NUTS 진단은 모든 모형에서 divergence 0, 최대 R-hat 1.00을 기록했다. 다섯
모형은 4,000 posterior draws, mixing이 느렸던 SVR은 8,000 draws를 사용했다.

---

## 프로젝트 구조

```
Demadn-Quadratic-Tilting/
├── ver2/                                  # 이번 실험의 설정·결과·재현 README
│   ├── configs/
│   └── results/
├── demand_quadratic_tilting/
│   ├── forecasting/                       # ML·DL 베이스라인 파이프라인
│   ├── model.py                           # HQT posterior fitting
│   ├── tilt.py                            # quadratic tilt 적용
│   └── pipeline.py                        # HQT end-to-end pipeline
├── scripts/train_baselines.py             # 168→24 베이스라인 학습 CLI
├── scripts/apply_hqt_to_baselines.py       # 모든 베이스라인에 동일 HQT 적용
├── tests/                                 # 누출·shape·재현성 테스트
├── docs/baseline_forecasting.md
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

PyTorch recurrent model과 PyMC(MCMC)는 지원 device가 다르므로 분리한다.

| 컴포넌트 | NVIDIA CUDA | Apple MPS | CPU |
|---|---|---|---|
| Seq2Seq LSTM (PyTorch) | ✅ | CPU fallback¹ | ✅ |
| Seq2Seq GRU (PyTorch) | ✅ | ✅ | ✅ |
| HQT NUTS (PyMC) | ❌ | ❌ | ✅ |
| HQT numpyro (JAX) | ✅ | ❌ | ✅ |

¹ 현재 환경의 PyTorch 2.8 MPS LSTM은 저장된 `state_dict`가 새 프로세스에서
학습 직후 예측을 재현하지 못해 CPU를 사용한다. MPS GRU는 재현성 검증을
통과했다.

### HQT 파이프라인 실행

```bash
uv run python scripts/apply_hqt_to_baselines.py \
  --output-dir artifacts/hqt_baselines \
  --chains 4 --draws 1000 --tune 1000 \
  --target-accept 0.99 --resume
```

커밋에 포함되는 고정 설정과 핵심 결과 CSV는 `ver2/configs/`와
`ver2/results/`에서 확인할 수 있다. 모델 체크포인트·전체 시간별 예측·posterior는
용량 때문에 `artifacts/`에 생성하되 Git에는 포함하지 않는다.

---

## 주요 설계 결정

| 항목 | 선택 | 이유 |
|---|---|---|
| 입력 윈도 | 168시간 → 24시간 | 직전 1주로 다음 하루를 예측하는 공통 정보집합 |
| 베이스라인 | XGBoost, LightGBM, Random Forest, SVR, Seq2Seq-LSTM/GRU | 특정 복합 모형이 아닌 서로 다른 모형군에서 tilting 효과 검증 |
| 틸트 형태 | 이차(quadratic) | 수요 감소-회복의 오목(concave) 곡선 포착 |
| 계층 prior | LKJ Cholesky | β 계수 간 공분산 구조 유연하게 모형화, 수치 안정적 |
| Non-centered reparam | β_i = μ_h + L_h ε_i | 적은 이벤트 수에서 NUTS mixing 개선 |
| 새 이벤트 β_new | μ_{h,s} + L_{h,s} @ ε_s | 사후 Cholesky를 직접 사용 → 논문 수식과 정확히 일치 |
| Apple LSTM device | CPU fallback | 저장 후 새 프로세스 재현성 보장 |
| MCMC device | MCMC_SAMPLER (numpyro/nuts) | PyMC는 MPS 미지원 → NVIDIA만 JAX 가속, 그 외 CPU |
