# HQT Reviewer 대응 실험 설계 기획서

작성일: 2026-08-26  
상태: 사용자 승인 설계안, 세부 구현 계획 작성 전 검토본

## 1. 목적

본 작업의 목적은 accept된 conference 논문의 Reviewer 1·2 의견 중 실험으로 직접 검증할 수 있는 항목을 최소한의 재계산으로 반영하는 것이다. 핵심은 휴일 정보를 모르는 기존 베이스라인만 제시하는 문제를 해소하고, 휴일 더미변수를 사용하는 현실적인 베이스라인에서도 HQT가 추가적인 개선을 제공하는지 검증하는 데 있다.

이번 작업은 다음 세 결과를 만든다.

1. 휴일 비인지 베이스라인 B0와 휴일 인지 베이스라인 B1의 성능 비교
2. B1 잔차로 다시 추정한 H0·H1·H2의 10-event LOEO 결과
3. 2020--2023 OOF 잔차만 사용해 2024년 두 명절을 평가하는 인과적 holdout 결과

AR(1), 시간대 효과, 확률예측 지표는 이번 conference 결과 반영 범위에 포함하지 않는다. 해당 항목의 기존 코드나 결과를 conference HQT 결과와 혼합하지 않는다.

## 2. 핵심 설계 결정

### 2.1 기존 B1을 reviewer용 holiday-aware baseline으로 사용한다

새 feature set을 만들지 않고 현재 구현된 B1을 사용한다. B1은 B0의 일반 시계열·달력 변수에 다음 일곱 개의 forecast-origin 가용 변수를 추가한다.

- `is_public_holiday`
- `official_sequence_position`
- `seollal_distance`
- `chuseok_distance`
- `is_substitute_or_temporary_holiday`
- `is_seollal`
- `is_chuseok`

`is_seollal`과 `is_chuseok`은 공식 연휴 구간에서만 1인 상호 배타적 더미변수다. HQT 분석 창인 공식 연휴 전후 1일까지 확장된 window dummy가 아니다. 논문과 reviewer 답변에는 이 의미를 그대로 기술한다.

별도의 `is_seollal_window`와 `is_chuseok_window`는 추가하지 않는다. 이를 추가하면 HQT가 사용하는 분석 창을 베이스라인이 직접 학습하게 되어 feature engineering과 residual correction의 역할이 겹치며, 기존 B1 산출물 전체를 무효화한다. Reviewer 답변에서 공식 연휴 전후 1일 더미를 명시적으로 약속하지 않는 한 현재 B1이 더 보수적이고 재현 가능한 비교다.

### 2.2 B0와 B1의 역할을 분리한다

- B0-H0: 휴일 정보를 사용하지 않는 기존 conference 베이스라인
- B1-H0: 휴일 더미와 운영상 알려진 달력 정보를 사용하는 reviewer 기준 베이스라인
- B1-H1: B1 잔차에 동일 명절 유형의 평균 잔차 이동을 적용한 상수 보정
- B1-H2: B1 잔차에 AR 없는 원 논문의 hierarchical quadratic tilting을 적용한 보정

주요 HQT 효과는 `B1-H2 대 B1-H0`으로 정의한다. `B0-H0 대 B1-H0`은 더미변수 자체의 기여이며 HQT의 기여로 해석하지 않는다. B0 잔차에서 학습한 H1/H2 보정계수는 B1에 재사용하지 않는다.

### 2.3 기존 베이스라인 산출물을 조건부 재사용한다

다섯 모델과 B0/B1에 대해 다음 산출물이 완전하고 manifest hash가 일치하면 베이스라인을 다시 학습하지 않는다.

- 2020 OOF: 2019 학습 후 2020 예측
- 2021 OOF: 2019--2020 학습 후 2021 예측
- 2022 OOF: 2019--2021 학습 후 2022 예측
- 2023 OOF: 2019--2022 학습 후 2023 예측
- 2024 final: 2019--2023 학습 후 2024 예측

검증 대상은 원자료, event registry, holiday calendar, model configuration, feature schema, seed, split, 예측 행 수와 timestamp coverage의 digest다. 하나라도 다르면 해당 context만 재생성한다. 단순히 파일이 존재한다는 이유로 캐시를 신뢰하지 않는다.

## 3. 통계 모델 계약

### 3.1 H0

H0는 보정하지 않은 각 베이스라인의 held-out 예측이다.

\[
\hat y_t^{(m,H0)}=\hat y_t^{(m)}.
\]

### 3.2 H1

H1은 LOEO 학습 집합에서 held-out event와 같은 명절 유형의 raw-MW 잔차 평균을 사용한다.

\[
\bar r_{h,-i}^{(m)}
=
\frac{1}{|\mathcal W_{h,-i}|}
\sum_{t\in\mathcal W_{h,-i}}r_t^{(m)},
\qquad
\hat y_t^{(m,H1)}=\hat y_t^{(m)}+\bar r_{h,-i}^{(m)}.
\]

Held-out event의 잔차는 평균 계산에 들어가지 않는다. 설날 평균을 추석에 사용하거나 반대로 사용하는 것도 금지한다.

### 3.3 H2

H2는 accept된 conference 논문의 원래 HQT를 그대로 사용한다.

\[
z_{i,t}^{(m)}
\sim
\mathcal N\!\left(
\beta_{i0}^{(m)}+\beta_{i1}^{(m)}\tau_{i,t}
+\beta_{i2}^{(m)}\tau_{i,t}^{2},
\sigma_{r,m}^{2}
\right),
\]

\[
\boldsymbol\beta_i^{(m)}
\sim
\mathcal N\!\left(
\boldsymbol\mu_{h(i)}^{(m)},
\boldsymbol\Sigma_{h(i)}^{(m)}
\right).
\]

고정 모델 계약은 다음과 같다.

- 명절 유형별 quadratic coefficient partial pooling
- `mu ~ Normal(0, 10)`
- between-event scale `HalfNormal(1)`
- correlation `LKJ(eta=2)`
- observation scale `HalfNormal(1)`
- iid Gaussian observation error
- 비중심화 parameterization
- pandemic covariate 없음
- hour-of-day profile 없음
- AR 계수 없음

점 보정은 새로운 event coefficient posterior draw의 quadratic correction을 평균하여 계산한다. H2의 사후분포와 보정값은 모델·feature set·held-out event마다 독립적으로 산출한다.

## 4. 평가 설계

### 4.1 10-event retrospective LOEO

평가 universe는 2020--2024년 설날과 추석 각 5회, 총 10개 occurrence다. 각 fold는 정확히 하나의 physical event를 제외하고 나머지 9개 event의 잔차로 H1과 H2를 학습한다.

예를 들어 `seollal-2022`를 평가할 때 학습 집합에는 다른 설날 4회와 추석 5회가 들어가며 `seollal-2022`의 어떤 시간별 행도 들어가지 않는다. 명절 유형별 partial pooling이므로 새로운 설날 correction의 평균은 설날 집단 posterior에서 생성된다. 두 명절 유형은 공통 observation scale `sigma_r`을 공유하지만 coefficient의 평균·공분산은 유형별로 분리하며, 새로운 공동 hyperprior는 도입하지 않는다.

이 LOEO는 event 간 일반화 성능을 확인하는 회고적 분석이다. 과거 event fold가 미래 연도의 event를 학습에 포함할 수 있으므로 실시간 배포 시뮬레이션 또는 causal backtest라고 부르지 않는다.

### 4.2 2024 causal holdout

운영 시점의 주장에는 별도의 2024 평가를 사용한다.

- 학습 잔차: expanding OOF로 생성한 2020--2023의 8개 event
- 평가 baseline: 2019--2023으로 학습한 final baseline의 2024 예측
- 평가 event: `seollal-2024`, `chuseok-2024`
- 금지 정보: 2024 관측 잔차, 2024 비이벤트 RMSE, 2024 event coefficient

2024 보정에 필요한 residual scale은 2024 관측값에서 추정하지 않고 마지막으로 이용 가능한 pre-2024 OOF scale을 사용한다. 2024 비이벤트 residual scale은 평가 종료 후 scale stability 진단에만 사용한다.

### 4.3 boundary taper 민감도

H2 본 결과는 taper를 적용하지 않은 원 논문 정의다. 공식 연휴 전후 1일 buffer에서 보정이 갑자기 시작·종료되는 문제를 확인하기 위해 cosine taper 결과를 `H2-taper`라는 보조 민감도 분석으로만 산출한다.

H2-taper가 H2보다 좋아도 본문에서 H2 결과를 대체하지 않는다. 두 결과를 함께 보고하여 경계 처리의 영향을 분리한다.

## 5. 평가 지표와 통계 요약

### 5.1 필수 point metrics

각 모델과 correction에 대해 다음을 계산한다.

- RMSE
- MAE
- MAPE
- SMAPE
- \(R^2\)
- H0 대비 RMSE 개선율

결과 범위는 다음 네 수준으로 분리한다.

1. 2024 전체 test period의 B0-H0와 B1-H0
2. 10-event LOEO 전체 pooled
3. 설날 pooled와 추석 pooled
4. 10개 개별 occurrence

개별 occurrence 표에는 baseline RMSE, corrected RMSE, 개선율을 포함한다. 개선율이 음수인 event 수와 최대 악화 폭을 반드시 보고한다.

### 5.2 event-level 불확실성

시간별 관측을 독립 표본처럼 취급하지 않는다. H2와 H0의 주요 비교 단위는 10개 occurrence다.

- 10개 event별 RMSE 개선율의 중앙값
- event 단위 bootstrap 95% 신뢰구간
- one-sided Wilcoxon signed-rank p-value

표본 수가 10개뿐이므로 p-value 단독으로 결론을 내리지 않는다. 효과크기, 신뢰구간, 개선·악화 event 수를 함께 보고한다. 시간별 Diebold--Mariano 결과가 필요하면 기존 conference 결과와의 연속성을 위한 보조 지표로만 두고 event-level 결과보다 앞세우지 않는다.

### 5.3 Transformer 예외 처리

Transformer에서 RMSE는 개선되지만 MAE 또는 SMAPE가 악화되는 경우 이를 그대로 보고한다. 모든 loss에서 일관되게 개선된다는 문장을 사용하지 않는다. 모델 순위가 correction 전후에 달라지는지도 실제 결과로만 판단한다.

## 6. 산출물 계약

각 `(model, feature_set)` context는 독립 디렉터리에 다음 파일을 생성한다.

```text
artifacts/hqt-reviewer/
├── manifest.json
├── baseline_feature_comparison.parquet
├── retrospective-loeo/
│   └── MODEL/FEATURE_SET/
│       ├── folds/OCCURRENCE_ID/
│       │   ├── input_identity.json
│       │   ├── posterior.nc
│       │   ├── hourly_predictions.parquet
│       │   ├── metrics.parquet
│       │   ├── manifest.json
│       │   └── COMPLETE
│       ├── event_metrics.parquet
│       ├── pooled_metrics.parquet
│       ├── event_improvements.parquet
│       ├── improvement_summary.parquet
│       └── scale_stability.parquet
├── causal-2024/
│   └── MODEL/B1/
│       ├── posterior.nc
│       ├── hourly_predictions.parquet
│       ├── event_metrics.parquet
│       ├── pooled_metrics.parquet
│       └── manifest.json
└── paper/
    ├── tables/
    └── figures/
```

모든 posterior와 결과 파일은 입력 digest, model specification, seed, sampler 설정에 묶인다. 완료된 fold는 `COMPLETE`와 manifest digest가 모두 유효할 때만 재사용한다. 중단된 실행은 완료된 fold 다음부터 자동 재개한다.

## 7. 계산 및 실행 정책

paper profile은 다음으로 고정한다.

- 4 chains
- chain당 1,000 warm-up
- chain당 1,000 retained draws
- `target_accept=0.99`
- seed를 root seed와 held-out occurrence id에서 결정적으로 파생
- `cores` 미지정 시 논리 CPU를 탐색하되 최대 4개 chain worker 사용

하나의 PyMC fit 내부에서는 chain 병렬화를 사용한다. 여러 context를 동시에 실행하여 각 프로세스가 다시 4개 chain을 만드는 중첩 병렬화는 기본값으로 금지한다. 모델 순서는 XGBoost, LightGBM, SVR, Seq2Seq-LSTM, Transformer로 고정하고 context 단위 checkpoint로 재개한다.

paper 결과에 사용할 posterior는 다음 조건을 모두 만족해야 한다.

- 최대 \(\hat R\leq1.01\)
- 최소 bulk ESS \(\geq400\)
- 최소 tail ESS \(\geq400\)
- divergent transition 0회

진단 실패 fold는 결과 집계에서 자동 제외하지 않고 실행을 실패시킨다. sampler 설정 변경 후 재실행할 경우 새 artifact identity를 사용한다.

## 8. 코드 변경 경계

현재 저장소에는 B1 특징 생성, expanding OOF, final-2024 baseline, AR 없는 legacy HQT와 retrospective LOEO 초안이 존재한다. 세부 구현 계획에서는 다음 경계만 수정한다.

- `hqrc_v3/src/hqrc_v3/features.py`: B1 더미 의미와 forecast-origin 계약 검증
- `hqrc_v3/src/hqrc_v3/baselines/config.py`: B0/B1 schema freeze와 manifest 검증
- `hqrc_v3/src/hqrc_v3/bayes/legacy_hqt.py`: 원 conference HQT 모델 계약 고정
- `hqrc_v3/src/hqrc_v3/legacy_hqt_loeo.py`: 10-event LOEO, 통계 요약, checkpoint 완성
- 신규 causal-2024 모듈: pre-2024 잔차만 사용하는 H1/H2 학습과 평가
- 신규 reviewer report 모듈: 논문 표·그림 입력 파일 생성
- `hqrc_v3/src/hqrc_v3/cli.py`: 전체 실행과 재개가 가능한 명령 연결
- 관련 unit/integration/slow tests와 `hqrc_v3/README.md`

기존 HQRC의 AR 진단·시간대 효과·확률예측 경로는 삭제하지 않지만 reviewer용 HQT command에서 import하거나 호출하지 않는다.

## 9. 테스트 전략

### 9.1 feature 및 leakage tests

- B0에 holiday/event 토큰을 가진 열이 없음을 검증
- B1에 일곱 개 추가 열이 정확한 순서로 존재함을 검증
- `is_seollal`과 `is_chuseok`이 공식 연휴에서만 1임을 검증
- 미래 target이나 관측 잔차가 B1 특징으로 들어가지 않음을 검증
- 임시공휴일은 발표일이 forecast origin보다 빠른 경우에만 사용됨을 검증

### 9.2 LOEO tests

- universe가 2020--2024의 정확히 10개 event인지 검증
- 각 fold가 held-out 1개와 training 9개로 구성되는지 검증
- held-out occurrence 행이 H1/H2 학습 집합에 한 행도 없는지 검증
- H1이 같은 명절 유형의 raw-MW 잔차만 평균하는지 검증
- H2 graph에 `phi`, hour profile, pandemic coefficient가 없는지 검증
- 동일 seed와 입력에서 correction draw 및 결과 digest가 재현되는지 검증

### 9.3 causal-2024 tests

- 학습 event가 2020--2023의 8개뿐인지 검증
- 2024 관측 잔차와 residual scale을 학습에서 요청하면 실패하는지 검증
- final baseline과 OOF residual의 model·feature set·seed가 다르면 실패하는지 검증

### 9.4 artifact 및 report tests

- 불완전하거나 hash가 다른 checkpoint를 재사용하지 않음
- pooled metric을 event metric의 단순 평균으로 잘못 계산하지 않음
- 음수 개선율이 표와 그림에서 유지됨
- B0→B1 개선과 B1-H0→B1-H2 개선이 서로 다른 열로 출력됨
- paper profile 진단 기준을 통과하지 못한 posterior가 결과 표에 들어가지 않음

## 10. 논문 반영 범위

실험 완료 후 conference 논문에는 실제 산출된 값만 다음 위치에 반영한다.

1. Methodology: B0/B1 정의와 B1 더미변수의 forecast-origin 가용성
2. Experimental Design: expanding OOF residual 생성, retrospective LOEO, causal-2024의 역할 구분
3. Results: B0-H0 대 B1-H0, B1의 H0/H1/H2, holiday별·event별 결과
4. Discussion: 더미변수로 회복되는 오차와 HQT의 추가 기여를 분리하여 해석
5. Limitations: LOEO의 회고적 성격, event 수 10개의 한계, 악화 event 공개

결과가 나오기 전에 “모든 모델 개선”, “두 대안보다 우수”, “통계적으로 유의” 같은 결론을 미리 고정하지 않는다. 실제 결과가 기존 conference 초록의 6.8--16.7%와 다르면 새 결과로 교체하고 차이를 설명한다.

## 11. 완료 조건

다음 조건을 모두 만족하면 reviewer 대응 실험이 완료된 것으로 본다.

1. 다섯 베이스라인의 B0/B1 2024 성능표가 생성된다.
2. 다섯 모델 B1에 대해 10개 LOEO fold의 H0/H1/H2 결과가 모두 존재한다.
3. 다섯 모델 B1에 대해 2024 causal holdout 결과가 존재한다.
4. 모든 paper posterior가 MCMC 진단 기준을 통과한다.
5. pooled, 설날, 추석, 개별 event 결과와 악화 event 수가 산출된다.
6. event-level bootstrap CI와 Wilcoxon 결과가 산출된다.
7. scale stability와 H2-taper 민감도 결과가 본 결과와 분리되어 산출된다.
8. 모든 결과가 입력·설정 digest와 실행 manifest로 재현 가능하다.
9. 논문 표와 그림이 정규화된 artifact에서 자동 생성되며 수작업 숫자 복사가 없다.
