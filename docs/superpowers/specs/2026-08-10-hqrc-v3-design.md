# HQRC v3 강건성 재설계 명세

작성일: 2026-08-10
상태: 구현 전 설계 검토본

## 1. 목적

이 작업은 기존 결과물과 실험 파일을 변경하지 않고 새 디렉터리 `hqrc_v3/`에 재현 가능한 HQRC 실험 파이프라인을 구축한다. 핵심 목표는 다음 네 가지다.

1. 2024년 성능을 한 번만 확인하는 정적 홀드아웃과, HQRC 학습에 필요한 정직한 out-of-fold(OOF) 잔차 생성을 분리한다.
2. AR(1)을 임의의 사전분포로 고정하지 않고, 각 베이스라인의 학습 구간 잔차 ACF/PACF 분석으로 Beta 사전분포를 보정한 뒤 Bayesian 모델에 사용한다.
3. 논문에 제시한 계층적 이차 추세, 원형 시간대 효과, 휴일 유형별 부분 풀링, 팬데믹 공변량, AR(1) 오차 구조를 보존한다.
4. Rust는 실제 병목에서만 사용한다. 데이터 처리에는 Rust 기반 Polars/Arrow를 사용하고, NUTS에는 선택적으로 Rust 샘플러인 nutpie를 비교한다. 별도 Rust 재작성은 벤치마크가 이득을 입증하기 전에는 하지 않는다.

본 설계의 일차 산출물은 연구 코드와 검증 가능한 중간 산출물이다. 논문 표의 최종 수치는 전체 실험을 실행한 뒤에만 채운다.

## 2. 설계 결정 요약

### 2.1 최종 홀드아웃과 expanding OOF의 역할

최종 평가는 다음 정적 구조를 사용한다.

- 모델/하이퍼파라미터 선택: 2019--2022 학습, 2023 검증
- 최종 베이스라인 재학습: 선택된 설정을 고정하고 2019--2023 재학습
- 최종 테스트: 2024년 1월 1일--10월 31일

반면 HQRC는 과적합된 in-sample 잔차가 아니라 배포 상황과 유사한 OOF 잔차가 필요하다. 따라서 HQRC 보정 학습용 잔차는 다음 expanding-origin 예측으로 생성한다.

| 평가 연도 | 베이스라인 학습 연도 |
|---|---|
| 2020 | 2019 |
| 2021 | 2019--2020 |
| 2022 | 2019--2021 |
| 2023 | 2019--2022 |
| 2024 | 2019--2023 |

2020--2023 OOF 잔차와 여덟 개 휴일 발생은 2024년 HQRC 학습 및 AR 사전분포 보정에 사용한다. 각 fold 모델은 OOF 잔차를 생성한 뒤 최종 예측에는 사용하지 않는다. 선택된 설정으로 2019--2023 전체를 다시 학습한 final baseline만 2024를 예측한다. 2024 예측 잔차는 최종 테스트와 10개 발생 LOEO 구조 분석에만 사용한다.

`2019--2022 / 2023 / 2024`는 모델 설정 선택과 최종 일반화 평가를 담당한다. Expanding OOF는 이 바깥 분할을 대체하는 별도 평가법이 아니라, pre-2024 구간에서 HQRC의 학습 target인 정직한 baseline residual을 만드는 내부 절차다.

### 2.2 계산 전략

하이퍼파라미터 탐색은 2019--2022 학습과 2023 검증에서 한 번만 수행하며, 기존 연구에서 확정된 versioned 설정을 사용할 경우 생략할 수 있다. 선택된 설정은 모든 OOF fold와 final baseline에서 고정한다. 연도별로 반복하는 것은 hyperparameter tuning이 아니라 fitted model parameter의 재추정이다.

각 `(baseline, feature_set, seed, fold)` 예측은 한 번 생성한 뒤 immutable Parquet artifact로 캐시한다. 이후 AR 진단, H0--H5, pooling ablation과 확률평가는 baseline을 다시 학습하지 않고 같은 OOF 잔차를 재사용한다. 독립 fold는 자원 과다할당을 막는 worker 제한 아래 병렬 실행한다.

최종적으로 필요한 baseline fit은 다음 다섯 개 연도 단계다.

| 목적 | 학습 연도 | 예측 연도 |
|---|---|---|
| OOF residual | 2019 | 2020 |
| OOF residual | 2019--2020 | 2021 |
| OOF residual | 2019--2021 | 2022 |
| OOF residual | 2019--2022 | 2023 |
| final forecast | 2019--2023 | 2024 |

Sliding window와 2020 제외 전용 실험은 필수 설계에 포함하지 않는다. 특정 occurrence의 영향은 이미 계획된 event-level LOEO에서 대칭적으로 진단한다.

### 2.3 AR(1)은 유지하되 잔차가 사전분포를 결정한다

AR(1)은 논문 모델에 포함한다. 다만 기존의

\[
(\phi+1)/2 \sim \mathrm{Beta}(5,2)
\]

를 그대로 사용하지 않는다. 예비 잔차에서 관찰된 높은 lag-1 상관과 이 사전분포의 중심이 맞지 않기 때문이다. 각 `(baseline, feature_set)` 조합에 대해 학습에 허용된 OOF 이벤트만 이용하여 ACF/PACF 진단과 수치 보정을 수행한다. `\phi`는 표본 ACF 값으로 고정하지 않고, 보정된 Beta prior 아래에서 posterior parameter로 계속 추정한다.

### 2.4 Rust 사용 범위

51,144시간 규모에서는 전체 모델을 Rust로 다시 작성할 근거가 부족하다. 주요 계산 비용은 Python 반복문이 아니라 XGBoost/LightGBM/신경망 학습 및 NUTS이다. 따라서 다음 경계를 사용한다.

- 열 지향 전처리와 집계: Polars lazy API 및 Arrow 메모리 사용
- 중간 자료: Parquet/Arrow, 불필요한 pandas 복사 금지
- CPU 병렬 처리: Polars, tree learner, 독립 baseline/seed 실행 단위
- Bayesian sampler: PyMC NUTS를 기준 구현으로 두고 nutpie를 동일 모델에서 선택 가능하게 함
- Rust 채택 판정: 동일 posterior와 seed 조건에서 wall time, peak RSS, bulk ESS/s, tail ESS/s를 비교

nutpie가 진단 품질을 유지하면서 기준 PyMC보다 wall time 또는 최소 bulk ESS/s에서 의미 있는 이득을 보일 때만 전체 실행 기본값으로 승격한다. 별도 PyO3/maturin 커널은 프로파일링에서 전처리 또는 예측 후처리가 전체 wall time의 20% 이상이고, 복사 비용이 병목으로 확인된 경우에만 후속 작업으로 고려한다.

## 3. 범위와 비범위

### 3.1 구현 범위

- 원본 시간별 자료의 스키마 검사와 일 단위 24시간 예측 샘플 생성
- B0/B1 특징 집합 생성 및 누수 검사
- 다섯 베이스라인: XGBoost, LightGBM, SVR, Seq2Seq-LSTM, Transformer
- 정적 train/validation/test와 expanding-origin OOF
- fold별 OOF 잔차 및 비휴일 잔차 척도 생성
- 이벤트 레지스트리와 휴일 창 생성
- ACF/PACF 기반 AR(1) 진단 및 prior 보정
- H0--H5 보정 변형과 H3 pooling ablation
- HQRC posterior와 point/probabilistic prediction
- 2024 causal holdout 및 10-event LOEO 평가
- event-level 통계 검정, 표, 그림용 정규화된 결과 파일
- sampler 성능 벤치마크와 실행 manifest

### 3.2 이번 구현의 비범위

- 기존 `demand_quadratic_tilting/`, `ver2/`, notebook, 그림, 모델 산출물 수정
- 원자료를 외부 소스에서 새로 수집하거나 기존 관측값을 임의로 보정
- 2024 테스트 결과에 맞춘 하이퍼파라미터 또는 분할 방식 선택
- 데이터 근거 없이 AR 차수를 자동으로 높이거나 Student-t를 주 likelihood로 교체
- 실험 실행 전에 논문 결과 숫자를 추정하여 채우는 작업
- 전체 Bayesian 모델의 순수 Rust 재작성

## 4. 새 코드베이스 구조

새 구현은 다음 경계를 가진 독립 프로젝트로 둔다.

```text
hqrc_v3/
├── README.md
├── pyproject.toml
├── configs/
│   ├── experiment.toml
│   ├── events.csv
│   ├── holiday_calendar.csv
│   └── model_spaces.toml
├── src/hqrc_v3/
│   ├── cli.py
│   ├── config.py
│   ├── contracts.py
│   ├── data.py
│   ├── events.py
│   ├── features.py
│   ├── splits.py
│   ├── residuals.py
│   ├── baselines/
│   │   ├── protocol.py
│   │   ├── boosting.py
│   │   ├── svr.py
│   │   └── sequence.py
│   ├── diagnostics/
│   │   └── ar.py
│   ├── bayes/
│   │   ├── model.py
│   │   ├── priors.py
│   │   ├── samplers.py
│   │   └── predictive.py
│   ├── corrections/
│   │   ├── variants.py
│   │   └── similar_day.py
│   ├── evaluation/
│   │   ├── metrics.py
│   │   ├── inference.py
│   │   └── reports.py
│   └── provenance.py
└── tests/
    ├── unit/
    ├── integration/
    └── fixtures/
```

`hqrc_v3`는 기존 모듈을 import하여 암묵적으로 동작하지 않는다. 재사용할 로직은 출처를 기록하고 새 계약에 맞춰 명시적으로 옮긴다. 원본 자료 경로는 config로 주입하며 새 폴더 안에 데이터를 복제하지 않는다.

## 5. 데이터 및 시간 계약

### 5.1 입력 스키마

필수 열은 timestamp, load MW, temperature, relative humidity, source holiday name, source public-holiday flag다. 기존 열 이름은 config의 명시적 rename map으로 표준 이름에 매핑한다. 고정 논문 자료에서는 `holiday_name`과 `is_holiday_dummies`를 각각 source holiday name과 source public-holiday flag로 보존한다. 파이프라인은 다음 조건을 만족하지 않으면 즉시 실패한다.

- timestamp 중복 없음
- 오름차순 정렬 후 1시간 간격
- 2019-01-01 00:00부터 2024-10-31 23:00까지 51,144개 관측
- load가 유한하고 양수
- 학습에 필요한 weather 값이 결측 처리 정책 이후 유한함
- public-holiday flag가 시간별 0/1이고 같은 날짜의 24개 행에서 동일함
- holiday name과 flag가 날짜 수준에서 일치하며, 고정 자료의 public-holiday 날짜가 107개임

결측 처리 통계는 fold 학습 구간에서만 계산한다. 보간 또는 대체가 일어난 timestamp와 방법은 별도 audit 파일에 기록한다.

### 5.2 예측 샘플

각 샘플의 forecast origin은 매일 00:00 직전이며 다음 구조를 가진다.

- 과거 입력: 직전 168시간
- 미래 목표: 당일 00:00--23:00의 24시간
- 미래 공변량: forecast origin에서 알려진 값만 허용

샘플의 전체 24시간 target이 한 partition 안에 있을 때만 해당 partition에 포함한다. 경계에서 평가 입력이 과거 학습 partition에 걸치는 것은 forecast origin에서 이미 관측된 정보이므로 허용하지만, 학습 target이 cutoff 이후로 넘어가면 학습 샘플에서 제외한다. 이 계약이면 미래 평가 target이 학습 feature나 target에 들어가지 않으므로 purge/embargo를 적용하지 않고 `gap_days=0`으로 고정한다.

### 5.3 특징 집합

B0는 휴일과 무관한 일반 예측 능력을 표현한다.

- lagged load/weather 168시간
- 알려진 미래 hour, day-of-week, weekend
- 알려진 미래 연간 및 주간 Fourier 항

이 자료에는 forecast origin 이전에 발행된 24시간 기상예보 vintage가 없다. 따라서 주 분석에서 temperature, humidity, heating/cooling degree hour는 과거 168시간에만 사용한다. target window에서 실현된 기상 관측치나 그 파생항을 `oracle_weather`로 이름만 바꾸어 입력하는 것도 금지한다. 별도 weather/oracle 민감도를 추가하려면 발행시각 또는 명시적 oracle 라벨과 별도 결과표가 필요하다.

B1은 B0 전체에 다음만 추가한다.

- source public-holiday flag
- 공식 연휴 내 signed day position
- 설날까지 signed distance
- 추석까지 signed distance
- substitute/temporary public-holiday flag
- holiday type

day-of-week는 일반 주간 예측 변수이므로 B0에 속하며 B1-only 특징으로 중복 기술하지 않는다. substitute/temporary flag는 source public-holiday 행 중 이름이 `Alternative holiday`로 시작하거나 `Temporary Public Holiday`와 같거나, 2024-10-01의 `Armed Forces Day`인 날짜에만 1이다. 고정 자료에서는 14개 날짜다. 2024-10-01은 B1 public/temporary flag에는 포함되지만 holiday type은 0이고 HQRC 보정 창에는 포함되지 않는다.

B0 특징 행렬에 holiday 이름, holiday flag, event-relative time, 이벤트 창 여부가 들어가면 테스트가 실패한다. 이벤트 정보는 B0 예측과 별개로 HQRC window 선택에만 사용한다. 공식 연휴 내 signed day position/type과 `official sequence +/- 1 day`인 HQRC 분석 창은 서로 다른 계약이다.

## 6. 이벤트 계약

`configs/events.csv`는 occurrence id, holiday type, central date, official sequence start/end, pandemic flag를 가진다. 2020--2024 설날과 추석 총 10개 correction/evaluation occurrence를 다음과 같이 명시한다. 날짜 범위는 현재 자료의 휴일 라벨과 대체공휴일을 포함한 공식 연휴 범위를 교차 확인한 값이다.

| occurrence | type | central | official start | official end | restriction flag |
|---|---|---|---|---|---:|
| seollal-2020 | Seollal | 2020-01-25 | 2020-01-24 | 2020-01-27 | 0 |
| chuseok-2020 | Chuseok | 2020-10-01 | 2020-09-30 | 2020-10-02 | 1 |
| seollal-2021 | Seollal | 2021-02-12 | 2021-02-11 | 2021-02-13 | 1 |
| chuseok-2021 | Chuseok | 2021-09-21 | 2021-09-20 | 2021-09-22 | 1 |
| seollal-2022 | Seollal | 2022-02-01 | 2022-01-31 | 2022-02-02 | 1 |
| chuseok-2022 | Chuseok | 2022-09-10 | 2022-09-09 | 2022-09-12 | 0 |
| seollal-2023 | Seollal | 2023-01-22 | 2023-01-21 | 2023-01-24 | 0 |
| chuseok-2023 | Chuseok | 2023-09-29 | 2023-09-28 | 2023-09-30 | 0 |
| seollal-2024 | Seollal | 2024-02-10 | 2024-02-09 | 2024-02-12 | 0 |
| chuseok-2024 | Chuseok | 2024-09-17 | 2024-09-16 | 2024-09-18 | 0 |

이벤트 창은 official start 하루 전부터 official end 하루 뒤까지의 모든 24시간이다. restriction flag는 해당 연휴에 정부의 이동·모임 관련 특별방역 또는 사적모임 제한이 실제 적용됐는지를 나타낸다. 이에 따라 2020 추석, 2021 설·추석뿐 아니라 전국 6인 사적모임 제한이 적용된 2022 설도 1이다. 근거는 대한민국 정책브리핑의 [2020 추석 특별방역](https://www.korea.kr/news/policyNewsView.do?newsId=148878262), [2021 설 거리두기](https://www.korea.kr/news/policyNewsView.do?newsId=148883362), [2021 추석 특별방역](https://www.korea.kr/news/policyNewsView.do?newsId=148892682), [2022 설 포함 거리두기](https://www.korea.kr/news/policyNewsView.do?newsId=148898047) 기록으로 versioning한다.

이벤트 레지스트리는 다음을 검증한다.

- occurrence id와 중앙일의 유일성
- start <= central <= end
- 이벤트 창 간 비중첩
- 각 창의 모든 timestamp 존재
- 휴일 유형별 연도당 정확히 한 occurrence
- pandemic flag가 실행 중 파생되지 않고 versioned input으로 고정됨

팬데믹 효과 `Delta_h`는 본 모델에 유지하되, 처리 이벤트가 네 개뿐이라는 약한 식별성을 posterior와 민감도 표에 명시한다. pandemic covariate 제거 모델도 민감도 분석으로 제공한다.

B1의 signed distance와 sequence-position 특징은 baseline 학습 첫해인 2019에도 필요하다. 따라서 `configs/holiday_calendar.csv`는 위 10개 occurrence에 2019 설날(central 2019-02-05, official 2019-02-04--2019-02-06)과 2019 추석(central 2019-09-13, official 2019-09-12--2019-09-14)을 추가한 12개 달력 occurrence를 가진다. 이 파일은 B1 특징 생성에만 사용하며, 2019 occurrence는 OOF residual, HQRC pooling, LOEO 이벤트 수에 포함하지 않는다.

## 7. 베이스라인 및 OOF 잔차

모든 베이스라인은 공통 protocol을 구현한다.

```python
fit(train_batch, validation_batch, seed) -> FittedBaseline
predict(fitted, forecast_batch) -> PredictionFrame
```

`PredictionFrame`은 sample origin, target timestamp, horizon 1--24, observed MW, predicted MW, model, feature set, seed, split id를 가진 long-form 자료다. 중복된 `(model, feature_set, seed, split_id, target_timestamp)`는 허용하지 않는다.

Tree/SVR은 horizon별 estimator 24개를 사용한다. 각 estimator는 동일한 `flatten(168시간 load/weather history) + flatten(24시간 known-future calendar path)`를 받는다. classical X는 estimator-fit 표본에만 적합한 StandardScaler로 변환하고, 모든 classical target은 같은 estimator-fit target StandardScaler로 표준화한 뒤 예측을 MW로 역변환한다. 이 좌표계에서 SVR의 `epsilon=0.05`는 0.05 MW가 아니라 학습 target 표준편차의 0.05다.

Seq2Seq-LSTM과 Transformer는 24시간을 공동 출력한다. history load와 target은 동일한 estimator-fit target scaler를 사용하고, history weather 및 future calendar는 estimator-fit 표본에만 적합한 featurewise scaler를 사용한다. 61일 early-stopping validation tail이나 evaluation year는 어떤 scaler 통계에도 포함되지 않는다. 신경망은 다섯 seed의 예측 평균을 한 baseline residual stream으로 사용하고 seed별 성능과 분산을 별도로 보존한다.

전처리 version, scaler 종류, scaler-fit partition, 24시간 future-path 길이, history/future 열 이름과 순서는 `model_spaces.toml`의 고정 설정과 baseline manifest에 포함한다. 이 중 하나가 없거나 달라지면 cache를 재사용하지 않고 실패한다.

하이퍼파라미터는 2019--2022/2023에서 한 번 선택하고 고정한다. 기존 baseline 연구에서 확정된 versioned 설정을 사용하면 탐색을 생략한다. OOF fold마다 다시 탐색하지 않으며, 고정된 설정으로 해당 fold train만 다시 학습한다. final baseline은 동일 설정으로 2019--2023 전체를 다시 학습해 2024를 예측한다.

fold `k`의 비휴일 OOF 척도는 다음과 같다.

\[
\sigma_{N,m}^{(k)} =
\sqrt{\frac{1}{|T_k^{ne}|}\sum_{t\in T_k^{ne}}r_{t,m}^2}.
\]

각 이벤트 잔차는 자신이 속한 fold의 척도로 표준화한다. 2024 최종 보정의 MW 변환에는 테스트를 보지 않는 가장 최근 완결 OOF fold인 2023 척도를 사용한다. 전체 2020--2023 척도를 사용한 결과는 민감도 분석으로 보존한다.

## 8. AR(1) 진단과 사전분포 보정

### 8.1 진단 잔차

원시 이벤트 잔차의 ACF는 휴일의 평균 곡선 자체 때문에 부풀 수 있다. 따라서 각 허용된 calibration occurrence에서 표준화 잔차에 다음 diagnostic-only 회귀를 적합한다.

\[
z_{i,t}=b_{i0}+b_{i1}\tau_{i,t}+b_{i2}\tau_{i,t}^2+
\sum_{j=1}^{J}\{a_j\sin(2\pi jh_t/24)+c_j\cos(2\pi jh_t/24)\}+u_{i,t},
\]

여기서 `J=3`이다. 잔차 `u`는 이벤트 경계에서 연결하지 않는다. 각 이벤트마다 다음을 저장한다.

- ACF lag 1--48
- PACF lag 1--48
- Bartlett 95% 참고 구간
- conditional least-squares AR(1) 추정치
- AR(1) 적용 후 innovation ACF/PACF와 Ljung--Box lag 24 통계

conditional estimate는

\[
\hat\phi_i=
\frac{\sum_{t=2}^{n_i}u_{i,t-1}u_{i,t}}
{\sum_{t=2}^{n_i}u_{i,t-1}^2}
\]

로 계산하고 수치 안정성을 위해 `[-0.98, 0.98]`로 제한한다.

### 8.2 Beta prior 보정

각 `(baseline, feature_set)`별로 calibration 이벤트의 `u_i=(phi_i+1)/2`를 계산한다. robust center는 `median(u_i)`, robust scale은 `1.4826 * MAD(u_i)`다. Beta concentration은 다음 moment 관계로 제안한다.

\[
\kappa^*=
\frac{\bar u(1-\bar u)}{s_u^2}-1,
\quad
\kappa=\operatorname{clip}(\kappa^*,8,40),
\]

여기서 `bar u = clip(median(u_i), 0.10, 0.975)`이고 `s_u^2`는 robust scale 제곱과 `0.025^2` 중 큰 값이다. 최종 파라미터는

\[
a=\max(1.05,\bar u\kappa),\qquad
b=\max(1.05,(1-\bar u)\kappa)
\]

로 만든 뒤 `a+b`가 40을 넘으면 비율을 유지해 40으로 축소한다. 이 규칙은 중심을 관측 잔차가 결정하게 하면서, 이벤트 수가 적어 분산이 우연히 작을 때 지나치게 강한 prior가 되는 것을 막는다.

진단 명령은 그림, occurrence별 추정치, 제안된 `(a,b)`, 사용 이벤트 id, 입력 잔차 hash를 `ar_calibration.json`에 기록한다. Bayesian fitting 명령은 이 파일과 명시적 승인 상태가 없으면 실행하지 않는다. 승인은 수치를 다시 입력하는 절차가 아니라, 진단 그림을 확인한 후 해당 hash를 동결하는 절차다.

2024 causal fit은 2020--2023 이벤트만 사용한다. LOEO fold는 held-out 이벤트를 제외하여 prior를 매번 다시 보정하고, 해당 fold의 calibration manifest를 저장한다. 어떤 경우에도 held-out 또는 미래 이벤트의 ACF가 prior에 들어가지 않는다.

### 8.3 AR(1) 적합성 게이트

다음 조건을 진단 보고서에 표시한다.

- 대부분의 calibration occurrence에서 PACF lag 1이 참고 구간 밖에 있음
- lag 2--6 PACF는 lag 1보다 실질적으로 작음
- AR(1) innovation ACF가 원 잔차 ACF보다 감소함

하나라도 명백히 실패하면 파이프라인은 AR(1)을 조용히 다른 차수로 바꾸지 않고 `diagnostic_warning`을 기록한다. 논문의 주모델은 그대로 AR(1)로 유지하고, AR(2)와 Student-t innovation은 사전 지정 민감도로 실행한다. 주 결과와 민감도 결론이 다르면 그 차이를 논문에 보고한다.

## 9. HQRC 통계 모델

각 baseline과 feature set에 별도 HQRC를 적합한다. occurrence `i`의 표준화 잔차는

\[
z_{i,t}=\beta_{i0}+\beta_{i1}\tau_{i,t}+\beta_{i2}\tau_{i,t}^2+
\gamma_{h(i),hr(t)}+e_{i,t}.
\]

시간대 효과는 휴일 유형별 원형 first-order random walk와 sum-to-zero 제약을 가진다.

\[
\gamma_{h,k}-\gamma_{h,k-1}\sim N(0,\sigma_{\gamma,h}^2),
\qquad \sum_{k=0}^{23}\gamma_{h,k}=0.
\]

AR(1) 오차는 이벤트 시작에서 reset한다.

\[
e_{i,t}=\phi e_{i,t-1}+\eta_{i,t},\qquad
\eta_{i,t}\sim N(0,\sigma_r^2),
\]

첫 오차는 정상분포 `N(0, sigma_r^2/(1-phi^2))`에서 시작한다. `phi=2u-1`, `u~Beta(a,b)`이며 `(a,b)`는 Section 8의 calibration artifact에서 읽는다.

발생별 계수는 holiday type과 pandemic covariate로 부분 풀링한다.

\[
\beta_i\sim N(\mu_{h(i)}+\Delta_{h(i)}x_i,\Sigma_{h(i)}),
\quad
\Sigma_h=\operatorname{diag}(s_h)R_h\operatorname{diag}(s_h).
\]

논문과 동일하게 `mu_h, Delta_h ~ N(0, 2^2 I)`, `s_h ~ HalfNormal(1)`, `R_h ~ LKJ(2)`, `sigma_gamma ~ HalfNormal(0.5)`, `sigma_r ~ HalfNormal(1)`를 사용한다. `beta_i`는 non-centered parameterization으로 구현한다.

발생 수가 적은 상태에서 full 3x3 covariance와 `Delta_h`가 약하게 식별될 수 있으므로 다음 민감도를 반드시 제공한다.

- diagonal between-event covariance
- pandemic covariate 제거
- LKJ eta 1과 4
- scale HalfNormal(2)
- AR(2) 또는 Student-t innovation 진단 모델

민감도는 주모델을 결과에 맞춰 교체하기 위한 탐색이 아니라 posterior 안정성을 공개하기 위한 분석이다.

## 10. 보정 변형과 pooling ablation

공통 데이터와 AR 처리 아래 다음을 구현한다.

- H0: 무보정 baseline
- H1: holiday-type constant mean residual shift
- H2: quadratic event-relative correction, hour profile 없음
- H3: quadratic + circular hour profile, full HQRC
- H4: random-walk-smoothed unrestricted day-position + hour profile
- H5: 과거 동일 holiday의 similar-day profile scaling

H3 pooling ablation은 다음과 같이 정의한다.

- complete: holiday type별 하나의 공통 shape
- partial: Section 9의 계층모델
- none: 각 과거 occurrence를 독립 적합하고, unseen occurrence의 point correction은 training occurrence posterior mean의 동일가중 mixture, predictive distribution은 mixture component를 먼저 뽑아 생성

이 정의로 no-pooling도 unseen event prediction이 가능하며, training occurrence 자신의 계수를 재사용하는 누수를 방지한다.

## 11. 예측분포

새 이벤트 posterior draw마다 `beta_new`를 holiday-type population에서 뽑고 `q_t`를 계산한다. corrected point forecast는 `y_hat + sigma_N * E[q_t]`다.

AR residual `e_t`는 이미 baseline holiday residual의 미설명 부분을 나타낸다. 따라서 corrected predictive draw는

\[
\tilde y_t=\hat y_t+\sigma_N(q_t+e_t)
\]

이다. 여기에 비휴일 bootstrap residual `epsilon_t`를 다시 더하지 않는다. 두 항을 함께 더하면 baseline residual uncertainty를 이중 계상한다.

H0 predictive distribution은 horizon을 보존한 비휴일 OOF residual block bootstrap으로 만든다. H0와 H3는 동일 draw 수와 평가 timestamp를 사용하되, 각 방법이 정의한 residual 생성 구조를 따른다. interval은 joint 24-hour/event trajectory draw에서 계산한다.

## 12. 평가 설계

### 12.1 일차 결과: causal 2024 holdout

- baseline hyperparameters는 2023까지만 보고 고정
- HQRC posterior와 AR prior는 2020--2023 OOF 이벤트만 사용
- 2024 설날과 추석에 한 번 적용
- 2024 임시공휴일은 correction 대상에서 제외
- 전체 테스트, pooled event window, 휴일 유형별 지표 보고

### 12.2 이차 결과: 10-event LOEO

2020--2024의 각 occurrence를 한 번씩 held out한다. baseline residual은 모두 해당 연도의 OOF 예측에서 오지만, correction training에는 held-out occurrence가 들어가지 않는다. 미래 occurrence가 과거 occurrence를 예측하는 fold도 있으므로 이 분석은 causal deployment가 아니라 event exchangeability와 pooling 구조를 검증하는 구조적 평가로 라벨링한다.

### 12.3 지표 및 추론

Point metric은 RMSE, MAE, MAPE, SMAPE, R2다. 확률 지표는 CRPS, 0.05--0.95 pinball loss, 50%/90% coverage다. 통계 단위는 hour가 아니라 occurrence다.

- 10개 paired event RMSE의 Wilcoxon signed-rank
- whole-event bootstrap median improvement CI
- secondary hourly DM test with HAC and event-block bootstrap
- family 내 p-value Holm adjustment
- correction이 악화시킨 occurrence 수와 최대 악화 폭

신뢰구간과 p-value는 모델별 결과와 전체 모델을 섞은 결과를 구분한다. 모델·이벤트 셀을 독립 event로 간주하지 않는다.

## 13. 산출물과 provenance

각 run은 `runs/<run_id>/`에만 기록한다.

```text
runs/<run_id>/
├── manifest.json
├── resolved_config.toml
├── data_audit.parquet
├── predictions/
├── residuals/
├── ar_diagnostics/
├── posterior/
├── metrics/
├── figures/
└── logs/
```

manifest는 git commit, dirty 여부, config hash, data file hash, event registry hash, library versions, CPU/GPU 정보, seed, split, sampler, 시작/종료 시간을 기록한다. 완료 marker는 모든 필수 artifact가 원자적으로 쓰이고 검증된 뒤에만 생성한다. 부분 실행은 재개할 수 있지만 다른 config/hash의 결과를 재사용하지 않는다.

## 14. CLI 실행 순서

사용자에게 노출되는 실행 순서는 다음과 같다.

1. `hqrc audit-data`: 스키마, 시간축, 이벤트 레지스트리 검사
2. `hqrc tune-baselines`: 필요할 때만 2019--2022/2023에서 설정을 한 번 선택
3. `hqrc generate-oof --through 2023`: expanding OOF residual 생성 및 캐시
4. `hqrc diagnose-ar --through 2023`: ACF/PACF와 prior 제안 생성
5. `hqrc approve-ar-calibration <artifact>`: 진단 hash 동결
6. `hqrc fit-final-baselines`: 2019--2023 재학습 및 2024 baseline 예측
7. `hqrc fit-corrections --evaluation causal-2024`: 2024 주 분석
8. `hqrc fit-corrections --evaluation loeo`: 10-event 구조 분석
9. `hqrc run-ablations`: H0--H5와 pooling/sensitivity
10. `hqrc benchmark-samplers`: PyMC와 nutpie 비교
11. `hqrc report`: 표·그림·논문 치환용 CSV 생성

각 단계는 선행 artifact의 schema와 hash를 검사한다. 누락되거나 미래 데이터가 포함된 artifact를 발견하면 자동 재사용하지 않고 실패한다.

## 15. 테스트 전략

구현은 test-driven development로 진행한다. 최소 테스트는 다음을 포함한다.

- split cutoff 이후 target이 train에 들어가지 않음
- expanding OOF와 final refit의 연도 계약이 정확함
- B0에 holiday-derived feature가 없음
- B1 signed distance와 sequence position의 경계값
- event window의 날짜와 24시간 완전성
- fold별 scale이 non-event OOF residual만 사용함
- AR lag pair가 이벤트 경계를 넘지 않음
- held-out event 제거 시 AR prior가 함께 재계산됨
- prior calibration이 유한한 `a,b > 1`을 생성하고 concentration cap을 지킴
- cyclic hour profile이 sum-to-zero임
- stationary AR initial log-likelihood가 수식과 일치함
- posterior predictive에 residual이 이중 가산되지 않음
- no-pooling unseen-event mixture가 held-out coefficient를 사용하지 않음
- synthetic data에서 partial pooling posterior가 유한하고 재현 가능함
- run manifest가 잘못된 data/config hash 재사용을 거부함

단위 테스트 외에 작은 synthetic 자료로 모든 CLI 단계가 연결되는 smoke test를 둔다. 실제 다섯 baseline 및 full NUTS 실행은 별도의 느린 테스트 marker로 분리한다.

## 16. 오류 처리와 중단 조건

다음 경우 명시적 오류로 중단한다.

- 데이터 시간축 또는 이벤트 registry 불일치
- train/validation/test target overlap
- fold scale이 0, 비유한 값, 또는 미래 residual을 포함
- AR calibration artifact가 현재 residual hash와 다름
- posterior에서 divergence, R-hat, ESS 기준을 통과하지 못했는데 완료로 표시하려는 경우
- 결과 행 수가 예측 대상 timestamp 수와 다름
- 2024 결과가 hyperparameter 또는 모델 구조 선택 입력으로 들어간 경우

기본 posterior 합격 기준은 4 chains, chain당 1,000 warmup과 1,000 retained draw, R-hat <= 1.01, bulk/tail ESS >= 400, divergence 0이다. 빠른 smoke profile은 더 작은 draw를 사용하지만 논문 결과로 표시할 수 없다.

## 17. 완료 기준

코드 구현은 다음이 모두 성립할 때 완료로 간주한다.

1. 새 `hqrc_v3/` 밖의 기존 연구 코드와 artifact를 변경하지 않는다.
2. unit/integration test가 통과하고 slow test는 명시적 명령으로 실행 가능하다.
3. expanding OOF와 final 2024 forecast가 동일 prediction contract를 사용한다.
4. AR prior 수치가 진단 artifact에서 유래하고 held-out/future event를 사용하지 않는다.
5. H3 모델이 quadratic, circular hour effect, partial pooling, pandemic covariate, stationary-reset AR(1)을 모두 포함한다.
6. corrected predictive draw가 baseline uncertainty를 이중 계상하지 않는다.
7. causal-2024와 10-event LOEO가 명확히 구분된 결과를 낸다.
8. sampler benchmark가 wall time, memory, ESS/s와 posterior agreement를 기록한다.
9. quick synthetic end-to-end run과 최소 한 baseline의 실제-data smoke run이 재현된다.
10. 전체 실행 명령, 예상 산출물, 논문 표 매핑이 README에 문서화된다.

## 18. 예상 구현 순서

1. 프로젝트 골격, config, provenance, 데이터 계약
2. event/feature/split 엔진과 누수 테스트
3. baseline protocol 및 정적/OOF 생성
4. expanding OOF 생성과 immutable prediction cache
5. AR 진단, prior calibration, 승인 artifact
6. HQRC H3 core와 predictive distribution
7. H0--H5, pooling, sensitivity
8. causal/LOEO 평가와 event-level inference
9. Polars/nutpie 성능 벤치마크
10. 실제-data smoke, 문서, 최종 코드 리뷰

이 순서는 잔차가 존재하기 전에 AR prior를 정하거나, 진단되지 않은 posterior 모델을 먼저 구축하는 일을 방지한다.
