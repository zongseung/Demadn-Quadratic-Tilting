# 168시간 입력 기반 24시간 전력수요 베이스라인

이 파이프라인은 HP filter와 Fourier 분해 없이 동일한 입력 정보로 다음 여섯
모델을 비교한다.

- XGBoost direct multi-horizon
- LightGBM direct multi-horizon
- Random Forest native multi-output
- RBF-SVR direct multi-horizon
- Seq2Seq-LSTM
- Seq2Seq-GRU

Apple Silicon에서는 GRU를 MPS로 학습하지만 LSTM은 CPU를 사용한다. 현재
환경의 PyTorch 2.8 MPS LSTM은 학습 중 packed weight와 저장되는
`state_dict`가 달라져 새 프로세스에서 예측을 재현하지 못했기 때문이다.
CPU LSTM과 MPS GRU는 각각 저장 전후 검증손실이 일치하는 것을 확인한다.

## 현재 베이스라인 결과

2024년 1월 1일~10월 31일 7,320시간 hold-out 결과는 다음과 같다. DL은
validation RMSE를 기준으로 설정을 선택했고 test는 선택 후 평가했다.

| 모델 | MAE (MW) | RMSE (MW) | WAPE (%) |
|---|---:|---:|---:|
| LightGBM | 2,254.3 | 3,575.3 | 3.488 |
| XGBoost | 2,256.2 | 3,571.7 | 3.491 |
| SVR | 2,329.0 | 3,585.1 | 3.604 |
| Seq2Seq-LSTM | 2,418.3 | 3,751.5 | 3.742 |
| Random Forest | 2,476.0 | 3,801.1 | 3.831 |
| Seq2Seq-GRU | 2,689.5 | 3,958.6 | 4.162 |

2024년 설날·추석 공식 7일(168시간)에서는 모든 모델이 과대예측했다. HQT를
적용하기 전 bias는 LSTM `+4,139.8 MW`, GRU `+5,839.0 MW`, SVR
`+5,954.9 MW`, XGBoost `+7,029.1 MW`, LightGBM `+7,434.1 MW`,
Random Forest `+7,630.3 MW`다.
따라서 전체 test 성능과 별개로 모델 공통의 명절 보정 문제가 남아 있다.

## HQT 보정 결과

2019–2023년 이벤트로 HQT를 적합한 뒤 2024년 공식 설날·추석 168시간에
적용했다. MAE는 Random Forest 22.9%, SVR 21.8%, GRU 20.2%, LightGBM
14.8%, XGBoost 11.2%, LSTM 9.2% 감소했다. 전체 7,320시간 MAE 개선률은
tilt가 264시간에만 적용되므로 0.15%~1.42%다.

모든 posterior는 divergence 0, 최대 R-hat 1.00을 충족했다. 결과 파일은
`artifacts/hqt_baselines/metrics.csv`, 모델별 posterior와 시간별 보정 예측은
`artifacts/hqt_baselines/<model>/`에 저장된다.

LSTM 후보 비교에서는 64-unit×2-layer 설정이 validation RMSE 2,802 MW,
기존 residual 모델의 128-unit×3-layer 설정이 2,863 MW였다. 따라서 더
단순하고 validation RMSE가 낮은 64×2 설정을 최종 사용했다.

## 데이터와 시간 분할

- 과거 입력: 전력수요, 기온(`ta`), 습도(`hm`), 봄·여름·가을·겨울 더미
- 입력 길이: 168시간
- 출력 길이: 24시간
- 예측 원점: 매일 00시, stride 24시간
- 학습: 2019–2022년
- 검증: 2023년
- 테스트: 2024년 1월–10월

기온과 습도는 과거 168시간에서만 사용한다. 계절 더미는 예측 시점에 이미
알려진 정보이므로 Seq2Seq 디코더와 ML 입력에 향후 24시간 값도 제공한다.
검증·테스트 윈도는 이전 구간의 과거 관측값을 사용할 수 있지만 목표값이
분할 경계를 넘는 윈도는 제거한다. 모든 scaler는 2019–2022년 자료로만
적합한다.

## 실행

전체 모델 학습:

```bash
uv run python scripts/train_baselines.py
```

현재 고정 결과의 LSTM(64 units × 2 layers)은 validation 비교 후 80 epoch,
patience 15로 다시 적합했다.

```bash
uv run python scripts/train_baselines.py \
  --models seq2seq_lstm \
  --max-epochs 80 --patience 15 --device cpu --resume
```

파이프라인 연결만 빠르게 점검:

```bash
uv run python scripts/train_baselines.py --quick \
  --output-dir /tmp/baseline-smoke
```

기존 결과를 유지한 채 특정 모델만 다시 학습하려면:

```bash
uv run python scripts/train_baselines.py \
  --models seq2seq_lstm --resume
```

일반 공휴일 여부를 베이스라인의 미래 사전정보에 추가하려면:

```bash
uv run python scripts/train_baselines.py \
  --known-covariates spring summer autoum winter is_holiday_dummies
```

## 산출물

기본 산출물 디렉터리는 `artifacts/baseline_168_24/`이다.

- `predictions.csv`: 시간별 실제값과 모든 베이스라인 예측. `holiday_name`을
  포함하므로 HQT의 `pred_col`만 선택해 바로 입력할 수 있다.
- `metrics.csv`: train/validation/test 전체 및 1–24시간 horizon별 지표
- `models/`: 모델 체크포인트
- `scalers.joblib`: 학습 구간에 적합한 target/weather scaler
- `config.json`, `training_summary.json`: 재현 설정과 학습 요약

HQT 연결 시 `predictions.csv`를 Polars로 읽고 split별로 나눈 뒤 모델 컬럼을
`pred_col`로 지정한다. 예를 들어 LightGBM 결과에는
`pred_col="lightgbm"`, `datetime_col="datetime"`을 사용한다.
