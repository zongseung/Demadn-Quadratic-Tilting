# Ver2: 168→24 Multi-Baseline HQT

이 디렉터리는 HP filter와 Fourier 분해를 제거한 두 단계 전력수요 예측 실험의
재현 번들이다. 수요와 모든 오차 지표의 단위는 MW이며, 별도의 강건성·민감도
검증은 이번 버전의 범위에 포함하지 않는다.

## 실험 범위

1. 동일한 정보집합으로 여섯 개 베이스라인을 학습한다.
   XGBoost, LightGBM, Random Forest, RBF-SVR, Seq2Seq-LSTM,
   Seq2Seq-GRU를 사용한다.
2. 각 베이스라인의 명절 잔차에 동일한 Hierarchical Quadratic Tilting(HQT)을
   적합한다.
3. 2024년 설날·추석을 학습에서 보지 않은 새 이벤트로 두고 보정 전후 성능을
   비교한다.

베이스라인 입력은 과거 168시간의 전력수요, 기온(`ta`), 습도(`hm`), 계절
더미(`spring`, `summer`, `autoum`, `winter`)다. `autoum`은 원자료의 컬럼명을
그대로 유지한 것이다. 출력은 다음 24시간 수요이며 예측 원점은 매일 00시,
stride는 24시간이다. 일반 공휴일 더미는 기본 베이스라인 입력에 포함하지
않는다.

데이터 분할은 시간 순서를 고정한다.

- train: 2019-01-01 00:00 ~ 2022-12-31 23:00
- validation: 2023-01-01 00:00 ~ 2023-12-31 23:00
- test: 2024-01-01 00:00 ~ 2024-10-31 23:00

모든 scaler는 train에만 적합한다. validation/test 윈도는 직전 구간의 과거
168시간을 입력으로 사용할 수 있지만, 24시간 목표가 분할 경계를 넘는 윈도는
제외한다. 따라서 2023년은 베이스라인 검증셋이고 2024년만 최종 test다.

## 파일 구성

이 폴더에는 Git으로 고정할 수 있는 설정과 요약 결과를 둔다. 대용량 산출물은
`artifacts/`에 생성되며 `.gitignore`로 제외된다.

```text
ver2/
├── README.md
├── configs/
│   ├── baseline.json
│   └── hqt.json
└── results/
    ├── baseline_test.csv
    ├── hqt_official_major_holiday.csv
    └── posterior_diagnostics.csv
```

실행 코드는 다음 경로가 이 버전의 기준 구현이다.

- `demand_quadratic_tilting/forecasting/`: 데이터 윈도, ML/DL 학습 및 평가
- `demand_quadratic_tilting/hqt_experiment.py`: 여섯 베이스라인 공통 HQT 실험
- `demand_quadratic_tilting/model.py`: HQT 적합 및 NUTS 진단
- `demand_quadratic_tilting/tilt.py`: 새 이벤트 posterior tilt 적용
- `scripts/train_baselines.py`: 베이스라인 실행 진입점
- `scripts/apply_hqt_to_baselines.py`: HQT 실행 진입점
- `tests/`: 데이터 누출, shape, 저장 재현성, 새 이벤트 이차 경로 테스트

## 실행

환경을 설치하고 전체 베이스라인을 학습한다.

```bash
uv sync --extra dev
uv run python scripts/train_baselines.py
```

고정 결과의 LSTM은 validation RMSE 비교 뒤 학습 예산만 늘려 다시 적합했다.

```bash
uv run python scripts/train_baselines.py \
  --models seq2seq_lstm \
  --max-epochs 80 --patience 15 --device cpu --resume
```

기본 결과는 `artifacts/baseline_168_24/`에 저장된다. 기존 결과를 이어서 특정
모델만 다시 학습할 때는 다음처럼 실행한다.

```bash
uv run python scripts/train_baselines.py \
  --models random_forest --resume
```

여섯 베이스라인에 HQT를 적용한다.

```bash
uv run python scripts/apply_hqt_to_baselines.py \
  --predictions artifacts/baseline_168_24/predictions.csv \
  --output-dir artifacts/hqt_baselines \
  --chains 4 --draws 1000 --tune 1000 \
  --target-accept 0.99 --resume
```

SVR의 고정 결과는 mixing을 개선하기 위해 별도로 2,000 draws와 2,000 tune,
`target_accept=0.995`로 재적합한 값이다.

```bash
uv run python scripts/apply_hqt_to_baselines.py \
  --models svr --chains 4 --draws 2000 --tune 2000 \
  --target-accept 0.995 --resume
```

전체 파이프라인 연결만 확인할 때는 각 실행 명령에 `--quick`을 사용한다.

검증 명령은 다음과 같다.

```bash
uv run pytest -q
uv run ruff check demand_quadratic_tilting/forecasting \
  demand_quadratic_tilting/hqt_experiment.py \
  demand_quadratic_tilting/model.py demand_quadratic_tilting/tilt.py scripts tests
uv run ruff format --check demand_quadratic_tilting/forecasting \
  demand_quadratic_tilting/hqt_experiment.py \
  demand_quadratic_tilting/model.py demand_quadratic_tilting/tilt.py scripts tests
```

## 결과 요약

2024년 전체 7,320시간의 베이스라인 성능은 다음과 같다.

| 모델 | MAE (MW) | RMSE (MW) | WAPE (%) |
|---|---:|---:|---:|
| LightGBM | 2,254.3 | 3,575.3 | 3.488 |
| XGBoost | 2,256.2 | 3,571.7 | 3.491 |
| SVR | 2,329.0 | 3,585.1 | 3.604 |
| Seq2Seq-LSTM | 2,418.3 | 3,751.5 | 3.742 |
| Random Forest | 2,476.0 | 3,801.1 | 3.831 |
| Seq2Seq-GRU | 2,689.5 | 3,958.6 | 4.162 |

2024년 공식 설날·추석 7일, 총 168시간에서는 모든 모델의 MAE와 RMSE가 HQT
적용 후 감소했다.

| 모델 | Baseline MAE | Tilted MAE | MAE 개선 | Tilted bias |
|---|---:|---:|---:|---:|
| Random Forest | 7,820.7 | 6,029.2 | 22.9% | +4,239.7 |
| SVR | 6,319.5 | 4,941.0 | 21.8% | +2,591.4 |
| Seq2Seq-GRU | 6,065.7 | 4,839.4 | 20.2% | +2,932.9 |
| LightGBM | 7,508.3 | 6,398.9 | 14.8% | +5,777.5 |
| XGBoost | 7,095.5 | 6,300.1 | 11.2% | +5,945.0 |
| Seq2Seq-LSTM | 4,846.8 | 4,401.8 | 9.2% | +2,899.3 |

모든 posterior는 divergence 0, 최대 R-hat 1.00을 기록했다. 세부 수치는
`results/posterior_diagnostics.csv`에 고정했다.

## 해석 제한

- HQT는 train과 validation의 2019–2023년 명절 잔차로 적합하고 2024년에는
  다시 적합하지 않는다.
- 2019–2022년 베이스라인 잔차는 현재 in-sample 예측에서 계산했다. 논문 최종
  결과에서는 rolling-origin OOF 예측 잔차로 교체하는 것이 바람직하다.
- HQT가 적용되는 test 구간은 앞뒤 패딩을 포함해 264/7,320시간이므로 전체
  MAE 개선률은 0.15%~1.42%로 작다.
- 공식 명절 집계에서는 여섯 모델 모두 개선됐지만 이벤트별 효과는 이질적이다.
  예를 들어 GRU는 추석 윈도 MAE가 23.9% 개선된 반면 설날 윈도는 22.0%
  악화됐다.
- 95% 예측구간의 실제 커버리지는 모델별 57.1%~88.7%이므로 불확실성 보정이
  완료됐다고 해석하지 않는다.
