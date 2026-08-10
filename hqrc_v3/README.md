# HQRC v3 operator boundary

The baseline commands now execute the five frozen manuscript models directly from
`configs/model_spaces.toml`: XGBoost, LightGBM, RBF-SVR, two-layer Seq2Seq-LSTM,
and a one-encoder/one-decoder-layer Transformer. There is no tuning command on this
path and no model substitution. XGBoost, LightGBM, and SVR fit 24 independent
horizon estimators; both neural models jointly predict all 24 hours from observed
history and known future covariates only.

Run these commands from the repository root. For a paper baseline run, calculate
the model-config digest and run both stages with all models and both feature sets:

```bash
MODEL_SHA256="$(openssl dgst -sha256 hqrc_v3/configs/model_spaces.toml | awk '{print $NF}')"

uv run --project hqrc_v3 hqrc generate-oof \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --frozen-model-hash "$MODEL_SHA256" \
  --event-registry hqrc_v3/configs/events.csv \
  --holiday-calendar hqrc_v3/configs/holiday_calendar.csv \
  --run-dir runs/RUN_ID \
  --cache-dir runs/RUN_ID/prediction-stream-cache \
  --model all --feature-set all --seed 7 --profile paper

uv run --project hqrc_v3 hqrc fit-final-baselines \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --frozen-model-hash "$MODEL_SHA256" \
  --event-registry hqrc_v3/configs/events.csv \
  --holiday-calendar hqrc_v3/configs/holiday_calendar.csv \
  --run-dir runs/RUN_ID \
  --cache-dir runs/RUN_ID/prediction-stream-cache \
  --model all --feature-set all --seed 7 --profile paper
```

The OOF stage is immutably fixed to `2019→2020`, `2019–2020→2021`,
`2019–2021→2022`, and `2019–2022→2023`. The final stage fits 2019–2023 and
produces the sole 2024 baseline. Boosting and neural fits reserve the final 61
complete pre-evaluation days for chronological early stopping; SVR uses the full
outer training range. Hyperparameters never vary by fold.

Successful execution publishes:

```text
runs/RUN_ID/predictions/oof_members.parquet
runs/RUN_ID/predictions/oof.parquet
runs/RUN_ID/predictions/final_2024_members.parquet
runs/RUN_ID/predictions/final_2024.parquet
runs/RUN_ID/predictions/baseline_manifest.json
```

Member files retain seeds 11, 23, 37, 41, and 53 for each neural model. Point
files identify the neural ensemble with seed 0 and contain its exact pointwise
arithmetic mean; classical rows are their direct forecasts. The manifest binds
the raw data, experiment, model registry, event registry, holiday calendar,
feature schemas, selected streams, seeds, profile, and output digests. Per-fold
stream caches are immutable and resumable. A partial pair, changed hash, changed
schema, altered seed/model identity, or tampered publication fails closed.

The opt-in real smoke uses the same LightGBM-B1 stage with a reduced round cap:

```text
uv run --project hqrc_v3 hqrc generate-oof \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --frozen-model-hash "$MODEL_SHA256" \
  --run-dir runs/SMOKE_ID \
  --cache-dir runs/SMOKE_ID/prediction-stream-cache \
  --model lightgbm --feature-set B1 --seed 7 \
  --profile smoke --oof-years 2020 --smoke-boosting-rounds 3
```

This is always recorded as `profile="smoke"` with its execution override and can
never populate paper numbers. A `paper` profile rejects model/feature/fold subsets
and round overrides.

Other concrete operator stages are:

```text
uv run --project hqrc_v3 hqrc audit-data --data power_demand_final.csv --fixed-bounds
uv run --project hqrc_v3 hqrc diagnose-ar --residuals runs/RUN_ID/inputs/standardized_residuals.parquet --output runs/RUN_ID/ar_diagnostics/proposed.json --residual-sha256 RESIDUAL_SHA256 --config-sha256 CONFIG_SHA256 --event-sha256 EVENT_SHA256 --through 2023
uv run --project hqrc_v3 hqrc approve-ar-calibration --proposal runs/RUN_ID/ar_diagnostics/proposed.json --output runs/RUN_ID/ar_diagnostics/approved.json --residual-sha256 RESIDUAL_SHA256 --config-sha256 CONFIG_SHA256 --event-sha256 EVENT_SHA256
uv run --project hqrc_v3 hqrc report --run-dir runs/RUN_ID --profile smoke
```

Before `report`, the run must contain the strict version-2 reporting manifest and
its run-local source audit, resolved config, event registry, standardized
residuals, validated OOF predictions, approved AR artifact, NetCDF InferenceData,
event metrics, and sampler benchmark. Reporting reloads every digest and diagnostic
gate before writing `COMPLETE`. Only a full `paper` run satisfying posterior
diagnostics may populate manuscript numbers.

Run fast contracts from the repository root with
`uv run --project hqrc_v3 pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m "not slow"`;
real-data and sampler checks are opt-in under `pytest -m slow`.
