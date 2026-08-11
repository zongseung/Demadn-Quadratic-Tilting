# HQRC v3 operator boundary

The baseline commands now execute the five frozen manuscript models directly from
`configs/model_spaces.toml`: XGBoost, LightGBM, RBF-SVR, two-layer Seq2Seq-LSTM,
and a one-encoder/one-decoder-layer Transformer. There is no tuning command on this
path and no model substitution. XGBoost, LightGBM, and SVR fit 24 independent
horizon estimators; each receives the same complete 168-hour history plus the same
complete 24-hour known-future path. Both neural models jointly predict all 24 hours
from observed history and known future covariates only.

## Causal data and feature contract

The fixed source covers 2019-01-01 00:00 through 2024-10-31 23:00: 51,144 hourly
rows, 2,131 complete dates, 107 source-labelled public-holiday dates, and 14
source-labelled substitute/temporary dates. The source `holiday_name` and
`is_holiday_dummies` fields are preserved as audited canonical fields. Every date
must have 24 consistent copies of one name/flag pair. The three exceptional
temporary dates also require a forecast-time availability record in
`configs/temporary_holiday_availability.csv`; each announcement predates the
holiday, and that registry's digest is part of artifact identity.

No historical forecast-vintage weather source is available. Therefore the main
paper matrix deliberately excludes target-window realized temperature, humidity,
heating degree hours, and cooling degree hours. Temperature and humidity are
observed inputs only inside the 168-hour history. The exact known-future schemas are:

```text
B0 future (7): hour, day_of_week, is_weekend,
               annual_sin, annual_cos, weekly_sin, weekly_cos
B1-only (7):   is_public_holiday, official_sequence_position,
               seollal_distance, chuseok_distance,
               is_substitute_or_temporary_holiday, is_seollal, is_chuseok
history:       load_mw, temperature_c, relative_humidity + that feature set's future schema
```

Thus B0 history/future widths are 10/7 and B1 widths are 17/14. Holiday type is
mutually exclusive one-hot, never an ordinal scalar. The feature calendar contains
14 rows: 2019--2024 Seollal/Chuseok plus 2018 Chuseok and 2025 Seollal as nearest-
distance boundary support. The correction registry remains the ten 2020--2024
events. In particular, 2024-10-01 is a known B1 public/temporary holiday, has both
holiday-type one-hots zero, and is not an HQRC correction window.

The real matrix contains exactly 2,124 midnight-origin daily samples. OOF evaluation
over 2020--2023 contains 1,461 samples. The final outer training partition contains
1,819 samples; boosting and neural models split this into 1,758 estimator-fit plus
61 chronological validation samples, whereas SVR fits all 1,819. The final
January--October 2024 evaluation contains 305 samples.

Classical preprocessing fits one featurewise population `StandardScaler` to
`flatten(history[168]) + flatten(future[24])` and one target scaler to all unique
estimator-fit target hours. Every horizon learns the standardized target and is
inverse-transformed to MW. Sequence preprocessing uses that same target scaler for
history load and output, fits weather scaling once per unique inferred observed-
history hour, and fits one calendar scaler on unique future hours for use on both
history and future calendar channels. Validation and evaluation rows never enter a
scaler. The frozen TOML and publication manifest bind this policy, exact column
order, scaler kinds, and each scaler population's count/start/end.

Run these commands from the repository root. Bootstrap the Python 3.11+ workspace
from the committed repository-root `uv.lock`; every invocation below carries its
own explicit lock check:

```bash
uv sync --project hqrc_v3 --locked
```

The root `pyproject.toml` owns the uv workspace and installs `hqrc_v3/` as the
editable PEP 517 package that provides the `hqrc` console script. Do not create or
use a member-local lock file; `uv.lock` at the repository root is the reproducible
runtime and test lock.

For a paper baseline run, calculate the model-config digest and run both stages
with all models and both feature sets:

```bash
MODEL_SHA256="$(openssl dgst -sha256 hqrc_v3/configs/model_spaces.toml | awk '{print $NF}')"

uv run --project hqrc_v3 --locked hqrc generate-oof \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --frozen-model-hash "$MODEL_SHA256" \
  --event-registry hqrc_v3/configs/events.csv \
  --holiday-calendar hqrc_v3/configs/holiday_calendar.csv \
  --temporary-holiday-availability hqrc_v3/configs/temporary_holiday_availability.csv \
  --run-dir runs/RUN_ID \
  --cache-dir runs/RUN_ID/prediction-stream-cache \
  --model all --feature-set all --seed 7 --profile paper

uv run --project hqrc_v3 --locked hqrc fit-final-baselines \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --frozen-model-hash "$MODEL_SHA256" \
  --event-registry hqrc_v3/configs/events.csv \
  --holiday-calendar hqrc_v3/configs/holiday_calendar.csv \
  --temporary-holiday-availability hqrc_v3/configs/temporary_holiday_availability.csv \
  --run-dir runs/RUN_ID \
  --cache-dir runs/RUN_ID/prediction-stream-cache \
  --model all --feature-set all --seed 7 --profile paper
```

The OOF stage is immutably fixed to `2019→2020`, `2019–2020→2021`,
`2019–2021→2022`, and `2019–2022→2023`. These four forecasts are not the final
score: they create honest 2020--2023 residual targets for HQRC and AR diagnostics.
The final stage then refits the unchanged baseline specification on the pre-2024
outer training range and produces the sole causal 2024 baseline. This second fit is
needed because the deployed 2024 forecaster may use all observations available
through 2023, while its HQRC layer must still be trained on out-of-sample residuals.
Boosting and neural fits reserve the final 61 complete pre-evaluation days for
chronological early stopping; SVR uses the full outer training range.
Hyperparameters never vary by fold and 2024 is never used for selection.

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
temporary-holiday availability registry, feature schemas, causal preprocessing
identity, exact scaler populations, selected streams, seeds, profile, and output
digests. Per-fold stream caches are immutable and resumable. A partial pair,
changed hash, changed feature/preprocessing schema, changed scaler population,
altered seed/model identity, or tampered publication fails closed.

Convert the complete OOF point publication into HQRC training targets with:

```text
uv run --project hqrc_v3 --locked hqrc prepare-residuals \
  --run-dir runs/RUN_ID \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --event-registry hqrc_v3/configs/events.csv \
  --holiday-calendar hqrc_v3/configs/holiday_calendar.csv \
  --temporary-holiday-availability hqrc_v3/configs/temporary_holiday_availability.csv \
  --profile paper
```

This publishes one canonical `inputs/standardized_residuals.parquet` containing
all five models and both feature sets. Each context contains the 1,032 event
hours from the eight 2020--2023 occurrences; no 2024 row enters calibration.
For every model, feature set, point-stream seed, and OOF split, the scale is the
RMS of residuals outside every registered correction window. The adjacent strict
manifest records all four scales and identifies `oof-2023` as the scale for the
later causal 2024 MW conversion. Neural point streams use ensemble seed 0;
classical point streams use the requested classical seed.

The opt-in real smoke uses the same LightGBM-B1 stage with a reduced round cap:

```text
uv run --project hqrc_v3 --locked hqrc generate-oof \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --frozen-model-hash "$MODEL_SHA256" \
  --event-registry hqrc_v3/configs/events.csv \
  --holiday-calendar hqrc_v3/configs/holiday_calendar.csv \
  --temporary-holiday-availability hqrc_v3/configs/temporary_holiday_availability.csv \
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
uv run --project hqrc_v3 --locked hqrc audit-data --data power_demand_final.csv --fixed-bounds --temporary-holiday-availability hqrc_v3/configs/temporary_holiday_availability.csv
uv run --project hqrc_v3 --locked hqrc prepare-residuals --run-dir runs/RUN_ID --data power_demand_final.csv --config hqrc_v3/configs/experiment.toml --frozen-model-config hqrc_v3/configs/model_spaces.toml --event-registry hqrc_v3/configs/events.csv --holiday-calendar hqrc_v3/configs/holiday_calendar.csv --temporary-holiday-availability hqrc_v3/configs/temporary_holiday_availability.csv --profile paper
uv run --project hqrc_v3 --locked hqrc diagnose-ar --run-dir runs/RUN_ID --config hqrc_v3/configs/experiment.toml --event-registry hqrc_v3/configs/events.csv --output runs/RUN_ID/ar_diagnostics/lightgbm-B1-proposed.json --through 2023 --model lightgbm --feature-set B1
uv run --project hqrc_v3 --locked hqrc approve-ar-calibration --proposal runs/RUN_ID/ar_diagnostics/lightgbm-B1-proposed.json --output runs/RUN_ID/ar_diagnostics/lightgbm-B1-approved.json --residual-sha256 RESIDUAL_SHA256 --config-sha256 CONFIG_SHA256 --event-sha256 EVENT_SHA256
uv run --project hqrc_v3 --locked hqrc fit-corrections --run-dir runs/RUN_ID --config hqrc_v3/configs/experiment.toml --approved-ar runs/RUN_ID/ar_diagnostics/lightgbm-B1-approved.json --evaluation causal-2024 --seed 20260811 --profile paper
uv run --project hqrc_v3 --locked hqrc fit-corrections --run-dir runs/RUN_ID --config hqrc_v3/configs/experiment.toml --approved-ar runs/RUN_ID/ar_diagnostics/lightgbm-B1-approved.json --evaluation causal-2024 --seed 20260811 --profile paper --draws 2000 --tune 1000 --chains 4
uv run --project hqrc_v3 --locked hqrc report --run-dir runs/RUN_ID --profile smoke
```

Run `diagnose-ar` separately for each of the ten model/feature contexts. The
selector resolves exactly one point-stream seed and fails if it is ambiguous;
`--seed` can be supplied as an additional assertion. Diagnosis only writes an
unapproved proposal. It never creates or updates the approval artifact.

`fit-corrections` runs one approved `(model, feature set, point-stream seed)`
context per command. Its `--seed` controls only PyMC and posterior-predictive
randomness; the baseline point-stream seed is read exclusively from the approved
AR context. The stage rebuilds source-derived matrices and invokes the public
final-baseline stage only as a validator, requiring `fit_count == 0`, so it never
refits a baseline. It fits only H3 partial pooling with full covariance,
restriction effects, and event-reset Gaussian AR(1), using the eight 2020--2023
OOF occurrences and the `oof-2023` residual scale. Evaluation is limited to the
264 registered 2024 Seollal/Chuseok hours; 1 October is unchanged. LOEO requires
separate fold-specific approvals and is intentionally unavailable in this stage.

Smoke correction runs must state their reduced sampler limits explicitly, for
example `--profile smoke --draws 20 --tune 20 --chains 2`. Paper runs enforce
four chains, at least 1,000 warm-up draws, at least 1,000 retained draws, target
acceptance 0.99, and the strict posterior diagnostic gate. Omitting paper limits
uses 1,000 warm-up and retained draws with four chains; explicit paper overrides
are accepted only when both draw counts remain at least 1,000 and chains remains
exactly four. A diagnostic retry can therefore retain its RNG seed while raising
the retained draw count, as in the 2,000-draw command above.
The command profile is the sampler profile. The source profile is inferred from
the immutable residual and baseline manifests and recorded separately: a paper
sampler requires paper sources, while a smoke sampler may read either paper or
smoke sources without relabeling those source artifacts.
Each completed context is namespaced as
`corrections/causal-2024/<model>/<feature-set>/seed-<baseline-seed>/<sampler-profile>/sampler-seed-<rng-seed>/draws-<D>-tune-<T>-chains-<C>/`.
The first seed remains exclusively the baseline seed from the approved AR
context; the final seed is the CLI `--seed`. Consequently smoke/paper runs and
different sampler seeds or sampler sizes never reuse or invalidate one another.

The boundary calendar dates are documented by the official Korea Astronomy and
Space Science Institute almanac releases for
[2018](https://www.kasi.re.kr/publication/post/newsMaterial/5971) and
[2025](https://www.kasi.re.kr/kor/publication/post/newsMaterial/30071). The three
temporary-holiday availability records point to their official Korea Policy
Briefing announcements:
[2020-08-17](https://www.korea.kr/news/policyNewsView.do?newsId=148874895),
[2023-10-02](https://www.korea.kr/news/policyNewsView.do?newsId=148919605), and
[2024-10-01](https://www.korea.kr/news/policyNewsView.do?newsId=148933400).

Before `report`, the run must contain the strict version-2 reporting manifest and
its run-local source audit, resolved config, event registry, standardized
residuals, validated OOF predictions, approved AR artifact, NetCDF InferenceData,
event metrics, and sampler benchmark. Reporting reloads every digest and diagnostic
gate before writing `COMPLETE`. Only a full `paper` run satisfying posterior
diagnostics may populate manuscript numbers.

Run fast contracts from the repository root with
`uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m "not slow"`;
real-data and sampler checks are opt-in under `pytest -m slow`.
