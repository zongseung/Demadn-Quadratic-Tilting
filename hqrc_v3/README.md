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

## Native H1--H3 accelerator runbook

The accelerated paper scope is exactly the four non-XGBoost models
`lightgbm`, `svr`, `seq2seq_lstm`, and `transformer`, both `B0` and `B1`,
`H1`, `H2`, and `H3`, and all ten LOEO folds. XGBoost is excluded from
accelerated scheduling and publication. On two CUDA devices, physical GPU 0
runs `lightgbm` then `seq2seq_lstm`, while physical GPU 1 runs `svr` then
`transformer`. The two model workers run concurrently, but every model fit
runs its four NUTS chains sequentially inside its worker.

CPU selection uses bounded model-level parallelism while preserving those
sequential Pyro chains inside each worker. For example, 24 logical CPUs and
the four paper models produce four workers with six threads each. Platform
defaults remain CUDA on Windows and Linux, and `auto` on macOS.

Run the following commands from the repository root or linked worktree. The
launchers perform a locked accelerator dependency sync before each action.
The setup shown below uses the current checkout when it contains
`artifacts.zip`; otherwise it resolves the source checkout two levels above a
`.worktrees/<name>` linked worktree. When using another layout, set
`SourceRoot`/`SOURCE_ROOT` explicitly to the canonical absolute source path
before running the import command.

### Windows PowerShell

```powershell
# Import the immutable correction source. An identical rerun validates and reuses it.
$SourceRoot = if (Test-Path -LiteralPath 'artifacts.zip' -PathType Leaf) { (Resolve-Path '.').Path } else { (Resolve-Path '..\..').Path }
scripts\run_hqrc_windows.ps1 import `
  -Archive (Join-Path $SourceRoot 'artifacts.zip') `
  -Data (Join-Path $SourceRoot 'power_demand_final copy.csv') `
  -RepositoryRoot $SourceRoot `
  -RunDir artifacts/hqrc-v3-paper-imported-20260819

# Publish/reuse the complete paper AR proposals. This does not sample.
scripts\run_hqrc_windows.ps1 proposal `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-loeo-accelerated-20260819 `
  -Devices 0,1

# Reduced two-model/B0/H1 hardware-path smoke. Without approval it remains proposal-only.
scripts\run_hqrc_windows.ps1 smoke `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-loeo-accelerated-20260819 `
  -Devices 0,1 -Models lightgbm,svr -FeatureSets B0 -Variants H1

# Run the complete paper matrix only after reviewing every proposal path and digest.
scripts\run_hqrc_windows.ps1 paper `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-loeo-accelerated-20260819 `
  -Devices 0,1 -ApproveDerivedAR

# Resume uses the same immutable identities and explicit approval.
scripts\run_hqrc_windows.ps1 resume `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-loeo-accelerated-20260819 `
  -Devices 0,1 -ApproveDerivedAR
```

Use explicit CPU selection for paper and resume without changing the AR or
artifact contracts:

```powershell
scripts\run_hqrc_windows.ps1 paper -Accelerator cpu `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-loeo-accelerated-20260819 -ApproveDerivedAR

scripts\run_hqrc_windows.ps1 resume -Accelerator cpu `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-loeo-accelerated-20260819 -ApproveDerivedAR
```

### Linux Bash

Linux uses the same fixed CUDA queues and physical device order:

```bash
SOURCE_ROOT="$(if [ -f artifacts.zip ]; then pwd -P; else cd ../.. && pwd -P; fi)"
scripts/run_hqrc_linux.sh import \
  --archive "$SOURCE_ROOT/artifacts.zip" \
  --data "$SOURCE_ROOT/power_demand_final copy.csv" \
  --repository-root "$SOURCE_ROOT" \
  --run-dir artifacts/hqrc-v3-paper-imported-20260819

scripts/run_hqrc_linux.sh proposal \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 \
  --devices 0 1

scripts/run_hqrc_linux.sh smoke \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 \
  --devices 0 1 --models lightgbm svr --feature-sets B0 --variants H1

scripts/run_hqrc_linux.sh paper \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 \
  --devices 0 1 --approve-derived-ar

scripts/run_hqrc_linux.sh resume \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 \
  --devices 0 1 --approve-derived-ar
```

For CPU paper runs and resumptions, set `HQRC_ACCELERATOR=cpu` on the launcher
invocation:

```bash
HQRC_ACCELERATOR=cpu scripts/run_hqrc_linux.sh paper \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 --approve-derived-ar

HQRC_ACCELERATOR=cpu scripts/run_hqrc_linux.sh resume \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 --approve-derived-ar
```

### macOS Bash

The macOS launcher uses `--accelerator auto`. On Apple Silicon, `auto` selects
MPS only after the HQRC float64/LKJ/AR gradient capability probe succeeds. If
MPS is unavailable or fails that probe, the scheduler selects CPU and records
the fallback reason. MPS executes the four models sequentially; explicit CPU
selection uses bounded model-level parallelism.

```bash
SOURCE_ROOT="$(if [ -f artifacts.zip ]; then pwd -P; else cd ../.. && pwd -P; fi)"
scripts/run_hqrc_macos.sh import \
  --archive "$SOURCE_ROOT/artifacts.zip" \
  --data "$SOURCE_ROOT/power_demand_final copy.csv" \
  --repository-root "$SOURCE_ROOT" \
  --run-dir artifacts/hqrc-v3-paper-imported-20260819

scripts/run_hqrc_macos.sh proposal \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819

scripts/run_hqrc_macos.sh smoke \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 \
  --models lightgbm svr --feature-sets B0 --variants H1

scripts/run_hqrc_macos.sh paper \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 \
  --approve-derived-ar

scripts/run_hqrc_macos.sh resume \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 \
  --approve-derived-ar
```

For CPU paper runs and resumptions, set `HQRC_ACCELERATOR=cpu` on the launcher
invocation:

```bash
HQRC_ACCELERATOR=cpu scripts/run_hqrc_macos.sh paper \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 --approve-derived-ar

HQRC_ACCELERATOR=cpu scripts/run_hqrc_macos.sh resume \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-accelerated-20260819 --approve-derived-ar
```

The AR boundary is mandatory on every platform: `import`, `proposal`, and an
unapproved `smoke` may publish or reuse unapproved proposals, but posterior
sampling starts only after a person reviews their plots and exact digests and
reruns with `--approve-derived-ar`/`-ApproveDerivedAR`. Never derive or forward
that approval automatically.

Worker output is streamed and appended to model-specific files under
`<output-root>/logs/`. A failing model stops the later model in its queue; the
other in-flight queue may finish. Partial evidence is preserved. Rerunning the
same `paper` or `resume` command revalidates completed immutable artifacts and
continues at the first incomplete identity; it does not delete mismatched or
partial evidence.

## One-command manuscript runner

`run-paper` is the portable operator entry point.  It first completes the shared
baseline source in its fixed order (all five models, B0 then B1, OOF then final
2024 forecast, then standardized residuals). It then completes HQRC contexts in
the following order, with every context's LOEO universe, AR diagnostics, and ten
folds each for H1, H2, and H3 completed before the next baseline starts:

```text
XGBoost -> LightGBM -> SVR -> Seq2Seq-LSTM -> Transformer
```

The default HQRC scope is B1, the holiday-aware baseline.  Use
`--hqrc-feature-set all` only when the B0 comparison is deliberately required.
Within each baseline context the default correction order is `H1 H2 H3`.
`--hqrc-variants` may select a manuscript-ordered subset for a diagnostic rerun;
the uncorrected H0 metrics are derived directly from the same held-out rows in
each H1/H2 aggregate, so H0 never invokes NUTS. H1 and H2 reuse the same reviewed
fold-specific AR prior as H3 but have separate random seeds and immutable output
namespaces. H1 is a constant holiday-type correction and H2 is quadratic in
event-relative time without the hour-of-day profile.
The baseline source must be generated on the computer that runs the sampler:
its immutable manifests bind input files by digest and absolute local path, so
copying an old prepared artifact to a different path is not a valid substitute
for rebuilding it from the raw input there.

```bash
MODEL_SHA256="$(openssl dgst -sha256 hqrc_v3/configs/model_spaces.toml | awk '{print $NF}')"

uv run --project hqrc_v3 --locked hqrc run-paper \
  --data power_demand_final.csv \
  --config hqrc_v3/configs/experiment.toml \
  --frozen-model-config hqrc_v3/configs/model_spaces.toml \
  --frozen-model-hash "$MODEL_SHA256" \
  --event-registry hqrc_v3/configs/events.csv \
  --holiday-calendar hqrc_v3/configs/holiday_calendar.csv \
  --temporary-holiday-availability hqrc_v3/configs/temporary_holiday_availability.csv \
  --run-dir artifacts/hqrc-v3-paper-local \
  --output-root artifacts/hqrc-v3-loeo-paper-local \
  --baseline-seed 7 --root-seed 20260813 \
  --hqrc-model all --hqrc-feature-set B1 --hqrc-variants H1 H2 H3 \
  --draws 5000 --tune 5000 --chains 4 --cores 4 \
  --init jitter+adapt_diag --target-accept 0.99 \
  --approve-derived-ar
```

The `--approve-derived-ar` flag is intentionally explicit.  Without it, the same
command builds/reuses all ACF/PACF plots and data-derived Beta-prior proposals,
prints their exact proposal digests, and stops before NUTS.  After reviewing those
plots, rerun the identical command with that flag; completed baseline, LOEO, and
AR artifacts are strictly reused.  To run only the correction half after a
baseline source already exists, use `run-loeo-primary` with `--source-run-dir`,
`--output-root`, `--variants H1 H2 H3`, and the same HQRC sampler arguments.

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
`corrections/causal-2024/<model>/<feature-set>/seed-<baseline-seed>/<sampler-profile>/sampler-seed-<rng-seed>/init-adapt_diag-geometry-noncentered-cyclic-hour-rw1-v1/draws-<D>-tune-<T>-chains-<C>/`.
The fixed PyMC contract starts from deterministic `adapt_diag` values and samples the cyclic
hour RW1 in an exactly equivalent non-centered geometry; both choices are recorded in the
posterior metadata and manifest. The prior, target acceptance, and strict diagnostic gate are
unchanged.
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
