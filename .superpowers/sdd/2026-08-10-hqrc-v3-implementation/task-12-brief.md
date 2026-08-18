# Task 12 — causal feature schema and train-only preprocessing

## Authority

- Implementation plan: `docs/superpowers/plans/2026-08-10-hqrc-v3-implementation.md`, Task 12.
- Method design: `docs/superpowers/specs/2026-08-10-hqrc-v3-design.md`, Sections 5--7.
- Start commit: `32b38fc` plus the Task 12 plan/spec commit created by the controller.

The user's chosen protocol remains unchanged: four expanding baseline OOF folds create honest
2020--2023 HQRC training residuals; the final baseline uses the pre-2024 outer training range and
evaluates January--October 2024. Hyperparameters are frozen, not tuned per fold. AR(1) remains
designed but its Beta prior numbers require later residual diagnostics and explicit user approval.

## Blocking defects to reproduce

1. `features.py` copies realized target-window weather to `oracle_*` and weather-derived degree
   hours. No forecast-vintage source exists, so this is future leakage in the main run.
2. B1 derives `is_holiday` only from twelve Seollal/Chuseok calendar rows and leaves the
   bridge/substitute flag constant zero. The real source contains 107 public-holiday dates and 14
   source-labelled substitute/temporary dates through 2024-10-31.
3. Classical adapters use raw mixed-unit X/raw MW y and append only `future[:, horizon, :]`.
   Frozen SVR `epsilon=0.05`, RBF geometry, boosting regularization, and the canonical full known
   24-hour path therefore have the wrong meaning.
4. Sequence scaling is train-only but uses a different load scale for history and target; the
   paper contract requires one shared target/load coordinate and explicit featurewise weather and
   calendar scales.
5. Baseline artifacts bind column names but not the complete preprocessing protocol.

## Required result

Implement Task 12 exactly, using TDD and preserving the protected untracked `hqrc_v3/uv.lock`.
Do not run the complete five-model paper experiment. A reduced real LightGBM smoke is required.
Do not change frozen model hyperparameters, OOF/final years, event windows, AR code, or correction
models. Do not add a custom Rust crate; Polars already supplies the Rust data engine and model
training is the expected bottleneck.

Key invariants:

- history: 168 hours of load/temperature/humidity plus the feature set's known-calendar columns;
- B0 future: exactly seven calendar/Fourier columns;
- B1 future: B0 plus six source/calendar holiday families represented by seven columns, with
  holiday type encoded as `is_seollal`/`is_chuseok` one-hot rather than scalar 0/1/2;
- no realized future weather or degree-hour feature in the main matrix;
- classical design: one full-path X per sample, shared by all 24 estimators;
- X/target scalers fit estimator-fit rows only; output is always MW;
- neural history load and y share the target scale; unique inferred history hours fit weather,
  while future unique hours fit one calendar scaler shared by past/future calendar channels;
- preprocessing contract is frozen in TOML SHA and baseline manifest;
- real source: 51,144 hours, 107 public-holiday dates, 14 substitute/temporary dates;
- October 1, 2024 is B1 holiday/temporary but not an HQRC event.
- exceptional temporary dates match a versioned three-row availability registry and were known
  before their holiday date; its hash is part of artifact identity.
- the feature calendar has 14 rows: 2018 Chuseok and 2025 Seollal are distance-only support;
  nearest-distance ties choose the earlier central date independent of input row order.

## Delivery and review

Record genuine RED and GREEN commands in `task-12-report.md`, update the SDD ledger, and commit in
small coherent commits if needed. Before handoff, run Task 12 focused tests, the complete non-slow
suite, Ruff, diff check, lock check, and the real causal matrix/LightGBM smoke. Report exact counts,
commits, changed contracts, and any remaining methodological uncertainty. Independent review is
mandatory before any full baseline computation.
