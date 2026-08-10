# Task 9 report

Implemented posterior predictive construction, H0--H5 correction variants, equal-occurrence
H5 profiles, explicit event-unit LOEO orchestration, and guarded event-level metrics and
inference.

## Contract decisions

- Corrected H1--H5 draws are exactly `baseline + sigma_N * (q + e)`; non-event bootstrap
  residuals are used only by H0 and cannot be added a second time to HQRC draws.
- Posterior AR(1) paths are sampled jointly for each event from their stationary initial
  distribution.  Every invocation resets at an event boundary.
- No-pooling predictions require explicit training occurrence indices and sample only those
  coefficients; a held-out event has no route into its own new-event correction.
- H0 resamples complete horizon-preserving non-event blocks.  H3 requires its circular hour
  profile; H4 requires its smoothed day profile plus the same circular hour profile.
- H5 uses only same-holiday training rows: normalize each occurrence/day profile by its daily
  mean, average occurrence profiles equally, then calculate a per-relative-day LS scale from
  training rows alone.
- LOEO is permanently labelled `causal=False`.  It either uses a backend's explicit
  pre-approved-calibration/load-fit-predict stages or a single injected atomic adapter; this
  module never creates or approves a fold calibration.
- Point metrics guard zero MAPE denominators and degenerate R2; probabilistic metrics include
  empirical CRPS, .05/.95 pinball loss, and 50%/90% coverage.  Wilcoxon and bootstrap use
  occurrence units, Holm adjusts only the supplied family, and HAC-DM records its bandwidth
  and event-block seed.

## TDD and verification

- RED: the required focused pytest command failed collection with four expected missing-module
  errors for `hqrc_v3.bayes.predictive`, `hqrc_v3.corrections`, and `hqrc_v3.evaluation`.
- GREEN: the focused Task 9 suite passed with `18 passed`.
- Regression: full HQRC v3 suite passed with `210 passed` (the existing tiny sampler warnings
  are expected ArviZ small-draw diagnostics).
- Ruff and `git diff --check` passed.

## Scope

- `hqrc_v3/src/hqrc_v3/bayes/__init__.py`
- `hqrc_v3/src/hqrc_v3/bayes/predictive.py`
- `hqrc_v3/src/hqrc_v3/corrections/`
- `hqrc_v3/src/hqrc_v3/evaluation/`
- Task 9 unit/integration tests
- this report

The pre-existing untracked `hqrc_v3/uv.lock` was not edited or staged.

## Fix round 1: predictive and evaluation contracts

- Posterior predictive construction now selects exactly one common posterior row per
  draw.  All required posterior variables are flattened and must have identical sample
  counts; coefficients, `gamma`, AR parameters, and sensitivity parameters consume that
  same index vector.  Full partial pooling uses the model's stacked
  `between_cholesky` and `L_h @ epsilon`; diagonal pooling uses `between_scale`.
- Added faithful normal AR(1), Student-t AR(1), and stationary stable normal AR(2)
  simulators.  Variant contexts declare their innovation family and fail when any
  required posterior parameter is absent.
- H0 accepts only a tokenized pool built from complete non-event residual horizons;
  each draw joins 24-hour blocks and truncates to the exact event horizon.  Contexts
  now require a unique chronological hourly timestamp grid whose tau/hour positions
  agree with those timestamps.
- H5 accepts only validated same-holiday training DataFrames with an explicit held-out
  id.  It rejects held-out rows, incomplete daily profiles, noninteger day positions,
  and zero LS profile denominators.
- LOEO accepts exactly the registered ten 2020--2024 Seollal/Chuseok frames, has no
  atomic fallback, reloads every supplied calibration through
  `require_approved_calibration`, and requires exact ordered input/output timestamp
  equality.
- CRPS now uses the sorted empirical identity, avoiding quadratic draw tensors.
  Aggregate and timestamp frames emit all 19 .05--.95 pinball levels plus their mean.
  HAC-DM now calculates its p-value from deterministic whole-event block bootstrap
  draws and records the draw count and event resampling unit.

### Verification

- Characterization RED: the prior focused suite failed with the old raw H0 block,
  short timestamp, and atomic-LOEO fixtures after the contracts were tightened.
- GREEN: expanded focused suite passed with `23 passed`.
- Full HQRC v3 regression: `215 passed`; existing tiny-draw ArviZ warnings remain
  expected. Ruff and diff checks passed.
