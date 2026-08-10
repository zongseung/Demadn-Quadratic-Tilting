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
