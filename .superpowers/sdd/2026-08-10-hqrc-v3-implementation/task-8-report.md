# Task 8 report

Implemented the hierarchical Bayesian HQRC correction model, event-reset AR
likelihoods, predeclared sensitivity structures, and sampler diagnostic gate.

## Model decisions

- `HQRCData` validates aligned standardized residual rows, dense occurrence
  identity, immutable occurrence context, and contiguous occurrence segments.
  The model never concatenates an AR transition across those segments.
- H1--H4 and complete/partial/none pooling are implemented.  H3/H4 use a
  centered 24-hour holiday-type profile with a circular first-difference
  potential; its named `gamma` profile is sum-to-zero.
- Partial pooling is non-centered.  It uses holiday-type means (`mu`), optional
  pandemic restriction effects (`delta`), either LKJ full covariance or diagonal
  scales, and preserves `between_scale` for both structures.
- The default AR model draws `u_phi ~ Beta(a, b)` from the approved calibration
  and defines `phi = 2*u_phi - 1`.  It uses the stationary initial density and
  conditional innovations separately for every event.
- Student-t AR(1) and stable PACF-parameterized normal AR(2) are explicitly
  selectable sensitivity models.  AR(2) resets its first two observations for
  each event.
- PyMC is the default sampler.  Nutpie is only imported when requested.
  Returned ArviZ data stores backend, dependency versions, elapsed time, and
  diagnostics; paper-profile runs fail on R-hat, ESS, or divergence thresholds.

## TDD evidence

- RED: the exact fast command failed collection because `hqrc_v3.bayes` was
  absent.
- GREEN: the exact fast command passed with `23 passed`.
- Slow smoke: `uv run pytest -c hqrc_v3/pyproject.toml
  hqrc_v3/tests/integration/test_hqrc_sampling.py -m slow -q` passed.
- Regression: the full HQRC v3 suite passed with `180 passed`.
- Ruff and `git diff --check` passed.

The tiny two-chain sampler emits expected ArviZ small-draw diagnostic runtime
warnings; it is a non-paper smoke test and therefore does not apply the paper
diagnostic gate.

## Scope

- `hqrc_v3/src/hqrc_v3/bayes/__init__.py`
- `hqrc_v3/src/hqrc_v3/bayes/model.py`
- `hqrc_v3/src/hqrc_v3/bayes/samplers.py`
- `hqrc_v3/tests/unit/test_ar_likelihood.py`
- `hqrc_v3/tests/unit/test_hqrc_model.py`
- `hqrc_v3/tests/integration/test_hqrc_sampling.py`
- `hqrc_v3/pyproject.toml` (registered the opt-in `slow` marker)
- `.superpowers/sdd/2026-08-10-hqrc-v3-implementation/task-8-report.md`

The pre-existing untracked `hqrc_v3/uv.lock` was not modified or staged.
