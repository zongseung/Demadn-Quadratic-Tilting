# Task 15C Step 3 implementation report

Status: IMPLEMENTATION AWAITING INDEPENDENT REVIEW

Implementation commit: `d32580f`

## Scope delivered

- Added the exact public operations `prepare_loeo_ar_proposal_set`,
  `approve_loeo_ar_proposal_set`, and `load_approved_loeo_ar_set` in
  `hqrc_v3.diagnostics.loeo_ar`.
- Preparation revalidates one current `LOEOPublication`, calls `load_loeo_fold` for every
  registry-ordered held-out occurrence, and passes only that physical causal-false nine-event
  frame to the existing `diagnose_event_residuals` and `calibrate_beta_prior` contracts.
- Each deterministic canonical proposal-set binds the exact source/diagnostic contexts, current
  LOEO manifest and universe, all ten physical training SHA-256 values, ten existing-format
  unapproved AR proposals, and ten deterministic SVG review plots. The set exposes `a`, `b`,
  `phi_center`, occurrence-local phi estimates, Ljung-Box statistics/p-values, and warning flags.
- SVG plots use one common correlation scale and show raw, independently detrended, and AR(1)
  innovation ACF/PACF for all nine training occurrences, including ACF confidence bands, PACF
  reference bounds, warnings, phi, and Ljung-Box values. Rendering and canonical file primitives
  live in focused private modules; no plotting dependency or public helper surface was added.
- Preparation always returns `approved=false` and never calls approval. Batch approval requires
  the caller's exact lowercase 64-character `confirm_proposal_set_sha256`, copies the ten existing
  proposals through the reviewed `approve_calibration` gate, and returns only the approved-set
  path. It never reruns diagnostics, recalibrates values, redraws plots, or accepts overrides.
- `load_approved_loeo_ar_set` revalidates the current LOEO source, every physical fold and exact
  `registry_ids - {held_out}` population, unapproved proposal semantics, deterministic plot bytes,
  approval identities, and all set/file hashes before constructing `ApprovedLOEOARSet` with the
  existing tokened `ApprovedARCalibration` values. `calibration_for()` repeats the complete set
  load before returning one calibration to a fit boundary.
- Publication is atomic, fsynced, restartable after injected generation/pointer/approval rename
  interruptions, exact-namespace/no-symlink/no-special-file, and fail-closed on incompatible or
  unsafe evidence.

No LOEO posterior fitting, baseline fitting, PyMC sampling, prediction, ablation, metric, CLI, or
paper-artifact write was added or executed. AR(1) remains order one; phi remains a later posterior
parameter under the approved fold-specific Beta prior.

## Genuine RED and GREEN evidence

Initial tests were written before the production module:

```text
env PYTHONPATH=hqrc_v3/src .venv/bin/python -m pytest \
  -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_loeo_ar.py -q

E   ModuleNotFoundError: No module named 'hqrc_v3.diagnostics.loeo_ar'
1 error in 0.49s, exit 2
```

The first implementation execution exposed the deliberately linear Task 15B fixture becoming
zero variance after the required detrending (`9 failed` with
`ARCalibrationError: event residual has zero lag variance`). Task 15C now wraps that fixture with
deterministic nonlinear residuals without changing Task 15B's fixture or production estimator.

Final focused GREEN after self-review, private-module extraction, semantic-tamper tests, and the
approved-path-only return change:

```text
env PYTHONPATH=hqrc_v3/src .venv/bin/python -m pytest \
  -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_loeo_ar.py -q
21 passed in 70.03s
```

Coverage includes all ten physical folds; held-out outcome mutation invariance; the existing
event-reset diagnostics and robust Beta rule; plot/warning serialization and hashes; prepare's
no-approval proof; malformed/missing/wrong/exact confirmation; approval diagnostic/calibration
call-count proof; exact ten-fold trusted loading/selection; causal eight-event, single approval,
fold swap, duplicate/missing/reordered/substituted registry, context, physical fold, rehashed
proposal, rehashed plot, unknown entry, symlink, and interrupted prepare/approval rejection or
safe retry.

## Final verification

Task 7/14/15 adjacent:

```text
env PYTHONPATH=hqrc_v3/src \
  PYTENSOR_FLAGS=compiledir=/private/tmp/hqrc-v3-pytensor-task15c-adjacent-final \
  .venv/bin/python -m pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests/unit/test_ar_diagnostics.py \
  hqrc_v3/tests/integration/test_ar_artifact.py \
  hqrc_v3/tests/unit/test_ar_likelihood.py \
  hqrc_v3/tests/unit/test_correction_source.py \
  hqrc_v3/tests/unit/test_correction_stage.py \
  hqrc_v3/tests/unit/test_hqrc_artifacts.py \
  hqrc_v3/tests/unit/test_loeo_diagnostics.py \
  hqrc_v3/tests/unit/test_loeo_ar.py -q
165 passed, 30 existing warnings in 84.34s
```

Full non-slow suite, run once after the final behavioral changes:

```text
env PYTHONPATH=hqrc_v3/src \
  PYTENSOR_FLAGS=compiledir=/private/tmp/hqrc-v3-pytensor-task15c-full \
  .venv/bin/python -m pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests -m 'not slow' -q
530 passed, 8 deselected, 77 existing warnings in 153.23s
```

Static and protected-file checks:

```text
.venv/bin/ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests
All checks passed!

git diff --check
(no output, exit 0)

env LC_ALL=C shasum -a 256 hqrc_v3/uv.lock
f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657  hqrc_v3/uv.lock
```

## Real temporary unapproved proposal

The final real check reused the existing no-fit paper source copy
`/tmp/hqrc-source-preflight.lKh7Yt`, current XGBoost/B1 LOEO publication
`/private/tmp/hqrc-v3-task15b-real-check`, and wrote only to
`/private/tmp/hqrc-v3-task15c-real-final-20260812`. It called source validation, LOEO loading, and
`prepare_loeo_ar_proposal_set`; it did not call approval. The output has no `generation/approval`.

```text
approved=false
proposal_set_sha256=db54721c0899f29312abe3d47d08977b33e02f53315d629c17fd319f6bfa83cd
```

Ten fold summaries in canonical held-out order:

| held_out | a | b | phi_center | warning |
|---|---:|---:|---:|:---:|
| seollal-2020 | 36.71951713959426 | 3.2804828604057423 | 0.8359758569797129 | false |
| chuseok-2020 | 36.73142339257454 | 3.268576607425455 | 0.8365711696287272 | false |
| seollal-2021 | 36.73142339257454 | 3.268576607425455 | 0.8365711696287272 | false |
| chuseok-2021 | 36.73142339257454 | 3.268576607425455 | 0.8365711696287272 | false |
| seollal-2022 | 36.73142339257454 | 3.268576607425455 | 0.8365711696287272 | false |
| chuseok-2022 | 36.71951713959426 | 3.2804828604057423 | 0.8359758569797129 | false |
| seollal-2023 | 36.71951713959426 | 3.2804828604057423 | 0.8359758569797129 | false |
| chuseok-2023 | 36.71951713959426 | 3.2804828604057423 | 0.8359758569797129 | false |
| seollal-2024 | 36.71951713959426 | 3.2804828604057423 | 0.8359758569797129 | false |
| chuseok-2024 | 36.73142339257454 | 3.268576607425455 | 0.8365711696287272 | false |

The original paper artifact's aggregate regular-file SHA was identical immediately before and
after all real checks:

```text
4ab54f253ed520e93475dd46381c913e289a32c03251f76a0da0fd74e3a5ee8e
```

The temporary validated source copy was also byte-identical before/after:

```text
7d637f91adf92ae02e741d05bd0ac6fe28a1c535e2e10c7aefa56bd2c63bcb64
```

## Self-review and concerns

- A first real import check found an eager package re-export cycle
  (`correction_source -> diagnostics.ar -> diagnostics.__init__ -> loeo_ar -> correction_source`)
  before output creation. The unnecessary package-level re-export was removed; the required API
  remains public in `hqrc_v3.diagnostics.loeo_ar`. Direct correction-source/API imports and all
  adjacent/full tests then passed.
- Approval intentionally returns only a path. A trusted `ApprovedLOEOARSet` can be obtained only
  through the semantic revalidating loader, so a rehashed reviewed-file mutation cannot escape as
  a trusted wrapper merely because its new set digest was confirmed.
- Proposal-set identity deliberately includes absolute current LOEO publication paths in addition
  to hashes. This makes publication substitution fail closed; moving an otherwise identical LOEO
  tree requires a new reviewed proposal set.
- Existing AR artifact private serialization validators are reused inside the same diagnostics
  package to avoid duplicating or weakening their robust-Beta and approval-token contracts.
- No unresolved implementation concern is known. Independent review should especially inspect
  the atomic recovery namespaces and the fit-boundary use of `calibration_for()` in Task 15D.

## Files changed

- `hqrc_v3/src/hqrc_v3/diagnostics/loeo_ar.py`
- `hqrc_v3/src/hqrc_v3/diagnostics/_loeo_ar_io.py`
- `hqrc_v3/src/hqrc_v3/diagnostics/_loeo_ar_plot.py`
- `hqrc_v3/tests/unit/test_loeo_ar.py`
- this report and the progress ledger line below

Task 15 Step 3 remains unchecked pending fresh independent review.
