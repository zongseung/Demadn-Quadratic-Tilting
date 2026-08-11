# Task 13 implementation report

Status: DONE

Commit: `5496b62` (`feat(hqrc-v3): prepare standardized OOF residuals`)

## Delivered

- Added production `prepare-residuals` CLI and a canonical combined
  `inputs/standardized_residuals.parquet` publication.
- Validates the strict baseline v2 manifest, its OOF point SHA, exact yearly coverage,
  stream identities, observed-load agreement, experiment hash, and fixed event-registry hash.
- Computes `observed_mw - predicted_mw`, excludes every registered correction-window hour
  from each `(model, feature_set, seed, split_id)` RMS, and standardizes only event rows.
- Attaches exact occurrence/type/restriction labels, fractional `tau_days` from central
  midnight, hour, fold scale, both timestamps, and the original prediction context.
- Publishes a canonical hash-bound manifest with all fold scales, non-event counts, and the
  latest complete OOF scale needed for later 2024 MW conversion.
- Added strict `diagnose-ar --model/--feature-set [--seed]` selection over the combined
  artifact. Diagnosis remains proposal-only and never auto-approves an AR prior.
- Added crash recovery for interrupted first publication and documented all-context usage.

## TDD evidence

- RED: the new focused suite failed during collection with
  `ModuleNotFoundError: No module named 'hqrc_v3.residual_stage'` before implementation.
- Focused GREEN:
  `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_residual_stage.py hqrc_v3/tests/integration/test_cli_oof.py -q`
  — `19 passed in 0.94s`.
- Adjacent residual/AR/paper-stage regression command exited 0.
- Full non-slow suite:
  `uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m 'not slow' -q`
  — `390 passed, 7 deselected, 47 pre-existing tiny-draw warnings in 63.93s`.
- Scoped Ruff for every changed Python file: clean.
- Full-tree Ruff still reports 37 pre-existing import-order findings in unrelated test files;
  no bulk rewrite was made.
- `git diff --check`: clean.
- Protected untracked `hqrc_v3/uv.lock` remained byte-identical at
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

## Concerns

- No production run artifacts were touched, as required. The controller should run the new
  CLI against the completed paper OOF publication only after independent review.
- Plot generation remains intentionally outside this bounded residual-preparation task.

## Independent-review fix round 1

Commit: `ae7c70f` (`fix(hqrc-v3): bind residual diagnostics to canonical OOF`)

- Replaced caller-selected Parquet and caller-supplied hashes in `diagnose-ar` with the
  canonical run directory, actual experiment config, and actual event registry. The command
  now reloads the strict residual manifest and derives every diagnostic provenance hash.
- Added a registry-exact diagnostic gate before ACF/AR estimation. Paper calibration is now
  restricted to numeric `through=2023`, `oof-2020` through `oof-2023`, the eight registered
  2020--2023 occurrences, and their exact timestamps, windows, `tau_days`, hours, holiday
  types, and restriction indicators. A fake 2024 occurrence relabeled `oof-2023` is rejected.
- Reused the baseline publisher's strict point/member validator. Residual preparation now
  verifies all six real source hashes and frozen source contracts, feature/preprocessing
  schemas, execution overrides, streams, expected coverage, scaler populations, member SHA,
  exact five-seed neural means, and point SHA before residual computation.
- Upgraded the residual manifest to v2. It binds point/member artifacts and a canonical
  immutable OOF semantic projection instead of the mutable whole baseline-manifest file, so
  a legitimate later `stages.final` append does not invalidate the completed residual stage.
- Updated CLI contracts and executable README examples with every required concrete source.

Fix-round evidence:

- Genuine RED: `5 failed, 17 passed` for tampered neural seeds, a non-mean neural point
  stream, mutable whole-manifest binding, fake occurrence metadata, and legacy arbitrary AR
  inputs.
- Focused GREEN: `23 passed in 1.19s` across the residual/CLI contracts plus the README
  executable-command check.
- Full non-slow GREEN: `393 passed, 7 deselected, 47 pre-existing tiny-draw warnings in
  66.43s` with a task-local writable PyTensor compile directory.
- A full run including slow tests reached `396 passed`; four environment-only failures were
  caused by sandbox-denied user PyTensor/uv caches, not changed code.
- Scoped Ruff and `git diff --check` are clean. The protected untracked nested lock remains
  byte-identical at
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.
