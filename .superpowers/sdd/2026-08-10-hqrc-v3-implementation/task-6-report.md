# Task 6 report

Implemented cache-aware expanding-origin OOF baseline orchestration, the exact
2019--2023 final refit / available-2024 prediction path, and a staged CLI.

## Decisions

- `generate_expanding_oof()` creates exactly the four immutable folds
  (2019→2020 through 2019--2022→2023), uses `validation=None`, performs no
  tuning/selection, and fails when either side of a fold has no complete
  targets.
- A cache key binds model name, B0/B1 feature set, seed, and fixed split.
  Cache compatibility also requires `config`, `data`, and `events` hashes.
  `config` is explicitly the hash of the complete frozen experiment plus
  selected model-config payload, so a frozen model change cannot reuse OOF
  artifacts accidentally.
- The declared feature set must exactly equal the matrix's ordered
  `feature_columns(feature_set)` future schema.  This rejects relabeling B1 as
  B0 and its associated holiday leakage.
- Each invocation owns one fitted seed stream.  Five-seed neural work invokes
  it once per seed and retains those artifacts; averaging streams into an HQRC
  residual stream is downstream evaluation/correction work.
- `OOFRunSummary` exposes each validated fold frame and the combined frame,
  exact evaluation years, fit/cache-hit counts, and model/feature/seed context.
  It rejects missing/duplicate folds and inconsistent counts or combined data.
- `fit_final_baseline()` returns `FinalBaselineResult`, containing the fitted
  model and validated `final-2024` prediction frame.  It never admits 2024
  targets into training.

## CLI

The `hqrc` console entry point exposes `audit-data`, `tune-baselines`,
`generate-oof`, and `fit-final-baselines`.  Parsers require explicit frozen
model path/hash for OOF/final stages, so they cannot route through tuning.
Handlers are injectable for tests; the default handlers deliberately fail
nonzero with a concise missing-handler error rather than succeeding as a
no-op.  Real stage-handler wiring is deferred to Task 10.  `audit-data
--fixed-bounds` supplies the fixed 2019-01-01 through 2024-10-31 / 51,144-row
contract to its handler.

## TDD evidence

- RED: the prescribed focused command failed collection because
  `hqrc_v3.oof` and `hqrc_v3.cli` did not exist.
- GREEN: after the first implementation, the prescribed focused command passed
  with `18 passed`.
- Review RED: three focused tests failed for the missing feature-schema binding,
  incomplete provenance hashes, and inconsistent summary count rejection.
- Review GREEN: the prescribed focused command passed with `22 passed`.
- Regression: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -q`
  passed with `135 passed`.
- `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`
  and `git diff --check` passed.

## Scope

- `hqrc_v3/src/hqrc_v3/oof.py`
- `hqrc_v3/src/hqrc_v3/cli.py`
- `hqrc_v3/pyproject.toml`
- `hqrc_v3/tests/integration/test_oof_pipeline.py`
- `hqrc_v3/tests/integration/test_cli_oof.py`
- `.superpowers/sdd/2026-08-10-hqrc-v3-implementation/task-6-report.md`

The pre-existing untracked `hqrc_v3/uv.lock` was not modified or staged.
