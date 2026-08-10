# Task 10 report — reproducible experiment workflow

## Delivered

- Added fail-closed report validation with manifest hash checks, approved-AR and
  posterior-diagnostic gates, normalized Parquet/CSV event metrics, a report figure,
  and atomic `COMPLETE` publication.
- Added a deterministic synthetic end-to-end run.  It explicitly creates a proposed
  calibration and then invokes the approval gate; production reporting never approves
  calibration automatically.
- Added structured sampler benchmark output, including a `nutpie` `not-installed`
  result when the optional backend is absent, its predeclared default-selection gate,
  Arrow/Polars metadata, and a measured-only Rust-rewrite gate.
- Added explicit CLI contracts for correction, ablation, sampler benchmark, and
  reporting stages.  Unwired compute stages fail honestly rather than pretending to
  execute an experiment; `report` is concrete.
- Added the operator README and a slow real-source smoke that audits all 51,144 rows,
  fits a small 2019 SVR OOF fold, and stores a schema-valid cached 2020 prediction.

## TDD evidence

The initial report/end-to-end test run failed at collection with
`ModuleNotFoundError: hqrc_v3.evaluation.reports`.  The implementation then made the
integration tests pass.

## Verification

- `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/integration/test_cli_oof.py hqrc_v3/tests/integration/test_report_pipeline.py hqrc_v3/tests/integration/test_end_to_end.py -q` — 16 passed.
- `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m "not slow" -q` — 227 passed, 2 deselected (two pre-existing ArviZ runtime warnings).
- `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/slow/test_real_data_smoke.py -m slow -q` — 1 passed.
- `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests` and `git diff --check` — clean.
