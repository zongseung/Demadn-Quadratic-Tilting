# Task 4 report: classical baseline adapters

## Original implementation

- Commit: `56da951 feat(hqrc-v3): add classical baseline adapters`
- RED: the required focused command failed collection because `hqrc_v3.baselines` did not exist.
- GREEN: the same focused command passed `8` tests and Ruff passed for `hqrc_v3/src` and `hqrc_v3/tests`.

## Contract-fix round 1

- RED command:

  ```text
  uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_baseline_protocol.py hqrc_v3/tests/integration/test_classical_baselines.py -q
  ```

  Result: `8 failed, 14 passed`; failures covered missing selection-record metadata,
  exact 168-history validation, prediction seed/context validation, and the existing
  LightGBM feature-name warnings.

- GREEN focused command: `23 passed in 0.67s`.
- Full verification:

  ```text
  uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit hqrc_v3/tests/integration -q
  ```

  Result: `77 passed in 1.53s`.

- Lint: `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests`
  passed with no diagnostics.
- Diff whitespace check: `git diff --check` passed.

## Contract coverage added

- Exact 168-hour history validation in both fit and predict paths.
- Per-horizon known-future feature isolation.
- Adapter-owned seed and single-thread settings, plus non-inspection of validation by fixed fit.
- Actionable optional-dependency errors.
- Ordered, JSON-serializable candidate score records with first-in-order tie resolution.
- Prediction-frame seed, shape, fold, and nonblank context validation.
- LightGBM now receives a consistent named feature representation for fit and predict, removing
  the prior feature-name warnings without global warning suppression.
