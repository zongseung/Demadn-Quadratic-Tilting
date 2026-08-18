# Task 7 report

Implemented event-reset AR diagnostics, robust data-derived Beta prior proposals,
provenance-bound proposal/approval artifacts, an approved-artifact loader, and staged
CLI commands for diagnosis and explicit approval.

## TDD evidence

- RED: the exact focused Task 7 command failed collection with two expected
  `ModuleNotFoundError` errors because `hqrc_v3.diagnostics` did not exist.
- Initial GREEN: the core diagnostic/artifact suite passed with `8 passed`.
- Final focused verification, including CLI: `17 passed`.
- Full HQRC v3 suite: `146 passed`.
- Ruff and cached diff checks: clean.

## Design decisions

- Each occurrence is detrended and diagnosed independently; event boundaries are never
  joined when estimating AR coefficients or innovation diagnostics.
- The proposal records suggested numeric Beta parameters derived from OOF diagnostic
  residuals. It does not fix phi and cannot be consumed before explicit approval.
- Approval verifies provenance and proposal integrity, copies the exact proposed values,
  and writes a distinct approved artifact.

## Scope

- `hqrc_v3/src/hqrc_v3/diagnostics/`
- `hqrc_v3/src/hqrc_v3/cli.py`
- Task 7 unit/integration tests and CLI routing tests

The pre-existing untracked `hqrc_v3/uv.lock` was not modified or staged.

## Fix round 1

- RED: `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_ar_diagnostics.py hqrc_v3/tests/integration/test_ar_artifact.py hqrc_v3/tests/integration/test_cli_oof.py -q` produced 11 expected failures before the hardening changes, covering multiple OOF split contexts, strict artifact content, and concurrent incompatible writers.
- GREEN: the same focused command passed with `23 passed`; the complete suite passed with `152 passed`; Ruff and `git diff --check` were clean.
- Added per-destination advisory locking for readers and writers, with the immutable overwrite decision re-checked under the exclusive lock and temporary-file cleanup on publication failure.
- Approval and loading now require all three current provenance hashes. Artifacts record sorted split IDs, validate strict calibration provenance and moment-rule recomputation, and serialize lag-specific Bartlett ACF bounds separately from the PACF reference width.

## Fix round 2

- RED: the focused Task 7 plus CLI command produced 3 expected failures: numeric `model`, numeric `feature_set`, and unknown string feature set `B2` were accepted by the previous context validator.
- GREEN: the focused command passed with `27 passed`; the full suite passed with `156 passed`; Ruff and `git diff --check` were clean.
- Context validation now requires string-like `model` and `feature_set` dtypes (including categorical/enum), nonblank single values, and `feature_set` exactly `B0` or `B1`; seed remains an integer-only contract.
