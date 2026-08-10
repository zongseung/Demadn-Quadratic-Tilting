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
