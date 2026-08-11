# Task 11 fix round 4/5 — explicit locked operator command

## Review finding

Task 11 fix round 3 made `hqrc_v3` installable and the exact locked command works in
an isolated environment, but the README and its subprocess helper rely on
`UV_LOCKED=1` instead of spelling the operator contract as:

```bash
uv run --project hqrc_v3 --locked hqrc ...
```

Independent verdict: **NOT READY** until documentation and executable regression
use that exact argv.

## Required implementation

1. Replace every repository-root README `uv run --project hqrc_v3 hqrc ...`
   example with `uv run --project hqrc_v3 --locked hqrc ...`.
2. Do not require or recommend `export UV_LOCKED=1`; the command itself must be
   self-contained.
3. Change the isolated subprocess helper/test to include `--locked` explicitly,
   remove `UV_LOCKED` from the clean environment, and add a regression that fails
   if a documented `hqrc` invocation omits the flag.
4. Re-run packaging/README tests, the two installed-console real-model smokes,
   Ruff, `uv lock --check`, and `git diff --check`.
5. Preserve `hqrc_v3/uv.lock` byte-for-byte at SHA-256
   `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657` and do not
   stage it.
6. Append final verification to the Task 11 report and create exactly one fix
   commit.

## Scope guard

Do not change model logic, dependency versions, feature construction, or the root
Python floor in this fix. Python 3.11+ is already explicit in the HQRC README and
member package; the independent reviewer recorded the root floor as a nonblocking
compatibility note to evaluate separately.

## Implementation record

- [x] Exact-argv RED captured both defects: the README still exported
  `UV_LOCKED=1`, and the isolated helper emitted
  `uv run --project hqrc_v3 hqrc ...` without `--locked`.
- [x] Every repository-root `hqrc` example now emits the self-contained argv
  `uv run --project hqrc_v3 --locked hqrc ...`; no operator environment variable
  supplies the lock contract.
- [x] The isolated subprocess helper passes `--locked` explicitly and removes
  `UV_LOCKED` while retaining its clean `UV_PROJECT_ENVIRONMENT` boundary.
- [x] Packaging/README verification passed (`4 passed`); the clean installed
  audit, real LightGBM console smoke, and real-source SVR smoke passed together
  (`3 passed, 1 deselected`).
- [x] Ruff, `uv lock --check` (134 packages), and `git diff --check` passed.
- [x] Model configuration, dependency metadata, and both Python floors remained
  unchanged. The protected nested lock remained unstaged and byte-identical at
  SHA-256 `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.
