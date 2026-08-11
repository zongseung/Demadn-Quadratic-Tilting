# Task 15B Step 2 implementation report

Status: READY FOR INDEPENDENT REVIEW

## Scope delivered

- Added `hqrc_v3.diagnostics.loeo` with an API that accepts only a
  `ValidatedCorrectionSource` and one exact `EventResidualContext`:
  `publish_loeo_universe`, `load_loeo_universe`, and `load_loeo_fold`.
- The source is reconstructed on every publication/load.  The 2020--2023 standardized OOF
  rows are preserved from the source artifact; the two 2024 event grids are joined only to the
  final point stream and use `residual_mw = observed_mw - predicted_mw` and the unique
  `oof-2023` `sigma_n_mw` for standardization.
- The canonical retrospective universe has all ten ordered registry occurrences, exactly 1,296
  rows, retained `STANDARDIZED_RESIDUAL_COLUMNS`, and explicit `causal=false` metadata.
- Publication uses a dedicated immutable output namespace with a lock, atomic staging-directory
  rename, canonical current pointer, canonical manifest/COMPLETE marker, one physical
  `universe.parquet`, and ten physical held-out Parquets.  Every loader re-reads and hashes the
  target Parquet; no API accepts a caller-owned filtered universe frame.
- Load/reuse fails closed on context/source/registry substitution, canonical-path or namespace
  changes, non-regular files and symlinks, missing/partial products, raw hash changes, rehashed
  semantic mutation, ordering/grid changes, and held-out occurrence reinsertion.  A fold residual
  SHA is the raw SHA-256 of that fold's physical nine-event Parquet.

## Genuine RED/GREEN evidence

Initial RED before the production module existed:

```text
uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_loeo_diagnostics.py -q
ModuleNotFoundError: No module named 'hqrc_v3.diagnostics.loeo'
```

GREEN after implementation and self-review additions:

```text
uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_loeo_diagnostics.py -q
12 passed in 1.76s

uv run pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests/unit/test_ar_diagnostics.py \
  hqrc_v3/tests/unit/test_correction_source.py \
  hqrc_v3/tests/unit/test_correction_stage.py \
  hqrc_v3/tests/unit/test_loeo_diagnostics.py -q
88 passed, 30 existing tiny-draw warnings in 4.58s
```

The focused contract tests cover the exact ten-event/1,296-hour universe, published 2020--2023
values, 2024 `oof-2023` scaling, every physical nine-event fold, both 2024 and OOF held-out-row
mutation invariance, physical-fold reinsertion, rehashed semantic universe mutation, incompatible
reuse, partial/crashed publication, context substitution, symlinks, unknown entries, and order
mutation.

## Real no-fit evidence

Using the existing paper source at
`/Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/artifacts/hqrc-v3-paper-20260811`,
the XGBoost/B1 context was source-validated/reused with no baseline fitting or PyMC sampling, and
published only to `/private/tmp/hqrc-v3-task15b-real-check`:

```text
{'universe_rows': 1296, 'fold_rows': 1152, 'events': 10, 'causal': False}
```

The first real check exposed a `datetime[ns]` source-artifact versus `datetime[us]` grid mismatch.
The implementation now constructs the canonical grid at the source artifact's nanosecond unit;
the repeated real check above passed.  The original paper artifact directory was not modified.

## Self-review and verification

- The full-universe reconstruction validator and the physical-fold validator deliberately have
  separate responsibilities: the former proves the ten-event source semantics; the latter proves
  each persisted fold is the exact source-derived universe minus its held-out occurrence.  The
  latter never receives an in-memory full-universe frame from a caller.
- Full Ruff after formatting: `All checks passed!`.
- `git diff --check`: clean.
- Protected untracked `hqrc_v3/uv.lock` remains byte-identical:
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.
- The requested full non-slow command was attempted three times.  This execution environment
  returned only partial pytest progress at 14% and no exit summary, while focused/adjacent suites
  completed normally.  It is therefore not represented as a passing full-suite result and needs
  rerun during independent review.

## Files changed

- `hqrc_v3/src/hqrc_v3/diagnostics/loeo.py`
- `hqrc_v3/tests/unit/test_loeo_diagnostics.py`
- this report and the progress ledger entry below.

No AR proposal/approval, model sampling, ablation/metric, CLI, or paper LOEO output was added.
Task 15 Step 2 remains unchecked pending independent review.
