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
- Full non-slow verification completed in the locked workspace virtual environment with explicit
  source and temporary PyTensor-cache paths after the sandbox denied `uv` cache access:

  ```text
  env PYTHONPATH=/Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/.worktrees/hqrc-v3/hqrc_v3/src \
    PYTENSOR_FLAGS=compiledir=/private/tmp/hqrc-v3-pytensor-task15b \
    .venv/bin/python -m pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m 'not slow' -q
  486 passed, 8 deselected, 77 existing warnings in 80.07s, exit 0
  ```

  The original `uv` command's only issue was sandbox access to its external cache; the protected
  lock was not touched.

## Files changed

- `hqrc_v3/src/hqrc_v3/diagnostics/loeo.py`
- `hqrc_v3/tests/unit/test_loeo_diagnostics.py`
- this report and the progress ledger entry below.

No AR proposal/approval, model sampling, ablation/metric, CLI, or paper LOEO output was added.
Task 15 Step 2 remains unchecked pending independent review.

## Fix round 1/5 — review findings and resolution

Important findings received verbatim:

1. `hqrc_v3/src/hqrc_v3/diagnostics/loeo.py:574`: Any process crash after creating `generations/` but before publishing `current.json` leaves entries that every later publication rejects as partial. A crash after moving staging to its final generation or while writing `.current.tmp` is likewise unrecoverable, so publication is fail-closed but not restartable; files/directories also are not fsynced around the rename boundary. Fix by cleaning validated staging debris, recovering a complete identity-matching generation or safely replacing incomplete generations under the lock, and fsyncing artifacts/directories before pointer publication.
2. `hqrc_v3/src/hqrc_v3/diagnostics/loeo.py:651`: `_load_publication` hash-checks and reads the fold, but `load_loeo_fold` then reads it again and returns that second frame after checking only occurrence ordering. A numerical mutation between those reads can therefore be returned with the old trusted digest. Return the already hash-validated physical read, or hash/check the exact second read before returning it.
3. `hqrc_v3/tests/unit/test_loeo_diagnostics.py:282`: Required boundary coverage is incomplete or passes for incidental reasons. Only one 2024 event and one pre-2024 event are checked; the symlink case at line 424 first adds an unknown root entry, while order mutation and held-out reinsertion are rejected by stale hashes rather than semantic validation. There are no genuine publication-boundary crash/retry, wrong-path, registry-substitution, or rehashed-fold mutation tests. Add independent all-event assertions, rehash semantic mutations where appropriate, and inject failures at publication boundaries followed by retries.

Resolution:

- Under the exclusive publication lock, transient `.current.tmp` files and no-symlink staging
  debris are removed; an identity/source/hash-validated completed generation is recovered by
  publishing its pointer, while a safely removable incomplete matching generation is replaced.
  Incompatible generation names remain fail-closed. Every Parquet, fold directory, staging
  directory, generation namespace, temporary pointer, and final pointer rename now has explicit
  `fsync` durability ordering around publication.
- `_load_publication` now returns its exact hash- and semantic-validated physical fold frame map;
  `load_loeo_fold` returns that frame rather than performing an untrusted second Parquet read.
- Tests now parameterize both 2024 scales and all ten held-outs, inject both generation-rename and
  pointer-rename interruptions followed by actual retry/reload, isolate symlink targets outside
  the publication root, and directly cover rehashed universe/fold semantic mutations, rehashed
  wrong paths, and registry substitution.

Genuine RED before the production repair:

```text
uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_loeo_diagnostics.py -q
3 failed, 12 passed
```

The failures were both retry boundaries (`generation-rename`, `pointer-rename`) and the former
two-read fold TOCTOU regression. GREEN after the implementation and expanded coverage:

```text
Task 15B focused: 29 passed in 2.67s
Task 14/15 adjacent: 105 passed, 30 existing warnings in 5.73s
```

Controller full fix-tree verification completed in a traceable session:

```text
env PYTHONPATH=/Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/.worktrees/hqrc-v3/hqrc_v3/src \
  PYTENSOR_FLAGS=compiledir=/private/tmp/hqrc-v3-pytensor-task15b-fix1 \
  .venv/bin/python -m pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m 'not slow' -q
503 passed, 8 deselected, 77 existing warnings in 79.84s, exit 0
```

No source artifacts, sampling, AR approval, or paper LOEO output was changed.

## Fix round 2/5 — open review findings

Open finding 1: `hqrc_v3/tests/unit/test_loeo_diagnostics.py:263-269` still checks exact
preserved pre-2024 standardized values for only `seollal-2023`, and `:384-397` still tests
held-out reinsertion with a stale hash rather than rehashing it to reach held-out semantic
validation. The new rehashed-fold test at `:439-453` mutates only numeric values.

Open finding 2: `hqrc_v3/src/hqrc_v3/diagnostics/loeo.py:160-194, 219-223, 694-698`: recovery
conflates incomplete debris with a completed-but-invalid generation. A hash/source/manifest-invalid
identity-named generation returns `None` and is recursively deleted, even when `current.json`
already points to it, before normal fail-closed validation runs. Publication then fails with a
dangling pointer and destroys evidence instead of refusing to overwrite an incompatible complete
generation.

Fixes:

- Exact standardized-residual preservation is asserted for each of all eight 2020--2023
  occurrences. Both 2024 event scale assertions remain. Held-out reinsertion now rehashes its
  fold and all manifest/COMPLETE/current bindings before asserting the specific held-out semantic
  rejection.
- Recovery reads any current pointer before cleanup. A same-identity generation with the complete
  namespace, or one named by `current.json`, is never removed when recovery validation fails; it
  now raises `LOEOError` and preserves all bytes. Only incomplete, unreferenced, no-symlink
  same-identity debris remains removable.

Genuine RED:

```text
uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/unit/test_loeo_diagnostics.py -q
2 failed, 29 passed
```

Both failures reproduced corrupted completed-generation destruction: without a pointer it was
silently replaced, and with a pointer its tree was removed before rejection. GREEN:

```text
31 passed in 2.72s
```

No source artifact, AR proposal/approval, sampling, or paper LOEO output was changed.
