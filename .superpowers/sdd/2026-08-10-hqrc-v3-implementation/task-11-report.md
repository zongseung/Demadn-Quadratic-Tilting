# Task 11 report — frozen paper baselines and concrete execution

## Outcome

Task 11 now has a hash-verified, strictly typed paper registry and concrete
`generate-oof` / `fit-final-baselines` execution. The implementation publishes
the complete five-model × B0/B1 contract without tuning or substitution:

- XGBoost, LightGBM, and RBF-SVR use 24 independent direct-horizon estimators.
- Seq2Seq-LSTM uses two recurrent encoder/decoder layers.
- Transformer uses one encoder and one decoder layer.
- Both neural models jointly emit 24 hours without target-demand decoder input.
- Neural members retain seeds 11, 23, 37, 41, and 53; point products use reserved
  ensemble seed 0 and the exact pointwise arithmetic mean.
- Boosting and neural fits receive the last 61 complete pre-evaluation days as a
  chronological early-stopping partition. SVR retains the full outer training set.
- Every estimator/member runs with one internal CPU thread.

The exact paper parameters live in `configs/model_spaces.toml` and are rejected if
any model, parameter, architecture depth, validation length, or seed changes—even
when a caller supplies the changed file's newly calculated digest. The expected
SHA-256 is also checked against the actual file before a stage begins.

## Concrete stage and publication contract

The OOF stage is fixed to evaluation years 2020–2023 and the final stage to 2024.
Paper execution requires all five models and both B0/B1; it cannot select a fold
subset or reduce rounds. The explicit smoke profile may select an immutable OOF
subset and cap boosting rounds. Its manifest records `profile="smoke"` and the
override, so it cannot be mistaken for paper output.

Each model/feature/seed/fold is cached independently with its complete context.
Publication resumes from complete stream caches, validates every stream, constructs
neural means only after all five seeds are present, and then publishes:

```text
predictions/oof_members.parquet
predictions/oof.parquet
predictions/final_2024_members.parquet
predictions/final_2024.parquet
predictions/baseline_manifest.json
```

The group publisher holds an advisory lock and writes a canonical, fsynced
transaction journal before replacing any staged Parquet/JSON file. Stage entry
recovers a verified interrupted commit or rolls it back to the embedded prior
manifest before making cache-hit decisions. Journal paths are restricted to exact
basenames in the predictions directory. Ordinary failures roll back immediately;
abrupt termination after members, point, or manifest replacement is recoverable.
Prior completed stages are preserved, and unmanifested OOF/final products fail as
orphans rather than being ignored. Every stage entry recomputes and applies the
full fold, coverage, stream-context, seed, ensemble, and Parquet validation to all
manifest-recorded stages, including a non-requested OOF/final sibling. Recovery
performs the same run-level validation before deleting its journal.

The manifest and stream metadata bind separate SHA-256 identities for raw data,
experiment config, frozen model config, event registry, and holiday calendar. They
also bind feature-column names/order, selected models/features, classical seed,
five neural seeds, ensemble identity, profile, smoke overrides, folds, exact
matrix-derived prediction coverage, and output digests. Every member and point
stream must match the complete `(origin, target_timestamp, horizon, split_id,
observed_mw)` set. Tests demonstrate fail-closed handling of:

- any one of the five changed input hashes;
- partial public artifacts or a partial stream cache pair;
- byte-level Parquet tampering;
- a hash-rebound wrong model identity;
- a hash-rebound changed neural seed set;
- a changed/reordered feature schema;
- uniform missing, extra, duplicate, misaligned, or changed-observation coverage;
- an interrupted multi-file publication or unsafe journal path.

## CLI and real-data path

Both concrete CLI handlers validate the experiment, correction event registry,
holiday calendar, and frozen model registry; audit the hourly source; independently
construct the selected B0/B1 matrices; calculate all five input hashes; and invoke
the shared cache/publication stage. Paper CLI execution requires the exact
2019-01-01 00:00 through 2024-10-31 23:00 range and 51,144 rows before feature
construction; only the explicit smoke profile is relaxed. No handler invokes the
tuning boundary. README commands declare repository-root execution and consistently
use `uv run --project hqrc_v3 --locked hqrc ...`; they compute the model hash
with OpenSSL and pass the quoted `"$MODEL_SHA256"` value.

The opt-in real smoke audited the 51,144-row source and executed actual LightGBM-B1
for OOF 2020 with three boosting rounds. It traversed the same loader, frozen
factory, 61-day validation-tail, prediction cache, semantic validation, and grouped
publication code as paper execution. It remained explicitly non-paper throughout;
no fallback estimator was used.

## TDD and verification evidence

- Phase A RED: the first focused run stopped at collection with
  `ModuleNotFoundError: hqrc_v3.baselines.config`.
- Phase A GREEN: 11 new frozen-config/factory/tail/early-stopping tests passed; 86
  related legacy and new baseline tests passed together.
- Phase B RED: the new integration suite stopped at collection because
  `run_paper_oof_stage` / `run_paper_final_stage` did not exist.
- Slow-smoke RED: the real test failed with an unexpected
  `smoke_boosting_rounds` argument before the non-paper override was implemented.
- Focused final: `30 passed` for the exact Task 11 unit and integration files.
- Full fast suite: `323 passed, 5 deselected` in 22.70 seconds. The 47 warnings are
  the pre-existing tiny-draw ArviZ/runtime warnings from Bayesian smoke fixtures.
- Real verification: the new LightGBM-B1 smoke and existing real-data SVR smoke
  passed together (`2 passed` in 1.73 seconds).
- Full Ruff: `All checks passed!` for `hqrc_v3/src` and `hqrc_v3/tests`.
- `git diff --check`: clean.

### Fix round 1

- Fixed-bound RED: a shortened continuous source reached feature construction,
  and the paper CLI passed `None` audit bounds. Exact paper bounds now fail before
  features/publication; smoke remains explicitly relaxed.
- Publication RED: all three abrupt replacement boundaries lacked a durable
  journal, an interrupted rerun stranded partial files, orphan final products were
  accepted during an OOF cache hit, and an unsafe journal was ignored. The final
  tests also remove a required staged file and prove deterministic rollback plus
  zero-refit republication from the immutable stream caches.
- Coverage RED: the manifest lacked expected coverage; uniformly truncated and
  changed-observation products could be rehashed into cache hits; changed supplied
  targets and divergent B0/B1 origins were not rebound on reload.
- README RED: commands mixed `uv run hqrc` and bare `hqrc` despite retaining paths
  relative to the repository root.
- Expanded focused suite: `47 passed` in 33.15 seconds.
- Full fast suite: `340 passed, 5 deselected` in 40.19 seconds; only the existing
  47 tiny-draw ArviZ/runtime warnings were emitted.
- Real verification: the paper-stage LightGBM smoke passed (`1 passed` in 1.51
  seconds), and the existing real-data smoke passed (`1 passed` in 1.29 seconds).
- Full Ruff: `All checks passed!`; `git diff --check`: clean.

### Fix round 2

- RED: nine targeted cases failed. Rehashed uniform OOF truncation was accepted
  before final execution; a rehashed wrong-model final artifact was accepted
  before OOF execution; paper sibling fold records could omit OOF 2023 or replace
  final-2024; smoke sibling split records accepted unknown, duplicate, and reordered
  identifiers; journal recovery accepted a consistently rebound corrupt sibling;
  and README commands used a placeholder/unexpanded hash token.
- GREEN: preflight reconstructs every recorded stage from the current matrix and
  canonical fold objects and invokes the complete stage validator for both OOF and
  final. Paper siblings require exactly OOF 2020–2023 and final-2024. Smoke sibling
  OOF splits are resolved only from a non-empty, known, unique, canonical-order
  manifest sequence before coverage is derived.
- Recovery now runs that same full preflight while the journal is still durable.
  Semantic failure rolls back the interrupted stage, restores the prior manifest,
  removes/fsyncs the journal, and preserves the fail-closed error.
- Focused suite: `55 passed` in 38.13 seconds.
- Full fast suite: `348 passed, 5 deselected` in 45.63 seconds; only the existing
  47 tiny-draw ArviZ/runtime warnings were emitted.
- Real verification: the paper-stage LightGBM smoke passed (`1 passed` in 1.92
  seconds), and the existing real-data smoke passed (`1 passed` in 1.52 seconds).
- Full Ruff: `All checks passed!`; `git diff --check`: clean.

### Fix round 3

- Packaging RED: from a clean `UV_PROJECT_ENVIRONMENT`, with `VIRTUAL_ENV` and
  `PYTHONPATH` removed, the documented repository-root command failed with
  `Failed to spawn: hqrc`. The old child metadata was a uv virtual project and
  therefore installed no distribution or console script.
- Packaging GREEN: `hqrc_v3/` now uses the supported `setuptools.build_meta`
  PEP 517 backend and exact, non-namespace discovery of `src/hqrc_v3`. Its
  metadata declares every direct production dependency for classical/neural
  baselines, diagnostics, PyMC sampling, evaluation, and Polars Parquet I/O;
  optional nutpie acceleration and the pytest/Ruff development group are also
  locked.
- The repository root now owns a uv workspace containing `hqrc_v3`, and its
  Python floor is aligned at 3.11. The tracked repository-root `uv.lock` is the
  sole committed lock used by `uv run --project hqrc_v3 ...`; it records the
  child as `source = { editable = "hqrc_v3" }`. The changed root-level files are
  `pyproject.toml` and `uv.lock`.
- Clean-install GREEN: the real 51,144-row fixed-bound audit passed through the
  installed console in an isolated locked environment. A Python `-I` inspection
  from an external temporary cwd resolved `hqrc_v3` to
  `hqrc_v3/src/hqrc_v3/__init__.py`, and distribution metadata exposed exactly
  `hqrc = hqrc_v3.cli:main`.
- Real installed-console GREEN: the non-paper LightGBM-B1 OOF-2020 smoke passed
  with three boosting rounds, published the expected Parquet/manifest products,
  and resolved the estimator to `lightgbm.sklearn.LGBMRegressor` rather than a
  fallback.
- Packaging-focused tests: `3 passed`; Task 11 focused tests: `58 passed`; full
  non-slow suite: `351 passed, 6 deselected` with the same 47 tiny-draw
  ArviZ/runtime warnings.
- Round 3's bootstrap used `uv sync --project hqrc_v3 --locked` and then exported
  `UV_LOCKED=1`; round 4 replaces that environment-dependent operator contract
  with an explicit flag on every command. The bootstrap resolved all 134 locked
  packages successfully.
- Both required real baseline smokes passed: installed-console LightGBM-B1
  (`1 passed` in 27.58 seconds) and the existing real-source SVR smoke (`1 passed`
  in 1.94 seconds). The isolated installed-console audit also passed (`1 passed`
  in 24.89 seconds).
- Ruff reported `All checks passed!`; `uv lock --check` resolved all 134 packages
  without changing the lock; `git diff --check` was clean. Generated
  `hqrc_v3.egg-info` was removed before publication.

### Fix round 4

- Exact-argv RED: the README contract test failed on the existing
  `export UV_LOCKED=1`, and the subprocess-helper regression captured
  `uv run --project hqrc_v3 hqrc ...` instead of the required literal argv with
  `--locked`. Both failures directly reproduced the review finding.
- GREEN: every README `hqrc` example is now
  `uv run --project hqrc_v3 --locked hqrc ...`, including paper OOF/final, smoke,
  audit, AR diagnostics/approval, and report commands. The test command also
  carries `--locked`; the README neither requires nor recommends `UV_LOCKED`.
- The isolated helper emits that same exact argv. Its clean environment removes
  `PYTHONPATH` and `VIRTUAL_ENV`, sets only the isolated
  `UV_PROJECT_ENVIRONMENT` plus progress control, and contains no `UV_LOCKED`
  escape hatch.
- The exact RED pair passed after the minimal change (`2 passed`), and the
  packaging plus README contracts passed (`4 passed`). The clean installed audit,
  actual LightGBM-B1 console smoke, and real-source SVR smoke passed together
  (`3 passed, 1 deselected` in 32.28 seconds).
- Ruff reported `All checks passed!`; `uv lock --check` resolved the unchanged
  134-package root lock; `git diff --check` was clean. No model configuration,
  dependency metadata, or Python floor changed.
- The protected `hqrc_v3/uv.lock` remained unstaged and byte-identical at
  SHA-256 `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

## Commits

- `f857e04 feat(hqrc-v3): freeze paper baseline factories`
- `4fad1ce feat(hqrc-v3): execute frozen paper baselines`
- `c4a30ec docs(hqrc-v3): report Task 11 verification`
- `ddf86ba fix(hqrc-v3): harden paper baseline execution`
- `bd5ede3 fix(hqrc-v3): validate complete baseline run state`
- `3ad2b4e fix(hqrc-v3): install workspace console package`
- `fix(hqrc-v3): make lock flag explicit` (fix round 4)

The protected workspace-local `hqrc_v3/uv.lock` was not staged or modified; its
SHA-256 remained
`f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`, and it
remains the only untracked path.
