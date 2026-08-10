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

The group publisher holds an advisory lock, fsyncs staged Parquet/JSON files,
publishes the manifest last, and rolls back newly visible files after an injected
boundary failure. On reuse, it reloads and semantically validates the complete
publication before returning a cache hit.

The manifest and stream metadata bind separate SHA-256 identities for raw data,
experiment config, frozen model config, event registry, and holiday calendar. They
also bind feature-column names/order, selected models/features, classical seed,
five neural seeds, ensemble identity, profile, smoke overrides, folds, and output
digests. Tests demonstrate fail-closed handling of:

- any one of the five changed input hashes;
- partial public artifacts or a partial stream cache pair;
- byte-level Parquet tampering;
- a hash-rebound wrong model identity;
- a hash-rebound changed neural seed set;
- a changed/reordered feature schema;
- an interrupted multi-file publication.

## CLI and real-data path

Both concrete CLI handlers validate the experiment, correction event registry,
holiday calendar, and frozen model registry; audit the hourly source; independently
construct the selected B0/B1 matrices; calculate all five input hashes; and invoke
the shared cache/publication stage. No handler invokes the tuning boundary.

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

## Commits

- `f857e04 feat(hqrc-v3): freeze paper baseline factories`
- `4fad1ce feat(hqrc-v3): execute frozen paper baselines`

The protected workspace-local `hqrc_v3/uv.lock` was not staged or modified and
remains the only untracked path.
