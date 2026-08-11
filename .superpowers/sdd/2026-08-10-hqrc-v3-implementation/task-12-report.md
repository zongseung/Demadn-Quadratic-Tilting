# Task 12 report — causal feature schema and train-only preprocessing

## Outcome

Task 12 closes the three paper-run blockers found after Task 11 without changing
the frozen baseline architectures, hyperparameters, OOF/final years, AR code, or
correction models.

- B0 future is the exact seven-column known-calendar path. B1 appends seven
  physical columns for six holiday families and represents holiday type with
  mutually exclusive `is_seollal` / `is_chuseok` indicators.
- History is 168 hours of load, temperature, and humidity followed by the same
  feature-set calendar schema. B0 history/future widths are 10/7 and B1 widths
  are 17/14.
- Target-window realized weather, degree hours, and all `oracle_*` variables are
  absent. No historical weather-forecast vintage exists, so future weather is
  excluded from the main paper analysis rather than treated as known.
- The source holiday name/flag pair is preserved and audited at both hour and
  date level. The fixed source has 51,144 hours, 2,131 complete dates, 107 public
  holiday dates, and 14 substitute/temporary dates.
- A frozen three-row availability registry proves that the exceptional temporary
  holidays were announced before their dates. Its SHA-256 is part of cache and
  publication identity.
- The feature calendar has 14 rows, with 2018 Chuseok and 2025 Seollal used only
  as nearest-distance boundary support. The correction registry remains the ten
  2020--2024 events. Signed-distance ties deterministically choose the earlier
  center.
- 2024-10-01 is B1 public/temporary, has both holiday-type indicators zero, and
  is outside every HQRC correction window.

The split design remains the approved one. Four expanding OOF baseline fits
produce honest 2020--2023 HQRC training residuals. The unchanged baseline
specification is then refitted on the pre-2024 outer range for the sole causal
2024 evaluation. This is not duplicate evaluation: the OOF fits create training
targets for the correction layer, while the final refit uses all baseline data
available before deployment.

## Preprocessing and model contract

Every classical horizon estimator now receives the same complete design row:

```text
flatten(history[168, history_width]) + flatten(future[24, future_width])
```

One featurewise population standardizer is fit on estimator-fit daily rows and
one target standardizer on their unique 24-hour target values. XGBoost,
LightGBM, and RBF-SVR learn standardized targets, and every output is inverse-
transformed to MW. In particular, frozen SVR `epsilon=0.05` now means 0.05
training-target standard deviations rather than 0.05 MW.

Sequence models share the train-target scaler with history `load_mw`. Weather
statistics use each unique inferred observed-history timestamp once and reject
inconsistent duplicates. One calendar scaler is fit on unique future target
hours and applied to calendar channels in both history and future. Validation
and evaluation data cannot alter any scaler.

The frozen model TOML is schema version 2 and binds the complete preprocessing
policy. The baseline manifest is also version 2 and binds preprocessing identity,
exact feature order, scaler kinds, scaler population count/start/end, and the
temporary-holiday availability digest. Rewritten or omitted preprocessing
metadata, changed scaler populations, and changed or omitted availability
identity all fail closed on reload.

The fixed real matrix has 2,124 daily samples. OOF evaluation contributes 1,461;
the final outer training partition has 1,819 (1,758 estimator-fit plus 61
chronological validation for early-stopped models), and final January--October
2024 evaluation has 305. SVR uses all 1,819 outer-final training samples.

## Rust decision

No custom Rust crate was added. Polars already supplies the Rust-parallel data
engine, feature construction remains lazy until one packed matrix conversion,
and the required scikit-learn/LightGBM/PyTorch estimators consume dense
NumPy/Torch arrays. A new binding would add another ownership/conversion boundary
without removing the dominant 24-model training work. The real LightGBM smoke
confirmed that the current boundary is operational; Rust remains a measured
optimization option, not part of the statistical implementation.

## TDD evidence

Genuine RED was recorded before production edits.

1. The focused data/features/preprocessing collection command failed with
   `ModuleNotFoundError: hqrc_v3.baselines.preprocessing` because the reusable
   train-only preprocessing module did not exist.
2. Running data and feature tests without that missing module produced
   `8 failed, 9 passed`: source holiday fields were not preserved/audited, B0
   still contained 11 future columns including realized weather/degree hours,
   canonical B1 names and one-hot type columns were absent, and changing actual
   target weather changed the future feature matrix.
3. The first full non-slow run after implementation produced
   `1 failed, 372 passed, 7 deselected`. The only failure was a stale Task 2 test
   that still expected history to contain only the three observed channels. It
   was updated to freeze the approved exact B0 168x10 / 24x7 and B1 168x17 /
   24x14 column-order contracts.

Final GREEN evidence:

- Task 12 focused suite: `109 passed in 53.97s`; no warnings.
- Full non-slow suite: `375 passed, 7 deselected, 47 warnings in 69.22s`.
  The 47 warnings are the existing tiny-draw ArviZ/runtime diagnostic warnings
  from synthetic Bayesian smoke fixtures; Task 12 focused tests emitted none.
- Real causal matrix/LightGBM smoke: `1 passed in 3.44s`; no warnings. It
  traversed the 51,144-row source audit, B0/B1 construction, actual
  `lightgbm.sklearn.LGBMRegressor` 24-horizon fit at three smoke rounds, target
  inverse transformation, 8,784 finite MW predictions for OOF 2020, manifest
  population/hash checks, and a semantic cache reload. The OOF-2020 LightGBM
  scaler populations were 297 estimator-fit daily X rows and 7,128 unique target
  hours.
- Full Ruff: `All checks passed!` for `hqrc_v3/src` and `hqrc_v3/tests`.
- `uv lock --check --project hqrc_v3`: `Resolved 134 packages in 8ms`.
- `git diff --check`: clean; tracked repository-root `uv.lock` has no diff.

The managed sandbox denied `uv` access to its global source cache for one Ruff
invocation. Final test/Ruff commands therefore used the already locked root
`.venv` executables directly; the operator README and package tests continue to
require the self-contained `uv run --project hqrc_v3 --locked ...` commands.

## Protected file and remaining gates

The protected workspace-local `hqrc_v3/uv.lock` was never staged or modified. It
remains untracked and byte-identical at SHA-256
`f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

No full five-model paper computation was run. Task 12 requires a fresh independent
methodology review before that computation. AR(1) remains designed as approved,
but its Beta prior values and coefficient evidence still require causal OOF
residual ACF/PACF diagnostics and explicit user approval; Task 12 did not invent
or change those values.

## Commits

- `03d9b68 feat(hqrc-v3): enforce causal baseline preprocessing`
- `docs(hqrc-v3): report Task 12 verification` (this report/ledger commit)

## Fix round 1 — actual fitted preprocessing provenance

Fresh review found one important provenance defect in the initial Task 12
implementation. The paper stage reconstructed the expected scaler timestamp
populations from the matrix and published those values as though they had been
reported by the fitted adapters. Although each fitted preprocessor already held
its actual population, OOF/final orchestration discarded it and prediction-cache
metadata did not preserve it across a process restart.

The fix makes the provenance path explicit and fail-closed:

- `FittedBaseline` now requires `population_contract()`. Both the classical and
  PyTorch fitted adapters delegate this method to their fitted train-only
  preprocessors, and the returned count/start/end/unit schema is strictly
  canonicalized.
- OOF and final results carry the actual fitted population for every fold. Cache
  hits restore that population without fitting, rather than deriving a
  replacement from the input matrix.
- prediction-cache metadata is now exact schema version 2. It binds artifact
  hashes, actual population, and the Parquet SHA-256 under a combined entry
  digest. Missing population fields, legacy metadata, metadata mutation, and a
  rewritten Parquet file all fail closed.
- the paper stage first requires all five neural seeds for a
  model/feature/fold to report the same actual population, then compares that
  agreed value with the independently matrix-derived expectation. Only the
  verified actual value is persisted in the manifest. A publication interrupted
  before completion can therefore be rebuilt from stream-cache entries while
  retaining and rechecking fitted provenance.
- a direct import of `hqrc_v3.oof` exposed a package initialization cycle; paper
  orchestration exports are now loaded lazily, so the public OOF module is
  independently importable.

Genuine RED evidence was recorded before the production changes:

1. `tests/unit/test_oof.py` produced `3 failed`: OOF/final results lacked actual
   population fields and cache metadata lacked `population_contract`.
2. The two new paper-stage regressions produced `2 failed`: neither a
   deliberately misreporting adapter nor inconsistent neural-seed populations
   were rejected.

Final GREEN evidence for fix round 1:

- amended cache/OOF suite: `32 passed in 1.19s`; after the final timestamp
  validator hardening, the expanded preprocessing/cache/OOF set was
  `38 passed in 1.20s`;
- original Task 12 focused suite: `111 passed in 55.57s`, no warnings;
- paper-stage integration suite: `52 passed in 54.46s`, no warnings;
- full non-slow suite: `383 passed, 7 deselected, 47 warnings in 70.20s`;
- real 51,144-row causal matrix/actual LightGBM smoke: `1 passed in 3.37s`.
  The smoke deleted the completed publication and successfully rebuilt it from
  the production stream cache, checking cache schema version 2, Parquet digest,
  actual population equality, and the republished manifest;
- full Ruff: `All checks passed!`; lock check resolved the unchanged 134-package
  lock in 9 ms; staged and working-tree diff checks were clean.

The 47 non-slow warnings are the same inherited tiny-draw ArviZ/runtime
diagnostic warnings described above. They are unrelated to this baseline
provenance fix and are explicitly deferred rather than hidden or repaired by
changing unrelated Bayesian fixtures.

The protected untracked `hqrc_v3/uv.lock` remains unstaged and byte-identical at
SHA-256 `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

Implementation commit: `1594873 fix(hqrc-v3): bind fitted preprocessing provenance`.
