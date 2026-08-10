# Task 10 report — reproducible experiment workflow

## Fix round 1: integrity tranche (not Task 10 completion)

- Replaced the permissive manifest with a strict version-2 schema that binds
  source audit, resolved config, event registry, standardized residuals, OOF
  predictions, approved AR calibration, posterior, metrics, and benchmark inputs
  to run-local digests.  It rejects unknown fields, unsafe paths, wrong types,
  invalid UTC timing, digest rewrites, and stale output maps.
- Reporting now removes a stale completion marker before any validation, reloads the
  approved AR artifact using the current residual/config/event hashes, opens the
  NetCDF InferenceData and checks sampler attributes, stages data-derived CSV,
  Parquet, and SVG output, writes output digests into the manifest, and finally
  writes a JSON `COMPLETE` marker bound to the final manifest digest.
- Synthetic coverage is now `prepare -> explicit external approval -> finalize`;
  it materializes actual prediction/residual/NetCDF artifacts and never approves
  AR calibration inside the pipeline.  Regression tests cover stale completion and
  an updated-manifest cross-run approval attack.
- `audit-data` and `diagnose-ar` are concrete.  The README now labels the still
  unwired correction/ablation/benchmark adapter boundary honestly.

Remaining for the next bounded tranche: process-isolated real PyMC/nutpie timing
and RSS/ESS/posterior-distance benchmarks; measured Arrow zero-copy and
parallel-vs-serial measurements; and serialized HQRCData/frozen-model loaders for
the correction, ablation, and sampler CLI routes.

## Adapter tranche foundation (still incomplete)

- Added `hqrc_v3.bayes.artifacts`: atomic, versioned NPZ plus strict JSON metadata
  for `HQRCData`.  Its file digest, metadata digest, exact array names, JSON-safe
  settings, occurrence IDs, `allow_pickle=False`, and reconstruction through
  `HQRCData` prevent detached or tampered worker input.
- `diagnose-ar` now compares the supplied residual SHA-256 to the bytes it reads and
  rejects residual split rows later than its explicit `--through` boundary.

This is only the safe-input foundation.  It does not yet provide the required
process-isolated sampler worker, measured Arrow/parallel benchmarks, or concrete
correction/ablation/benchmark CLI execution.

## Sampler-worker tranche

- Added canonical digest-bound sampler request/result JSON contracts and a parent
  runner that launches `sys.executable -m hqrc_v3.bayes.sampler_worker`, enforces a
  timeout, checks the exact child PID, and rejects nonzero, missing, malformed,
  mismatched, nonfinite, or stale results.
- The child reloads the hash-bound HQRC NPZ/metadata and approved AR artifact using
  the current residual/config/event hashes, executes the existing real sampler,
  measures child wall time and normalized peak RSS, revalidates inference diagnostics,
  and atomically emits per-element posterior means/SDs and software versions.
- Backend comparison runs required PyMC and only launches nutpie when installed. It
  computes the maximum per-element mean distance in pooled-SD units, audits every
  parameter element, and applies the existing strict eligibility gate. Zero pooled SD
  requires exact mean equality.
- RED: the focused worker test initially failed collection with
  `ModuleNotFoundError: hqrc_v3.bayes.benchmark`.
- GREEN: focused worker/report tests passed (`10 passed`), the non-slow suite passed
  (`236 passed, 3 deselected`), the real PyMC child-process smoke passed (`1 passed`),
  and Ruff plus `git diff --check` were clean.

Remaining Task 10 work is the measured Arrow/Polars and serial/parallel data
microbenchmark tranche plus production CLI wiring; those are intentionally not part of
this commit.

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

- `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/integration/test_cli_oof.py hqrc_v3/tests/integration/test_report_pipeline.py hqrc_v3/tests/integration/test_end_to_end.py -q` — 15 passed.
- `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests -m "not slow" -q` — 226 passed, 2 deselected (two pre-existing ArviZ runtime warnings).
- `uv run pytest -c hqrc_v3/pyproject.toml hqrc_v3/tests/slow/test_real_data_smoke.py -m slow -q` — 1 passed.
- `uv run ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests` and `git diff --check` — clean.
