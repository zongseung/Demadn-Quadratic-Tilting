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

## Polars data-benchmark tranche

- Added a canonical, digest-bound request for an existing numeric Parquet artifact.
  The contract binds the current file SHA-256, unique selected columns, at least three
  repetitions, at least two parallel workers, the seed, and the fixed workload
  identity. Parent and child independently revalidate the request and current bytes.
- Added a minimal fresh-process worker. The parent sets `POLARS_MAX_THREADS` before
  launch; the worker imports Polars only after checking that environment, records
  `pl.thread_pool_size()`, and the parent verifies the exact spawned PID, timeout,
  exit status, canonical result, request/result digests, dimensions, timings, RSS,
  versions, and checksums.
- The worker reads and rechunks finite, null-free `Float64` Series and measures
  `to_numpy(allow_copy=False)` separately from explicit `np.array(..., copy=True)`.
  It requires every zero-copy view to be non-owning and read-only, every explicit copy
  to be owning and writeable, records deterministic allocated bytes, and rejects any
  path where Polars would need to copy or the byte checksums differ.
- Serial (`POLARS_MAX_THREADS=1`) and parallel (`N`) workers execute the identical
  deterministic lazy Polars numeric-summary workload. The published artifact records
  full wall-time distributions and medians, peak RSS, actual thread counts, dimensions,
  checksums, speed ratios, and Python/NumPy/Polars versions; no speedup threshold is
  imposed.
- The final canonical benchmark revalidates the supplied sampler benchmark's file
  SHA-256, schema, canonical bytes, embedded digest, and measured PyMC wall time. It
  derives explicit-copy and total data-processing shares solely from measured medians.
  The predeclared custom-Rust candidate gate requires both data-processing share
  `>= 20%` and explicit-copy share `>= 50%`. It states that Polars is already Rust and
  that no custom Rust implementation was made, regardless of candidacy.
- RED: the focused test initially failed collection with
  `ModuleNotFoundError: hqrc_v3.evaluation.data_benchmark`.
- GREEN: the focused integration suite passed (`16 passed`); the complete non-slow
  suite passed (`252 passed, 4 deselected`); the dedicated audited 51,144-row real-source
  serial/parallel smoke passed (`1 passed`); and repository-wide Ruff plus
  `git diff --check` were clean. Tests cover invalid arguments/hash/schema, noncanonical
  and tampered artifacts, forced-copy/null rejection, checksum mismatch, wrong PID,
  timeout, nonzero exit, malformed JSON, and recomputed-but-false derived shares.

Remaining Task 10 work is production CLI wiring and the separate correction/ablation/
fit-handler and manuscript execution paths. Those boundaries remain intentionally
unwired in this tranche.

## Fix round 2: integrity review findings

- The Polars workload checksum now serializes the computed aggregate cell values,
  rather than relying on workload metadata. Every value receives an explicit type tag;
  finite floats use exact hexadecimal encoding and null, NaN, signed infinity, integer,
  boolean, and string values have deterministic canonical representations. A regression
  changes only one aggregate value between simulated serial and parallel results and
  proves the checksums differ while independently created NaNs hash identically.
- `write_hqrc_data()` now prepares a hash-bound NPZ/JSON generation and holds a stable
  per-artifact `flock` exclusively across both atomic replacements. `load_hqrc_data()`
  holds the same lock shared across metadata parsing, NPZ digest verification, archive
  loading, and `HQRCData` reconstruction. The deterministic concurrency regression
  pauses writer A immediately after its NPZ replacement, observes writer B blocked on
  the exclusive lock and a reader blocked on the shared lock, then proves the reader
  sees one coherent generation, writer B publishes the final coherent pair, and no
  thread is stranded.
- `build_report()` now parses the manifest-bound sampler benchmark before publication.
  It requires exact canonical JSON, the complete top-level and per-backend schemas,
  sorted unique PyMC/nutpie rows, finite typed measurements, consistent optional-backend
  eligibility, and the declared Arrow/parallel/Rust fields. Malformed JSON, unknown
  schema, noncanonical bytes, and semantically tampered measurements remain rejected
  even when an attacker updates the manifest file hash.
- RED: the three focused files produced five intended failures: missing aggregate
  canonicalization, missing pair locking, and three accepted malformed/tampered sampler
  artifacts.
- GREEN: the focused suite passed (`28 passed`), the complete non-slow suite passed
  (`258 passed, 4 deselected`), and repository-wide Ruff plus `git diff --check` were
  clean. The warnings were the existing synthetic ArviZ diagnostic warnings.

## Fix round 3: crash-consistent publication and canonical sampler reporting

- Replaced cooperative NPZ/JSON pair locking with immutable version-2 generations and
  one canonical, digest-bound `*.current.json` pointer. Each generation NPZ and metadata
  file is fsynced before publication, the generation directory is fsynced before the
  pointer swap, and the containing directory is fsynced afterward. A logical-path read
  resolves exactly the pointed generation; direct return values identify the real
  immutable NPZ/JSON files used by sampler requests, so an older returned tuple remains
  coherent after another writer advances the pointer.
- The pointer loader rejects noncanonical bytes, unknown fields or versions, malformed
  generation identities, missing/tampered hashes, symlinked generation namespaces,
  missing pointers, absolute paths, traversal, and targets outside the exact generation
  directory. Legacy version-1 direct pairs remain readable, while new logical namespaces
  fail closed instead of silently falling back when their pointer is removed.
- Deterministic failure injection covers every publication boundary. Failures before the
  pointer swap leave the prior generation current; failures after the swap expose the
  complete new generation. A gated reader/writer regression proves readers remain
  nonblocking while an unpublished writer pauses and uses unconditional bounded thread
  cleanup.
- `bayes.benchmark.load_sampler_benchmark()` is now the single strict production loader
  shared by reporting and the Polars data benchmark. It checks canonical bytes, exact
  schema/version, embedded digest, backend identity/status, child PID/request/environment
  audit, positive timing/RSS/ESS/R-hat measurements, divergences, every posterior-audit
  row, recomputed row and maximum distances, and recomputed nutpie eligibility.
- `build_report()` now accepts the actual artifact emitted by
  `benchmark_sampler_processes()`. Integration coverage launches controlled PyMC and
  nutpie child processes, binds their production artifact into the run manifest, and
  completes reporting; a re-digested false eligibility claim is rejected. The synthetic
  `write_benchmark()` format is separately versioned as
  `legacy-smoke-sampler-benchmark`, digest-bound, canonical, and explicitly prohibited
  from paper reporting.
- RED: the focused artifact/report run produced `13 failed, 9 passed`, exercising stale
  returned tuples, all pointer boundaries, path traversal, nonblocking readers, injected
  worker commands, production artifact acceptance, and semantic eligibility tampering.
- GREEN: the focused artifact/sampler/data/report suite passed (`48 passed`), the full
  non-slow suite passed (`272 passed, 4 deselected`), and the real sampler child-process
  smoke passed (`1 passed`). Repository-wide Ruff and `git diff --check` were clean.

Baseline/correction CLI wiring remains outside this fix round and was not changed.

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
