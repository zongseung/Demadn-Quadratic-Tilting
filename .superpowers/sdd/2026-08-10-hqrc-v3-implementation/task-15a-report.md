# Task 15A Step 1 implementation report

Status: READY FOR INDEPENDENT REVIEW

Implementation commit:

- `78e2c6b` (`refactor(hqrc-v3): validate correction sources once`)

## Scope delivered

- Added `correction_source.py` with the frozen, slot-based
  `ValidatedCorrectionSource`. It stores no mutable in-memory DataFrame, exposes detached manifest
  copies, wraps source/hash maps in read-only proxies, and re-reads hash-bound Parquet streams for
  each requested context.
- Moved the common raw-source/config/calendar reconstruction, residual/baseline manifest
  verification, source-derived OOF residual reconstruction, final-baseline validation/reuse, and
  source-derived final target audit out of `correction_stage.py`.
- The preflight calls the public final baseline stage and requires `fit_count == 0`. It binds the
  exact canonical OOF member/point and final member/point paths and hashes, requires the OOF and
  final context populations to equal the residual publication, and rejects source/config path
  substitution.
- Canonical source namespaces contain no unknown entries or symlinks. The requested config, all
  six source files, both publication directories, and all bound files use no-follow type checks.
  Context loaders re-check the namespaces and artifact hashes, closing post-validation mutation.
- `ValidatedCorrectionSource` exposes exact standardized-OOF, baseline-OOF point, and final-2024
  point stream loaders. A model/feature/seed context absent from the validated publication fails
  before any frame is returned.
- `prepare_causal_correction_inputs` is now only a Task 14 adapter over the common source object.
  It still loads the approval after the residual digest is known, selects the same approved
  context, builds the same eight-event `HQRCData`, applies the same `oof-2023` scale, and builds
  the same two causal-2024 prediction segments.
- No LOEO universe, fold-specific AR proposal/approval set, LOEO sampler, functional variant, or
  pooling implementation was added.

## Genuine RED/GREEN evidence

Initial RED, before production code existed:

```text
uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests/unit/test_correction_source.py -q
```

Result: collection failed with
`ModuleNotFoundError: No module named 'hqrc_v3.correction_source'`.

The first implementation made seven source-contract tests pass. A self-review then added two
additional defect reproductions before their fixes:

- a requested experiment-config symlink to the correct file was accepted;
- an unknown publication entry added after source validation was ignored by a context loader.

RED result: `2 failed, 7 passed`. After adding no-follow requested-config validation and
namespace revalidation at every stream load, the same suite was GREEN: `9 passed`.

## Regression and real-source evidence

- Task 14 approval/source/stage/CLI focused suite:
  `57 passed, 30 expected tiny-draw warnings in 4.72s`.
- Full non-slow suite on the final code:
  `452 passed, 8 deselected, 77 existing tiny-draw warnings in 65.62s`.
- Full project Ruff: `All checks passed!`.
- `git diff --check`: clean.
- Protected untracked `hqrc_v3/uv.lock` remains byte-identical at
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

The completed real paper source publication was copied to
`/tmp/hqrc-source-preflight.lKh7Yt`; validation and lock activity occurred only in that temporary
copy. The common preflight completed without fitting, preserving source profile `paper`, all ten
contexts, residual SHA
`895767248bccd9f0326e0373a5db5b947803e45c77ae5928267b53a1234171d2`, and final-point SHA
`87f042da96ae16430288aeb6287657f6ff419db8b3c1b8e3ee111ba8fdd399e3`.

The refactored Task 14 adapter on the temporary copy and the existing XGBoost-B1 approval returned
the same contract facts: eight ordered training events / 1,032 rows, two ordered evaluation
events / 264 rows, and `oof-2023` scale `2290.0780128512224` MW. Existing Task 14 tests continued
to prove event-only prediction semantics, deterministic namespace identity, posterior-checkpoint
resume, completed-result reuse, and rejection of rehashed semantic products.

## Boundaries and concerns

- No PyMC smoke or paper fit was run. No original paper artifact was written or republished.
- The exact `inputs/` and `predictions/` namespaces are now part of the source trust boundary.
  Task 15 downstream universe/fold products must therefore live in their own output namespace,
  rather than being placed beside the canonical residual or baseline publications.
- This step intentionally retains Task 14's public causal API and product namespace; the new
  abstraction is ready for the later ten-event universe and LOEO stages but does not implement
  them.

## Independent-review fix round 1 — complete split-context identity

Review finding: `ValidatedCorrectionSource` stored and checked availability only as
`(model, feature_set, seed)`. A caller could therefore supply an `EventResidualContext` with a
subset, reordered list, duplicate, or substitution in `split_ids`; the standardized and final
loaders ignored the change, while the OOF loader caught only some mutations incidentally after
frame selection.

Genuine RED was recorded before the production fix. The amended test matrix passes all four
split mutations to each of the standardized-residual, OOF-point, and final-point loaders. It also
requires the source to expose the complete manifest-derived `EventResidualContext`, rather than
the former three-field tuple.

```text
uv run --project hqrc_v3 --locked pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests/unit/test_correction_source.py -q
```

RED result: `12 failed, 9 passed`. The failures showed all four mutations bypassing the
standardized/final loaders, reordered and duplicate identities bypassing the OOF loader, and the
source retaining only a partial context key.

Fix commit: `bdc46ea` (`fix(hqrc-v3): bind correction split context`).

- `available_contexts` now contains complete immutable `EventResidualContext` values parsed from
  the validated residual manifest.
- Manifest and requested contexts must use a non-empty, unique, canonically ordered subsequence
  of the four immutable OOF split IDs.
- Every loader requires exact equality with one complete available context before reading or
  filtering any Parquet stream. Model/feature/seed equality alone is insufficient.
- Duplicate manifest records are rejected both by complete context and by
  model/feature/seed key, so different split populations cannot masquerade as separate contexts.

Final evidence on the fix:

- amended source suite: `21 passed in 1.03s`;
- Task 14 approval/source/stage/CLI focused suite: `69 passed, 30 expected warnings in 4.85s`;
- full non-slow suite: `464 passed, 8 deselected, 77 existing warnings in 65.23s`;
- full Ruff and `git diff --check`: clean;
- protected nested lock remains
  `f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

The no-fit Task 14 adapter was rechecked against the existing temporary paper-publication copy
and XGBoost-B1 approval. It retained the exact approved split context
`oof-2020`--`oof-2023`, source profile `paper`, 1,032 training rows, 264 event rows, and scale
`2290.0780128512224` MW. No original artifact or paper fit was touched.
