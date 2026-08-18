# Task 2 Report: Portable LOEO and Posterior Publication

## Status

DONE_WITH_CONCERNS. The Task 2 implementation and all bounded affected-test batches are
green. The brief's single combined focused command exceeded a 30-minute hard bound without
emitting a test failure, and the full Windows collection gate is blocked by two pre-existing,
out-of-scope direct `resource` imports. Those concerns are recorded verbatim below.

## Implementation summary

- Added `TrustedDirectory`, `trusted_directory`, `guard_trusted_directory`,
  `atomic_write_bytes`, `replace_entry`, and `unlink_entry` to `publication_fs.py`.
- Kept POSIX publication descriptor-relative with `O_DIRECTORY` / `O_NOFOLLOW`, held
  directory identities, descriptor-relative stat/replace/unlink, and directory fsync.
- Added the spec-approved Windows trusted-local-workspace backend: raw parent traversal,
  symlink, and junction rejection; trusted-root containment; directory identity checks before
  and after critical writes; same-directory temporary files; writable file fsync; atomic
  replace; and owned-temporary cleanup.
- Changed `LOEOPublicationHandle.directory_fd` to
  `LOEOPublicationHandle.directory: TrustedDirectory` and routed LOEO/posterior/HQRC
  generation publication through the adapter.
- Removed the direct `fcntl` import. LOEO locking now keeps the historical persistent
  `.loeo-fold.lock` evidence inode and uses the approved cross-platform lock on a transient,
  trusted-root hashed guard path. Placing the transient guard at the trusted root also permits
  Windows namespace-swap hardening tests to rename an intermediate component.
- Preserved immutable-prefix, mismatch, COMPLETE, generation, hash, partial-evidence, and
  resume behavior. Publication payloads are staged and fsynced before the prewrite boundary;
  final-name insertion therefore cannot replace foreign evidence planted at that boundary.
- Ported posterior and primary serialization temporaries away from `/private/tmp`, and closed
  the validated COMPLETE handle before Windows unlink while retaining ownership validation.
- Adapted owned tests for forced Windows backend coverage, junction fallback, unavailable
  file-symlink/FIFO capability skips, portable fixture fsync, and persistent fixture lock
  evidence. Added an end-to-end forced-Windows fold test covering write, canonical load,
  namespace identity validation, and reuse.

## TDD evidence

### RED

Command (brief exact command):

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_publication_fs.py hqrc_v3/tests/unit/test_loeo_stage.py -q
```

Result: exit 1 during collection. `test_publication_fs.py` could not import
`atomic_write_bytes` / `TrustedDirectory`, and `_loeo_publication.py` failed with
`ModuleNotFoundError: No module named 'fcntl'`. This was recorded before production changes.

Additional real REDs found and fixed during bounded fail-fast runs:

- Input-only resume omitted persistent `.loeo-fold.lock` on Windows because Windows
  `filelock` removes its synchronization file on release.
- Primary COMPLETE withdrawal left COMPLETE behind because Windows denied unlink while the
  validated read handle remained open.
- Intermediate namespace rename failed because the transient lock was initially below the
  swapped component.
- Namespace-swap failures were wrapped as generic publication failure instead of preserving
  the namespace-changed contract.
- Publication boundary tests found no staged temporary because the first adapter version
  invoked the prewrite boundary before staging/fsync.
- The owned lock-hardening test used the POSIX-only `/private/tmp` path.

### GREEN and diagnostic commands

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_publication_fs.py -q
```

Result: `8 passed in 0.78s`.

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_hqrc_artifacts.py -q
```

Result: `20 passed in 5.44s`.

```powershell
uv run --project hqrc_v3 --locked pytest "hqrc_v3/tests/unit/test_loeo_primary.py::test_publication_time_semantic_mutation_never_leaves_complete[matrix-complete-prewrite]" -vv -x --durations=0
```

Result: `1 passed in 24.52s`; setup 10.51s, call 11.84s. This exact node is intrinsically
fixture-heavy, not hung.

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_primary.py -vv -x --durations=0 -k "postpublish_complete or interrupted_ordered or partial_matrix or final_name_insertion or training_psis"
```

Result: `8 passed, 2 skipped, 22 deselected in 207.96s`. Skips are unprivileged Windows file
symlink and unavailable FIFO cases.

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py -vv -x --durations=0 -k "fully_rehashed_checkpoint or recovery_preserves_verified or recovery_rejects_and_preserves or sampler_cores or paper_v2_jitter"
```

Result: `10 passed, 52 deselected in 325.69s`. The slowest real node was the fully rehashed
posterior semantic-mutation test at 65.80s call time.

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py -q -x -k "publication_class_uses or publication_primitive or windows_backend_fold"
```

Result: `7 passed, 2 skipped, 53 deselected in 110.67s`; skips are unavailable Windows file
symlink/FIFO capabilities. Directory junction cases ran.

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py -q -x -k "fold_lock_rejects or strict_diagnostic"
```

Result: `2 passed, 2 skipped, 58 deselected in 57.50s`.

The preceding boundary batch completed all ten boundary parameters before reaching the
then-POSIX-only lock test: four regular-file cases passed and six unavailable Windows file
symlink/FIFO cases skipped. After porting the lock test, the command above passed its remaining
supported cases.

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_primary.py::test_publication_time_semantic_mutation_never_leaves_complete -q -x
```

Result: `3 passed in 69.93s`.

Final compact post-refactor regression command:

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_publication_fs.py hqrc_v3/tests/unit/test_hqrc_artifacts.py hqrc_v3/tests/unit/test_loeo_stage.py::test_windows_backend_fold_publication_writes_loads_validates_and_reuses "hqrc_v3/tests/unit/test_loeo_stage.py::test_publication_primitive_never_replaces_foreign_target_inserted_at_prewrite[regular]" "hqrc_v3/tests/unit/test_loeo_stage.py::test_every_publication_boundary_preserves_foreign_target_inserted_at_prewrite[hqrc-generation-npz-prewrite-regular-0]" "hqrc_v3/tests/unit/test_loeo_primary.py::test_publication_time_semantic_mutation_never_leaves_complete[matrix-complete-prewrite]" -q
```

Result: `32 passed in 47.86s`.

### Brief focused gate concern

Command (brief exact command, run once after the controller's single-process ruling):

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_publication_fs.py hqrc_v3/tests/unit/test_hqrc_artifacts.py hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_primary.py -q
```

Result: exit 124 after `1804s` (30-minute hard bound), with no emitted test failure. The shell
timeout left child PIDs 7224, 8808, and 19776; their exact command lines were verified and only
that process tree was terminated. No Task 2 pytest/uv process remained afterward. Bounded
batches above cover the affected tests and demonstrate cumulative fixture/semantic-validation
cost rather than one hanging node, but the exact combined gate did not complete within its
bound.

### Full Windows collection gate concern

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests --collect-only -q
```

Result: `633 tests collected, 2 errors in 9.03s`. Task 2's direct `fcntl` collection error is
removed. The remaining collection errors are out of Task 2 scope:

- `tests/integration/test_data_benchmark.py` ->
  `hqrc_v3/evaluation/data_benchmark_worker.py: import resource`
- `tests/integration/test_sampler_worker.py` ->
  `hqrc_v3/bayes/sampler_worker.py: import resource`

Both fail on Windows with `ModuleNotFoundError: No module named 'resource'`. Per the task
ownership ruling, those modules were not changed.

### Static verification

```powershell
uv run --project hqrc_v3 --locked ruff check hqrc_v3/src/hqrc_v3/publication_fs.py hqrc_v3/src/hqrc_v3/_loeo_publication.py hqrc_v3/src/hqrc_v3/_loeo_primary_publication.py hqrc_v3/src/hqrc_v3/bayes/artifacts.py hqrc_v3/tests/unit/test_publication_fs.py hqrc_v3/tests/unit/test_hqrc_artifacts.py hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_primary.py
```

Result: `All checks passed!`.

```powershell
uv run --project hqrc_v3 --locked python -m compileall -q hqrc_v3/src/hqrc_v3/publication_fs.py hqrc_v3/src/hqrc_v3/_loeo_publication.py hqrc_v3/src/hqrc_v3/_loeo_primary_publication.py hqrc_v3/src/hqrc_v3/bayes/artifacts.py
```

Result: exit 0.

```powershell
git diff --check
```

Result: exit 0; only Windows LF-to-CRLF working-copy notices.

## Controller interruption and scratch cleanup

During an earlier combined run, controller inspection found three overlapping pytest/uv
process trees created by timed-out wrappers and terminated them after PID/command-line
verification. On resume, no matching process remained. The exact `.pytest-task2` scratch
directory created by the Task 2 `--basetemp` diagnostic was resolved and inspected, then
deleted with its explicit absolute path. It is not recoverable. No user files were removed.

## Files changed

- `hqrc_v3/src/hqrc_v3/publication_fs.py`
- `hqrc_v3/src/hqrc_v3/_loeo_publication.py`
- `hqrc_v3/src/hqrc_v3/_loeo_primary_publication.py`
- `hqrc_v3/src/hqrc_v3/bayes/artifacts.py`
- `hqrc_v3/tests/unit/test_publication_fs.py`
- `hqrc_v3/tests/unit/test_hqrc_artifacts.py`
- `hqrc_v3/tests/unit/test_loeo_stage.py`
- `hqrc_v3/tests/unit/test_loeo_primary.py`

## Self-review

- Confirmed no direct `fcntl` import, `/private/tmp`, or `directory_fd` remains in owned files.
- Confirmed Windows adapter paths reject raw `..`, links, and junctions via Task 1 leaf checks,
  then revalidate containment and final-directory identity around writes.
- Confirmed POSIX adapter operations retain held directory descriptors, no-follow opens,
  descriptor-relative stat/replace/unlink, and directory fsync.
- Confirmed existing-target replace failure preserves the old target and owned temporary
  cleanup checks the recorded device/inode before unlink.
- Confirmed prewrite boundary ordering stages and fsyncs the owned temporary before final-name
  publication, preserving foreign evidence and partial recovery semantics.
- Confirmed COMPLETE withdrawal validates ownership and closes the Windows handle before the
  guarded unlink.
- Confirmed generation and pointer publication remain separately hash-bound and cleanup does
  not mask an original namespace-identity failure.
- Confirmed test-only platform adaptations skip only unavailable file-symlink/FIFO primitives;
  Windows directory junction rejection and forced backend behavior execute.

## Concerns

1. The brief's exact combined focused gate exceeded 30 minutes on this Windows host, although
   all bounded affected-test batches and the final 32-test regression set passed.
2. Full Windows collection remains blocked by two out-of-scope `resource` imports listed
   above. Task 2's `fcntl` collection blocker is resolved.
3. Windows test setup still needs test-local substitution of Task 1 portable fsync and
   persistent lock-evidence behavior for older diagnostics/LOEO-AR fixture modules; changing
   those production modules would exceed Task 2 ownership.
