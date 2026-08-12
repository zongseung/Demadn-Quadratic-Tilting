# Task 15D Step 4 implementation report

Status: FIX ROUND 1 IMPLEMENTATION AWAITING FRESH INDEPENDENT RE-REVIEW

Implementation commit: `28e7731`

Fix round 1 code/tests commit: `5650a41`

## Fix round 1 review response

The four Important review findings and the scale-provenance Minor are addressed without starting
Task 15 Step 5 or any full paper LOEO fit:

1. **Trusted approved-set wrapper.** Preparation now reloads the complete approved LOEO AR set
   from the physical source/publication, compares the supplied wrapper's complete public payload
   and loader token with that trusted reload, and stores/uses only the trusted reloaded set and
   calibration. Retained-token `dataclasses.replace` substitutions of approved paths/hashes,
   training/proposal/plot hashes, proposal digests, or calibrations all fail closed.
2. **Central H3 posterior semantics.** `_loeo_posterior.py` now owns checkpoint validation for the
   exact nine-event H3 posterior. It requires the model variables and exact trailing shapes,
   finite samples, sampler-declared chain/draw sizes, positive scales and Cholesky diagonals,
   lower-triangular full-covariance factors, `u_phi`/`phi` support, exact
   `phi = 2*u_phi - 1`, and a nine-event log-likelihood vector. Checkpoint write, recovery, and
   complete reuse all pass through the same validator. A fully rehashed NetCDF/product/manifest
   mutation of `phi` is rejected semantically.
3. **Immutable ordered recovery.** Recovery recognizes only the ordered verified prefix
   `hourly -> metrics -> posterior summary -> manifest`. Existing Parquets must match regenerated
   schema and values and JSON must be canonical and exactly equal. The entire prefix is validated
   before mutation; verified files are skipped rather than rewritten, preserving their inodes.
   Foreign bytes, gaps, and manifest-only evidence fail closed and remain byte/inode-identical.
4. **Race-resistant namespace and lock.** Namespace creation walks canonical components below an
   absolute trusted output root with directory FDs plus `O_DIRECTORY|O_NOFOLLOW`. A namespace
   records trusted-root and final `(device,inode)` identities. `fold_lock` re-walks the complete
   no-follow chain, holds the final-directory FD for the full lock/publication lifetime, opens the
   lock relative to that FD with `O_NOFOLLOW|O_NONBLOCK`, and accepts only an `fstat`-verified
   regular file. Publication guards compare the held final FD with a fresh trusted-root-relative
   no-follow walk before writes and before/after injectable boundaries. Dangling lock symlinks,
   FIFOs, and a non-regular special-node substitute reject without escape or blocking.
5. **Evaluation versus scale-source identity (Minor).** Inputs and manifest identity now record
   `evaluation_split_id` and `scale_source_split_id` separately. For 2020--2023 they are the same
   OOF split; for either 2024 event they are respectively `final-2024` and `oof-2023`.

The operating assumption is an absolute, user-controlled local `output_root` whose ownership is
trusted. Under that model, the held final FD, repeated no-follow re-walks, and before/after boundary
guards close the reviewed intermediate-component swap threat. Existing multi-step HQRCData helper
I/O is guarded before entry and immediately after its publication boundary; this report does not
claim safety against a privileged hostile actor continuously replacing a trusted root inside an
individual helper syscall sequence.

### Fix-round RED evidence

Tests were added before the production fixes. The approval-wrapper/scale group was genuinely RED:

```text
8 failed, 6 deselected in 57.22s
```

The posterior/recovery/namespace group was genuinely RED:

```text
9 failed, 13 deselected in 148.51s
```

Those failures included accepted retained-token provenance substitutions; absent 2024
evaluation/scale-source identity; accepted fully rehashed `phi` mutation; blind overwrite of all
four valid recovery-prefix boundaries; acceptance of foreign/gapped/manifest-only evidence; and
following an intermediate namespace symlink. Direct lock probes additionally showed a dangling
symlink could create its external target and a FIFO could block. AF_UNIX socket creation is denied
by this macOS sandbox, so the non-regular-node branch is exercised with a directory while the FIFO
test separately proves nonblocking behavior. The child process records only lock-call monotonic
time (`<1s`), with a 10-second parent startup/import safety timeout.

The later self-review discovered a second TOCTOU window after the first full verification: a
namespace could be safely created, then an intermediate component could be replaced before
`fold_lock` re-opened the Path. The exact post-namespace swap test was RED because the fit did not
raise and wrote through the replacement. A second injected swap at `hqrc-data-published` reached
the sampler and then surfaced a raw `FileNotFoundError` rather than a guarded namespace error:

```text
test_fit_rejects_intermediate_swap_after_namespace_without_touching_outside
FAILED: DID NOT RAISE (1 failed in 27.84s)

test_fit_rejects_intermediate_swap_at_boundary_before_sampler_or_outside_write
FAILED: raw FileNotFoundError after the boundary (1 failed in 27.46s)
```

After the held-FD/re-walk guards, both exact race regressions passed. The first proves sampler zero
and an unchanged outside snapshot; the second proves an immediate typed rejection after the hook
and no post-swap write into the outside tree:

```text
2 passed in 33.92s
```

### Final post-guard verification

All earlier suite results preceded the final TOCTOU guard and are retained below only as
chronology. Every acceptance layer was rerun after that code change; these are the final results:

```text
# focused one-fold module, including the two TOCTOU regressions
27 passed in 401.97s

# impacted LOEO/AR/HQRC adjacent files
136 passed, 1 deselected, 2 warnings in 183.79s

# actual XGBoost-B1, two-chain 3-tune/3-draw PyMC fit and semantic reuse
1 passed, 3 warnings in 185.12s

# complete non-slow suite
572 passed, 9 deselected, 77 warnings in 669.78s
```

The actual smoke again compared complete relative-path/file-hash maps for the source, LOEO,
approved-AR, and original paper trees before and after the fit/reuse; all four maps were identical.
Its only writes were pytest output under `/private/tmp`. The warnings are the expected tiny-draw
ArviZ/all-NaN diagnostic warnings, not failures.

Final post-import-cleanup and static evidence:

```text
6 passed, 21 deselected in 41.60s
.venv/bin/ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests
All checks passed!
.venv/bin/ruff format --check --config hqrc_v3/pyproject.toml <6 changed source/test files>
6 files already formatted
git diff --check
(no output, exit 0)
openssl dgst -sha256 hqrc_v3/uv.lock
f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657
```

The protected nested lock remains an intentionally untracked 127-byte file with its original
`2026-08-10T21:28:30+0900` mtime. The code commit staged exactly six source/test files; its
post-commit status contained only `?? hqrc_v3/uv.lock`.

## Scope delivered

- Added typed `prepare_loeo_fold_inputs`, `fit_loeo_fold`, and
  `load_loeo_fold_result` operations for exactly one retrospective LOEO fold. There is no matrix,
  CLI, ablation, baseline fit, or AR proposal/approval operation in this tranche.
- Preparation accepts only the Task 15A/B/C trusted wrappers. It reloads and semantically
  validates the complete physical LOEO publication, calls
  `ApprovedLOEOARSet.calibration_for(held_out)` (which revalidates the complete approved set),
  loads the physical nine-event training fold, and re-reads the held-out event from the validated
  universe. The resulting `HQRCData` contains exactly nine contiguous event-reset segments and no
  held-out row.
- The fit is exactly H3, partial pooling, full covariance, restriction included, non-centered
  cyclic-hour RW1, Gaussian event-reset AR(1), `adapt_diag`, and PyMC. The approved fold-specific
  Beta prior remains live: `u_phi` is a free sampled RV and posterior `phi=2*u_phi-1` is retained;
  phi is neither fixed nor replaced with an ACF plug-in.
- Evaluation uses the unique source-frozen `sigma_n_mw` attached to the held-out physical frame.
  Callers cannot supply a scale or an outcome-derived substitute. Both 2024 events are evaluated
  on `final-2024` rows but use the pre-existing `oof-2023` scale frozen before correction fitting.
- Held-out point predictions are exactly `baseline + sigma_eval * posterior_mean(q)`. Predictive
  draws are exactly `baseline + sigma_eval * (q + e)`, with one stationary AR(1) simulation whose
  state starts at the first held-out hour. No baseline bootstrap, residual resampling, or second
  noise term is present.
- Published fold-local products contain the 144-hour held-out frame, point and probabilistic
  metrics, and a compact posterior-phi summary. Full posterior draws remain only in the immutable
  NetCDF checkpoint. HQRCData settings, NetCDF attrs, posterior checkpoint, hourly/metric rows,
  posterior summary, manifest/identity, and COMPLETE marker all record `causal=false`.
- Identity binds the complete source paths/hashes/manifests/registry, current LOEO
  universe/fold/manifest and event populations, proposal/approved-set/per-fold approval hashes,
  evaluation scale, exact model/geometry/init, profile, root/derived RNG seeds, sampler sizes,
  target acceptance, and `causal=false`. Seeds use canonical SHA-256 labels, never Python
  `hash()`.
- Publication uses atomic/fsynced HQRCData, posterior checkpoint, Parquets, JSON, manifest, and
  COMPLETE boundaries. Complete reuse semantically regenerates and compares products and invokes
  the sampler zero times. A valid checkpoint may recover only downstream products; invalid,
  foreign, partial, rehashed, aliased, symlink, special-file, or unknown evidence fails closed and
  is preserved.

## Genuine RED and GREEN evidence

The focused test module was created before production code. The first collection was genuine RED:

```text
.venv/bin/pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests/unit/test_loeo_stage.py -q

E   ModuleNotFoundError: No module named 'hqrc_v3.loeo_stage'
1 error in 0.04s
```

An initial `uv run --project hqrc_v3 ...` attempt could not access the sandbox-external uv cache
at `~/.cache/uv/sdists-v9/.git`; it did not sync, lock, install, or modify either lockfile. All
recorded test evidence therefore uses the already-provisioned workspace `.venv`.

Final focused GREEN after artifact-wide `causal=false` hardening and import-only Ruff cleanup:

```text
PYTENSOR_FLAGS=base_compiledir=/private/tmp/hqrc-v3-task15d-pytensor \
  .venv/bin/pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests/unit/test_loeo_stage.py -q

10 passed in 75.87s
```

The focused coverage includes trusted-wrapper/API and stable-seed contracts; exact nine-event
training; contiguous event-reset segments; representative OOF and both 2024 evaluation scales;
raw/unapproved/incomplete approval rejection; actual PyMC graph variables/options; held-out
restriction; exact point/predictive formula and no-double-noise regression; one AR reset;
checkpoint/product crash recovery; zero-sampler reuse/load; namespace separation; strict paper
diagnostic failure; unknown/symlink artifacts; and self-rehashed semantic product mutation.

## Adjacent and full verification

The Task 8/14/15 adjacent set covered HQRC graph/artifacts/predictive/sampling, correction source
and stage, and LOEO publication/AR/stage modules. Its first default-cache run produced 211 passes
and two environment-only failures because PyTensor attempted to create its compiler cache outside
the sandbox. Re-running precisely those two tests with the temporary compiler directory passed:

```text
2 passed, 24 warnings in 105.67s
```

The complete non-slow suite then passed with the same temporary PyTensor cache:

```text
PYTENSOR_FLAGS=base_compiledir=/private/tmp/hqrc-v3-task15d-pytensor \
  .venv/bin/pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests -m 'not slow' -q

555 passed, 9 deselected, 77 warnings in 250.46s
```

Final static and protected-file checks after the import-only amend:

```text
.venv/bin/ruff check --config hqrc_v3/pyproject.toml hqrc_v3/src hqrc_v3/tests
All checks passed!

.venv/bin/ruff format --check --config hqrc_v3/pyproject.toml <8 changed files>
8 files already formatted

git diff --check
(no output, exit 0)

openssl dgst -sha256 \
  /Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/.worktrees/hqrc-v3/hqrc_v3/uv.lock
f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657
```

The protected lock is the nested, intentionally untracked 127-byte
`.../.worktrees/hqrc-v3/hqrc_v3/uv.lock` (mtime 2026-08-10 21:28:30 +0900). An intermediate status
message incorrectly reported `c6858ef...` because it hashed the distinct tracked 467,230-byte
worktree-root `.../.worktrees/hqrc-v3/uv.lock`. Absolute-path revalidation established that the
protected nested lock remained exactly `f07f294...`; neither file was modified, restored, deleted,
synced, or relocked. Final status after the code commit contained only `?? hqrc_v3/uv.lock`; no
generated result or compiled-cache path was staged or untracked. PyTensor and real-smoke outputs
were directed to `/private/tmp`.

## Actual reduced PyMC smoke

The final slow test used the reviewed actual inputs:

- source: `/tmp/hqrc-source-preflight.lKh7Yt`
- LOEO: `/private/tmp/hqrc-v3-task15b-real-check`
- approved AR set: `/private/tmp/hqrc-v3-task15c-fix1-real.6DHoD4`
- proposal SHA: `db54721c0899f29312abe3d47d08977b33e02f53315d629c17fd319f6bfa83cd`
- approved SHA: `cb385e3a5a261819612d333b05c287462b99f790767f0aadcc4f795aae68581c`
- protected paper tree:
  `/Users/ijongseung/Documents/GitHub/arima-type/Demadn-Quadratic-Tilting/artifacts/hqrc-v3-paper-20260811`

It held out `seollal-2024`, fit the other nine events (1,152 rows), evaluated 144 held-out hours,
and used source-frozen `sigma_eval=2290.0780128512224`. The held-out id was absent from both
`HQRCData.occurrence_ids` and the approved calibration event ids. Root seed `20260812` produced
sampler seed `1065505343` and predictive seed `1030185397`.

```text
PYTENSOR_FLAGS=base_compiledir=/private/tmp/hqrc-v3-task15d-pytensor \
HQRC_V3_REAL_RUN_DIR=/tmp/hqrc-source-preflight.lKh7Yt \
HQRC_V3_REAL_LOEO_DIR=/private/tmp/hqrc-v3-task15b-real-check \
HQRC_V3_REAL_LOEO_AR_DIR=/private/tmp/hqrc-v3-task15c-fix1-real.6DHoD4 \
HQRC_V3_REAL_PAPER_ARTIFACT_DIR=.../artifacts/hqrc-v3-paper-20260811 \
  .venv/bin/pytest -c hqrc_v3/pyproject.toml \
  hqrc_v3/tests/slow/test_real_loeo_fold.py -m slow \
  --basetemp=/private/tmp/hqrc-v3-task15d-real-smoke-protected-final -vv

1 passed, 3 warnings in 27.07s
```

The first real-smoke attempt failed before sampling because the test assumed 1,164/132 physical
rows. Inspection of the already-validated physical artifacts showed the correct nine-event/
held-out counts were 1,152/144; only those test expectations were corrected. The final test
traversed the actual Task 14 PyMC model with two chains, three tune iterations, and three retained
draws, then repeated the exact fit call and asserted `reused=true`, `sampler_fit_count=0`. This is
an execution smoke only, not a paper diagnostic result or evidence about model quality.

The posterior retained six `u_phi`/`phi` draws with exact `phi=2*u_phi-1` equality and the approved
fold prior `a=36.71951713959426`, `b=3.2804828604057423`:

```text
phi mean       0.9268789011292848
phi SD         0.015783520960219885
phi 2.5%       0.9052437689054733
phi 97.5%      0.946028417662693
```

The one held-out metric row was `causal=false` and included RMSE 10192.2173, MAE 8393.6441, CRPS
5739.9233, 50% coverage 0.2361, and 90% coverage 0.7153. These values are deliberately not treated
as paper estimates because the smoke retains only six posterior draws.

The output namespace was:

```text
/private/tmp/hqrc-v3-task15d-real-smoke-protected-final/
  test_real_xgboost_b1_one_fold_0/loeo-h3/xgboost/B1/seed-7/
  seollal-2024/smoke/
  identity-0c2718df28766544bcf6fda773799b1759d6de7c59ca5068874325b20e9b2433
```

Key output SHA-256 values:

```text
COMPLETE                    cb8913f6d3247fad6c0a4e6cb6f07784a2a8b66c9a1d41caaf2a9c09ac425b20
hourly_predictions.parquet  43310afa2e715320ee9d7097c93ad20ae1a5281691888f0964e55e47c6cdd346
manifest.json               1851bb2155cd4f90635ccb17ea5a56e97e7d4abf4a2df9519adf69357bb1814c
metrics.parquet              32868e272f203d6951103d16446eb5fa48d96d9bd3069511e897b1d5fd5cae41
posterior.checkpoint.json    632aede4bbddb32ec7cfa547f6f7dac6c6832b733a73519e96cbac0a6ef09c83
posterior.nc                 86c31b87ef7d8724c1e1facf63787aa7908375b424e9d0c094e50a05f49ccf47
posterior_summary.json       d7b464da23c39370194983a5f510d12b9d3ab48a660743147cf71bdf2fa30022
```

The slow test captured each protected tree as a complete relative-path-to-file-SHA map before the
fit and asserted exact map equality afterward. SHA-256 over a canonical JSON rendering of each
unchanged map was:

| protected tree | before | after |
|---|---|---|
| source | `f0390150d444c04f6f811cad3528881b5ef87257422be089400f2e176da5bab1` | same |
| LOEO | `cd2fcb96c9523a5dda9d3f8abc960285c9ead834c6e92618d3dedaca0e4bccf8` | same |
| approved AR | `9cf0fe502ddd2ae7ab0f12cab05597b2c6f526e44e4c486da9e8ded362b7140c` | same |
| original paper artifacts | `78a8e0278ed7e4e86cc8db003a3ca6bbf558b45de1c13fe964bbdb3239bae9a1` | same |

## Self-review and concerns

- The no-double-noise test replaces the new-event coefficient draw and the sole AR simulator with
  known arrays, captures the arguments to the reviewed `corrected_predictive_draws`, and proves
  exact point and predictive values. It also asserts exactly one AR simulator call over the
  held-out horizon. Production imports no baseline bootstrap/residual draw helper.
- The safe held-out loader intentionally revalidates the complete universe and all ten physical
  folds before filtering one event; it does not expose a cached unchecked frame.
- `phi` wording is precise: the constrained posterior parameter is a deterministic transform of
  sampled free `u_phi`, so its posterior is estimated under the approved Beta prior rather than
  fixed. The actual model and NetCDF tests check both variables and the exact transform.
- The 717-line private publication module is the largest new unit. It is cohesive around identity,
  atomic checkpoint/product state, and semantic reuse, while public orchestration and deterministic
  products are separate. This is a minor maintainability point for whole-branch review, not a
  known correctness defect.
- No unresolved Critical/Important concern is known. Independent review should focus on the
  publication/recovery state machine, identity completeness, and the physical scale provenance.

## Files changed

- `hqrc_v3/src/hqrc_v3/diagnostics/loeo.py`
- `hqrc_v3/src/hqrc_v3/_loeo_contract.py`
- `hqrc_v3/src/hqrc_v3/_loeo_products.py`
- `hqrc_v3/src/hqrc_v3/_loeo_publication.py`
- `hqrc_v3/src/hqrc_v3/_loeo_types.py`
- `hqrc_v3/src/hqrc_v3/loeo_stage.py`
- `hqrc_v3/tests/unit/test_loeo_stage.py`
- `hqrc_v3/tests/slow/test_real_loeo_fold.py`
- this report and the progress-ledger line below

Task 15 Step 4 remains unchecked pending fresh independent review.
