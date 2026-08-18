# Task 15E Step 5 implementation report

Status: IMPLEMENTATION AWAITING FRESH INDEPENDENT REVIEW

Code/tests commit: `63f956e`

Fix round 1 code/tests commit: `72b69bf`

## Fix round 1 review response

The reproduced Critical and two Important findings are fixed without changing the H3 statistical
contract, starting the ten-fold paper fit, or implementing later Task 15 steps.

1. **Aggregate completion now has validation barriers on both sides.** Before `COMPLETE`, the
   held-dirfd reader revalidates every product's exact semantics, all hashes, manifest identity,
   rows, and persisted fold execution. After exclusive no-replace `COMPLETE` publication it runs
   the full completed-publication validator again. A failure removes only the exact newly owned
   regular COMPLETE inode; a substituted foreign entry is preserved and the call fails closed.
2. **Paper preflight reloads all physical inputs before the first fold call.** The shared Task 15D
   source validator completely reloads the current ten-fold LOEO universe/folds and approved AR
   set. Missing/substituted evidence, plus lower-level `LOEOError`, `TypeError`, and `ValueError`,
   become public `LOEOPrimaryError` with zero `fit_loeo_fold` calls.
3. **Fit/reuse status is immutable provenance.** Schema version 2 manifests store one canonical
   `fold_execution` row per selected fold with `fit_status` and exact `sampler_fit_count`. Initial
   fit/recovery status is retained byte-for-byte across later zero-fit aggregate reuse instead of
   being replaced by transient reuse status.

### Fix round 1 RED and GREEN

Four direct regressions were RED before production changes:

```text
4 failed, 23 deselected in 70.16s
```

They proved a missing final physical paper fold still reached the first kernel, `fold_execution`
was absent, and both manifest/COMPLETE prewrite mutations returned a corrupt successful complete
result. Targeted post-fix GREEN was `4 passed, 23 deselected in 63.15s`; public lower-level error
wrapping then passed `3 passed, 28 deselected in 34.01s`.

Final focused coverage also includes mutation after hourly publication and after COMPLETE,
owned-COMPLETE withdrawal, foreign COMPLETE substitution preservation, schema-v2 status
validation, and byte-stable status reuse:

```text
31 passed in 281.34s
```

JUnit: `/private/tmp/hqrc-v3-task15e-r1-focused-final.xml`.

The original reviewer vulnerability reproducer now raises at each formerly successful mutation
boundary and blocks the paper kernel before its old vulnerable assertions can hold. No corrupt
owned `COMPLETE` remains.

### Fix round 1 final verification

```text
adjacent Task 15B/C/D + metrics: 166 passed in 918.46s
actual XGBoost-B1 two-fold smoke: 1 passed, 124 warnings in 188.01s
full non-slow: 632 passed, 10 deselected, 77 warnings in 1019.14s
```

JUnit files are `/private/tmp/hqrc-v3-task15e-r1-adjacent.xml`,
`/private/tmp/hqrc-v3-task15e-r1-real.xml`, and
`/private/tmp/hqrc-v3-task15e-r1-full.xml`. The actual run again used only
`seollal-2024` and `chuseok-2024` under `/private/tmp`, performed real fits and immutable reuse,
and retained the slow test's exact before/after protected-tree equality assertions. This is not a
paper estimate.

The actual schema-v2 matrix identity was
`a91952030657600f72ade6b1a84cd1d802d398507f086bcfca0e04f04b36936e`; manifest digest
`b49d4b4f101036cbfe5c403c81d32f95f6d8bfe7338dc311b3f813c396f28cf7`. It records both selected
folds as `fit` with sampler-fit count 1; the second aggregate call returned zero transient fits
while preserving those manifest bytes.

Final full Ruff passed, all five changed source/test files were already formatted,
`git diff --check` passed, and the aggregate modules contain zero rename/replace calls. The only
target unlink is held-dirfd relative and requires exact `(device,inode)` ownership of the newly
published regular COMPLETE. The protected untracked nested `hqrc_v3/uv.lock` remains 127 bytes,
inode `97849780`, mtime `2026-08-10T21:28:30+0900`, SHA-256
`f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

Task 15 Step 5 remains unchecked pending fresh independent re-review. No full ten-fold paper
matrix, Step 6/7/8, CLI, graph, or performance publication was run or implemented.

## Scope and outcome

Implemented only the immutable primary H3 LOEO matrix. `fit_loeo_primary` composes the reviewed
Task 15D `fit_loeo_fold` kernel and its held-dirfd semantic loader; it does not clone the PyMC
model, implement H0--H5/pooling/testing matrices, add CLI wiring, or run the ten-fold paper fit.

Paper preflight accepts exactly all ten canonical occurrence ids, a paper source, and the paper
sampler contract before any fold call. Smoke preflight accepts only an explicit non-empty unique
canonical-order subset. Every selected fold receives the same matrix root seed; the reviewed
kernel derives and records distinct fold and predictive seeds.

The aggregate publication contains:

- `hourly_predictions.parquet`, preserving held-out rows and adding fold references and
  `causal=false`;
- `per_event_metrics.parquet` and `aggregate_metrics.parquet`, with point metrics recomputed from
  hourly observations/forecasts and probabilistic metrics weighted by physical held-out hours;
- compact `posterior_summaries.parquet`, with no posterior-draw duplication;
- `training_psis_loo.parquet`, explicitly labelled `training_nine_event_psis_loo`, requiring nine
  finite Pareto-k values per fold;
- an identity-bound `manifest.json` and `COMPLETE` marker.

Complete aggregate reuse re-invokes every fold kernel in zero-fit reuse mode and securely reloads
each current fold before aggregate validation. Fold failure leaves earlier immutable fold
checkpoints available but no aggregate `COMPLETE`.

## Genuine RED and GREEN

The focused test was written first against the absent production module. Initial RED:

```text
ModuleNotFoundError: No module named 'hqrc_v3.loeo_primary'
1 error in 0.04s
```

Final focused result:

```text
23 passed in 200.92s
```

JUnit: `/private/tmp/hqrc-v3-task15e-focused-final.xml`.

The focused suite covers paper/smoke preflight and zero calls, a fake ten-fold/1,296-row matrix,
canonical order and aggregation formulas, all `causal=false` fields, seed provenance, fold
references without draw copying, fold failure/retry, zero-fit inode-stable complete reuse,
current-fold revalidation, fully rehashed semantic mutation, ordered-prefix recovery with inode
preservation, unknown/gap/symlink/FIFO preservation, no-replace final insertion, and finite
nine-event PSIS validation.

## Verification

Adjacent Task 15B/C/D plus metrics/PSIS/correction coverage:

```text
166 passed, 30 warnings in 736.49s
```

JUnit: `/private/tmp/hqrc-v3-task15e-adjacent.xml`.

Final actual reduced smoke, with XGBoost/B1 seed 7, root seed `20260812`, and the explicit subset
`seollal-2024`, `chuseok-2024`:

```text
1 passed, 124 warnings in 268.09s
```

JUnit: `/private/tmp/hqrc-v3-task15e-real3.xml`. This used two chains, 50 draws, and 20 tune steps
per fold under `/private/tmp`. It traversed two real Task 15D fits, secure semantic loads, finite
PSIS, aggregate publication, and zero-fit/inode-stable reuse. The warnings are expected for this
small diagnostic smoke; its metrics are not paper estimates.

Final full non-slow suite:

```text
624 passed, 10 deselected, 77 warnings in 1393.50s
```

JUnit: `/private/tmp/hqrc-v3-task15e-full.xml`.

Static verification:

```text
ruff check hqrc_v3/src hqrc_v3/tests: All checks passed
ruff format --check <9 changed files>: 9 files already formatted
git diff --check: clean
Task 15E publication rename/replace scan: zero
```

The syscall audit confirms aggregate publication uses the reviewed fsynced temporary plus
exclusive `os.link` no-replace path, a held final-directory descriptor, `O_NOFOLLOW` reads, and a
regular nonblocking lock. Aggregate recovery validates an exact ordered prefix before adding a
new artifact and preserves foreign evidence.

## Actual subset identities and outputs

Aggregate namespace:

```text
/private/tmp/hqrc-v3-task15e-real3-run/test_real_xgboost_b1_two_fold_0/
  loeo-primary-h3/xgboost/B1/seed-7/smoke/
  selection-63ab5256942fd47d7dd769b88694571d2166837a47e245a033f9fd302af64a90/
  identity-dfd14753e99545e43e3afa8d4e41f1f701b8fe9a331a91b4f990bb5d9df63476
```

The matrix manifest digest is
`420b608fd3d3137989c42ece94c899a0e71752bb0e3ee1b06e10642589ad5759`.
Its approved-set identity is
`cb385e3a5a261819612d333b05c287462b99f790767f0aadcc4f795aae68581c` and its LOEO universe
identity is `7fa17b0b53c61c8678dacd8ce5621de6472c341603a86bc6b4a2016c3f39f16a`.

Fold evidence:

- `seollal-2024`: identity
  `57741ec00808c38d2c24c8c8819c3455c0b9ea97d18b4f3d687b3e6590ec81a2`, sampler seed
  `1065505343`, predictive seed `1030185397`, 144 held-out rows.
- `chuseok-2024`: identity
  `a11157ef3c920b3c13ede35e57896662361380348b6d4f62bfb71f8b1c18e3f3`, sampler seed
  `2111133628`, predictive seed `1967958863`, 120 held-out rows.

Output shapes and SHA-256 values:

| Artifact | Shape | SHA-256 |
| --- | ---: | --- |
| `hourly_predictions.parquet` | 264 x 26 | `f8f997a0f6c6312c2e3d77cfc7f4800cbacf8b891e83e92e9b48a9064ce7e0d1` |
| `per_event_metrics.parquet` | 2 x 41 | `a545f4d1d4707df50f809004f809e330c87e4de7e2f32dea01ee62ea19dd7d83` |
| `aggregate_metrics.parquet` | 3 x 32 | `0c6ea1829bd6c377d24185243da864a57f943bd3c16eb05d86d6a703c02e2c5f` |
| `posterior_summaries.parquet` | 2 x 14 | `933b2dcc7f1a61d3960a9046f5c585fe667b75655c53bac7f10680217692c95e` |
| `training_psis_loo.parquet` | 2 x 15 | `1a75b9736b03884ea689968cc0fe3f4c4f33fd58b2970102974e201448686559` |
| `manifest.json` | - | `400f8786d630f4e7b7dbc8bc87804af1a8369b367ff4076f1fb3e428a5fd2235` |
| `COMPLETE` | - | `57611413383329c5aae5b4752efdd45dfba2def45e765064ca72171aeb653588` |

The pooled smoke values were RMSE `5328.093761935238`, MAE `4021.0007828330185`, CRPS
`2828.9932024118098`, 50% coverage `0.5492424242424242`, and 90% coverage
`0.8977272727272727`. The PSIS maxima were `1.5527095367113977` and `1.6324805873806791`;
these are explicitly training diagnostics from a tiny smoke posterior, not held-out accuracy or
paper-quality estimates.

## Protected inputs

The actual test compared complete before/after regular-file hash maps and found exact equality.
Independent canonical map digests after the run were:

- source/original paper tree: 618 files,
  `cd74dcc47e4619028fd23c3b49bd559883a173c6239bf8ccb7d55bea70d24c27`;
- LOEO tree: 15 files,
  `d084552c2041743ac5e35a81fe312654b51b53bca2d8a9cd1fa46ad4dc19411e`;
- approved-AR tree: 58 files,
  `378c01a275ef854aa5baf22e4261737c2a6db1254fee11ebea9998416452511b`.

The untracked protected `hqrc_v3/uv.lock` remains 127 bytes, inode `97849780`, mtime
`2026-08-10T21:28:30+0900`, SHA-256
`f07f2944707750a9b0753690e6fca2d9c83dbf1d29340e3628483573a2766657`.

## Files and self-review

Added focused primary types, deterministic product generation, immutable publication, public
orchestration, unit acceptance coverage, and an opt-in actual-smoke test. The Task 15D changes are
limited to a typed securely loaded fold material, a generalized secure namespace/lock helper,
and a public loader; historical result-only behavior remains covered by the adjacent suite.

No Critical or Important concern remains in self-review. One maintainability concern is deferred:
the focused aggregate publisher reuses several reviewed underscored Task 15D held-dirfd helpers
instead of first promoting a larger shared public publication layer. This avoids duplicating or
weakening the security primitives, but those internal imports should be reconsidered during the
final whole-branch modularity review. The custom aggregate lock can also still emit Task 15D's
generic "fold" wording on lock errors; this does not affect locking semantics.

Task 15 Step 5 remains unchecked pending fresh independent review. No full ten-fold paper matrix
was run.
