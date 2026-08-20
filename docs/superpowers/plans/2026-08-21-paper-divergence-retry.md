# Paper Divergence Retry Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single provenance-safe paper-profile retry for divergence-only LOEO failures across H1, H2, and H3 without invalidating existing successful folds.

**Architecture:** Keep the zero-divergence gate and base sampler contract unchanged. Attach structured diagnostics to `SamplingError`, derive one immutable retry contract in the LOEO publication layer, then let H1/H2 and H3 orchestration resolve base or retry publications while recording the first failure in retry manifests.

**Tech Stack:** Python 3.12, PyMC/Pyro, ArviZ, pytest, Polars, immutable JSON/NetCDF publication contracts.

**Spec:** `docs/superpowers/specs/2026-08-21-paper-divergence-retry.md`

## Global Constraints

- The paper diagnostic gate remains zero divergences.
- Base paper identity remains byte-for-byte unchanged: target acceptance `0.99`, default tune/draws `1000`, four chains.
- Retry is paper-only, divergence-only, exactly once, target acceptance `0.999`, tune `max(2000, base_tune)`, with draws/chains/cores/backend/device/init unchanged.
- Retry seed is deterministic and distinct; retry provenance includes the base sampler digest and first failure diagnostics.
- Existing complete base folds are reused; no failed or divergent posterior is published.
- H1, H2, and H3 aggregates accept valid mixed base/retry fold contracts.
- No new dependency and no OS-specific branch.

---

### Task 1: Structured diagnostics and immutable retry sampler contract

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/bayes/samplers.py`
- Modify: `hqrc_v3/src/hqrc_v3/_loeo_publication.py`
- Test: `hqrc_v3/tests/integration/test_hqrc_sampling.py`
- Test: `hqrc_v3/tests/unit/test_loeo_stage.py`

**Interfaces:**
- Produces: `SamplingError.diagnostics: SamplingDiagnostics | None`.
- Produces: `retryable_divergence(error: BaseException) -> SamplingDiagnostics | None`.
- Produces: `retry_sampler_contract(base: Mapping[str, object], *, variant: str, held_out_occurrence_id: str) -> dict[str, object]`.
- Produces: common `retry_failure_payload(...)` and `validate_retry_failure(...)` helpers used by both H1/H2 and H3 manifests.
- The retry contract carries a stable `retry` object containing attempt `1`, reason `divergence-only`, and `base_sampler_sha256`.

- [ ] **Step 1: Write failing structured-error tests**

Add tests proving `validate_inference_data(..., paper_profile=True)` exposes the exact `SamplingDiagnostics` on `SamplingError`, and that missing diagnostics/non-divergence diagnostic failures are not retryable.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/integration/test_hqrc_sampling.py -k "diagnostic or target_accept" -q`

Expected: failure because `SamplingError` has no structured diagnostics and paper target acceptance `0.999` is rejected.

- [ ] **Step 3: Implement the minimal structured error behavior**

Extend `SamplingError` with an optional diagnostics attribute and raise it from the strict diagnostic gate as:

```python
raise SamplingError(str(diagnostics), diagnostics=diagnostics)
```

Allow an explicit paper target acceptance of either `0.99` or `0.999`; keep smoke fixed at `0.9` and paper's default at `0.99`.

- [ ] **Step 4: Write failing retry-contract tests**

Add literal assertions that a default paper base contract is unchanged and its retry contract has target acceptance `0.999`, tune `2000`, unchanged draws/chains/cores/backend/device/init, a distinct deterministically derived seed, and a stable base sampler digest. Assert smoke contracts and failures that are not divergence-only cannot enter retry.

- [ ] **Step 5: Run retry-contract tests and verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py -k "retry_sampler or paper_v2" -q`

Expected: failure because the retry contract does not exist.

- [ ] **Step 6: Implement the minimal retry contract**

Build the retry by copying the base contract and replacing only `seed`, `tune`, and `target_accept`, then add:

```python
"retry": {
    "attempt": 1,
    "reason": "divergence-only",
    "base_sampler_sha256": sha_json(base),
}
```

The seed label is `{variant}-fold-sampler-retry-1:{held_out_occurrence_id}`.
Add common builders/validators for the signed retry-failure record so H1/H2 and H3 do not duplicate its eligibility rules.

- [ ] **Step 7: Run focused and adjacent tests**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/integration/test_hqrc_sampling.py hqrc_v3/tests/unit/test_loeo_stage.py -q`

Expected: pass.

- [ ] **Step 8: Commit**

```text
feat: define paper divergence retry contract
```

---

### Task 2: H1/H2 fold retry, provenance, and reuse

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/loeo_ablation.py`
- Test: `hqrc_v3/tests/unit/test_loeo_ablation.py`

**Interfaces:**
- Consumes: Task 1 `SamplingError.diagnostics`, `retryable_divergence`, and `retry_sampler_contract`.
- Consumes: Task 1 retry-failure payload builder/validator.
- Produces: retry manifests with `retry_failure` containing the base sampler digest and literal diagnostic fields.
- Produces: `_fit_fold` behavior that prefers a valid base publication, then a valid retry publication, then samples base and at most one retry.

- [ ] **Step 1: Write failing H1/H2 behavior tests**

Add tests for: divergence-only failure followed by successful retry; exact retry arguments; first-failure diagnostics in the signed manifest; second call reusing the retry without sampling; base success remaining preferred; smoke and R-hat/ESS/non-diagnostic failures not retrying; retry failure propagating after exactly two attempts.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_ablation.py -k retry -q`

Expected: failure because `_fit_fold` currently makes one attempt and manifests have no retry provenance.

- [ ] **Step 3: Implement one-attempt orchestration**

Extract the existing single-sampler body into a private attempt helper. Compute stable base and retry directories without publishing either; reuse base first, retry second. Catch only `SamplingError`, call `retryable_divergence`, and invoke one retry only for an eligible paper failure.

- [ ] **Step 4: Record and validate retry provenance**

Allow `_publish_directory` to add a signed `retry_failure` only for retry identities. The record contains:

```python
{
    "reason": "divergence-only",
    "base_sampler_sha256": retry_sampler["retry"]["base_sampler_sha256"],
    "diagnostics": {
        "max_rhat": diagnostics.max_rhat,
        "min_bulk_ess": diagnostics.min_bulk_ess,
        "min_tail_ess": diagnostics.min_tail_ess,
        "divergences": diagnostics.divergences,
    },
}
```

Require a valid eligible record through Task 1's common validator whenever the identity uses a retry contract and reject the field for a base identity.

- [ ] **Step 5: Run H1/H2 tests**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_ablation.py -q`

Expected: pass.

- [ ] **Step 6: Commit**

```text
feat: retry divergent ablation folds once
```

---

### Task 3: H3 fold retry and mixed-contract primary aggregation

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/_loeo_publication.py`
- Modify: `hqrc_v3/src/hqrc_v3/loeo_stage.py`
- Modify: `hqrc_v3/src/hqrc_v3/loeo_primary.py`
- Test: `hqrc_v3/tests/unit/test_loeo_stage.py`
- Test: `hqrc_v3/tests/unit/test_loeo_primary.py`

**Interfaces:**
- Consumes: Task 1 retry contract and common `retry_failure` schema.
- Produces: H3 fit/load resolution across base and retry namespaces.
- Produces: primary aggregate validation that accepts each fold only when its sampler is the exact base contract or the exact deterministic retry contract for that held-out occurrence.

- [ ] **Step 1: Write failing H3 retry tests**

Add tests matching Task 2 behavior at the public `fit_loeo_fold`/`load_loeo_fold_material` boundary, including reuse without replaying the failed base and rejection of malformed retry provenance.

- [ ] **Step 2: Run focused H3 tests and verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py -k "divergence_retry or retry_manifest" -q`

Expected: failure because H3 resolves only the base namespace.

- [ ] **Step 3: Implement H3 retry resolution and manifest validation**

Extract the existing H3 single-contract fit/load body. Resolve a complete base first and a complete retry second; otherwise sample the base and invoke one retry only for an eligible structured failure. Extend H3 manifest construction and validation with the same signed `retry_failure` schema as H1/H2.

- [ ] **Step 4: Write failing mixed-aggregate tests**

Build ten literal fold materials where one uses its deterministic retry sampler and nine use base samplers. Assert `_matrix_identity` accepts them and records every effective sampler. Mutate retry seed, target acceptance, tune, or base digest independently and assert rejection.

- [ ] **Step 5: Run mixed-aggregate tests and verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_primary.py -k retry -q`

Expected: failure because `_matrix_identity` requires uniform samplers and only base seed labels.

- [ ] **Step 6: Implement exact per-fold sampler validation**

For every held-out occurrence, reconstruct its base contract from the recorded root seed/profile/settings and its deterministic retry contract. Accept only exact equality with one candidate. Remove only the obsolete cross-fold equality assumption; keep all source/model/seed/predictive-seed checks and fold references unchanged.

- [ ] **Step 7: Run LOEO regression tests**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/tests/unit/test_loeo_primary.py hqrc_v3/tests/integration/test_hqrc_sampling.py -q`

Expected: pass.

- [ ] **Step 8: Commit**

```text
feat: retry divergent H3 folds once
```

---

### Task 4: Cross-surface verification and operational handoff

**Files:**
- Modify only if a regression test exposes a defect in Tasks 1-3.
- Verify: current output tree and running process state without mutation.

**Interfaces:**
- Consumes: completed Tasks 1-3.
- Produces: verification evidence that existing base artifacts validate and retry behavior is reachable through the existing Windows/macOS/Linux CLI path.

- [ ] **Step 1: Run targeted scheduler and CLI tests**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/tests/integration/test_cli_corrections.py -q`

Expected: pass.

- [ ] **Step 2: Run the complete relevant regression set**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/tests/unit/test_loeo_primary.py hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/tests/integration/test_hqrc_sampling.py hqrc_v3/tests/integration/test_cli_corrections.py -q`

Expected: pass. If CPU contention causes a timeout, preserve the timeout evidence and rerun after the current two model workers finish.

- [ ] **Step 3: Verify repository and artifact invariants**

Run `git diff --check`, inspect `git status --short`, count existing `posterior.nc` files, and confirm no existing manifest changed.

- [ ] **Step 4: Commit verification documentation only if new durable evidence is added**

Do not commit `.omo` runtime evidence or mutate current artifacts.
