# HQRC H1--H3 Dual-GPU Execution Design

## 1. Purpose

Reuse the validated baseline and standardized-residual publications in
`artifacts.zip`, exclude XGBoost, and execute every H1, H2, and H3 LOEO
correction for the remaining manuscript baselines on both installed GPUs.

The complete target matrix is:

- models: `lightgbm`, `svr`, `seq2seq_lstm`, `transformer`
- feature sets: `B0`, `B1`
- variants: `H1`, `H2`, `H3`
- held-out occurrences: all ten LOEO folds
- profile: `paper` for the final run, preceded by a reduced `smoke` qualification

This is 24 model/feature/variant contexts and 240 fold fits. XGBoost must not be
scheduled, sampled, or published by the dual-GPU command.

## 2. Existing Constraints

The current LOEO pipeline is sequential. `run_paper_loeo_pipeline` processes one
baseline context at a time and every LOEO fitting path calls `sample_hqrc` with
`backend="pymc"`. PyMC's default NUTS path is CPU-bound. The installed Windows
environment also contains the CPU-only Torch build, so setting
`CUDA_VISIBLE_DEVICES` alone cannot accelerate H1--H3.

JAX does not support NVIDIA CUDA on native Windows. GPU sampling therefore runs
inside the installed WSL2 Ubuntu distribution with JAX's CUDA 13 package and
PyMC's NumPyro NUTS interface. The local RTX 5060 Ti cards report compute
capability 12.0 and driver 591.59, which satisfy JAX's published CUDA 13 minimums.

Relevant upstream contracts:

- https://docs.jax.dev/en/latest/installation.html
- https://docs.jax.dev/en/latest/gpu_memory_allocation.html
- https://www.pymc.io/projects/docs/en/stable/api/generated/pymc.sampling.jax.sample_numpyro_nuts.html

## 3. Architecture

The change has three narrow components:

1. a hash-checked importer for the archived immutable correction source;
2. an explicit NumPyro sampler backend carried through LOEO provenance;
3. a two-worker model scheduler that assigns one model to each GPU at a time.

The existing correction data preparation, AR proposal and approval, fold
identity, posterior publication, prediction generation, aggregation, and resume
contracts remain the source of truth. The new code must call those paths rather
than reimplement them.

## 4. Archived Source Import

### 4.1 Imported namespace

The importer accepts the repository `artifacts.zip` and extracts only these
members from `artifacts/hqrc-v3-paper-20260811/` into a new run directory:

- `predictions/`
- `inputs/`
- `prediction-stream-cache/`

It ignores `__MACOSX`, causal correction outputs, reports, diagnostics,
quarantine data, and every unrelated artifact tree. Every ZIP member is resolved
and checked before extraction; absolute paths and `..` traversal are rejected.
The original ZIP is never modified.

The destination is published atomically. An absent destination is created from a
temporary sibling directory. An existing destination is accepted only when its
canonical publications validate exactly; mismatched or partial content fails
without deletion.

### 4.2 Source restoration and rebinding

The archived residual manifest binds six source files by SHA-256 and by the
original macOS absolute path. The data digest matches
`power_demand_final copy.csv`. The five configuration/registry digests are still
available as Git objects in this repository:

| Source | Git revision | Archived SHA-256 |
|---|---|---|
| experiment config | `07e771fd` | `526416f140e3e2fe15777ee72e3e0155c9dff7d629c13d9ff88f1e032cc29c9f` |
| model config | `03d9b68a` | `b06d00880cd4311c5a0ff73c427800bd330f238d757c11165627d70f6465a1e9` |
| event registry | `3ed23767` | `147a823ba37501240460cd4cd53c914ca529c353dd43cf3cc302a65026206e7b` |
| holiday calendar | `03d9b68a` | `a72d93b49c9eaedd469ff04fb14534fcc718c893ac509ebc2152f532609c398e` |
| temporary-holiday availability | `03d9b68a` | `4b695cbbd5e737c43733e842eefb92ce19a6c75d6b0886853696818525fb2b0d` |

The importer materializes these exact Git blobs under `<run>/sources/`, copies
the matching data file there, verifies every digest, and replaces only the six
absolute source paths in `standardized_residuals_manifest.json`. It then
recomputes the manifest's canonical `manifest_sha256`. Prediction Parquet files,
their hashes, baseline semantics, standardized residual values, and context
records are unchanged.

The import succeeds only after `validate_correction_source` rebuilds source truth
and proves that the final baseline reuses the publication with `fit_count == 0`.
No baseline is retrained during import.

## 5. NumPyro GPU Sampler

`sample_hqrc` gains a `numpyro` backend alongside the existing `pymc` and
`nutpie` backends. It uses PyMC's JAX NumPyro NUTS interface with:

- the existing draws, tune, chain count, seed, and target acceptance;
- `chain_method="vectorized"`, because each worker exposes exactly one GPU;
- CPU post-processing so completed draws are converted and written outside GPU
  memory;
- the same ArviZ log-likelihood group and convergence gate used by CPU sampling.

Paper runs retain exactly four chains, at least 1,000 tune iterations, at least
1,000 retained draws, target acceptance 0.99, zero divergences, maximum R-hat
1.01, and minimum bulk/tail ESS 400. Backend, device identity, JAX version,
NumPyro version, initialization mode, and chain method are recorded in sampler
metadata and immutable fold identity.

NumPyro does not use PyMC's `adapt_diag` initialization parameter. Its recorded
initialization is therefore explicit and backend-specific rather than pretending
to use the CPU initialization contract. Existing PyMC artifacts remain reusable
only by the PyMC identity; GPU and CPU outputs cannot collide.

The optional `gpu-sampler` dependency group is Linux-only and installs the CUDA
13 JAX runtime plus NumPyro. Native Windows invocation fails with an actionable
message directing the operator to WSL2.

## 6. Two-GPU Model Scheduler

A new `run-loeo-dual-gpu` command performs a read-only source preflight before
launching workers. A `paper` invocation rejects model, feature-set, variant, or
fold subsets and is fixed to the approved matrix. A `smoke` invocation may select
an ordered subset of the four allowed models, B0/B1, and H1/H2/H3 for hardware
qualification, but it still rejects XGBoost. Production uses this deterministic
assignment:

```text
GPU 0: lightgbm -> seq2seq_lstm
GPU 1: svr      -> transformer
```

Each worker exposes one physical card through `CUDA_VISIBLE_DEVICES` before JAX
is imported. Within a worker, one model completes `B0`, then `B1`; within each
feature set it completes `H1`, then `H2`, then `H3`; each variant retains the
existing ten-fold sequential order. The two workers operate concurrently, so at
most two models are active and each active model owns one GPU.

The scheduler uses standard-library subprocesses and calls the existing
`run-loeo-primary` command with an explicit `--backend numpyro`. It does not add a
distributed framework. Child output is streamed to the console and copied to a
model-specific log under the output root.

The parent sets `XLA_PYTHON_CLIENT_PREALLOCATE=false` for each worker. This avoids
JAX's default large reservation and makes per-process memory usage visible,
without allowing workers to share a card.

## 7. AR Review and Execution Flow

The scientific review boundary remains intact:

1. import and validate the archived correction source;
2. run the selected matrix without approval to publish/reuse all LOEO universes,
   ACF/PACF plots, and proposal digests;
3. review the proposals;
4. rerun with the explicit `--approve-derived-ar` flag;
5. execute H1--H3 sampling and aggregate publications.

The dual-GPU scheduler forwards the approval flag but never invents approval.
Completed proposal, fold, and aggregate artifacts are reused through their
existing immutable identities.

## 8. Failure and Resume Behavior

Before launching, the command requires Linux under WSL2, exactly two requested
GPU indices, two JAX-visible CUDA devices in the parent preflight, a validated
correction source, and output paths that do not overlap the source publication.

Each child validates that exactly one local GPU is visible and records its
physical assignment. A model failure prevents that worker's later model from
starting. The other in-flight model may finish, and all completed immutable
artifacts are preserved. The parent exits nonzero and prints the failed model,
device, log path, and restart command. A rerun reuses every completed fold and
continues from the first incomplete identity.

No cleanup command deletes partial data automatically. Namespace or hash
mismatches fail closed for manual inspection.

## 9. Verification

Implementation follows test-driven development with these checks:

1. importer tests reject ZIP traversal, unknown namespace entries, hash
   mismatches, and incomplete destinations;
2. importer integration verifies path rebinding, canonical manifest digest, and
   `validate_correction_source` with zero baseline fits;
3. sampler tests prove NumPyro arguments, metadata, backend-specific identity,
   log-likelihood publication, and diagnostic gates;
4. scheduler tests prove the fixed model-to-device mapping, XGBoost exclusion,
   one-model-per-worker behavior, environment setup, failure propagation, and
   resume command;
5. existing unit and integration suites continue to pass for the default PyMC
   backend;
6. a WSL2 hardware preflight runs one tiny sampler process on each GPU
   concurrently and confirms one CUDA device per child;
7. a reduced two-model H1 smoke run exercises the real archive, both GPUs,
   posterior serialization, prediction products, and resume;
8. only after smoke succeeds is the paper H1--H3 matrix launched.

## 10. Non-Goals

- Retraining archived baseline models is not part of this change.
- LightGBM or Torch baseline GPU training is not added because the archived
  baseline publication is reused.
- One posterior is not split across both cards; parallelism is model-level as
  requested.
- XGBoost data already present in the shared source is not deleted, because the
  source's paper-profile all-context integrity requires it. It is excluded from
  the new correction schedule and outputs.
- Existing causal-2024 H3 results in the ZIP are not rewritten or presented as
  LOEO H1--H3 results.
