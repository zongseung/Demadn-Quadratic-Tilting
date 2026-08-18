# HQRC H1--H3 Cross-Platform Accelerator Design

## 1. Purpose

Reuse the immutable correction source in `artifacts.zip`, exclude XGBoost, and
run every H1, H2, and H3 LOEO correction on Windows, macOS, and Linux from one
shared implementation. The current Windows host must use both installed NVIDIA
GPUs concurrently, with one model context active on each physical GPU.

The paper matrix is fixed:

- models: `lightgbm`, `svr`, `seq2seq_lstm`, `transformer`;
- feature sets: `B0`, `B1`;
- variants: `H1`, `H2`, `H3`;
- held-out occurrences: all ten LOEO folds;
- profile: `paper`, after a reduced `smoke` qualification.

This is 24 model/feature/variant contexts and 240 fold fits. XGBoost is never
scheduled or published by the accelerated command.

## 2. Decisions

The implementation uses one PyTorch/Pyro H1--H3 model on all three operating
systems. It does not maintain separate NumPyro and Pyro translations of the
same scientific model.

Device selection is explicit:

- Windows and Linux may use NVIDIA CUDA devices;
- Apple Silicon macOS may use MPS when a runtime capability probe succeeds;
- every platform may use CPU;
- `auto` may fall back from MPS to CPU and records the reason;
- an explicitly requested `cuda:N` or `mps` device never silently falls back.

The existing PyMC and nutpie backends remain available and retain their current
artifact identities. The new Pyro backend has a distinct immutable identity.

Relevant upstream contracts:

- https://pytorch.org/get-started/locally/
- https://docs.pytorch.org/docs/stable/notes/mps.html
- https://docs.pyro.ai/en/stable/inference.html
- https://docs.pyro.ai/en/dev/_modules/pyro/infer/mcmc/api.html
- https://docs.astral.sh/uv/guides/integration/pytorch/

## 3. Architecture

The change has five focused components:

1. a hash-checked importer for the archived correction source;
2. one small cross-platform file-lock adapter;
3. a Pyro translation of the exact paper H1--H3 model and sampler;
4. a device resolver and model-level scheduler;
5. three thin operator launchers around one shared CLI.

Existing correction preparation, AR proposal and approval, fold identity,
posterior publication, prediction generation, aggregation, and resume code
remain authoritative. The new paths call those components rather than
reimplementing them.

## 4. Archived Correction Source

### 4.1 Safe import

`hqrc import-paper-source` accepts `artifacts.zip` and extracts only these
members from `artifacts/hqrc-v3-paper-20260811/`:

- `predictions/`;
- `inputs/`;
- `prediction-stream-cache/`.

Every ZIP member is validated before the destination is created. Absolute
paths, drive-qualified paths, `..` traversal, links, and unexpected namespaces
are rejected. `__MACOSX`, causal results, reports, diagnostics, quarantine
data, and unrelated artifact trees are ignored. The ZIP is never modified.

An absent destination is assembled in a sibling staging directory and renamed
atomically. An existing destination is reused only when its complete canonical
publication validates. Partial or mismatched evidence is preserved and
rejected; it is never deleted automatically.

### 4.2 Historical source restoration

The archived residual manifest identifies six source files by SHA-256 but uses
old absolute macOS paths. The importer restores the exact bytes as follows:

| Source | Source revision | SHA-256 |
|---|---|---|
| demand data | `power_demand_final copy.csv` | `ccb27f1d5ab7c5e2315bcd4cfc82f3c64d439b1b8d746ba65439fc89062969f9` |
| experiment config | `07e771fd` | `526416f140e3e2fe15777ee72e3e0155c9dff7d629c13d9ff88f1e032cc29c9f` |
| model config | `03d9b68a` | `b06d00880cd4311c5a0ff73c427800bd330f238d757c11165627d70f6465a1e9` |
| event registry | `3ed23767` | `147a823ba37501240460cd4cd53c914ca529c353dd43cf3cc302a65026206e7b` |
| holiday calendar | `03d9b68a` | `a72d93b49c9eaedd469ff04fb14534fcc718c893ac509ebc2152f532609c398e` |
| temporary-holiday availability | `03d9b68a` | `4b695cbbd5e737c43733e842eefb92ce19a6c75d6b0886853696818525fb2b0d` |

The exact Git blobs and demand data are materialized under `<run>/sources/`.
Only those six source paths are rebound in the residual manifest, whose
canonical digest is then recomputed. Archived Parquet bytes, baseline manifest
semantics, standardized residual values, and context records stay unchanged.

The import succeeds only when `validate_correction_source` proves that the
final baseline publication is reused with `fit_count == 0`. A second identical
import is hash-stable and reports reuse.

## 5. Cross-Platform Locking

Direct `fcntl` imports currently prevent package collection on Windows. One
`hqrc_v3.locking` context manager replaces direct OS-specific calls in the
publication and diagnostic modules touched by the pipeline.

The adapter uses a conservative exclusive inter-process lock on all platforms.
The pipeline does not need concurrent shared readers for correctness, so a
shared-lock abstraction is not introduced. Lock paths and critical-section
boundaries remain unchanged. Tests exercise contention and release in spawned
processes on Windows and POSIX.

## 6. Shared Pyro H1--H3 Model

### 6.1 Mathematical contract

The Pyro model reproduces the paper path of `build_hqrc_model`:

- H1 occurrence intercept;
- H2 occurrence-specific quadratic demand tilt;
- H3 H2 plus the centered cyclic hourly RW1 profile;
- partial pooling with restriction effects and full LKJ covariance;
- approved Beta calibration for stationary event-reset normal AR(1);
- occurrence-level `event_log_likelihood`.

Only the exact options used by the H1--H3 paper matrix are accepted by the
Pyro backend. Other sensitivity options continue to use PyMC/nutpie and fail
clearly if requested with Pyro. H4 is not added to the accelerated command.

The port uses Torch double precision on CUDA and CPU. MPS is accepted only if a
runtime probe successfully evaluates the model's required distributions,
linear algebra, gradients, and finite log density at the selected precision.
An unsupported operation cannot be hidden by per-operation CPU fallback.

### 6.2 Sampler and publication

`sample_hqrc(..., backend="pyro", device=...)` runs Pyro NUTS with the existing
draw count, tune count, seed family, and target acceptance. Each fit retains
exactly four independent chains. Chains run sequentially within a worker to
avoid nested CUDA/MPS multiprocessing and Windows spawn instability; the two
model workers provide the requested hardware concurrency.

The result is converted to the existing ArviZ schema, including posterior,
sample statistics, occurrence log likelihood, and diagnostics. Paper gates are
unchanged:

- exactly four chains;
- at least 1,000 tune iterations and 1,000 retained draws;
- target acceptance 0.99;
- zero divergences;
- maximum R-hat 1.01;
- minimum bulk and tail ESS 400.

Backend, resolved device kind, logical and physical device identifiers, Torch
and Pyro versions, dtype, chain execution mode, and capability-probe result are
recorded in sampler metadata and immutable fold identity.

### 6.3 Scientific equivalence gate

Before hardware smoke testing, deterministic tests compare PyMC and Pyro at
fixed latent values for H1, H2, and H3. They check the derived mean, centered
hour profile, stationary AR parameters, and occurrence log-likelihood against
independently calculated fixtures. A small seeded smoke fit checks posterior
shape and diagnostic publication, not bitwise draw equality across platforms.

## 7. Device Resolution

The public selector is `--device auto|cpu|mps|cuda:N`.

`auto` resolves in this order:

1. CUDA devices when Torch reports them usable;
2. MPS on macOS when it is built, available, and passes the HQRC probe;
3. CPU.

Explicit devices must pass the same probe or the command exits before artifact
creation. CUDA workers set `CUDA_VISIBLE_DEVICES` before importing Torch; each
child therefore sees one logical `cuda:0`, while `HQRC_PHYSICAL_DEVICE` records
the requested physical index. The child verifies that exactly one CUDA device
is visible and that a test tensor executes there.

## 8. Scheduler

`hqrc run-loeo-accelerated` performs a read-only source, output, scope, and
device preflight before creating logs or workers.

On a host with at least two selected CUDA devices, production assignment is:

```text
physical GPU 0: lightgbm -> seq2seq_lstm
physical GPU 1: svr      -> transformer
```

One long-lived worker process owns each physical GPU. A worker completes B0
then B1 for its current model; each feature set completes H1, H2, then H3; each
variant retains the existing ten-fold order. Only after a model is complete
does that worker start the second model in its queue.

A single selected CUDA device or one MPS device runs all four models
sequentially. CPU mode is sequential. The accelerated paper command accepts
only the four approved models, B0/B1, H1/H2/H3, and all ten folds. Smoke mode
may reduce models, feature sets, and variants, but never folds. Any XGBoost
request fails before workers start.

Child output is streamed to the console and copied to a model-specific log. If
one worker fails, its later model does not start; the other in-flight worker may
finish. The parent exits nonzero with the failed model, physical device, log,
and restart command. Completed immutable artifacts remain resumable.

## 9. AR Approval Boundary

The scientific review workflow remains explicit:

1. import and validate the correction source;
2. publish or reuse all LOEO universes and AR proposals without approval;
3. review proposal plots and digests;
4. rerun with `--approve-derived-ar`;
5. sample and aggregate H1--H3.

The scheduler forwards approval but never invents it. Completed proposals,
folds, and aggregates are reused only through exact identities.

## 10. Packaging and Launchers

`pyro-ppl` is an optional accelerator dependency. The locked Torch source uses
the official CUDA 13.0 index on Windows and Linux and the standard macOS wheel
on Darwin, which contains the MPS backend. CPU remains a valid explicit mode.

There is one Python CLI. Three thin launchers provide native shell syntax and
dependency setup without duplicating pipeline logic:

- `scripts/run_hqrc_windows.ps1`;
- `scripts/run_hqrc_linux.sh`;
- `scripts/run_hqrc_macos.sh`.

Each launcher supports import, proposal, smoke, paper, and resume modes by
forwarding to the same CLI. Platform-specific values are limited to shell
syntax and dependency/device preflight.

## 11. Verification

Implementation follows test-driven development:

1. safe-import tests cover traversal, namespace, digest, atomic publication,
   validation, and reuse;
2. locking tests cover cross-process exclusion and release;
3. model tests cover H1--H3 deterministic mathematical equivalence;
4. sampler tests cover four-chain assembly, ArviZ schema, metadata, diagnostic
   gates, and unsupported options;
5. resolver tests cover Windows/Linux CUDA, macOS MPS and CPU fallback, explicit
   failure, and physical/logical identity;
6. scheduler tests cover fixed dual-GPU queues, single-device behavior,
   XGBoost rejection, complete paper scope, logs, failure, and resume;
7. parser tests cover the shared CLI and all three launchers;
8. existing default-PyMC tests continue to pass;
9. the current Windows host runs one CUDA tensor and one tiny H1 fit on each GPU
   concurrently, followed by a two-model H1 archive smoke;
10. the full paper launch begins only after the smoke passes.

macOS and Linux platform behavior is covered by deterministic unit tests and
documented native commands. Real MPS/Linux CUDA qualification requires those
hosts and is not claimed from the Windows machine.

## 12. Non-Goals

- Archived baseline models are not retrained.
- One posterior is not split across two GPUs; parallelism is model-level.
- XGBoost bytes already required by the archived all-context source are not
  deleted, but XGBoost is excluded from accelerated scheduling and outputs.
- H4 and non-paper Pyro sensitivity modes are not implemented.
- MPS use is not forced when its runtime probe fails.
- Existing causal-2024 H3 results are not rewritten as LOEO H1--H3 results.
