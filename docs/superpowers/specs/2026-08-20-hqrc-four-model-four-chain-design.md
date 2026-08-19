# HQRC Four-Model/Four-Chain CPU Design

## Goal

Run the existing non-XGBoost H1--H3 LOEO paper workload as four concurrent
model processes, with four concurrent single-threaded Pyro chains inside each
model process. On the target Intel Core i7-13700 host this creates sixteen
independent chain workers for the sixteen physical cores instead of four
mostly single-core model workers.

## Scope and invariants

- Keep the paper scope exactly `lightgbm`, `svr`, `seq2seq_lstm`, and
  `transformer`; B0/B1; H1/H2/H3; ten LOEO folds; XGBoost excluded.
- Keep four chains, 1,000 warm-up draws, 1,000 retained draws,
  `target_accept=0.99`, the existing per-chain seeds, float64, diagnostics,
  AR approval, and posterior shapes.
- Change only CPU Pyro chain execution. CUDA and MPS retain one sequential
  chain runner per model and their existing device isolation.
- Add no dependency. Use Python's process primitives and the existing
  single-chain Pyro implementation.
- Preserve all sequential-run artifacts and logs. Parallel results use a
  different sampler identity and therefore never silently reuse or overwrite
  sequential posterior checkpoints.

## Architecture

The accelerated scheduler continues to launch the four model commands in
parallel. For a resolved CPU run it passes `--cores 4` to every model and sets
all numerical-library thread environment variables to `1`. Thus those
variables are hard caps per chain process, not unused six-thread allowances.

The Pyro sampler keeps `_run_pyro_chain()` as the only code that runs NUTS.
When `cores=4`, `_sample_pyro()` submits four module-level worker calls through
a Windows-compatible `spawn` process pool. Each worker receives serializable
HQRC data, approved calibration, model options, chain index, and sampler
settings; constructs its model inside the child; sets Torch intra-op and
interop thread counts to one; runs `num_chains=1`; reconstructs deterministic
values; and returns NumPy arrays plus divergence flags. The parent collects
results by chain index and creates the same `(chain, draw, ...)` ArviZ layout
as the sequential path.

The worker must be module-level because the current Pyro model is a nested
function and cannot be pickled directly by Windows `spawn`. Native
`MCMC(num_chains=4)` is therefore not used.

## Contracts and provenance

Pyro continues to require exactly four chains. It accepts `cores=1` for the
existing sequential path and `cores=4` for the new parallel path; other Pyro
core counts are rejected. Metadata records the actual core count and
`chain_execution` as `sequential` or `parallel`.

The LOEO sampler contract already includes `cores`, so `cores=4` produces a
different identity without changing the identity of existing `cores=1`
outputs. `chain_execution` is derived from `cores` and validated in posterior
metadata rather than added as a new identity field. Existing
`cores=1`/`sequential` outputs therefore remain valid but are not candidates
for a `cores=4`/`parallel` run. The new full run uses a new output root for
clear operational separation even though the identity hash also prevents
collisions.

If any child fails, the parent cancels pending work, terminates the pool, and
raises the original failure before publishing a posterior checkpoint. Existing
atomic publication behavior preserves completed folds.

## Resource gate and execution

Before the paper run, execute a CPU smoke workload covering all four models at
once so the process fan-out reaches sixteen chain workers. Observe peak
committed memory, working set, process count, CPU utilization, exit status,
and sampler metadata. Proceed with the paper run only when:

- all four model jobs finish without traceback;
- each sampled posterior reports four chains and `chain_execution=parallel`;
- sixteen chain workers are observed while all four models are sampling;
- peak committed memory remains below 28 GiB on the 32 GiB host; and
- the machine does not enter sustained paging.

If the memory gate fails, preserve the evidence and stop before the paper run.
The next safe design is two concurrent models with four chains each; do not
silently switch topology or weaken draws, tuning, chains, dtype, or diagnostic
thresholds.

## Platform behavior

Windows, Linux, and macOS launchers keep their existing accelerator selection.
When CPU resolves, the scheduler selects four chain workers per model and one
numerical-library thread per worker. The implementation uses `spawn` explicitly
so Windows is the portability baseline and macOS/Linux exercise the same code
path.

## Verification

- Unit tests prove Pyro accepts only one or four cores, preserves sequential
  behavior at one core, and aggregates four out-of-order parallel results in
  deterministic chain order.
- A real small-draw integration test exercises the spawn path with the actual
  Pyro model and validates posterior shapes, seeds, divergences, and metadata.
- Scheduler tests prove a CPU paper job receives `--cores 4` and all five
  numerical thread variables equal `1`, while CUDA/MPS remain sequential.
- LOEO tests prove sequential checkpoints cannot be reused as parallel
  checkpoints and that parallel checkpoints resume cleanly.
- The focused suite, full non-slow suite, four-model smoke resource gate, and
  Windows paper launch must all pass before completion is claimed.
