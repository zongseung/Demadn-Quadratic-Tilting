# HQRC CPU All-Core Design

## Scope

Run the existing non-XGBoost H1--H3 LOEO workload on CPU without changing its
Pyro sampler, paper settings, AR approval boundary, artifact identities, or
resume behavior. Preserve CUDA and MPS support.

## Scheduling

- CPU runs at most one model worker per requested model and never more workers
  than the detected logical CPU count.
- Requested models are distributed across those queues in manuscript order.
- Logical CPUs are divided as evenly as possible between active model workers;
  the first queues receive any remainder. For 24 logical CPUs and four paper
  models this is four concurrent workers with six threads each.
- Each CPU job sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
  `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and
  `VECLIB_MAXIMUM_THREADS` to its allocation before Torch is imported.
- CUDA retains the fixed two-device queues. MPS retains one sequential queue.
- A failed worker stops later work that has not started, while in-flight work
  and partial evidence are preserved under the existing contract.

## Launchers

- Windows exposes `-Accelerator auto|cuda|mps|cpu` and retains `cuda` as its
  backward-compatible default.
- Linux reads `HQRC_ACCELERATOR`, defaulting to `cuda`.
- macOS reads `HQRC_ACCELERATOR`, defaulting to `auto`.
- Setting the platform control to `cpu` selects the CPU all-core scheduler.

## Verification

- Unit tests prove CPU concurrency and exact thread allocation, including a
  machine with fewer logical CPUs than requested models.
- Launcher tests execute the real wrappers with a fake `uv` boundary and prove
  that explicit CPU selection reaches the CLI.
- Existing CUDA/MPS scheduling tests remain green.
- After regression tests and review, resume the already-approved paper output
  on native Windows with explicit CPU selection and observe its exact process
  tree, CPU use, memory use, and logs before leaving it running.
