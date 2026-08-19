# HQRC Four-Model/Four-Chain CPU Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the non-XGBoost H1--H3 CPU paper workload as four concurrent model processes with four concurrent, single-threaded Pyro chain processes per model.

**Architecture:** Keep the existing model-level scheduler and single-chain NUTS function. Add a Windows-`spawn` process boundary whose module-level worker reloads the approved AR artifact, constructs one Pyro model, runs one chain, and returns NumPy posterior arrays; aggregate the four results in deterministic chain order. Resolve CPU jobs to four chain workers with one numerical-library thread each while preserving sequential CUDA/MPS behavior and immutable sampler identities.

**Tech Stack:** Python 3.12, `concurrent.futures.ProcessPoolExecutor`, `multiprocessing`, PyTorch, Pyro, ArviZ, pytest, PowerShell, POSIX shell

**Spec:** `docs/superpowers/specs/2026-08-20-hqrc-four-model-four-chain-design.md`

## Global Constraints

- Paper scope remains exactly `lightgbm`, `svr`, `seq2seq_lstm`, and `transformer`; B0/B1; H1/H2/H3; ten LOEO folds; XGBoost excluded.
- Keep four chains, 1,000 tune, 1,000 draws, `target_accept=0.99`, float64, existing chain seeds, AR approval, diagnostics, and posterior dimensions.
- CPU Pyro accepts `cores=1` or `cores=4`; no intermediate topology. CUDA and MPS remain `cores=1` and sequential.
- Use explicit `spawn`, construct the nested Pyro model inside the child, and use only standard-library process scheduling.
- Existing `cores=1` output identities and files remain valid and untouched; `cores=4` creates a distinct identity.
- Preserve untracked `.omo/` evidence and the stopped run's artifacts.

---

### Task 1: Spawn-Safe Four-Chain Pyro Sampling

**Files:**
- Modify: `hqrc_v3/tests/integration/test_hqrc_sampling.py`
- Modify: `hqrc_v3/src/hqrc_v3/bayes/samplers.py`

**Interfaces:**
- Consumes: `HQRCData`, `ApprovedARCalibration` artifact coordinates, `HQRCModelOptions`, `_run_pyro_chain()`, `reconstruct_pyro_deterministics()`.
- Produces: `_PyroChainRequest`, `_PyroChainResult`, `_initialize_pyro_chain_worker()`, `_run_pyro_chain_worker(request)`, and `_sample_pyro(..., cores)`.

- [ ] **Step 1: Write failing core-topology and parallel-aggregation tests**

Replace the test that rejects every Pyro `cores > 1` with literal cases proving `cores=2`, `3`, and `5` fail while `cores=4` reaches the Pyro sampling boundary. Add a test around `sample_hqrc(..., backend="pyro", cores=4)` using a fake `ProcessPoolExecutor` only at the OS-process boundary. Capture these required executor options:

```python
{
    "max_workers": 4,
    "mp_context": multiprocessing.get_context("spawn"),
    "initializer": _initialize_pyro_chain_worker,
}
```

Have the fake executor apply a fake chain worker to four requests and return results with chain indexes `0, 1, 2, 3`. Assert request seeds are exactly `17, 18, 19, 20`, each request carries the approved artifact path and current hashes, the resulting posterior shape is `(4, draws, ...)`, and metadata contains `cores=4` and `chain_execution="parallel"`.

- [ ] **Step 2: Verify RED**

Run:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/integration/test_hqrc_sampling.py -k "pyro and (core or parallel)" -q
```

Expected: failure because Pyro currently rejects `cores=4` and loops over four chains in the parent.

- [ ] **Step 3: Add serializable request/result boundaries**

In `samplers.py`, add frozen private dataclasses with these fields:

```python
@dataclass(frozen=True)
class _PyroChainRequest:
    chain: int
    data: HQRCData
    approved_ar_path: Path
    residual_sha256: str
    config_sha256: str
    event_sha256: str
    variant: Variant
    pooling: Pooling
    options: HQRCModelOptions | None
    draws: int
    tune: int
    seed: int
    target_accept: float
    device: str

@dataclass(frozen=True)
class _PyroChainResult:
    chain: int
    posterior: dict[str, np.ndarray]
    divergences: np.ndarray
```

Implement `_initialize_pyro_chain_worker()` to call `torch.set_num_threads(1)` and `torch.set_num_interop_threads(1)` once when each spawned process starts. Implement `_run_pyro_chain_worker()` to reload the calibration through `load_approved_calibration()` using the request's artifact path and hashes, build the model inside the child, call `_run_pyro_chain(..., num_chains=1)`, reconstruct deterministics, remove `_PRIVATE_PYRO_SITES`, copy values to CPU NumPy arrays, and return `_PyroChainResult`.

- [ ] **Step 4: Implement the minimal sequential/parallel branch**

Pass `cores` from `sample_hqrc()` into `_sample_pyro()`. Keep the current single-model sequential loop unchanged for `cores=1`. For `cores=4`, build four requests and execute:

```python
context = multiprocessing.get_context("spawn")
with ProcessPoolExecutor(
    max_workers=4,
    mp_context=context,
    initializer=_initialize_pyro_chain_worker,
) as pool:
    results = tuple(pool.map(_run_pyro_chain_worker, requests))
```

Validate returned chain indexes are exactly `0, 1, 2, 3`, order by index, and feed the same ArviZ aggregation code used by the sequential branch. Accept only Pyro `cores in {1, 4}` and record the actual value plus derived execution mode in both JSON and top-level attributes.

- [ ] **Step 5: Verify GREEN**

Run the RED command again, then the whole sampling file:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/integration/test_hqrc_sampling.py -q
```

Expected: focused and file-level tests pass.

- [ ] **Step 6: Add and run the real Windows spawn integration test**

Add a `@pytest.mark.slow` test that creates a real approved calibration artifact for nine two-row occurrences, uses the matching H1 `HQRCData`, calls `sample_hqrc(..., backend="pyro", device="cpu", draws=4, tune=4, chains=4, cores=4)`, and asserts four chains, four draws, finite values, integer divergence flags, `cores=4`, and `chain_execution="parallel"`. Run its exact node ID so the slow mark does not skip it:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/integration/test_hqrc_sampling.py::test_pyro_four_chain_spawn_runs_real_model -q
```

Expected: one actual four-process Pyro sample passes on native Windows.

- [ ] **Step 7: Commit**

```powershell
git add hqrc_v3/tests/integration/test_hqrc_sampling.py hqrc_v3/src/hqrc_v3/bayes/samplers.py
git commit -m "feat: run Pyro chains in parallel on CPU"
```

---

### Task 2: Parallel LOEO Identity and Resume Validation

**Files:**
- Modify: `hqrc_v3/tests/unit/test_loeo_stage.py`
- Modify: `hqrc_v3/tests/unit/test_loeo_ablation.py`
- Modify: `hqrc_v3/src/hqrc_v3/_loeo_publication.py`

**Interfaces:**
- Consumes: `sampler_contract()`, `validate_posterior_provenance()`, existing `cores` identity field.
- Produces: `_chain_execution(sampler)` returning `"sequential"` for one core and `"parallel"` for four cores; Pyro sampler contracts that accept only those two topologies.

- [ ] **Step 1: Write failing contract and reuse tests**

Add table-driven tests proving:

```python
cores=1 -> chain_execution="sequential"
cores=4 -> chain_execution="parallel"
cores in (2, 3, 5) -> LOEOFoldError
```

Build a `cores=4` fake posterior whose sampler JSON and top-level attribute both say `parallel`; assert provenance validation accepts it. Mutate either field to `sequential` and assert validation rejects it. Assert `sha_json(_fold_identity(... sampler cores=1))` differs from the otherwise identical `cores=4` identity, without changing the literal `cores=1` identity fixture.

- [ ] **Step 2: Verify RED**

Run:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_ablation.py -k "pyro or parallel or identity" -q
```

Expected: `cores=4` is rejected or incorrectly validated as sequential.

- [ ] **Step 3: Derive execution mode from existing identity data**

Allow Pyro sampler contracts only when chains equal four and cores are one or four. Add a private helper equivalent to:

```python
def _chain_execution(sampler: Mapping[str, object]) -> str:
    return "parallel" if sampler["cores"] == 4 else "sequential"
```

Use it for expected sampler JSON and `hqrc_chain_execution`. Do not add a new sampler-contract key: `cores` already changes the fold identity, and omitting a new key preserves existing sequential identities.

- [ ] **Step 4: Verify GREEN and commit**

Run both entire unit files, then commit:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_ablation.py -q
git add hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/src/hqrc_v3/_loeo_publication.py
git commit -m "feat: validate parallel Pyro fold identities"
```

---

### Task 3: Resolve CPU Jobs to Four Single-Threaded Chains

**Files:**
- Modify: `hqrc_v3/tests/unit/test_accelerated_scheduler.py`
- Modify: `hqrc_v3/src/hqrc_v3/accelerated_scheduler.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Modify: `scripts/run_hqrc_windows.ps1`
- Modify: `scripts/run_hqrc_linux.sh`
- Modify: `scripts/run_hqrc_macos.sh`
- Modify: `hqrc_v3/README.md`

**Interfaces:**
- Consumes: resolved accelerator kind, `_build_job()`, `_cpu_queue_specs()`, existing launcher accelerator controls.
- Produces: `_chain_workers(request, accelerator_kind)` and job commands with an explicit resolved `--cores`.

- [ ] **Step 1: Write failing CPU topology tests**

Update scheduler tests to prove that a resolved CPU paper run starts all four model jobs concurrently and that every job has:

```python
command option: --cores 4
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
NUMEXPR_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1
```

Assert CUDA and MPS jobs receive `--cores 1` and retain existing queues. Add mismatch cases: explicit CPU `cores=1` and CUDA/MPS `cores=4` fail before any model job starts. Update real launcher tests so smoke commands no longer force `--cores 1`; the scheduler owns device-specific resolution.

- [ ] **Step 2: Verify RED**

Run:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/unit/test_accelerated_scheduler.py -q
```

Expected: CPU jobs still receive six-thread budgets and no resolved four-core chain option.

- [ ] **Step 3: Implement device-specific core resolution**

Permit `request.cores` to be omitted during profile-shape validation. After preflight, resolve CPU to four and CUDA/MPS to one. If an explicit value disagrees, raise `AcceleratedRunError` before `_run_job()`. Pass the resolved value into `_build_job()` and emit it as `--cores` regardless of whether the CLI request supplied a value.

Change CPU queue specifications to keep four concurrent model queues but return a thread budget of one for every queue. Keep the five environment variables and durable log header. Update CLI help to describe CPU four-chain versus accelerator sequential behavior.

- [ ] **Step 4: Remove launcher-owned core topology and update the runbook**

Remove the hard-coded smoke `--cores 1` from all three platform launchers. Document `4 models x 4 chains x 1 thread`, the 16-physical-core target, the 28 GiB memory gate, and the new-output-root requirement. Keep Windows/macOS/Linux accelerator defaults unchanged.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/tests/integration/test_cli_oof.py -q
git add hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/src/hqrc_v3/accelerated_scheduler.py hqrc_v3/src/hqrc_v3/cli.py scripts/run_hqrc_windows.ps1 scripts/run_hqrc_linux.sh scripts/run_hqrc_macos.sh hqrc_v3/README.md
git commit -m "feat: schedule sixteen CPU chain workers"
```

---

### Task 4: Regression, Resource Gate, and Paper Launch

**Files:**
- Create: `.superpowers/sdd/2026-08-20-hqrc-four-model-four-chain-implementation/cpu-parallel-smoke-report.md`
- Runtime output only: a new directory below `artifacts/`

**Interfaces:**
- Consumes: native Windows launcher, imported paper source, parallel sampler and scheduler.
- Produces: test evidence, a four-model resource report, and a safely detached paper run only if the resource gate passes.

- [ ] **Step 1: Run focused and full verification**

Run:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/integration/test_hqrc_sampling.py hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/tests/integration/test_cli_oof.py -q
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests -m "not slow" -q
```

Expected: zero failures.

- [ ] **Step 2: Run the real four-model CPU smoke gate**

Use a new smoke output root and the native Windows launcher with `-Accelerator cpu`, four models, B0, H1, and the existing smoke draws/tune. While all models sample, record process tree, per-process CPU, total working set and committed memory, system paging, logs, exit code, posterior dimensions, and sampler metadata in the runtime report.

Run from the worktree root:

```powershell
& .\scripts\run_hqrc_windows.ps1 smoke `
  -Accelerator cpu `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-four-chain-smoke-20260820 `
  -Models lightgbm,svr,seq2seq_lstm,transformer `
  -FeatureSets B0 `
  -Variants H1 `
  -ApproveDerivedAR
```

Pass criteria are exactly: all four models exit zero; sixteen chain children are observed; all produced posteriors say `cores=4` and `parallel`; peak committed memory is below 28 GiB; no sustained paging or traceback.

- [ ] **Step 3: Stop on a failed resource gate**

If any gate fails, preserve logs and report the observed value. Do not launch paper work or silently reduce model concurrency.

- [ ] **Step 4: Launch the paper run on a new root after a passing gate**

Start the native Windows `resume` command detached and hidden with explicit `-Accelerator cpu`, the existing imported source/config, and a new output root named for the parallel run. Capture root PID plus stdout/stderr paths. After at least sixty seconds, verify the parent, four model processes, and sixteen chain children remain alive with no traceback, then record the evidence.

Use these exact run paths:

```powershell
$stdout = '.omo/evidence/cpu-four-chain-paper-20260820.stdout.log'
$stderr = '.omo/evidence/cpu-four-chain-paper-20260820.stderr.log'
$arguments = @(
  '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File',
  (Resolve-Path 'scripts/run_hqrc_windows.ps1'), 'resume',
  '-Accelerator', 'cpu',
  '-SourceRunDir', 'artifacts/hqrc-v3-paper-imported-20260819',
  '-Config', 'artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml',
  '-OutputRoot', 'artifacts/hqrc-v3-loeo-cpu-four-chain-20260820',
  '-ApproveDerivedAR'
)
$process = Start-Process powershell.exe -ArgumentList $arguments -PassThru `
  -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr
$process.Id
```

- [ ] **Step 5: Commit durable implementation evidence**

```powershell
git add .superpowers/sdd/2026-08-20-hqrc-four-model-four-chain-implementation/cpu-parallel-smoke-report.md docs/superpowers/plans/2026-08-20-hqrc-four-model-four-chain-implementation.md
git commit -m "test: verify four-model four-chain CPU run"
```
