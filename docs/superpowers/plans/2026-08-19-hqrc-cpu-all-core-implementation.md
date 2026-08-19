# HQRC CPU All-Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the approved non-XGBoost H1--H3 Pyro workload across all detected CPU threads while preserving existing accelerator and resume behavior.

**Architecture:** Extend the existing accelerated scheduler only at its queue boundary: CPU gets bounded model-level parallel queues and each child receives an even thread budget through standard numerical-library environment variables. Keep CUDA and MPS queue behavior unchanged, and expose CPU selection through the existing platform launchers.

**Tech Stack:** Python 3.12, `concurrent.futures`, Pyro/Torch, pytest, PowerShell, POSIX shell

**Spec:** `docs/superpowers/specs/2026-08-19-hqrc-cpu-all-core-design.md`

## Global Constraints

- The paper scope remains exactly `lightgbm`, `svr`, `seq2seq_lstm`, and `transformer`; B0/B1; H1/H2/H3; ten folds; XGBoost excluded.
- Do not change Pyro's four sequential chains, paper draws/tune/target-accept, AR approval, immutable artifact identities, or resume semantics.
- Preserve CUDA's fixed queues and MPS's sequential queue.
- Add no dependency and use only standard-library scheduling and environment variables.
- Preserve the user's untracked `.omo/` evidence.

---

### Task 1: CPU Model Queues and Thread Budgets

**Files:**
- Modify: `hqrc_v3/tests/unit/test_accelerated_scheduler.py`
- Modify: `hqrc_v3/src/hqrc_v3/accelerated_scheduler.py`

**Interfaces:**
- Consumes: `AcceleratedRequest.models`, `os.cpu_count()`, `_build_job()`, and the existing queue runner.
- Produces: private `_cpu_queue_specs(models, logical_cpus=None)` returning ordered `(models, threads)` queue specifications used by `run_accelerated_loeo()`.

- [ ] **Step 1: Write failing CPU scheduling tests**

Add tests that run the real scheduler boundary with `_run_job` replacing only the external subprocess. For four requested models and `os.cpu_count() == 24`, use a four-party barrier to prove all model jobs are in flight concurrently and assert that each job receives all five thread variables with value `"6"`. Add a literal boundary test proving two logical CPUs yield two ordered queues, one thread per queue, without losing or duplicating models. Existing MPS and CUDA tests must continue to assert their prior queue behavior.

- [ ] **Step 2: Verify RED**

Run:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/unit/test_accelerated_scheduler.py -q
```

Expected: the new CPU concurrency/thread assertions fail because CPU currently has one sequential queue and no per-job CPU thread budget.

- [ ] **Step 3: Implement the minimum scheduler change**

Implement `_cpu_queue_specs()` with:

```python
available = max(1, logical_cpus if logical_cpus is not None else (os.cpu_count() or 1))
worker_count = min(len(models), available)
base, remainder = divmod(available, worker_count)
queues = tuple(tuple(models[index::worker_count]) for index in range(worker_count))
threads = tuple(base + (index < remainder) for index in range(worker_count))
```

Use these queues only when `resolved.kind == "cpu"`. Pass the queue's integer thread budget into `_build_job()` and set `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `NUMEXPR_NUM_THREADS`, and `VECLIB_MAXIMUM_THREADS` to that same decimal value. CUDA and MPS pass no CPU budget and retain current behavior.

- [ ] **Step 4: Verify GREEN and regression scope**

Run:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/tests/integration/test_cli_oof.py -q
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```powershell
git add hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/src/hqrc_v3/accelerated_scheduler.py
git commit -m "feat: use all CPUs for accelerated scheduling"
```

---

### Task 2: Cross-Platform CPU Selection and Runbook

**Files:**
- Modify: `hqrc_v3/tests/unit/test_accelerated_scheduler.py`
- Modify: `scripts/run_hqrc_windows.ps1`
- Modify: `scripts/run_hqrc_linux.sh`
- Modify: `scripts/run_hqrc_macos.sh`
- Modify: `hqrc_v3/README.md`

**Interfaces:**
- Consumes: the existing `run-loeo-accelerated --accelerator` CLI option.
- Produces: Windows `-Accelerator` and POSIX `HQRC_ACCELERATOR` controls.

- [ ] **Step 1: Write failing launcher tests**

Extend the real-wrapper tests so Windows invocation with `-Accelerator cpu` records exactly one `--accelerator cpu` in the fake `uv` command. Add Linux and macOS cases with `HQRC_ACCELERATOR=cpu` that likewise record CPU selection. Preserve tests for the existing platform defaults.

- [ ] **Step 2: Verify RED**

Run the launcher test selections from `hqrc_v3/tests/unit/test_accelerated_scheduler.py`; expect failure because the launchers currently hard-code CUDA or auto.

- [ ] **Step 3: Implement the minimum launcher controls**

Add a validated PowerShell parameter whose default is `cuda` and substitute it for the hard-coded value. In Linux use `${HQRC_ACCELERATOR:-cuda}`; in macOS use `${HQRC_ACCELERATOR:-auto}`. Do not add another parser or dependency.

- [ ] **Step 4: Update the runbook**

Document the four-worker/six-thread allocation on this 24-thread host and show explicit CPU `paper`/`resume` commands for Windows, Linux, and macOS. State that platform defaults remain unchanged and that CPU selection preserves the Pyro/proposal/resume contracts.

- [ ] **Step 5: Verify GREEN and commit**

Run:

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/tests/integration/test_cli_oof.py -q
```

Then commit:

```powershell
git add hqrc_v3/tests/unit/test_accelerated_scheduler.py scripts/run_hqrc_windows.ps1 scripts/run_hqrc_linux.sh scripts/run_hqrc_macos.sh hqrc_v3/README.md docs/superpowers/specs/2026-08-19-hqrc-cpu-all-core-design.md docs/superpowers/plans/2026-08-19-hqrc-cpu-all-core-implementation.md
git commit -m "docs: add cross-platform CPU run mode"
```

---

### Task 3: Qualification and Native Windows Resume

**Files:**
- Verify only: repository tests and `artifacts/hqrc-v3-loeo-accelerated-20260819`

**Interfaces:**
- Consumes: reviewed Tasks 1--2 and the previously approved AR artifacts.
- Produces: fresh test evidence and a live CPU paper-resume process with recorded PID and resource evidence.

- [ ] **Step 1: Run full regression**

```powershell
uv run --project hqrc_v3 --extra accelerator --locked pytest hqrc_v3/tests -q -m "not slow"
```

- [ ] **Step 2: Run an independent whole-diff review**

Require no open Critical or Important findings before runtime launch.

- [ ] **Step 3: Launch native Windows CPU resume**

Start `scripts/run_hqrc_windows.ps1 resume` with the existing imported source, config, output root, `-Accelerator cpu`, and `-ApproveDerivedAR`. Record the exact parent PID without matching unrelated Python processes.

- [ ] **Step 4: Observe the live workload**

Verify four model workers, each with the expected six-thread environment in its restart command/provenance, nonzero aggregate CPU use, bounded memory, advancing logs, and no CUDA-bound HQRC processes. Preserve the run in the background if healthy; stop only on a concrete failure or user request.
