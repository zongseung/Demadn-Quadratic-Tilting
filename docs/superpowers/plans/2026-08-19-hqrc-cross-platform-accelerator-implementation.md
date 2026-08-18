# HQRC H1--H3 Cross-Platform Accelerator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the archived correction source safely and run the complete non-XGBoost H1--H3 LOEO matrix through one Windows/macOS/Linux implementation that uses both CUDA cards on the current Windows host.

**Architecture:** Preserve the existing PyMC path and immutable publication contracts, add a portable publication boundary, and implement the paper H1--H3 model once in PyTorch/Pyro. A shared scheduler resolves CUDA, MPS, or CPU and uses two fixed CUDA model queues when two cards are selected.

**Tech Stack:** Python 3.12, PyTorch CUDA/MPS, Pyro NUTS, ArviZ, filelock, pathlib/zipfile, subprocess, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-19-hqrc-cross-platform-accelerator-design.md`

## Global Constraints

- Final models are exactly `lightgbm`, `svr`, `seq2seq_lstm`, and `transformer`; XGBoost is never scheduled or published by the accelerated command.
- Final feature sets are exactly `B0` and `B1`; variants are exactly `H1`, `H2`, and `H3`; every paper context has all ten LOEO folds.
- Windows and Linux support CUDA, Apple Silicon macOS may use MPS after a capability probe, and every platform supports CPU.
- Explicit `cuda:N` or `mps` selection fails closed; only `auto` may fall back from MPS to CPU, recording the reason.
- Two-CUDA assignment is fixed: GPU 0 owns `lightgbm` then `seq2seq_lstm`; GPU 1 owns `svr` then `transformer`.
- One worker exposes one physical CUDA GPU before importing Torch. Four NUTS chains run sequentially inside a fit; model workers run concurrently.
- Paper sampling retains four chains, at least 1,000 tune iterations, at least 1,000 draws, target acceptance 0.99, zero divergences, R-hat at most 1.01, and bulk/tail ESS at least 400.
- The Pyro backend supports only H1--H3 paper options: partial pooling, restriction effects, full LKJ covariance, and stationary normal AR(1).
- Archived Parquet bytes, baseline manifest semantics, residual values, and the original ZIP remain unchanged.
- Existing PyMC identities and artifacts remain reusable; Pyro outputs have distinct backend/device-bound identities.
- AR proposals are never approved implicitly. Sampling begins only after proposal review and an explicit `--approve-derived-ar` invocation.
- Partial or mismatched evidence is preserved and rejected; no automatic cleanup is added.
- POSIX retains descriptor-relative publication hardening. Windows rejects symlinks and junctions, enforces trusted-root containment, locks exclusively, and uses same-directory `os.replace` publication.

## File Structure

- Create `hqrc_v3/src/hqrc_v3/publication_fs.py`: portable locks, trusted paths, and atomic publication primitives.
- Create `hqrc_v3/tests/unit/test_publication_fs.py`: Windows/POSIX contract and contention tests.
- Modify current publication modules to consume `publication_fs` and remove direct `fcntl` imports.
- Create `hqrc_v3/src/hqrc_v3/artifact_import.py`: safe archive import and historical source restoration.
- Create `hqrc_v3/tests/unit/test_artifact_import.py`: importer trust-boundary tests.
- Create `hqrc_v3/src/hqrc_v3/bayes/pyro_model.py`: exact H1--H3 Torch/Pyro model and deterministic reconstruction.
- Create `hqrc_v3/tests/unit/test_pyro_model.py`: fixed-value model-equivalence tests.
- Create `hqrc_v3/src/hqrc_v3/accelerators.py`: lazy device resolution, probe, and provenance.
- Modify `hqrc_v3/src/hqrc_v3/bayes/samplers.py`: sequential four-chain Pyro NUTS and ArviZ conversion.
- Modify LOEO modules and CLI to carry backend/device through immutable identities.
- Create `hqrc_v3/src/hqrc_v3/accelerated_scheduler.py`: scope validation, queues, subprocess isolation, logs, and failure reporting.
- Create `scripts/run_hqrc_windows.ps1`, `scripts/run_hqrc_linux.sh`, and `scripts/run_hqrc_macos.sh`: thin native launchers.
- Modify `hqrc_v3/pyproject.toml` and `uv.lock`: direct `filelock`, Pyro extra, and platform-marked Torch sources.
- Modify `hqrc_v3/README.md`: exact import, proposal, smoke, paper, and resume commands.

---

### Task 1: Portable Locks and Basic Publication Operations

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/publication_fs.py`
- Create: `hqrc_v3/tests/unit/test_publication_fs.py`
- Modify: `hqrc_v3/src/hqrc_v3/baselines/paper.py`
- Modify: `hqrc_v3/src/hqrc_v3/correction_stage.py`
- Modify: `hqrc_v3/src/hqrc_v3/residual_stage.py`
- Modify: `hqrc_v3/src/hqrc_v3/diagnostics/ar.py`
- Modify: `hqrc_v3/src/hqrc_v3/diagnostics/loeo.py`
- Modify: `hqrc_v3/src/hqrc_v3/diagnostics/loeo_ar.py`
- Modify: `hqrc_v3/src/hqrc_v3/residuals.py`
- Modify: `hqrc_v3/tests/integration/test_ar_artifact.py`
- Modify: `hqrc_v3/tests/unit/test_cache.py`
- Modify: `hqrc_v3/pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: existing lock paths and their current critical-section boundaries.
- Produces: `PublicationFSError`, `exclusive_lock(path: Path) -> ContextManager[None]`, `require_local_entry(path: Path, *, kind: Literal["file", "directory"]) -> Path`, and `require_within(root: Path, candidate: Path) -> Path`.

- [ ] **Step 1: Write failing portable-lock tests**

Add behavior tests that do not import `fcntl` and spawn a child process to prove exclusion:

```python
def test_exclusive_lock_blocks_a_spawned_writer(tmp_path):
    lock_path = tmp_path / "publication.lock"
    with exclusive_lock(lock_path):
        assert _spawn_lock_attempt(lock_path, timeout=0.2) == "blocked"
    assert _spawn_lock_attempt(lock_path, timeout=2.0) == "acquired"


def test_require_local_entry_rejects_symlink_or_junction(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = _make_platform_link(tmp_path / "link", target)
    with pytest.raises(PublicationFSError, match="link or junction"):
        require_local_entry(link, kind="directory")


def test_require_within_rejects_escape(tmp_path):
    with pytest.raises(PublicationFSError, match="outside trusted root"):
        require_within(tmp_path / "root", tmp_path / "escape")
```

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_publication_fs.py -q`

Expected: collection fails because `hqrc_v3.publication_fs` does not exist.

- [ ] **Step 3: Implement the minimal platform boundary**

Use `filelock.FileLock` for the conservative exclusive lock. Reject each existing path component when `Path.is_symlink()` or Python 3.12 `Path.is_junction()` is true. Resolve both root and candidate with `strict=False` and compare with `Path.is_relative_to`:

```python
@contextmanager
def exclusive_lock(path: Path, *, timeout: float = -1) -> Iterator[None]:
    checked = require_within(Path(path).parent, Path(path))
    if checked.exists():
        require_local_entry(checked, kind="file")
    with FileLock(str(checked), timeout=timeout):
        require_within(checked.parent, checked)
        yield
```

Keep the module free of imports from any publication module so it remains a leaf dependency.

- [ ] **Step 4: Replace basic `fcntl` blocks**

For each listed source module, remove `import fcntl` and preserve the surrounding validation while changing only the lock body:

```python
with exclusive_lock(lock_path):
    yield
```

Convert shared diagnostic/cache reads to exclusive locks. Update the two tests to use `exclusive_lock` contention rather than importing `fcntl`.

- [ ] **Step 5: Lock dependencies and run tests**

Add `filelock>=3.18.0` as a direct dependency, run `uv lock`, then run:

```powershell
uv run --project hqrc_v3 --locked pytest `
  hqrc_v3/tests/unit/test_publication_fs.py `
  hqrc_v3/tests/integration/test_ar_artifact.py `
  hqrc_v3/tests/unit/test_cache.py -q
```

Expected: PASS with no import or warning noise on Windows.

- [ ] **Step 6: Commit**

```powershell
git add hqrc_v3/src/hqrc_v3/publication_fs.py hqrc_v3/src/hqrc_v3 hqrc_v3/tests hqrc_v3/pyproject.toml uv.lock
git commit -m "feat: add cross-platform publication locks"
```

### Task 2: Portable LOEO and Posterior Publication

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/publication_fs.py`
- Modify: `hqrc_v3/src/hqrc_v3/_loeo_publication.py`
- Modify: `hqrc_v3/src/hqrc_v3/_loeo_primary_publication.py`
- Modify: `hqrc_v3/src/hqrc_v3/bayes/artifacts.py`
- Modify: `hqrc_v3/tests/unit/test_loeo_stage.py`
- Modify: `hqrc_v3/tests/unit/test_loeo_primary.py`
- Modify: `hqrc_v3/tests/unit/test_hqrc_artifacts.py`

**Interfaces:**
- Consumes: Task 1 `exclusive_lock`, `require_local_entry`, and `require_within`.
- Produces: `TrustedDirectory`, `trusted_directory(root: Path, path: Path, *, backend: Literal["posix", "windows"] | None = None) -> TrustedDirectory`, `guard_trusted_directory(directory: TrustedDirectory) -> None`, `atomic_write_bytes(directory: TrustedDirectory, name: str, content: bytes) -> Path`, `replace_entry(directory: TrustedDirectory, source_name: str, target_name: str) -> Path`, and `unlink_entry(directory: TrustedDirectory, name: str, *, missing_ok: bool = False) -> None`.

- [ ] **Step 1: Write failing forced-Windows publication tests**

Inject `backend="windows"` into `trusted_directory` so Windows behavior is testable on every OS:

```python
def test_windows_backend_publishes_atomically_without_dir_fd(tmp_path):
    root = tmp_path / "root"
    target = root / "fold"
    target.mkdir(parents=True)
    directory = trusted_directory(root, target, backend="windows")
    atomic_write_bytes(directory, "COMPLETE", b"ok\n")
    assert (target / "COMPLETE").read_bytes() == b"ok\n"
    assert not list(target.glob(".publication-*.tmp"))


def test_windows_backend_preserves_existing_target_on_replace_failure(tmp_path, monkeypatch):
    directory = _trusted_windows_directory(tmp_path)
    (directory.path / "result.json").write_bytes(b"old")
    monkeypatch.setattr(os, "replace", _raise_permission_error)
    with pytest.raises(PublicationFSError):
        atomic_write_bytes(directory, "result.json", b"new")
    assert (directory.path / "result.json").read_bytes() == b"old"
```

Add one end-to-end fold publication test that forces the Windows backend and performs write, load, identity validation, and reuse.

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_publication_fs.py hqrc_v3/tests/unit/test_loeo_stage.py -q`

Expected: FAIL because `TrustedDirectory` and backend selection do not exist.

- [ ] **Step 3: Implement `TrustedDirectory` and atomic helpers**

```python
@dataclass
class TrustedDirectory:
    root: Path
    path: Path
    identity: tuple[int, int]
    backend: Literal["posix", "windows"]
    descriptor: int | None = None

    def close(self) -> None:
        if self.descriptor is not None:
            os.close(self.descriptor)
            self.descriptor = None
```

POSIX opens directories with the current `O_DIRECTORY|O_NOFOLLOW` path. Windows stores a resolved path and `st_dev/st_ino`, rejects links/junctions component by component, and revalidates root containment and identity before and after each write. `atomic_write_bytes` creates a same-directory `mkstemp`, writes and fsyncs, calls `os.replace`, fsyncs the directory only where supported, and removes only its own temporary file on failure.

- [ ] **Step 4: Route existing publishers through the adapter**

Change `LOEOPublicationHandle.directory_fd` to `LOEOPublicationHandle.directory: TrustedDirectory`. Replace descriptor-relative `os.open`, `os.stat`, `os.mkdir`, `os.replace`, and `os.unlink` calls in the three modules with the adapter methods defined above. Keep all existing `_safe_name`, hash, `COMPLETE`, generation, and namespace checks. Do not relax mismatch handling or delete partial evidence.

- [ ] **Step 5: Run focused and Windows collection tests**

```powershell
uv run --project hqrc_v3 --locked pytest `
  hqrc_v3/tests/unit/test_publication_fs.py `
  hqrc_v3/tests/unit/test_hqrc_artifacts.py `
  hqrc_v3/tests/unit/test_loeo_stage.py `
  hqrc_v3/tests/unit/test_loeo_primary.py -q
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests --collect-only -q
```

Expected: focused tests pass and full Windows collection has no POSIX-import error.

- [ ] **Step 6: Commit**

```powershell
git add hqrc_v3/src/hqrc_v3/publication_fs.py hqrc_v3/src/hqrc_v3/_loeo_publication.py hqrc_v3/src/hqrc_v3/_loeo_primary_publication.py hqrc_v3/src/hqrc_v3/bayes/artifacts.py hqrc_v3/tests
git commit -m "feat: publish LOEO artifacts on Windows and POSIX"
```

### Task 3: Hash-Checked Archived Correction-Source Import

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/artifact_import.py`
- Create: `hqrc_v3/tests/unit/test_artifact_import.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Modify: `hqrc_v3/tests/integration/test_cli_oof.py`

**Interfaces:**
- Consumes: `validate_correction_source`, `file_sha256`, Task 1 locks, and the approved six source digests.
- Produces: `ImportedCorrectionSource(run_dir: Path, config_path: Path, reused: bool)`, `import_archived_correction_source(archive_path: Path, data_path: Path, repository_root: Path, run_dir: Path, *, profile: str = "paper") -> ImportedCorrectionSource`, and `hqrc import-paper-source`.

- [ ] **Step 1: Write importer trust-boundary tests**

Cover traversal before destination creation, unexpected namespace rejection, source hash mismatch, atomic failure preservation, successful path rebinding, zero-fit validation, and stable reuse:

```python
def test_import_rejects_traversal_before_destination_creation(tmp_path):
    archive = tmp_path / "artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr("artifacts/hqrc-v3-paper-20260811/../../escape", b"x")
    destination = tmp_path / "run"
    with pytest.raises(ArchiveImportError, match="unsafe archive member"):
        import_archived_correction_source(archive, tmp_path / "data.csv", tmp_path, destination)
    assert not destination.exists()


def test_import_rebinds_only_six_sources_and_reuses_identical_destination(fixture):
    first = import_archived_correction_source(**fixture)
    hashes = _tree_hashes(first.run_dir)
    second = import_archived_correction_source(**fixture)
    assert first.reused is False
    assert second.reused is True
    assert _tree_hashes(second.run_dir) == hashes
```

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_artifact_import.py -q`

Expected: collection fails because `hqrc_v3.artifact_import` does not exist.

- [ ] **Step 3: Implement staged import and historical Git restoration**

Validate the complete ZIP member table before creating staging. Extract only the three approved subtrees, use `git show <revision>:<path>` without a shell, verify every SHA-256, write the six files under `sources/`, rebind exactly six manifest paths, recompute the canonical manifest digest, validate staging with staging paths, rewrite paths to final paths immediately before atomic rename, then validate final. Existing destinations are read-only validation paths.

- [ ] **Step 4: Add CLI routing**

Add exact arguments `--archive`, `--data`, `--repository-root`, `--run-dir`, and optional `--profile` defaulting to `paper`. Print `run_dir`, `config_path`, and `reused` as stable `key=value` lines.

- [ ] **Step 5: Run focused tests and commit**

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_artifact_import.py hqrc_v3/tests/integration/test_cli_oof.py -q
git add hqrc_v3/src/hqrc_v3/artifact_import.py hqrc_v3/src/hqrc_v3/cli.py hqrc_v3/tests
git commit -m "feat: import archived correction source safely"
```

### Task 4: Exact Pyro H1--H3 Model

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/bayes/pyro_model.py`
- Create: `hqrc_v3/tests/unit/test_pyro_model.py`
- Modify: `hqrc_v3/pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: `HQRCData`, `HQRCModelOptions`, `ApprovedARCalibration`, and the PyMC equations in `bayes/model.py`.
- Produces: `PyroModelError`, `build_pyro_hqrc_model(data: HQRCData, calibration: ApprovedARCalibration, *, variant: Variant = "H3", pooling: Pooling = "partial", options: HQRCModelOptions | None = None, device: str = "cpu") -> Callable[[], None]`, `reconstruct_pyro_deterministics(samples: Mapping[str, Tensor], data: HQRCData, variant: Variant) -> dict[str, Tensor]`, and fixed-value Torch helpers for the mean and AR likelihood.

- [ ] **Step 1: Write fixed-value mathematical tests before importing Pyro**

Use literal tensors and hand-calculated expectations for H1, H2, H3, centered cyclic hour profile, and event-reset stationary AR(1):

```python
@pytest.mark.parametrize(
    ("variant", "expected"),
    [("H1", [1.0, 1.0]), ("H2", [1.0, 1.75]), ("H3", [1.25, 1.50])],
)
def test_torch_mean_matches_hand_calculated_fixture(variant, expected, tiny_data):
    actual = hqrc_mean_torch(tiny_data, beta=_beta_fixture(), variant=variant, gamma=_gamma_fixture())
    assert actual.cpu().tolist() == pytest.approx(expected)


def test_event_reset_ar1_matches_numpy_reference(tiny_data):
    torch_value = event_reset_ar1_logp_torch(_residual_fixture(), tiny_data, phi=0.25, sigma=0.8)
    numpy_value = event_reset_ar1_logp_numpy(_segments_fixture(), phi=0.25, sigma=0.8)
    np.testing.assert_allclose(torch_value.detach().cpu().numpy(), numpy_value, rtol=1e-10, atol=1e-10)
```

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_pyro_model.py -q`

Expected: collection fails because `bayes.pyro_model` does not exist.

- [ ] **Step 3: Implement the paper-only Torch/Pyro model**

Use `torch.float64`, Pyro sample sites for `mu`, `delta`, `beta_offset`, per-type scale and LKJ correlation Cholesky factors, `sigma_gamma`, non-centered hour innovations, `sigma_r`, and `u_phi`. Construct `beta`, `gamma`, `phi`, and `event_log_likelihood` deterministically. Add the AR log likelihood with `pyro.factor`. Reject H4, non-partial pooling, diagonal covariance, disabled restriction, and non-`normal_ar1` innovation with exact `PyroModelError` messages.

- [ ] **Step 4: Add Pyro dependency and run tests**

Add `pyro-ppl>=1.9.1` to optional dependency group `accelerator`, run `uv lock`, then:

```powershell
uv sync --project hqrc_v3 --extra accelerator --locked
uv run --project hqrc_v3 --extra accelerator pytest hqrc_v3/tests/unit/test_pyro_model.py -q
```

Expected: all fixed-value tests pass without CUDA.

- [ ] **Step 5: Commit**

```powershell
git add hqrc_v3/src/hqrc_v3/bayes/pyro_model.py hqrc_v3/tests/unit/test_pyro_model.py hqrc_v3/pyproject.toml uv.lock
git commit -m "feat: add shared Pyro H1-H3 model"
```

### Task 5: Device Resolution and Pyro Sampler

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/accelerators.py`
- Create: `hqrc_v3/tests/unit/test_accelerators.py`
- Modify: `hqrc_v3/src/hqrc_v3/bayes/samplers.py`
- Modify: `hqrc_v3/tests/integration/test_hqrc_sampling.py`
- Modify: `hqrc_v3/tests/integration/test_packaging.py`
- Modify: `hqrc_v3/pyproject.toml`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: Task 4 model/reconstruction and existing `validate_inference_data`.
- Produces: `ResolvedDevice`, `DeviceProbe`, `resolve_device(request: str) -> ResolvedDevice`, `probe_hqrc_device(device: ResolvedDevice) -> DeviceProbe`, and new `sample_hqrc` parameters `backend: Literal["pymc", "nutpie", "pyro"]` plus `device: str = "cpu"`.

- [ ] **Step 1: Write resolver tests with a fake Torch surface**

```python
def test_auto_prefers_cuda_and_records_physical_identity(fake_torch, monkeypatch):
    monkeypatch.setenv("HQRC_PHYSICAL_DEVICE", "1")
    fake_torch.cuda.available = True
    resolved = resolve_device("auto", torch_module=fake_torch, platform_name="Windows")
    assert resolved.kind == "cuda"
    assert resolved.logical_device == "cuda:0"
    assert resolved.physical_device == "1"


def test_auto_mps_probe_failure_falls_back_but_explicit_mps_fails(fake_torch):
    fake_torch.mps.available = True
    fake_torch.probe_error = RuntimeError("unsupported op")
    assert resolve_device("auto", torch_module=fake_torch, platform_name="Darwin").kind == "cpu"
    with pytest.raises(DeviceResolutionError, match="unsupported op"):
        resolve_device("mps", torch_module=fake_torch, platform_name="Darwin")
```

- [ ] **Step 2: Write a failing sequential-chain sampler test**

Patch `_run_pyro_chain` to return complete fake constrained samples and divergence arrays. Assert four calls with seeds `seed + 0..3`, one chain per call, the ArviZ chain dimension of four, `event_log_likelihood`, and device metadata. Also assert PyMC default metadata bytes remain unchanged.

- [ ] **Step 3: Run tests to verify RED**

Run: `uv run --project hqrc_v3 --extra accelerator pytest hqrc_v3/tests/unit/test_accelerators.py hqrc_v3/tests/integration/test_hqrc_sampling.py -q`

Expected: missing resolver and unsupported `pyro` backend failures.

- [ ] **Step 4: Implement lazy resolution and capability probe**

Do not import Torch at module import time. `resolve_device` imports it only inside the worker, validates exactly one visible CUDA device when `HQRC_PHYSICAL_DEVICE` is present, and runs a float64 gradient plus the model's LKJ/AR primitives. Auto MPS failure returns CPU with `fallback_reason`; explicit selection raises before artifact creation.

- [ ] **Step 5: Implement four sequential Pyro chains and ArviZ conversion**

Refactor `sample_hqrc` so PyMC models are built only for PyMC/nutpie. For Pyro, create `kernel = NUTS(model, target_accept_prob=resolved_target_accept)` and call `MCMC(kernel, num_samples=draws, warmup_steps=tune, num_chains=1)` four times, reconstruct deterministics, stack constrained posterior arrays on a leading chain axis, convert to ArviZ, and populate integer `sample_stats.diverging`. Reuse the existing diagnostics gate and log-likelihood group.

Record `hqrc_torch_version`, `hqrc_pyro_version`, resolved/physical device, dtype, probe result, and `chain_execution="sequential"`. The existing PyMC JSON remains byte-for-byte unchanged.

- [ ] **Step 6: Configure platform Torch sources and verify packaging**

Use the official `pytorch-cu130` explicit index for `sys_platform == 'win32' or sys_platform == 'linux'`; fall back to the standard macOS wheel on Darwin. Run `uv lock` and the packaging test, which asserts `pyro-ppl`, `filelock`, platform markers, and the accelerator extra.

- [ ] **Step 7: Run tests and commit**

```powershell
uv run --project hqrc_v3 --extra accelerator pytest hqrc_v3/tests/unit/test_accelerators.py hqrc_v3/tests/unit/test_pyro_model.py hqrc_v3/tests/integration/test_hqrc_sampling.py hqrc_v3/tests/integration/test_packaging.py -q
git add hqrc_v3/src/hqrc_v3/accelerators.py hqrc_v3/src/hqrc_v3/bayes/samplers.py hqrc_v3/tests hqrc_v3/pyproject.toml uv.lock
git commit -m "feat: sample H1-H3 with portable accelerators"
```

### Task 6: Carry Backend and Device Through LOEO Provenance

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/_loeo_publication.py`
- Modify: `hqrc_v3/src/hqrc_v3/loeo_stage.py`
- Modify: `hqrc_v3/src/hqrc_v3/loeo_primary.py`
- Modify: `hqrc_v3/src/hqrc_v3/loeo_ablation.py`
- Modify: `hqrc_v3/src/hqrc_v3/paper_pipeline.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Modify: `hqrc_v3/tests/unit/test_loeo_stage.py`
- Modify: `hqrc_v3/tests/unit/test_loeo_primary.py`
- Modify: `hqrc_v3/tests/unit/test_loeo_ablation.py`
- Modify: `hqrc_v3/tests/unit/test_paper_pipeline.py`
- Modify: `hqrc_v3/tests/integration/test_cli_oof.py`

**Interfaces:**
- Consumes: Task 5 `sample_hqrc` backend/device contract.
- Produces: backend/device-aware `sampler_contract` and new `run_paper_loeo_pipeline` keyword parameters `backend: str = "pymc"` and `device: str = "cpu"`, exposed as `run-loeo-primary --backend/--device`.

- [ ] **Step 1: Write failing identity and propagation tests**

Assert the existing default PyMC contract exactly equals its frozen fixture. Assert Pyro contracts differ for physical device `0` and `1`, and every pipeline layer forwards backend/device without changing fold order. Test that reload rejects a sampler metadata mismatch while preserving the artifact.

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run --project hqrc_v3 --extra accelerator pytest hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_primary.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/tests/unit/test_paper_pipeline.py -q`

Expected: unexpected keyword or identity equality failures.

- [ ] **Step 3: Add backend/device parameters with compatible defaults**

Thread `backend` and `device` from CLI through paper pipeline, primary/ablation aggregation, fold fit, sampler, and reload. Default values preserve PyMC behavior. Bind Pyro identities to `backend`, resolved kind, and physical device. Treat runtime-only probe/fallback details as validated metadata rather than sampler-contract input.

- [ ] **Step 4: Run tests and commit**

```powershell
uv run --project hqrc_v3 --extra accelerator pytest hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_primary.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/tests/unit/test_paper_pipeline.py hqrc_v3/tests/integration/test_cli_oof.py -q
git add hqrc_v3/src/hqrc_v3 hqrc_v3/tests
git commit -m "feat: bind LOEO artifacts to accelerator identity"
```

### Task 7: Accelerated Scheduler, CLI, and Native Launchers

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/accelerated_scheduler.py`
- Create: `hqrc_v3/tests/unit/test_accelerated_scheduler.py`
- Create: `scripts/run_hqrc_windows.ps1`
- Create: `scripts/run_hqrc_linux.sh`
- Create: `scripts/run_hqrc_macos.sh`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py`
- Modify: `hqrc_v3/tests/integration/test_cli_oof.py`

**Interfaces:**
- Consumes: Task 3 imported run/config and Task 6 `run-loeo-primary --backend pyro --device cuda:0`.
- Produces: `AcceleratedRequest`, `AcceleratedResult`, `model_queues(devices: tuple[int, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]`, `run_accelerated_loeo(request: AcceleratedRequest) -> AcceleratedResult`, and `hqrc run-loeo-accelerated`.

- [ ] **Step 1: Write queue, scope, environment, and failure tests**

```python
def test_two_cuda_devices_receive_fixed_model_queues():
    assert model_queues((0, 1)) == (
        ("cuda:0", ("lightgbm", "seq2seq_lstm")),
        ("cuda:1", ("svr", "transformer")),
    )


def test_paper_scope_rejects_xgboost_or_partial_folds():
    with pytest.raises(SchedulerError, match="XGBoost"):
        validate_scope(profile="paper", models=("xgboost",), folds=ALL_FOLDS)
    with pytest.raises(SchedulerError, match="all ten LOEO folds"):
        validate_scope(profile="paper", models=APPROVED_MODELS, folds=ALL_FOLDS[:-1])


def test_cuda_child_environment_exposes_one_physical_gpu():
    env = child_environment(os.environ, physical_device=1)
    assert env["CUDA_VISIBLE_DEVICES"] == "1"
    assert env["HQRC_PHYSICAL_DEVICE"] == "1"
```

Use a fake subprocess factory to prove that one failed queue stops its later model, the peer in-flight queue is joined, the parent fails, and completed logs remain.

- [ ] **Step 2: Run tests to verify RED**

Run: `uv run --project hqrc_v3 --extra accelerator pytest hqrc_v3/tests/unit/test_accelerated_scheduler.py -q`

Expected: missing module failure.

- [ ] **Step 3: Implement deterministic scheduling**

Preflight source/output non-overlap, correction-source validity, exact paper scope, and device probe subprocesses before logs. Use only standard-library subprocesses and threads. Each CUDA child receives a single physical GPU through its environment and invokes the existing CLI with `--backend pyro --device cuda:0`; MPS/CPU use one sequential queue. Tee stdout/stderr to console and model log.

- [ ] **Step 4: Add CLI and thin launchers**

The command accepts `--source-run-dir`, `--config`, `--output-root`, `--devices`, `--profile`, `--models`, `--feature-sets`, `--variants`, and `--approve-derived-ar`. Paper defaults are complete and immutable; smoke allows only ordered model/feature/variant subsets.

Each launcher supports a `Mode`/positional mode of `import`, `proposal`, `smoke`, `paper`, or `resume`, performs native `uv sync --extra accelerator --locked`, and forwards all scientific arguments to the shared CLI. No launcher implements model logic.

- [ ] **Step 5: Run parser/launcher behavior tests and commit**

```powershell
uv run --project hqrc_v3 --extra accelerator pytest hqrc_v3/tests/unit/test_accelerated_scheduler.py hqrc_v3/tests/integration/test_cli_oof.py -q
git add hqrc_v3/src/hqrc_v3/accelerated_scheduler.py hqrc_v3/src/hqrc_v3/cli.py hqrc_v3/tests scripts
git commit -m "feat: schedule H1-H3 across native accelerators"
```

### Task 8: Documentation, Regression, and Windows Hardware Qualification

**Files:**
- Modify: `hqrc_v3/README.md`
- Generated, gitignored: `artifacts/hqrc-v3-paper-imported-20260819/`
- Generated, gitignored: `artifacts/hqrc-v3-loeo-accelerated-20260819/`

**Interfaces:**
- Consumes: all Tasks 1--7 and root `artifacts.zip` plus `power_demand_final copy.csv`.
- Produces: exact three-platform operator instructions, validated import, AR proposals, dual-GPU smoke evidence, and a resumable paper launch command.

- [ ] **Step 1: Document exact native commands**

Document Windows PowerShell, Linux Bash, and macOS Bash setup and all launcher modes. State that macOS auto uses MPS only after the HQRC probe and otherwise records CPU fallback. Include the fixed two-GPU queues, four sequential chains, explicit AR review gate, failure/resume behavior, and XGBoost exclusion.

- [ ] **Step 2: Run formatting and full regression**

```powershell
uv run --project hqrc_v3 --extra accelerator ruff check hqrc_v3/src hqrc_v3/tests
uv run --project hqrc_v3 --extra accelerator pytest hqrc_v3/tests -m "not slow" -q
```

Expected: clean output. If a pre-existing platform test is intentionally skipped, record its exact node and reason in the task report; do not hide new failures.

- [ ] **Step 3: Import the real archive twice**

```powershell
uv run --project hqrc_v3 --extra accelerator hqrc import-paper-source `
  --archive "..\..\artifacts.zip" `
  --data "..\..\power_demand_final copy.csv" `
  --repository-root "..\.." `
  --run-dir artifacts/hqrc-v3-paper-imported-20260819
```

Expected: first run prints `reused=False`, second `reused=True`, and a tree-hash comparison is identical.

- [ ] **Step 4: Verify both Windows CUDA devices independently and concurrently**

Run the accelerator probe once with physical GPU 0 and once with GPU 1, then launch a tiny H1 sampler on each concurrently. Evidence must show one visible logical `cuda:0` per child, physical identities `0` and `1`, CUDA tensors, finite log density, four chains, and serialized ArviZ output.

- [ ] **Step 5: Run the two-model H1 archive smoke**

```powershell
scripts\run_hqrc_windows.ps1 smoke `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-loeo-accelerated-20260819 `
  -Devices 0,1 -Models lightgbm,svr -FeatureSets B0 -Variants H1
```

Expected: both model logs overlap in wall-clock time, bind to different physical GPUs, preserve ten folds, and reuse completed folds on an identical rerun.

- [ ] **Step 6: Publish proposals and stop at the scientific review gate**

Run the complete paper command without `--approve-derived-ar`. Expected: all proposal artifacts are published/reused and sampling does not start. Report proposal paths and digests to the user. Do not execute the approval command until the user explicitly approves those generated proposals.

- [ ] **Step 7: Prepare the resumable approved paper command**

After explicit proposal approval, the exact launch is:

```powershell
scripts\run_hqrc_windows.ps1 paper `
  -SourceRunDir artifacts/hqrc-v3-paper-imported-20260819 `
  -Config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml `
  -OutputRoot artifacts/hqrc-v3-loeo-accelerated-20260819 `
  -Devices 0,1 -ApproveDerivedAR
```

Do not claim completion merely because the long-running command started. Record its PID, logs, completed fold count, and restart command.

- [ ] **Step 8: Commit documentation**

```powershell
git add hqrc_v3/README.md
git commit -m "docs: add native accelerator runbook"
```
