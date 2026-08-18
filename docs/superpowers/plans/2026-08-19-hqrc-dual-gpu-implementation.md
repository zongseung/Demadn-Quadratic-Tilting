# HQRC H1--H3 Dual-GPU Implementation Plan

> Superseded by
> `docs/superpowers/plans/2026-08-19-hqrc-cross-platform-accelerator-implementation.md`.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Import the archived immutable correction source, add a provenance-safe NumPyro GPU sampler, and run every non-XGBoost B0/B1 H1--H3 LOEO context with one model per GPU.

**Architecture:** A hash-checked importer creates a local correction source without retraining baselines. The existing LOEO stack receives an explicit sampler backend and physical device identity, while a standard-library scheduler launches two isolated model workers with one CUDA device visible to each process.

**Tech Stack:** Python 3.11+, PyMC, JAX CUDA 13, NumPyro, ArviZ, Polars, `zipfile`, `subprocess`, WSL2, pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-19-hqrc-dual-gpu-design.md`

## Global Constraints

- Final models are exactly `lightgbm`, `svr`, `seq2seq_lstm`, and `transformer`; XGBoost is never scheduled or published by the dual-GPU command.
- Final feature sets are exactly `B0` and `B1`; variants are exactly `H1`, `H2`, and `H3`; each paper context has all ten LOEO folds.
- GPU sampling runs only in WSL2 Linux with JAX CUDA 13; native Windows GPU sampling must fail clearly.
- GPU 0 owns `lightgbm` then `seq2seq_lstm`; GPU 1 owns `svr` then `transformer`.
- Each worker exposes one GPU before importing JAX and uses NumPyro NUTS with `chain_method="vectorized"` and CPU post-processing.
- Paper sampling retains four chains, at least 1,000 tune iterations, at least 1,000 draws, target acceptance 0.99, zero divergences, R-hat at most 1.01, and bulk/tail ESS at least 400.
- Archived Parquet bytes, baseline manifest semantics, residual values, and the original ZIP remain unchanged.
- Existing PyMC identities and completed artifacts remain reusable; NumPyro outputs have distinct immutable identities.
- AR proposals are not approved implicitly. Sampling begins only through an explicit `--approve-derived-ar` invocation after proposal review.
- Partial or mismatched evidence is preserved and rejected; no automatic deletion or cleanup is added.

## File Structure

- Create `hqrc_v3/src/hqrc_v3/artifact_import.py`: safe ZIP extraction, historical source restoration, manifest rebinding, and correction-source validation.
- Create `hqrc_v3/tests/unit/test_artifact_import.py`: importer path, hash, atomic publication, and reuse contracts.
- Modify `hqrc_v3/src/hqrc_v3/bayes/samplers.py`: lazy NumPyro GPU sampling and backend-specific metadata.
- Modify `hqrc_v3/tests/integration/test_hqrc_sampling.py`: NumPyro call and diagnostic contracts.
- Modify `hqrc_v3/src/hqrc_v3/_loeo_publication.py`: backend/device-aware sampler identity without changing the PyMC default identity.
- Modify `hqrc_v3/src/hqrc_v3/loeo_stage.py`: pass backend/device into H3 fold fit and reload.
- Modify `hqrc_v3/src/hqrc_v3/loeo_primary.py`: pass backend/device through the H3 ten-fold aggregate.
- Modify `hqrc_v3/src/hqrc_v3/loeo_ablation.py`: pass backend/device through H1/H2 fold and aggregate paths.
- Modify `hqrc_v3/src/hqrc_v3/paper_pipeline.py`: pass backend/device through each selected context.
- Create `hqrc_v3/src/hqrc_v3/gpu_scheduler.py`: WSL/GPU preflight, fixed model queues, subprocess tee logging, and failure reporting.
- Create `hqrc_v3/tests/unit/test_gpu_scheduler.py`: fixed assignment, environment isolation, scope, and failure tests.
- Modify `hqrc_v3/src/hqrc_v3/cli.py`: `import-paper-source`, backend/device options, and `run-loeo-dual-gpu`.
- Modify `hqrc_v3/tests/integration/test_cli_oof.py`: parser routing for both new commands.
- Modify `hqrc_v3/tests/unit/test_loeo_stage.py`, `test_loeo_primary.py`, `test_loeo_ablation.py`, and `test_paper_pipeline.py`: propagation and backward-compatibility assertions.
- Modify `hqrc_v3/pyproject.toml` and `uv.lock`: Linux-only `gpu-sampler` optional dependency.
- Modify `hqrc_v3/README.md`: import, WSL setup, proposal review, smoke, paper run, and resume commands.

---

### Task 1: Hash-Checked Archived Correction-Source Import

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/artifact_import.py`
- Create: `hqrc_v3/tests/unit/test_artifact_import.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py:396-659`
- Modify: `hqrc_v3/tests/integration/test_cli_oof.py:10-170`

**Interfaces:**
- Consumes: `validate_correction_source(run_dir: Path, config_path: Path, profile: str)` and `file_sha256(path: Path) -> str`.
- Produces: `ImportedCorrectionSource`, `import_archived_correction_source(...)`, and the `hqrc import-paper-source` command. Later tasks consume `result.run_dir` and `result.config_path`.

- [ ] **Step 1: Write importer failure and success tests**

Add tests that construct a small ZIP with the approved prefix, a canonical residual manifest, and fake source bytes. Patch the source specification table and final validator so the test isolates importer behavior.

```python
def test_import_rejects_traversal_before_destination_creation(tmp_path, monkeypatch):
    archive = tmp_path / "artifacts.zip"
    with ZipFile(archive, "w") as bundle:
        bundle.writestr(
            "artifacts/hqrc-v3-paper-20260811/../../escape.txt",
            b"escape",
        )
    destination = tmp_path / "run"

    with pytest.raises(ArchiveImportError, match="unsafe archive member"):
        import_archived_correction_source(
            archive_path=archive,
            data_path=tmp_path / "data.csv",
            destination=destination,
            repository_root=tmp_path,
        )

    assert not destination.exists()
    assert not (tmp_path / "escape.txt").exists()
```

```python
def test_import_rebinds_sources_and_validates_without_refitting(
    archived_source_zip, tmp_path, monkeypatch
):
    validated = []
    monkeypatch.setattr(
        artifact_import,
        "validate_correction_source",
        lambda **kwargs: validated.append(kwargs) or object(),
    )

    result = import_archived_correction_source(
        archive_path=archived_source_zip.archive,
        data_path=archived_source_zip.data,
        destination=tmp_path / "run",
        repository_root=archived_source_zip.repository,
    )
    manifest = json.loads((result.run_dir / "inputs/standardized_residuals_manifest.json").read_bytes())

    assert result.config_path == result.run_dir / "sources/experiment.toml"
    assert all(Path(manifest["inputs"][name]["path"]).is_absolute() for name in SOURCE_NAMES)
    assert manifest["manifest_sha256"] == residual_manifest_digest(manifest)
    assert len(validated) == 2
    assert result.reused is False
```

- [ ] **Step 2: Run the importer tests and confirm the missing module failure**

Run:

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_artifact_import.py -q
```

Expected: collection fails because `hqrc_v3.artifact_import` does not exist.

- [ ] **Step 3: Implement the importer with exact source bindings**

Create the public types and fixed source catalog:

```python
@dataclass(frozen=True, slots=True)
class SourceSpec:
    name: str
    revision: str
    repository_path: str
    output_name: str
    sha256: str


@dataclass(frozen=True, slots=True)
class ImportedCorrectionSource:
    run_dir: Path
    config_path: Path
    reused: bool


SOURCE_SPECS = (
    SourceSpec("experiment_config", "07e771fd", "hqrc_v3/configs/experiment.toml", "experiment.toml", "526416f140e3e2fe15777ee72e3e0155c9dff7d629c13d9ff88f1e032cc29c9f"),
    SourceSpec("model_config", "03d9b68a", "hqrc_v3/configs/model_spaces.toml", "model_spaces.toml", "b06d00880cd4311c5a0ff73c427800bd330f238d757c11165627d70f6465a1e9"),
    SourceSpec("event_registry", "3ed23767", "hqrc_v3/configs/events.csv", "events.csv", "147a823ba37501240460cd4cd53c914ca529c353dd43cf3cc302a65026206e7b"),
    SourceSpec("holiday_calendar", "03d9b68a", "hqrc_v3/configs/holiday_calendar.csv", "holiday_calendar.csv", "a72d93b49c9eaedd469ff04fb14534fcc718c893ac509ebc2152f532609c398e"),
    SourceSpec("temporary_holiday_availability", "03d9b68a", "hqrc_v3/configs/temporary_holiday_availability.csv", "temporary_holiday_availability.csv", "4b695cbbd5e737c43733e842eefb92ce19a6c75d6b0886853696818525fb2b0d"),
)
```

Implement `import_archived_correction_source` with this exact sequence:

1. resolve archive, data, repository, destination, and reject destination/source overlap;
2. verify the data digest is `ccb27f1d5ab7c5e2315bcd4cfc82f3c64d439b1b8d746ba65439fc89062969f9`;
3. enumerate every ZIP member before writing and accept only files below `predictions/`, `inputs/`, or `prediction-stream-cache/` under the fixed archive prefix;
4. reject absolute, drive-qualified, empty, or `..` member paths;
5. extract accepted regular files with `ZipFile.open` plus `shutil.copyfileobj`, never `ZipFile.extract`;
6. materialize each Git source through `subprocess.run(["git", "show", f"{revision}:{repository_path}"], check=True, stdout=PIPE)` and verify its digest;
7. copy the data bytes to `sources/power_demand_final.csv` and verify again;
8. replace only the six source `path` values and recompute the residual manifest digest using canonical JSON;
9. validate the staging run, rewrite the six paths to their final destination, atomically rename staging to destination, and validate the final run;
10. if destination exists, validate it and return `reused=True` without writing.

Use this digest helper so its bytes match `residual_stage.py`:

```python
def residual_manifest_digest(manifest: Mapping[str, object]) -> str:
    unsigned = {key: manifest[key] for key in sorted(manifest) if key != "manifest_sha256"}
    raw = json.dumps(unsigned, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()
```

Add `import-paper-source` to `cli.py` with required `--archive`, `--data`, `--repository-root`, and `--run-dir` arguments. The handler prints the absolute run and config paths returned by the importer.

- [ ] **Step 4: Run importer and CLI routing tests**

Run:

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_artifact_import.py hqrc_v3/tests/integration/test_cli_oof.py -q
```

Expected: all selected tests pass; the injected CLI handler receives all four paths unchanged.

- [ ] **Step 5: Commit the importer**

```powershell
git add hqrc_v3/src/hqrc_v3/artifact_import.py hqrc_v3/src/hqrc_v3/cli.py hqrc_v3/tests/unit/test_artifact_import.py hqrc_v3/tests/integration/test_cli_oof.py
git commit -m "feat: import archived HQRC correction source"
```

### Task 2: NumPyro GPU Sampling Backend

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/bayes/samplers.py:85-205`
- Modify: `hqrc_v3/tests/integration/test_hqrc_sampling.py:88-228`
- Modify: `hqrc_v3/pyproject.toml:20-24`
- Modify: `hqrc_v3/tests/integration/test_packaging.py:50-84`
- Modify: `uv.lock`

**Interfaces:**
- Consumes: existing `HQRCData`, `ApprovedARCalibration`, and `validate_inference_data`.
- Produces: `sample_hqrc(..., backend="numpyro", device="cuda:N")` with backend-specific metadata. Task 3 binds these values into LOEO identity.

- [ ] **Step 1: Write failing NumPyro backend tests**

Add a test that patches the lazy GPU helper and preserves the existing fake ArviZ data shape:

```python
def test_numpyro_backend_uses_vectorized_single_gpu_and_records_provenance(
    hqrc_data, approved_calibration, monkeypatch
):
    captured = {}
    fake = az.from_dict(
        posterior={"event_log_likelihood": np.zeros((2, 3, 2))},
        sample_stats={"diverging": np.zeros((2, 3), dtype=np.int8)},
    )
    monkeypatch.setattr(
        samplers,
        "_sample_numpyro_gpu",
        lambda model, **kwargs: captured.update(kwargs) or (fake, {"device_kind": "NVIDIA GeForce RTX 5060 Ti"}),
    )
    monkeypatch.setattr(samplers, "validate_inference_data", lambda *_args, **_kwargs: SamplingDiagnostics(1.0, 800.0, 700.0, 0))
    monkeypatch.setattr(samplers.importlib.metadata, "version", lambda name: {"pymc": "5", "arviz": "1", "jax": "1", "numpyro": "1"}[name])

    idata = sample_hqrc(
        hqrc_data,
        approved_calibration,
        backend="numpyro",
        device="cuda:1",
        init="numpyro-jitter",
        draws=3,
        tune=2,
        chains=2,
        cores=1,
    )

    sampler = json.loads(idata.attrs["hqrc_sampler_json"])
    assert captured["chain_method"] == "vectorized"
    assert captured["postprocessing_backend"] == "cpu"
    assert sampler["device"] == "cuda:1"
    assert sampler["init"] == "numpyro-jitter"
    assert idata.attrs["hqrc_backend"] == "numpyro"
```

Add rejection cases for native Windows, missing `HQRC_PHYSICAL_GPU`, more than one JAX-visible GPU, `cores != 1`, and a device argument that differs from the scheduler environment.

- [ ] **Step 2: Run the sampler test and confirm the unsupported backend failure**

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/integration/test_hqrc_sampling.py -k numpyro -q
```

Expected: failure because `sample_hqrc` accepts only `pymc` and `nutpie`.

- [ ] **Step 3: Add a lazy single-GPU helper and backend-specific metadata**

Add this public contract and keep imports of JAX and NumPyro inside the helper:

```python
SamplerBackend = Literal["pymc", "nutpie", "numpyro"]


def _sample_numpyro_gpu(
    model,
    *,
    draws: int,
    tune: int,
    chains: int,
    seed: int,
    target_accept: float,
    chain_method: str,
    postprocessing_backend: str,
) -> tuple[az.InferenceData, dict[str, str]]:
    import jax
    from pymc.sampling.jax import sample_numpyro_nuts

    devices = jax.devices("gpu")
    if len(devices) != 1:
        raise SamplingError("NumPyro worker requires exactly one visible GPU")
    idata = sample_numpyro_nuts(
        draws=draws,
        tune=tune,
        chains=chains,
        target_accept=target_accept,
        random_seed=seed,
        jitter=True,
        model=model,
        progressbar=False,
        chain_method=chain_method,
        postprocessing_backend=postprocessing_backend,
        compute_convergence_checks=False,
    )
    return idata, {"device_kind": str(devices[0].device_kind)}
```

In `sample_hqrc`, require Linux, `cores == 1`, `init == "numpyro-jitter"`, and `device == f"cuda:{os.environ['HQRC_PHYSICAL_GPU']}"` before calling the helper. Record `chain_method`, `postprocessing_backend`, `device`, `device_kind`, JAX version, and NumPyro version only for NumPyro. Do not add keys to the existing PyMC or nutpie sampler JSON.

Add this optional dependency group:

```toml
[project.optional-dependencies]
sampler-acceleration = ["nutpie"]
gpu-sampler = [
    "jax[cuda13]>=0.6.2; sys_platform == 'linux'",
    "numpyro>=0.19.0; sys_platform == 'linux'",
]
```

Update the packaging assertion to require `{"jax", "numpyro", "nutpie"} <= optional`, then run `uv lock` to update the workspace lock.

- [ ] **Step 4: Run sampler, packaging, and default-backend regression tests**

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/integration/test_hqrc_sampling.py hqrc_v3/tests/integration/test_packaging.py -q
```

Expected: NumPyro mock tests pass and every pre-existing PyMC test still passes without metadata changes.

- [ ] **Step 5: Commit the GPU backend**

```powershell
git add hqrc_v3/src/hqrc_v3/bayes/samplers.py hqrc_v3/tests/integration/test_hqrc_sampling.py hqrc_v3/tests/integration/test_packaging.py hqrc_v3/pyproject.toml uv.lock
git commit -m "feat: add NumPyro GPU sampler backend"
```

### Task 3: Carry Backend and Device Through LOEO Provenance

**Files:**
- Modify: `hqrc_v3/src/hqrc_v3/_loeo_publication.py:82-148,263-311`
- Modify: `hqrc_v3/src/hqrc_v3/loeo_stage.py:245-510`
- Modify: `hqrc_v3/src/hqrc_v3/loeo_primary.py:26-260`
- Modify: `hqrc_v3/src/hqrc_v3/loeo_ablation.py:214-505`
- Modify: `hqrc_v3/src/hqrc_v3/paper_pipeline.py:98-205`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py:298-318,396-432,575-593`
- Modify: `hqrc_v3/tests/unit/test_loeo_stage.py`
- Modify: `hqrc_v3/tests/unit/test_loeo_primary.py`
- Modify: `hqrc_v3/tests/unit/test_loeo_ablation.py`
- Modify: `hqrc_v3/tests/unit/test_paper_pipeline.py`
- Modify: `hqrc_v3/tests/integration/test_cli_oof.py`

**Interfaces:**
- Consumes: Task 2 `sample_hqrc` backend/device contract.
- Produces: `run_paper_loeo_pipeline(..., backend="pymc", device_id=None)` and `run-loeo-primary --backend/--device-id`. Task 4 invokes that CLI in isolated workers.

- [ ] **Step 1: Write failing provenance and propagation tests**

Add a sampler identity test proving backward compatibility and GPU separation:

```python
def test_numpyro_sampler_identity_is_device_bound_without_changing_pymc_default():
    default = sampler_contract(
        "smoke",
        root_seed=9,
        held_out_occurrence_id="seollal-2024",
        draws=2,
        tune=2,
        chains=2,
    )
    gpu = sampler_contract(
        "smoke",
        root_seed=9,
        held_out_occurrence_id="seollal-2024",
        draws=2,
        tune=2,
        chains=2,
        cores=1,
        init="numpyro-jitter",
        backend="numpyro",
        device_id=1,
    )

    assert default["backend"] == "pymc"
    assert "device" not in default
    assert gpu["backend"] == "numpyro"
    assert gpu["device"] == "cuda:1"
    assert sha_json(default) != sha_json(gpu)
```

Extend pipeline fakes to assert that H1, H2, and H3 receive `backend="numpyro"` and `device_id=1`. Extend CLI routing to assert `--backend numpyro --device-id 1` reaches the pipeline unchanged.

- [ ] **Step 2: Run focused LOEO tests and observe missing keyword failures**

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_primary.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/tests/unit/test_paper_pipeline.py hqrc_v3/tests/integration/test_cli_oof.py -q
```

Expected: new tests fail because the public functions do not accept `backend` or `device_id`.

- [ ] **Step 3: Extend the sampler contract while preserving PyMC bytes**

Change the signature to:

```python
def sampler_contract(
    profile: str,
    *,
    root_seed: int,
    held_out_occurrence_id: str,
    variant: str = "H3",
    draws: int | None,
    tune: int | None,
    chains: int | None,
    cores: int | None = None,
    init: str | None = None,
    target_accept: float | None = None,
    backend: str = "pymc",
    device_id: int | None = None,
) -> dict[str, object]:
```

For `pymc`, reject a device and return the current dictionary byte-for-byte. For `numpyro`, require a non-negative integer device, `cores == 1`, and `init == "numpyro-jitter"`, then add exactly:

```python
{
    "backend": "numpyro",
    "device": f"cuda:{device_id}",
    "chain_method": "vectorized",
    "postprocessing_backend": "cpu",
}
```

Update posterior metadata validation to add `device`, `chain_method`, and
`postprocessing_backend` to `expected_sampler`. Before exact dictionary
comparison, remove `device_kind` from the recorded NumPyro sampler JSON and
require it to be a non-empty string. The physical card model is runtime evidence,
while the requested physical index remains part of immutable identity.

- [ ] **Step 4: Thread the two values through every H1--H3 caller**

Add `backend: str = "pymc"` and `device_id: int | None = None` to these functions and pass them unchanged:

```text
run_paper_loeo_pipeline
  -> fit_loeo_ablation -> _fit_fold -> sampler_contract -> sample_hqrc
  -> fit_loeo_primary -> fit_loeo_fold/load_loeo_fold_material
                      -> sampler_contract -> sample_hqrc
```

Replace hardcoded `backend="pymc"` calls with:

```python
backend=str(sampler["backend"]),
device=None if "device" not in sampler else str(sampler["device"]),
```

Add `--backend {pymc,numpyro}` and `--device-id INT` only to `run-loeo-primary`. Keep `run-paper` on its existing implicit PyMC backend. Validate that NumPyro requires a device and PyMC rejects one before loading source data.

- [ ] **Step 5: Run the LOEO regression set**

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_primary.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/tests/unit/test_paper_pipeline.py hqrc_v3/tests/integration/test_cli_oof.py -q
```

Expected: all tests pass, including the existing assertion that default H3 uses PyMC.

- [ ] **Step 6: Commit provenance propagation**

```powershell
git add hqrc_v3/src/hqrc_v3/_loeo_publication.py hqrc_v3/src/hqrc_v3/loeo_stage.py hqrc_v3/src/hqrc_v3/loeo_primary.py hqrc_v3/src/hqrc_v3/loeo_ablation.py hqrc_v3/src/hqrc_v3/paper_pipeline.py hqrc_v3/src/hqrc_v3/cli.py hqrc_v3/tests/unit/test_loeo_stage.py hqrc_v3/tests/unit/test_loeo_primary.py hqrc_v3/tests/unit/test_loeo_ablation.py hqrc_v3/tests/unit/test_paper_pipeline.py hqrc_v3/tests/integration/test_cli_oof.py
git commit -m "feat: bind GPU sampler to LOEO provenance"
```

### Task 4: Deterministic Two-GPU Model Scheduler

**Files:**
- Create: `hqrc_v3/src/hqrc_v3/gpu_scheduler.py`
- Create: `hqrc_v3/tests/unit/test_gpu_scheduler.py`
- Modify: `hqrc_v3/src/hqrc_v3/cli.py:396-659`
- Modify: `hqrc_v3/tests/integration/test_cli_oof.py`

**Interfaces:**
- Consumes: Task 3 `run-loeo-primary --backend numpyro --device-id N` and an imported source/config path from Task 1.
- Produces: `run_dual_gpu_loeo(request: DualGPURequest) -> DualGPUResult` and `hqrc run-loeo-dual-gpu`.

- [ ] **Step 1: Write scheduler assignment, scope, and failure tests**

```python
def test_assignments_put_one_fixed_model_queue_on_each_gpu():
    assert model_queues(
        models=("lightgbm", "svr", "seq2seq_lstm", "transformer"),
        devices=(0, 1),
    ) == (
        (0, ("lightgbm", "seq2seq_lstm")),
        (1, ("svr", "transformer")),
    )
```

```python
def test_worker_environment_exposes_one_physical_gpu(tmp_path):
    job = build_job(
        model="svr",
        physical_device=1,
        source_run_dir=tmp_path / "source",
        config_path=tmp_path / "experiment.toml",
        output_root=tmp_path / "output",
        profile="smoke",
        feature_set="B1",
        variants=("H1",),
        root_seed=17,
        draws=2,
        tune=2,
        chains=2,
        approve_derived_ar=True,
    )

    assert job.environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert job.environment["HQRC_PHYSICAL_GPU"] == "1"
    assert job.environment["XLA_PYTHON_CLIENT_PREALLOCATE"] == "false"
    assert job.command[-4:] == ("--backend", "numpyro", "--device-id", "1")
    assert "xgboost" not in job.command
```

Add tests that paper rejects any subset, smoke accepts ordered non-XGBoost subsets, duplicate devices fail, native Windows fails before source validation, a failed first model skips its later queue item, and the other in-flight queue may finish.

- [ ] **Step 2: Run scheduler tests and confirm the missing module failure**

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_gpu_scheduler.py -q
```

Expected: collection fails because `hqrc_v3.gpu_scheduler` does not exist.

- [ ] **Step 3: Implement immutable request/job/result types and fixed queues**

```python
GPU_MODELS = ("lightgbm", "svr", "seq2seq_lstm", "transformer")


@dataclass(frozen=True, slots=True)
class GPUJob:
    model: str
    physical_device: int
    command: tuple[str, ...]
    environment: Mapping[str, str]
    log_path: Path


@dataclass(frozen=True, slots=True)
class DualGPUResult:
    completed_models: tuple[str, ...]
    failed_models: tuple[str, ...]
    logs: tuple[Path, ...]
```

Implement `model_queues` with the approved fixed grouping, filtering only for smoke. Validate paper scope before creating output or logs.

- [ ] **Step 4: Implement WSL/JAX preflight and subprocess tee execution**

Before creating logs, the parent resolves the source and output roots, rejects
either directory containing the other, and calls `validate_correction_source`
once with the supplied imported config. It then verifies
`sys.platform.startswith("linux")` and
`"microsoft" in platform.release().lower()`. For each physical device, run a
short child with the same environment as its model worker:

```python
GPU_PROBE = (
    "import json, jax; "
    "d=jax.devices('gpu'); "
    "assert len(d)==1, d; "
    "print(json.dumps({'id': d[0].id, 'kind': d[0].device_kind}))"
)
```

Use two standard-library threads, one per device queue. Each thread runs its models sequentially through `subprocess.Popen`, merges stderr into stdout, prints each line with `[GPU N model]`, and writes the same line to `output_root/logs/<model>-gpu<N>.log`. Stop only that queue after a nonzero exit. After both queues finish, raise `DualGPURunError` if any model failed, including the log and exact command in the message.

- [ ] **Step 5: Add and route `run-loeo-dual-gpu`**

The command requires source/config/output paths, exactly two `--devices`, and standard sampler arguments. It accepts `--models` only for smoke, reuses `--feature-set` and `--variants`, and forwards `--approve-derived-ar`. It does not expose a backend flag because this command is always NumPyro.

Use a lazy import in the handler so normal Windows CLI startup never imports GPU dependencies:

```python
def run_loeo_dual_gpu_handler(arguments: argparse.Namespace) -> object:
    from hqrc_v3.gpu_scheduler import request_from_namespace, run_dual_gpu_loeo

    return run_dual_gpu_loeo(request_from_namespace(arguments))
```

- [ ] **Step 6: Run scheduler and CLI tests**

```powershell
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests/unit/test_gpu_scheduler.py hqrc_v3/tests/integration/test_cli_oof.py -q
```

Expected: fixed mapping, environment, scope, failure, and routing tests pass.

- [ ] **Step 7: Commit the scheduler**

```powershell
git add hqrc_v3/src/hqrc_v3/gpu_scheduler.py hqrc_v3/src/hqrc_v3/cli.py hqrc_v3/tests/unit/test_gpu_scheduler.py hqrc_v3/tests/integration/test_cli_oof.py
git commit -m "feat: schedule HQRC models across two GPUs"
```

### Task 5: Operator Documentation and Full Regression

**Files:**
- Modify: `hqrc_v3/README.md:72-301`

**Interfaces:**
- Consumes: all new CLI commands and flags from Tasks 1--4.
- Produces: one reproducible operator sequence for this repository and archive.

- [ ] **Step 1: Add exact Windows import and WSL2 execution commands**

Document these commands with the real repository paths and explain that the imported config path, not the current working-tree config, must be used:

```powershell
uv run --project hqrc_v3 --locked hqrc import-paper-source `
  --archive artifacts.zip `
  --data "power_demand_final copy.csv" `
  --repository-root . `
  --run-dir artifacts/hqrc-v3-paper-imported-20260819
```

```bash
uv sync --project hqrc_v3 --extra gpu-sampler --locked
uv run --project hqrc_v3 --extra gpu-sampler hqrc run-loeo-dual-gpu \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-dual-gpu-20260819 \
  --devices 0 1 --profile paper \
  --feature-set all --variants H1 H2 H3 \
  --draws 1000 --tune 1000 --chains 4 --cores 1
```

Document the proposal-only first run, reduced smoke rerun with explicit approval, full paper rerun with explicit approval, log locations, nonzero failure behavior, and identical resume command.

- [ ] **Step 2: Run formatting and the complete non-slow test suite**

```powershell
uv run --project hqrc_v3 --locked ruff check hqrc_v3/src hqrc_v3/tests
uv run --project hqrc_v3 --locked pytest hqrc_v3/tests -m "not slow" -q
git diff --check
```

Expected: Ruff exits 0, all non-slow tests pass, and `git diff --check` prints nothing.

- [ ] **Step 3: Commit documentation and regression-complete state**

```powershell
git add hqrc_v3/README.md
git commit -m "docs: add dual-GPU HQRC operator flow"
```

### Task 6: Real Archive, Two-GPU Smoke, and Paper Launch

**Files:**
- Generated, gitignored: `artifacts/hqrc-v3-paper-imported-20260819/`
- Generated, gitignored: `artifacts/hqrc-v3-loeo-dual-gpu-20260819/`

**Interfaces:**
- Consumes: committed CLI and current `artifacts.zip` plus `power_demand_final copy.csv`.
- Produces: validated imported source, AR proposals, two-GPU smoke artifacts, and a resumable paper H1--H3 run.

- [ ] **Step 1: Import and validate the real archive on Windows**

```powershell
uv run --project hqrc_v3 --locked hqrc import-paper-source `
  --archive artifacts.zip `
  --data "power_demand_final copy.csv" `
  --repository-root . `
  --run-dir artifacts/hqrc-v3-paper-imported-20260819
```

Expected: the command prints `reused=False`, the imported config path, and a successful zero-fit correction-source validation. A second identical invocation prints `reused=True` and does not change artifact hashes.

- [ ] **Step 2: Install the locked WSL2 GPU environment and verify both isolated devices**

From WSL2 in `/mnt/c/Users/new92/orca/Demadn-Quadratic-Tilting`:

```bash
uv sync --project hqrc_v3 --extra gpu-sampler --locked
CUDA_VISIBLE_DEVICES=0 uv run --project hqrc_v3 --extra gpu-sampler python -c "import jax; print(jax.devices('gpu'))"
CUDA_VISIBLE_DEVICES=1 uv run --project hqrc_v3 --extra gpu-sampler python -c "import jax; print(jax.devices('gpu'))"
```

Expected: each command prints exactly one RTX 5060 Ti GPU.

- [ ] **Step 3: Generate every AR proposal without sampling**

```bash
uv run --project hqrc_v3 --extra gpu-sampler hqrc run-loeo-dual-gpu \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-dual-gpu-20260819 \
  --devices 0 1 --profile paper \
  --feature-set all --variants H1 H2 H3 \
  --draws 1000 --tune 1000 --chains 4 --cores 1
```

Expected: eight non-XGBoost model/feature contexts report `AR_REVIEW_REQUIRED`; no posterior file is created.

- [ ] **Step 4: Review proposal plots and digests**

For every model and B0/B1 context, verify the ACF/PACF plot is readable, the proposal set contains ten folds, all proposed Beta parameters are finite and positive, and the logged proposal digest equals the staged proposal-set digest. Record the eight accepted digests in the execution notes before using the approval flag.

- [ ] **Step 5: Run a real two-model H1 smoke on both GPUs**

```bash
uv run --project hqrc_v3 --extra gpu-sampler hqrc run-loeo-dual-gpu \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-dual-gpu-20260819 \
  --devices 0 1 --profile smoke \
  --models lightgbm svr --feature-set B1 --variants H1 \
  --draws 20 --tune 20 --chains 2 --cores 1 \
  --approve-derived-ar
```

Expected: GPU 0 runs LightGBM and GPU 1 runs SVR concurrently; both produce ten H1 fold posteriors and complete aggregates. Repeating the command reports zero sampler fits and unchanged hashes.

- [ ] **Step 6: Launch the full resumable paper matrix**

```bash
uv run --project hqrc_v3 --extra gpu-sampler hqrc run-loeo-dual-gpu \
  --source-run-dir artifacts/hqrc-v3-paper-imported-20260819 \
  --config artifacts/hqrc-v3-paper-imported-20260819/sources/experiment.toml \
  --output-root artifacts/hqrc-v3-loeo-dual-gpu-20260819 \
  --devices 0 1 --profile paper \
  --feature-set all --variants H1 H2 H3 \
  --draws 1000 --tune 1000 --chains 4 --cores 1 \
  --approve-derived-ar
```

Expected: XGBoost never appears in a scheduled job or new output path. GPU 0 processes LightGBM then Seq2Seq-LSTM, GPU 1 processes SVR then Transformer, with one model per card. Every completed fold passes the paper diagnostic gate and is reusable after interruption.

- [ ] **Step 7: Capture final verification evidence**

```bash
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv
find artifacts/hqrc-v3-loeo-dual-gpu-20260819 -name COMPLETE -type f | wc -l
grep -R "xgboost" artifacts/hqrc-v3-loeo-dual-gpu-20260819/logs || true
```

Expected during sampling: two worker PIDs, one per GPU. Expected after completion: no scheduled XGBoost log entry and complete markers for every requested fold and aggregate namespace.
