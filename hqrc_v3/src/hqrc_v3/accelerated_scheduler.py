"""Native subprocess scheduling for the H1--H3 accelerator workload."""

from __future__ import annotations

import json
import math
import os
import platform
import shlex
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PAPER_MODELS = ("lightgbm", "svr", "seq2seq_lstm", "transformer")
PAPER_FEATURE_SETS = ("B0", "B1")
PAPER_VARIANTS = ("H1", "H2", "H3")
CPU_THREAD_ENVIRONMENT_NAMES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class AcceleratedRequest:
    source_run_dir: Path
    config_path: Path
    output_root: Path
    profile: str = "paper"
    models: tuple[str, ...] = PAPER_MODELS
    feature_sets: tuple[str, ...] = PAPER_FEATURE_SETS
    variants: tuple[str, ...] = PAPER_VARIANTS
    devices: tuple[int, ...] = (0, 1)
    accelerator: str = "auto"
    root_seed: int = 20260813
    draws: int | None = None
    tune: int | None = None
    chains: int | None = None
    cores: int | None = None
    init: str | None = None
    target_accept: float | None = None
    approve_derived_ar: bool = False


@dataclass(frozen=True, slots=True)
class AcceleratedResult:
    completed_models: tuple[str, ...]
    failed_models: tuple[str, ...]
    logs: tuple[Path, ...]
    accelerator: str
    logical_device: str
    physical_devices: tuple[int, ...]
    probe_detail: str
    fallback_reason: str | None


class AcceleratedRunError(ValueError):
    """Raised when accelerator preflight or one scheduled model fails."""

    def __init__(self, message: str, *, result: AcceleratedResult | None = None):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True, slots=True)
class _ResolvedAccelerator:
    kind: str
    logical_device: str
    physical_device: str | None
    probe_detail: str
    fallback_reason: str | None


@dataclass(frozen=True, slots=True)
class _Job:
    model: str
    physical_device: int | None
    command: tuple[str, ...]
    environment: dict[str, str]
    log_path: Path


@dataclass(frozen=True, slots=True)
class _JobOutcome:
    job: _Job
    returncode: int


def model_queues(devices: tuple[int, ...]) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """Return one sequential queue or the fixed two-device manuscript queues."""

    if (
        len(devices) not in {1, 2}
        or len(set(devices)) != len(devices)
        or any(
            isinstance(device, bool) or not isinstance(device, int) or device < 0
            for device in devices
        )
    ):
        raise AcceleratedRunError("CUDA scheduling requires one or two unique non-negative devices")
    if len(devices) == 1:
        return ((f"cuda:{devices[0]}", PAPER_MODELS),)
    if devices != (0, 1):
        raise AcceleratedRunError(
            "two-device scheduling requires exactly physical devices 0 then 1"
        )
    return (
        ("cuda:0", ("lightgbm", "seq2seq_lstm")),
        ("cuda:1", ("svr", "transformer")),
    )


def _cpu_queue_specs(
    models: tuple[str, ...], logical_cpus: int | None = None
) -> tuple[tuple[tuple[str, ...], int], ...]:
    available = max(1, logical_cpus if logical_cpus is not None else (os.cpu_count() or 1))
    worker_count = min(len(models), available, 2)
    if not worker_count:
        return ()
    return tuple((tuple(models[index::worker_count]), 1) for index in range(worker_count))


def _ordered_subset(value: tuple[str, ...], complete: tuple[str, ...], description: str) -> None:
    if (
        not value
        or len(set(value)) != len(value)
        or any(item not in complete for item in value)
        or tuple(item for item in complete if item in value) != value
    ):
        raise AcceleratedRunError(f"{description} must be a non-empty ordered subset")


def _exact_int(value: object, *, minimum: int) -> bool:
    return type(value) is int and value >= minimum


def _exact_target(value: object, expected: float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and float(value) == expected
    )


def _validate_request(request: AcceleratedRequest) -> None:
    if not isinstance(request, AcceleratedRequest):
        raise TypeError("request must be an AcceleratedRequest")
    if "xgboost" in request.models:
        raise AcceleratedRunError("XGBoost is excluded from the accelerated paper scope")
    model_queues(request.devices)
    if request.profile == "paper":
        if request.models != PAPER_MODELS:
            raise AcceleratedRunError("paper models must match the exact accelerated scope")
        if request.feature_sets != PAPER_FEATURE_SETS:
            raise AcceleratedRunError("paper feature sets must be exactly B0 then B1")
        if request.variants != PAPER_VARIANTS:
            raise AcceleratedRunError("paper variants must be exactly H1, H2, H3")
    elif request.profile == "smoke":
        _ordered_subset(request.models, PAPER_MODELS, "smoke models")
        _ordered_subset(request.feature_sets, PAPER_FEATURE_SETS, "smoke feature sets")
        _ordered_subset(request.variants, PAPER_VARIANTS, "smoke variants")
        if (
            not _exact_int(request.draws, minimum=1)
            or not _exact_int(request.tune, minimum=1)
            or not _exact_int(request.chains, minimum=4)
            or request.chains != 4
            or request.cores is not None
            and (not _exact_int(request.cores, minimum=1) or request.cores not in {1, 4})
            or not _exact_target(request.target_accept, 0.9)
        ):
            raise AcceleratedRunError(
                "smoke Pyro requires positive draws/tune, chains=4, "
                "optional cores=1 or cores=4, and target_accept=0.9"
            )
    else:
        raise AcceleratedRunError("profile must be paper or smoke")
    if request.profile == "paper" and (
        request.draws is not None
        and not _exact_int(request.draws, minimum=1000)
        or request.tune is not None
        and not _exact_int(request.tune, minimum=1000)
        or request.chains is not None
        and (not _exact_int(request.chains, minimum=4) or request.chains != 4)
        or request.cores is not None
        and (not _exact_int(request.cores, minimum=1) or request.cores not in {1, 4})
        or request.target_accept is not None
        and not _exact_target(request.target_accept, 0.99)
    ):
        raise AcceleratedRunError(
            "paper Pyro requires draws/tune >=1000, chains=4, "
            "optional cores=1 or cores=4, and target_accept=0.99"
        )
    if request.init is not None:
        raise AcceleratedRunError("accelerated Pyro does not accept an init override")
    if not isinstance(request.approve_derived_ar, bool):
        raise AcceleratedRunError("approve_derived_ar must be boolean")
    if (
        isinstance(request.root_seed, bool)
        or not isinstance(request.root_seed, int)
        or request.root_seed < 0
    ):
        raise AcceleratedRunError("root seed must be a non-negative integer")
    if request.accelerator not in {"auto", "cuda", "mps", "cpu"}:
        raise AcceleratedRunError("accelerator must be auto, cuda, mps, or cpu")


def _resolved_paths(request: AcceleratedRequest) -> tuple[Path, Path, Path]:
    try:
        source = Path(request.source_run_dir).expanduser().resolve(strict=True)
        config = Path(request.config_path).expanduser().resolve(strict=True)
        output = Path(request.output_root).expanduser().resolve()
    except OSError as error:
        raise AcceleratedRunError("accelerated source path is missing or unreadable") from error
    if config != source / "sources" / "experiment.toml":
        raise AcceleratedRunError("accelerated runs require the imported config path")
    if output == source or output.is_relative_to(source) or source.is_relative_to(output):
        raise AcceleratedRunError("source and output roots must not overlap")
    return source, config, output


_SOURCE_VALIDATION = (
    "import sys; from pathlib import Path; "
    "from hqrc_v3.correction_source import validate_correction_source; "
    "validate_correction_source(run_dir=Path(sys.argv[1]), config_path=Path(sys.argv[2]), "
    "profile=sys.argv[3])"
)


def validate_correction_source(*, run_dir: Path, config_path: Path, profile: str) -> None:
    """Validate the imported source without importing Torch into the scheduler parent."""

    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("HQRC_PHYSICAL_DEVICE", None)
    completed = subprocess.run(
        (sys.executable, "-c", _SOURCE_VALIDATION, str(run_dir), str(config_path), profile),
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown validation error"
        raise AcceleratedRunError(f"imported correction source validation failed: {detail}")


_DEVICE_PROBE = (
    "import json,sys; from hqrc_v3.accelerators import resolve_device; "
    "d=resolve_device(sys.argv[1]); print(json.dumps({"
    "'kind':d.kind,'logical_device':d.logical_device,'physical_device':d.physical_device,"
    "'probe':d.probe.detail,'fallback_reason':d.fallback_reason}))"
)


def _probe_device(request: str, *, physical_device: int | None = None) -> _ResolvedAccelerator:
    environment = os.environ.copy()
    if physical_device is None:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.pop("HQRC_PHYSICAL_DEVICE", None)
    else:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        environment["HQRC_PHYSICAL_DEVICE"] = str(physical_device)
    completed = subprocess.run(
        (sys.executable, "-c", _DEVICE_PROBE, request),
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown probe error"
        raise AcceleratedRunError(f"{request} accelerator probe failed: {detail}")
    try:
        line = next(line for line in reversed(completed.stdout.splitlines()) if line.strip())
        payload = json.loads(line)
        return _ResolvedAccelerator(
            str(payload["kind"]),
            str(payload["logical_device"]),
            None if payload["physical_device"] is None else str(payload["physical_device"]),
            str(payload["probe"]),
            None if payload["fallback_reason"] is None else str(payload["fallback_reason"]),
        )
    except (KeyError, TypeError, ValueError, StopIteration, json.JSONDecodeError) as error:
        raise AcceleratedRunError("accelerator probe returned invalid provenance") from error


def _preflight_accelerator(
    request: str, devices: tuple[int, ...], *, platform_name: str | None = None
) -> _ResolvedAccelerator:
    system = platform.system() if platform_name is None else platform_name
    if request == "cuda":
        probes = tuple(_probe_device("cuda:0", physical_device=device) for device in devices)
        if any(
            probe.kind != "cuda"
            or probe.logical_device != "cuda:0"
            or probe.physical_device != str(device)
            for probe, device in zip(probes, devices, strict=True)
        ):
            raise AcceleratedRunError("isolated CUDA probe identity differs")
        return _ResolvedAccelerator("cuda", "cuda:0", None, probes[0].probe_detail, None)
    if request == "auto" and system != "Darwin":
        probes: list[_ResolvedAccelerator] = []
        try:
            for device in devices:
                probes.append(_probe_device("cuda:0", physical_device=device))
        except AcceleratedRunError as error:
            cpu = _probe_device("cpu")
            return _ResolvedAccelerator(
                "cpu", "cpu", None, cpu.probe_detail, f"CUDA fallback: {error}"
            )
        if all(
            probe.kind == "cuda"
            and probe.logical_device == "cuda:0"
            and probe.physical_device == str(device)
            for probe, device in zip(probes, devices, strict=True)
        ):
            return _ResolvedAccelerator("cuda", "cuda:0", None, probes[0].probe_detail, None)
        cpu = _probe_device("cpu")
        return _ResolvedAccelerator(
            "cpu", "cpu", None, cpu.probe_detail, "CUDA fallback: isolated identity differs"
        )
    resolved = _probe_device(request)
    if resolved.kind == "cuda":
        raise AcceleratedRunError("native CUDA scheduling requires two isolated physical devices")
    return resolved


def _feature_argument(feature_sets: tuple[str, ...]) -> str:
    return "all" if feature_sets == PAPER_FEATURE_SETS else feature_sets[0]


def _chain_workers(request: AcceleratedRequest, resolved: _ResolvedAccelerator) -> int:
    required = 4 if resolved.kind == "cpu" else 1
    if request.cores is not None and request.cores != required:
        raise AcceleratedRunError(
            f"resolved {resolved.kind} Pyro requires cores={required}, not cores={request.cores}"
        )
    return required


def _build_job(
    request: AcceleratedRequest,
    model: str,
    *,
    physical_device: int | None,
    device: str,
    chain_workers: int | None = None,
    thread_budget: int | None = None,
) -> _Job:
    command = [
        sys.executable,
        "-m",
        "hqrc_v3.cli",
        "run-loeo-primary",
        "--source-run-dir",
        str(Path(request.source_run_dir).resolve()),
        "--config",
        str(Path(request.config_path).resolve()),
        "--output-root",
        str(Path(request.output_root).resolve()),
        "--model",
        model,
        "--feature-set",
        _feature_argument(request.feature_sets),
        "--variants",
        *request.variants,
        "--profile",
        request.profile,
        "--root-seed",
        str(request.root_seed),
    ]
    for option, value in (
        ("--draws", request.draws),
        ("--tune", request.tune),
        ("--chains", request.chains),
        ("--cores", request.cores if chain_workers is None else chain_workers),
        ("--init", request.init),
        ("--target-accept", request.target_accept),
    ):
        if value is not None:
            command.extend((option, str(value)))
    if request.approve_derived_ar:
        command.append("--approve-derived-ar")
    command.extend(("--backend", "pyro", "--device", device))

    environment = os.environ.copy()
    environment.pop("CUDA_VISIBLE_DEVICES", None)
    environment.pop("HQRC_PHYSICAL_DEVICE", None)
    if physical_device is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(physical_device)
        environment["HQRC_PHYSICAL_DEVICE"] = str(physical_device)
        suffix = f"cuda-{physical_device}"
    else:
        suffix = device.replace(":", "-")
    if thread_budget is not None:
        for name in CPU_THREAD_ENVIRONMENT_NAMES:
            environment[name] = str(thread_budget)
    return _Job(
        model=model,
        physical_device=physical_device,
        command=tuple(command),
        environment=environment,
        log_path=Path(request.output_root).resolve() / "logs" / f"{model}-{suffix}.log",
    )


def _run_job(job: _Job) -> _JobOutcome:
    job.log_path.parent.mkdir(parents=True, exist_ok=True)
    label = (
        f"CUDA {job.physical_device}"
        if job.physical_device is not None
        else job.command[-1].upper()
    )
    prior_evidence = job.log_path.is_file() and job.log_path.stat().st_size > 0
    with job.log_path.open("a", encoding="utf-8", buffering=1) as log:
        if prior_evidence:
            log.write("\n=== HQRC WORKER ATTEMPT ===\n")
        if job.command[-1] == "cpu":
            log.write("=== HQRC CPU ENVIRONMENT ===\n")
            for name in CPU_THREAD_ENVIRONMENT_NAMES:
                log.write(f"{name}={job.environment[name]}\n")
        try:
            process = subprocess.Popen(
                job.command,
                env=job.environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for raw_line in process.stdout:
                line = f"[{label} {job.model}] {raw_line.rstrip()}\n"
                print(line, end="", flush=True)
                log.write(line)
            returncode = process.wait()
        except OSError as error:
            line = f"[{label} {job.model}] worker launch failed: {error}\n"
            print(line, end="", flush=True)
            log.write(line)
            returncode = 1
        log.flush()
        os.fsync(log.fileno())
    return _JobOutcome(job, returncode)


def _restart_command(command: tuple[str, ...], *, platform_name: str | None = None) -> str:
    return (
        subprocess.list2cmdline(command)
        if (os.name if platform_name is None else platform_name) == "nt"
        else shlex.join(command)
    )


def _restart_environment(job: _Job) -> str:
    if job.command[-1] == "cpu":
        return ",".join(f"{name}={job.environment[name]}" for name in CPU_THREAD_ENVIRONMENT_NAMES)
    if job.physical_device is not None:
        return (
            f"CUDA_VISIBLE_DEVICES={job.environment.get('CUDA_VISIBLE_DEVICES', 'unset')},"
            f"HQRC_PHYSICAL_DEVICE={job.environment.get('HQRC_PHYSICAL_DEVICE', 'unset')}"
        )
    return "none"


def _result(
    outcomes: list[_JobOutcome], request: AcceleratedRequest, resolved: _ResolvedAccelerator
) -> AcceleratedResult:
    by_model = {outcome.job.model: outcome for outcome in outcomes}
    ordered = tuple(by_model[model] for model in PAPER_MODELS if model in by_model)
    return AcceleratedResult(
        completed_models=tuple(item.job.model for item in ordered if item.returncode == 0),
        failed_models=tuple(item.job.model for item in ordered if item.returncode != 0),
        logs=tuple(item.job.log_path for item in ordered),
        accelerator=resolved.kind,
        logical_device=resolved.logical_device,
        physical_devices=request.devices if resolved.kind == "cuda" else (),
        probe_detail=resolved.probe_detail,
        fallback_reason=resolved.fallback_reason,
    )


def run_accelerated_loeo(request: AcceleratedRequest) -> AcceleratedResult:
    """Preflight and execute the requested model queues without loading Torch in the parent."""

    _validate_request(request)
    source, config, _output = _resolved_paths(request)
    validate_correction_source(run_dir=source, config_path=config, profile=request.profile)
    resolved = _preflight_accelerator(request.accelerator, request.devices)
    chain_workers = _chain_workers(request, resolved)

    if resolved.kind == "cuda":
        queue_specs = tuple(
            (
                int(device.removeprefix("cuda:")),
                tuple(model for model in models if model in request.models),
                None,
            )
            for device, models in model_queues(request.devices)
        )
    elif resolved.kind == "cpu":
        queue_specs = tuple(
            (None, models, threads) for models, threads in _cpu_queue_specs(request.models)
        )
    else:
        queue_specs = ((None, request.models, None),)

    stop = threading.Event()
    lock = threading.Lock()
    outcomes: list[_JobOutcome] = []

    def run_queue(
        physical_device: int | None, models: tuple[str, ...], thread_budget: int | None
    ) -> None:
        for model in models:
            if stop.is_set():
                break
            job = _build_job(
                request,
                model,
                physical_device=physical_device,
                device="cuda:0" if physical_device is not None else resolved.logical_device,
                chain_workers=chain_workers,
                thread_budget=thread_budget,
            )
            outcome = _run_job(job)
            with lock:
                outcomes.append(outcome)
            if outcome.returncode:
                stop.set()
                break

    active = tuple((device, models, threads) for device, models, threads in queue_specs if models)
    with ThreadPoolExecutor(max_workers=len(active), thread_name_prefix="hqrc-accelerator") as pool:
        futures = tuple(
            pool.submit(run_queue, device, models, threads) for device, models, threads in active
        )
        for future in futures:
            future.result()

    result = _result(outcomes, request, resolved)
    if result.failed_models:
        failures = [outcome for outcome in outcomes if outcome.returncode]
        details = []
        for item in failures:
            physical = item.job.physical_device
            details.append(
                f"{item.job.model} exit={item.returncode} "
                f"physical_device={physical if physical is not None else 'none'} "
                f"log={item.job.log_path} restart_env={_restart_environment(item.job)} "
                f"restart_command={_restart_command(item.job.command)}"
            )
        detail = "; ".join(details)
        raise AcceleratedRunError(f"accelerated LOEO failed: {detail}", result=result)
    return result


def request_from_namespace(arguments: Any) -> AcceleratedRequest:
    """Build an immutable scheduler request from the shared CLI namespace."""

    return AcceleratedRequest(
        source_run_dir=Path(arguments.source_run_dir),
        config_path=Path(arguments.config),
        output_root=Path(arguments.output_root),
        profile=arguments.profile,
        models=tuple(arguments.models),
        feature_sets=tuple(arguments.feature_sets),
        variants=tuple(arguments.variants),
        devices=tuple(arguments.devices),
        accelerator=arguments.accelerator,
        root_seed=arguments.root_seed,
        draws=arguments.draws,
        tune=arguments.tune,
        chains=arguments.chains,
        cores=arguments.cores,
        init=arguments.init,
        target_accept=arguments.target_accept,
        approve_derived_ar=arguments.approve_derived_ar,
    )


__all__ = [
    "AcceleratedRequest",
    "AcceleratedResult",
    "AcceleratedRunError",
    "model_queues",
    "request_from_namespace",
    "run_accelerated_loeo",
]
