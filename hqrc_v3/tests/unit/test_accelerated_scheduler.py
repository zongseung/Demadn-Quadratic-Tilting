"""Focused contracts for native HQRC accelerator scheduling."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
from fractions import Fraction
from pathlib import Path

import pytest

import hqrc_v3.accelerated_scheduler as scheduler
from hqrc_v3.accelerated_scheduler import (
    AcceleratedRequest,
    AcceleratedRunError,
    model_queues,
    run_accelerated_loeo,
)

PAPER_MODELS = ("lightgbm", "svr", "seq2seq_lstm", "transformer")


def _request(tmp_path: Path, **overrides) -> AcceleratedRequest:
    source = tmp_path / "imported"
    config = source / "sources" / "experiment.toml"
    config.parent.mkdir(parents=True)
    config.write_text("[experiment]\n")
    values = {
        "source_run_dir": source,
        "config_path": config,
        "output_root": tmp_path / "output",
        "profile": "paper",
        "models": PAPER_MODELS,
        "feature_sets": ("B0", "B1"),
        "variants": ("H1", "H2", "H3"),
        "devices": (0, 1),
    }
    values.update(overrides)
    return AcceleratedRequest(**values)


def _validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(scheduler, "validate_correction_source", lambda **_kwargs: object())


def test_fixed_two_cuda_queues_exclude_xgboost():
    assert model_queues((0, 1)) == (
        ("cuda:0", ("lightgbm", "seq2seq_lstm")),
        ("cuda:1", ("svr", "transformer")),
    )
    assert all("xgboost" not in models for _, models in model_queues((0, 1)))
    with pytest.raises(AcceleratedRunError, match="two unique"):
        model_queues((0, 0))


def test_single_cuda_device_runs_all_models_in_one_sequential_queue():
    assert model_queues((3,)) == (("cuda:3", PAPER_MODELS),)


def test_cpu_two_logical_cpus_create_ordered_one_thread_queues():
    assert scheduler._cpu_queue_specs(PAPER_MODELS, logical_cpus=2) == (
        (("lightgbm", "seq2seq_lstm"), 1),
        (("svr", "transformer"), 1),
    )


def test_cpu_scheduler_runs_all_models_concurrently_with_even_thread_budgets(
    tmp_path, monkeypatch
):
    request = _request(tmp_path, accelerator="cpu")
    _validated(monkeypatch)
    monkeypatch.setattr(
        scheduler,
        "_preflight_accelerator",
        lambda *_args, **_kwargs: scheduler._ResolvedAccelerator("cpu", "cpu", None, "probe", None),
    )
    monkeypatch.setattr(scheduler.os, "cpu_count", lambda: 24)
    all_started = threading.Barrier(4)
    environments = {}

    def run_job(job):
        environments[job.model] = job.environment
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        job.log_path.write_text(f"{job.model}\n")
        all_started.wait(timeout=2)
        return scheduler._JobOutcome(job, 0)

    monkeypatch.setattr(scheduler, "_run_job", run_job)
    result = run_accelerated_loeo(request)

    assert result.completed_models == PAPER_MODELS
    assert set(environments) == set(PAPER_MODELS)
    for environment in environments.values():
        assert {
            name: environment[name]
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        } == {
            "OMP_NUM_THREADS": "6",
            "MKL_NUM_THREADS": "6",
            "OPENBLAS_NUM_THREADS": "6",
            "NUMEXPR_NUM_THREADS": "6",
            "VECLIB_MAXIMUM_THREADS": "6",
        }


@pytest.mark.parametrize("devices", ((1, 0), (2, 3)))
def test_two_cuda_devices_require_physical_zero_then_one_before_validation_or_logs(
    tmp_path, monkeypatch, devices
):
    calls = []
    monkeypatch.setattr(
        scheduler, "validate_correction_source", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        scheduler,
        "_preflight_accelerator",
        lambda *_args, **_kwargs: scheduler._ResolvedAccelerator("cpu", "cpu", None, "probe", None),
    )
    monkeypatch.setattr(scheduler, "_run_job", lambda job: scheduler._JobOutcome(job, 0))
    request = _request(tmp_path, devices=devices)
    with pytest.raises(AcceleratedRunError, match="exactly physical devices 0 then 1"):
        run_accelerated_loeo(request)
    assert calls == []
    assert not request.output_root.exists()


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"models": ("xgboost", *PAPER_MODELS)}, "XGBoost"),
        ({"models": PAPER_MODELS[:-1]}, "paper models"),
        ({"feature_sets": ("B1", "B0")}, "paper feature sets"),
        ({"variants": ("H1", "H3")}, "paper variants"),
    ],
)
def test_paper_scope_is_exact_and_rejected_before_logs(tmp_path, overrides, match):
    request = _request(tmp_path, **overrides)
    with pytest.raises(AcceleratedRunError, match=match):
        run_accelerated_loeo(request)
    assert not request.output_root.exists()


def test_smoke_accepts_only_ordered_non_xgboost_subsets(tmp_path):
    good = _request(
        tmp_path,
        profile="smoke",
        models=("svr", "transformer"),
        feature_sets=("B1",),
        variants=("H1", "H3"),
        draws=4,
        tune=4,
        chains=4,
        cores=1,
        target_accept=0.9,
    )
    scheduler._validate_request(good)
    for bad in (
        _request(tmp_path / "a", profile="smoke", models=("transformer", "svr")),
        _request(tmp_path / "b", profile="smoke", models=("xgboost",)),
        _request(tmp_path / "c", profile="smoke", variants=("H3", "H1")),
    ):
        with pytest.raises(AcceleratedRunError):
            scheduler._validate_request(bad)


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"draws": 0, "tune": 4, "chains": 4, "cores": 1, "target_accept": 0.9},
        {"draws": 4, "tune": 4, "chains": 2, "cores": 1, "target_accept": 0.9},
        {"draws": 4, "tune": 4, "chains": 4, "cores": 2, "target_accept": 0.9},
        {"draws": 4, "tune": 4, "chains": 4, "cores": 1, "target_accept": 0.99},
    ],
)
def test_invalid_smoke_sampler_contract_fails_before_validation_or_logs(
    tmp_path, monkeypatch, overrides
):
    calls = []
    monkeypatch.setattr(
        scheduler, "validate_correction_source", lambda **kwargs: calls.append(kwargs)
    )
    request = _request(tmp_path, profile="smoke", models=("svr",), **overrides)
    with pytest.raises(AcceleratedRunError, match="smoke Pyro"):
        run_accelerated_loeo(request)
    assert calls == []
    assert not request.output_root.exists()


@pytest.mark.parametrize(
    ("profile", "overrides"),
    [
        ("paper", {"draws": 1000.0}),
        ("paper", {"draws": "1000"}),
        ("paper", {"chains": 4.0}),
        ("paper", {"cores": 1.0}),
        ("paper", {"target_accept": float("nan")}),
        ("paper", {"target_accept": "0.99"}),
        ("paper", {"target_accept": Fraction(99, 100)}),
        (
            "smoke",
            {"draws": 4.0, "tune": 4, "chains": 4, "cores": 1, "target_accept": 0.9},
        ),
        (
            "smoke",
            {"draws": 4, "tune": 4, "chains": 4.0, "cores": 1, "target_accept": 0.9},
        ),
        (
            "smoke",
            {"draws": 4, "tune": 4, "chains": 4, "cores": 1.0, "target_accept": 0.9},
        ),
        ("paper", {"approve_derived_ar": "yes"}),
    ],
)
def test_programmatic_sampler_types_fail_before_validation_or_logs(
    tmp_path, monkeypatch, profile, overrides
):
    calls = []
    monkeypatch.setattr(
        scheduler, "validate_correction_source", lambda **kwargs: calls.append(kwargs)
    )
    monkeypatch.setattr(
        scheduler,
        "_preflight_accelerator",
        lambda *_args, **_kwargs: scheduler._ResolvedAccelerator("cpu", "cpu", None, "probe", None),
    )
    monkeypatch.setattr(scheduler, "_run_job", lambda job: scheduler._JobOutcome(job, 0))
    request = _request(tmp_path, profile=profile, **overrides)
    with pytest.raises(AcceleratedRunError):
        run_accelerated_loeo(request)
    assert calls == []
    assert not request.output_root.exists()


def test_source_output_overlap_and_nonimported_config_fail_before_validation(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        scheduler, "validate_correction_source", lambda **kwargs: calls.append(kwargs)
    )
    source = tmp_path / "source"
    config = source / "experiment.toml"
    source.mkdir()
    config.write_text("x")
    with pytest.raises(AcceleratedRunError, match="imported config"):
        run_accelerated_loeo(
            AcceleratedRequest(
                source_run_dir=source,
                config_path=config,
                output_root=tmp_path / "output",
            )
        )
    assert calls == []

    imported = source / "sources" / "experiment.toml"
    imported.parent.mkdir()
    imported.write_text("x")
    with pytest.raises(AcceleratedRunError, match="overlap"):
        run_accelerated_loeo(
            AcceleratedRequest(
                source_run_dir=source,
                config_path=imported,
                output_root=source / "products",
            )
        )
    assert calls == []


def test_probe_runs_in_subprocess_with_isolated_cuda_environment(monkeypatch):
    calls = []

    def fake_run(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "kind": "cuda",
                    "logical_device": "cuda:0",
                    "physical_device": "1",
                    "probe": "float64-gradient-lkj-ar",
                    "fallback_reason": None,
                }
            )
            + "\n",
            stderr="",
        )

    monkeypatch.setattr(scheduler.subprocess, "run", fake_run)
    resolved = scheduler._probe_device("cuda:0", physical_device=1)
    command, options = calls[0]
    assert command[:2] == (sys.executable, "-c")
    assert "resolve_device" in command[2]
    assert command[-1] == "cuda:0"
    assert options["env"]["CUDA_VISIBLE_DEVICES"] == "1"
    assert options["env"]["HQRC_PHYSICAL_DEVICE"] == "1"
    assert resolved.logical_device == "cuda:0"


def test_scheduler_module_does_not_import_torch_in_parent():
    command = (
        sys.executable,
        "-c",
        "import hqrc_v3.accelerated_scheduler,hqrc_v3.cli,sys; print('torch' in sys.modules)",
    )
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    assert result.stdout.strip() == "False"


def test_auto_cuda_preflight_falls_back_to_sequential_cpu_with_provenance(monkeypatch):
    calls = []

    def probe(request, *, physical_device=None):
        calls.append((request, physical_device))
        if physical_device is not None:
            raise AcceleratedRunError("CUDA unavailable")
        return scheduler._ResolvedAccelerator("cpu", "cpu", None, "cpu-probe", None)

    monkeypatch.setattr(scheduler, "_probe_device", probe)
    resolved = scheduler._preflight_accelerator("auto", (0, 1), platform_name="Windows")
    assert calls == [("cuda:0", 0), ("cpu", None)]
    assert resolved.kind == "cpu"
    assert "CUDA unavailable" in resolved.fallback_reason


def test_cuda_job_isolates_physical_gpu_and_uses_logical_cuda_zero(tmp_path):
    request = _request(
        tmp_path,
        profile="smoke",
        models=("svr",),
        draws=4,
        tune=4,
        chains=4,
        cores=1,
        target_accept=0.9,
    )
    job = scheduler._build_job(request, "svr", physical_device=1, device="cuda:0")
    assert job.environment["CUDA_VISIBLE_DEVICES"] == "1"
    assert job.environment["HQRC_PHYSICAL_DEVICE"] == "1"
    assert job.command[-4:] == ("--backend", "pyro", "--device", "cuda:0")
    assert "xgboost" not in job.command


def test_failed_queue_stops_later_model_joins_peer_and_preserves_logs(tmp_path, monkeypatch):
    request = _request(tmp_path)
    _validated(monkeypatch)
    monkeypatch.setattr(
        scheduler,
        "_preflight_accelerator",
        lambda *_args, **_kwargs: scheduler._ResolvedAccelerator(
            "cuda", "cuda:0", None, "probe", None
        ),
    )
    started = []
    both_started = threading.Barrier(2)
    failed = threading.Event()

    def run_job(job):
        started.append(job.model)
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        job.log_path.write_text(f"{job.model}\n")
        if job.model in {"lightgbm", "svr"}:
            both_started.wait(timeout=2)
        if job.model == "lightgbm":
            failed.set()
            return scheduler._JobOutcome(job, 7)
        if job.model == "svr":
            assert failed.wait(timeout=2)
            threading.Event().wait(0.05)
        return scheduler._JobOutcome(job, 0)

    monkeypatch.setattr(scheduler, "_run_job", run_job)
    with pytest.raises(AcceleratedRunError, match="lightgbm") as error:
        run_accelerated_loeo(request)

    assert set(started) == {"lightgbm", "svr"}
    assert error.value.result.completed_models == ("svr",)
    assert error.value.result.failed_models == ("lightgbm",)
    assert {path.name for path in error.value.result.logs} == {
        "lightgbm-cuda-0.log",
        "svr-cuda-1.log",
    }
    assert all(path.is_file() for path in error.value.result.logs)
    failure = str(error.value)
    assert "physical_device=0" in failure
    assert "CUDA_VISIBLE_DEVICES=0" in failure
    assert "HQRC_PHYSICAL_DEVICE=0" in failure
    assert "restart_command=" in failure


def test_worker_tees_stdout_and_stderr_to_console_and_durable_log(tmp_path, monkeypatch, capsys):
    request = _request(
        tmp_path,
        profile="smoke",
        models=("svr",),
        draws=4,
        tune=4,
        chains=4,
        cores=1,
        target_accept=0.9,
    )
    job = scheduler._build_job(request, "svr", physical_device=1, device="cuda:0")

    class FakeProcess:
        stdout = iter(("hello\n", "problem\n"))

        def wait(self):
            return 3

    calls = []

    def fake_popen(command, **kwargs):
        calls.append((tuple(command), kwargs))
        return FakeProcess()

    monkeypatch.setattr(scheduler.subprocess, "Popen", fake_popen)
    outcome = scheduler._run_job(job)
    expected = "[CUDA 1 svr] hello\n[CUDA 1 svr] problem\n"
    assert outcome.returncode == 3
    assert capsys.readouterr().out == expected
    assert job.log_path.read_text() == expected
    assert calls[0][1]["stderr"] is subprocess.STDOUT
    assert calls[0][1]["env"]["CUDA_VISIBLE_DEVICES"] == "1"


def test_worker_second_attempt_appends_delimiter_and_preserves_first_log(
    tmp_path, monkeypatch, capsys
):
    request = _request(tmp_path)
    job = scheduler._build_job(request, "svr", physical_device=1, device="cuda:0")
    outputs = iter((("first\n",), ("second\n",)))

    class FakeProcess:
        def __init__(self):
            self.stdout = iter(next(outputs))

        def wait(self):
            return 0

    monkeypatch.setattr(scheduler.subprocess, "Popen", lambda *_args, **_kwargs: FakeProcess())
    scheduler._run_job(job)
    scheduler._run_job(job)

    assert job.log_path.read_text() == (
        "[CUDA 1 svr] first\n\n=== HQRC WORKER ATTEMPT ===\n[CUDA 1 svr] second\n"
    )
    assert capsys.readouterr().out == "[CUDA 1 svr] first\n[CUDA 1 svr] second\n"


def test_restart_command_quotes_adversarial_paths_for_each_native_shell():
    command = (
        "python",
        "--config",
        "/tmp/source path/'quoted'/experiment.toml",
        "literal;not-a-command",
    )
    assert scheduler._restart_command(command, platform_name="nt") == subprocess.list2cmdline(
        command
    )
    assert scheduler._restart_command(command, platform_name="posix") == shlex.join(command)


def test_windows_workers_import_and_report_positive_peak_rss():
    from hqrc_v3 import peak_rss
    from hqrc_v3.bayes import sampler_worker
    from hqrc_v3.evaluation import data_benchmark_worker

    assert peak_rss.peak_rss_mb() > 0
    assert sampler_worker._rss_mb() > 0
    assert data_benchmark_worker._peak_rss_mb() > 0


@pytest.mark.parametrize(
    ("script", "shell", "accelerator"),
    [
        ("run_hqrc_windows.ps1", "powershell", "cuda"),
        ("run_hqrc_linux.sh", r"C:\Program Files\Git\bin\bash.exe", "cuda"),
        ("run_hqrc_macos.sh", r"C:\Program Files\Git\bin\bash.exe", "auto"),
    ],
)
@pytest.mark.parametrize(
    ("action", "expected"),
    [
        ("import", "hqrc import-paper-source --archive fixture.zip"),
        ("proposal", "hqrc run-loeo-accelerated --profile paper --source-run-dir imported"),
        ("smoke", "hqrc run-loeo-accelerated --profile smoke --source-run-dir imported"),
        ("paper", "hqrc run-loeo-accelerated --profile paper --source-run-dir imported"),
        ("resume", "hqrc run-loeo-accelerated --profile paper --source-run-dir imported"),
    ],
)
def test_native_launchers_sync_locked_and_only_forward_to_shared_cli(
    tmp_path, script, shell, accelerator, action, expected
):
    if not Path(shell).exists() and shell != "powershell":
        pytest.skip(f"native test shell is unavailable: {shell}")
    scripts = Path(__file__).parents[3] / "scripts"
    capture = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv.cmd").write_text(f'@echo %*>>"{capture}"\n')
    uv_sh = fake_bin / "uv"
    uv_sh.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{capture.as_posix()}"\n')
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    forwarded = (
        ["--archive", "fixture.zip"]
        if action == "import"
        else [
            "--source-run-dir",
            "imported",
        ]
    )
    command = (
        [shell, "-NoProfile", "-File", str(scripts / script), action, *forwarded]
        if shell == "powershell"
        else [shell, str(scripts / script), action, *forwarded]
    )
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    if action != "import":
        smoke_defaults = (
            " --draws 4 --tune 4 --chains 4 --cores 1 --target-accept 0.9"
            if action == "smoke"
            else ""
        )
        expected = expected.replace(
            " --source-run-dir",
            f" --accelerator {accelerator}{smoke_defaults} --source-run-dir",
        )
    assert capture.read_text().splitlines() == [
        "sync --project hqrc_v3 --extra accelerator --locked",
        f"run --project hqrc_v3 --extra accelerator --locked {expected}",
    ]


@pytest.mark.parametrize(
    ("script", "shell"),
    [
        ("run_hqrc_windows.ps1", "powershell"),
        ("run_hqrc_linux.sh", r"C:\Program Files\Git\bin\bash.exe"),
        ("run_hqrc_macos.sh", r"C:\Program Files\Git\bin\bash.exe"),
    ],
)
@pytest.mark.parametrize("override", (("--profile", "smoke"), ("--profile=smoke",)))
def test_native_launchers_reject_raw_profile_override_before_uv_sync(
    tmp_path, script, shell, override
):
    scripts = Path(__file__).parents[3] / "scripts"
    capture = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv.cmd").write_text(f'@echo %*>>"{capture}"\n')
    uv_sh = fake_bin / "uv"
    uv_sh.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{capture.as_posix()}"\n')
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    command = (
        [shell, "-NoProfile", "-File", str(scripts / script), "paper", *override]
        if shell == "powershell"
        else [shell, str(scripts / script), "paper", *override]
    )
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 2
    assert "profile is owned by the launcher action" in result.stderr
    assert not capture.exists()


@pytest.mark.parametrize(
    ("script", "shell"),
    [
        ("run_hqrc_windows.ps1", "powershell"),
        ("run_hqrc_linux.sh", r"C:\Program Files\Git\bin\bash.exe"),
        ("run_hqrc_macos.sh", r"C:\Program Files\Git\bin\bash.exe"),
    ],
)
def test_native_launchers_forward_profile_abbreviation_to_shared_parser(tmp_path, script, shell):
    scripts = Path(__file__).parents[3] / "scripts"
    capture = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv.cmd").write_text(f'@echo %*>>"{capture}"\n')
    uv_sh = fake_bin / "uv"
    uv_sh.write_text(f'#!/bin/sh\nprintf "%s\\n" "$*" >> "{capture.as_posix()}"\n')
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    command = (
        [shell, "-NoProfile", "-File", str(scripts / script), "paper", "--prof", "smoke"]
        if shell == "powershell"
        else [shell, str(scripts / script), "paper", "--prof", "smoke"]
    )
    result = subprocess.run(command, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    calls = capture.read_text().splitlines()
    assert calls[0] == "sync --project hqrc_v3 --extra accelerator --locked"
    assert calls[1].endswith("--prof smoke")


def test_windows_launcher_translates_approved_named_parameters(tmp_path):
    scripts = Path(__file__).parents[3] / "scripts"
    capture = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv.cmd").write_text(f'@echo %*>>"{capture}"\n')
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    invocation = (
        f"& '{scripts / 'run_hqrc_windows.ps1'}' smoke "
        "-SourceRunDir imported -Config imported/sources/experiment.toml "
        "-OutputRoot products -Devices 0,1 -Models svr,transformer "
        "-FeatureSets B1 -Variants H1,H3 -ApproveDerivedAR"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", invocation],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [
        "sync --project hqrc_v3 --extra accelerator --locked",
        "run --project hqrc_v3 --extra accelerator --locked hqrc run-loeo-accelerated "
        "--profile smoke --accelerator cuda --draws 4 --tune 4 --chains 4 --cores 1 "
        "--target-accept 0.9 --source-run-dir imported "
        "--config imported/sources/experiment.toml --output-root products "
        "--devices 0 1 --models svr transformer --feature-sets B1 --variants H1 H3 "
        "--approve-derived-ar",
    ]


def test_windows_launcher_translates_named_import_parameters(tmp_path):
    scripts = Path(__file__).parents[3] / "scripts"
    capture = tmp_path / "calls.txt"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    (fake_bin / "uv.cmd").write_text(f'@echo %*>>"{capture}"\n')
    env = os.environ.copy()
    env["PATH"] = f"{fake_bin}{os.pathsep}{env['PATH']}"
    invocation = (
        f"& '{scripts / 'run_hqrc_windows.ps1'}' import "
        "-Archive fixture.zip -Data demand.csv -RepositoryRoot . -RunDir imported"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", invocation],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [
        "sync --project hqrc_v3 --extra accelerator --locked",
        "run --project hqrc_v3 --extra accelerator --locked hqrc import-paper-source "
        "--archive fixture.zip --data demand.csv --repository-root . --run-dir imported",
    ]
