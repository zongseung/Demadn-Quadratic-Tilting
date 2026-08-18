"""Real installed-console smoke for the concrete paper-baseline execution path."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import polars as pl
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parent
SOURCE = REPOSITORY_ROOT / "power_demand_final.csv"
ROOT_LOCK = REPOSITORY_ROOT / "uv.lock"


@pytest.fixture(scope="module")
def clean_project_environment(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[tuple[dict[str, str], Path, Path]]:
    """Return one isolated uv environment with no ambient import/install escape hatch."""

    temporary_root = tmp_path_factory.mktemp("installed-hqrc")
    environment_path = temporary_root / "environment"
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    environment.pop("UV_LOCKED", None)
    environment.pop("VIRTUAL_ENV", None)
    environment.update(
        {
            "UV_NO_PROGRESS": "1",
            "UV_PROJECT_ENVIRONMENT": str(environment_path),
        }
    )
    lock_before = ROOT_LOCK.read_bytes()
    yield environment, environment_path, temporary_root
    assert ROOT_LOCK.read_bytes() == lock_before


def _run_repository_command(
    arguments: Sequence[str], *, environment: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["uv", "run", "--project", "hqrc_v3", "--locked", "hqrc", *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )


def _assert_success(result: subprocess.CompletedProcess[str]) -> None:
    assert result.returncode == 0, (
        f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
    )


def test_repository_command_passes_locked_in_the_exact_uv_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def capture_run(
        arguments: Sequence[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        captured.append(list(arguments))
        return subprocess.CompletedProcess(arguments, 0, "", "")

    monkeypatch.setattr(subprocess, "run", capture_run)
    _run_repository_command(
        ["audit-data", "--data", "power_demand_final.csv", "--fixed-bounds"],
        environment={"UV_PROJECT_ENVIRONMENT": "/isolated/environment"},
    )

    assert captured == [
        [
            "uv",
            "run",
            "--project",
            "hqrc_v3",
            "--locked",
            "hqrc",
            "audit-data",
            "--data",
            "power_demand_final.csv",
            "--fixed-bounds",
        ]
    ]


@pytest.mark.slow
def test_clean_locked_install_exposes_console_and_audits_real_source(
    clean_project_environment: tuple[dict[str, str], Path, Path],
) -> None:
    """Regress the old virtual metadata, which never created the ``hqrc`` script."""

    if not SOURCE.is_file():
        pytest.skip("repository power_demand_final.csv source is absent")
    environment, environment_path, temporary_root = clean_project_environment
    assert "UV_LOCKED" not in environment

    result = _run_repository_command(
        ["audit-data", "--data", "power_demand_final.csv", "--fixed-bounds"],
        environment=environment,
    )
    _assert_success(result)

    console = environment_path / "bin" / "hqrc"
    installed_python = environment_path / "bin" / "python"
    assert console.is_file()
    assert os.access(console, os.X_OK)
    assert installed_python.is_file()

    inspection = subprocess.run(
        [
            str(installed_python),
            "-I",
            "-c",
            (
                "import importlib.metadata as metadata, importlib.util, json; "
                "spec = importlib.util.find_spec('hqrc_v3'); "
                "entry_points = [entry.value for entry in "
                "metadata.distribution('hqrc-v3').entry_points "
                "if entry.group == 'console_scripts' and entry.name == 'hqrc']; "
                "print(json.dumps({'origin': spec.origin if spec else None, "
                "'entry_points': entry_points}))"
            ),
        ],
        cwd=temporary_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    _assert_success(inspection)
    package = json.loads(inspection.stdout)
    assert Path(package["origin"]).resolve() == (
        PROJECT_ROOT / "src/hqrc_v3/__init__.py"
    ).resolve()
    assert package["entry_points"] == ["hqrc_v3.cli:main"]


@pytest.mark.slow
def test_real_lightgbm_b1_one_fold_uses_installed_non_paper_console(
    tmp_path: Path,
    clean_project_environment: tuple[dict[str, str], Path, Path],
) -> None:
    if not SOURCE.is_file():
        pytest.skip("repository power_demand_final.csv source is absent")
    environment, environment_path, temporary_root = clean_project_environment
    model_path = PROJECT_ROOT / "configs/model_spaces.toml"
    model_digest = hashlib.file_digest(model_path.open("rb"), "sha256").hexdigest()

    result = _run_repository_command(
        [
            "generate-oof",
            "--data",
            "power_demand_final.csv",
            "--config",
            "hqrc_v3/configs/experiment.toml",
            "--frozen-model-config",
            "hqrc_v3/configs/model_spaces.toml",
            "--frozen-model-hash",
            model_digest,
            "--run-dir",
            str(tmp_path / "run"),
            "--cache-dir",
            str(tmp_path / "stream-cache"),
            "--model",
            "lightgbm",
            "--feature-set",
            "B1",
            "--seed",
            "7",
            "--profile",
            "smoke",
            "--oof-years",
            "2020",
            "--smoke-boosting-rounds",
            "3",
        ],
        environment=environment,
    )
    _assert_success(result)

    predictions = tmp_path / "run/predictions"
    point = pl.read_parquet(predictions / "oof.parquet")
    members = pl.read_parquet(predictions / "oof_members.parquet")
    manifest = json.loads(
        (predictions / "baseline_manifest.json").read_text(encoding="utf-8")
    )
    assert point["model"].unique().to_list() == ["lightgbm"]
    assert point["feature_set"].unique().to_list() == ["B1"]
    assert point["split_id"].unique().to_list() == ["oof-2020"]
    assert point.height > 0
    assert members.is_empty()
    assert manifest["profile"] == "smoke"
    assert manifest["execution_overrides"] == {"boosting_rounds": 3}

    implementation = subprocess.run(
        [
            str(environment_path / "bin/python"),
            "-I",
            "-c",
            (
                "import json; "
                "from hqrc_v3.baselines.classical import _estimator_class; "
                "estimator = _estimator_class('lightgbm'); "
                "print(json.dumps([estimator.__module__, estimator.__name__]))"
            ),
        ],
        cwd=temporary_root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    _assert_success(implementation)
    assert json.loads(implementation.stdout) == ["lightgbm.sklearn", "LGBMRegressor"]
