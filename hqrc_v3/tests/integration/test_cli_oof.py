"""CLI routing checks that do not load real forecasting data."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from hqrc_v3 import cli
from hqrc_v3.provenance import file_sha256


def test_help_succeeds(capsys):
    with pytest.raises(SystemExit) as error:
        cli.main(["--help"])
    assert error.value.code == 0
    assert "generate-oof" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("command", "arguments"),
    [
        ("audit-data", ["--data", "input.parquet"]),
        ("tune-baselines", ["--data", "input.parquet", "--config", "experiment.toml"]),
        (
            "generate-oof",
            [
                "--data",
                "input.parquet",
                "--config",
                "experiment.toml",
                "--frozen-model-config",
                "model.toml",
                "--frozen-model-hash",
                "abc",
                "--cache-dir",
                "cache",
                "--feature-set",
                "B0",
                "--seed",
                "9",
            ],
        ),
        (
            "fit-final-baselines",
            [
                "--data",
                "input.parquet",
                "--config",
                "experiment.toml",
                "--frozen-model-config",
                "model.toml",
                "--frozen-model-hash",
                "abc",
                "--feature-set",
                "B1",
                "--seed",
                "9",
            ],
        ),
        (
            "prepare-residuals",
            [
                "--run-dir",
                "run",
                "--config",
                "experiment.toml",
                "--event-registry",
                "events.csv",
                "--profile",
                "paper",
            ],
        ),
        (
            "fit-corrections",
            [
                "--run-dir",
                "run",
                "--config",
                "experiment.toml",
                "--approved-ar",
                "approved.json",
                "--evaluation",
                "causal-2024",
                "--seed",
                "9",
                "--profile",
                "smoke",
            ],
        ),
        (
            "run-ablations",
            [
                "--run-dir",
                "run",
                "--config",
                "experiment.toml",
                "--approved-ar",
                "approved.json",
                "--seed",
                "9",
                "--profile",
                "smoke",
            ],
        ),
        (
            "benchmark-samplers",
            [
                "--run-dir",
                "run",
                "--config",
                "experiment.toml",
                "--approved-ar",
                "approved.json",
                "--seed",
                "9",
                "--draws",
                "10",
                "--tune",
                "10",
                "--chains",
                "1",
                "--profile",
                "smoke",
            ],
        ),
    ],
)
def test_commands_route_to_injected_handler(command, arguments):
    received = []
    assert cli.main([command, *arguments], handlers={command: received.append}) == 0
    assert len(received) == 1
    assert received[0].command == command


def test_generate_oof_requires_explicit_frozen_config(capsys):
    result = cli.main(
        [
            "generate-oof",
            "--data",
            "input.parquet",
            "--config",
            "experiment.toml",
            "--cache-dir",
            "cache",
            "--feature-set",
            "B0",
            "--seed",
            "9",
        ]
    )
    assert result != 0
    assert "frozen-model-config" in capsys.readouterr().err


def test_concrete_audit_reports_missing_input_clearly(capsys):
    assert cli.main(["audit-data", "--data", "input.parquet"]) != 0
    assert "unable to read hourly data" in capsys.readouterr().err


def test_audit_fixed_bounds_are_provided_to_its_handler():
    received = []
    assert (
        cli.main(
            ["audit-data", "--data", "input.parquet", "--fixed-bounds"],
            handlers={"audit-data": received.append},
        )
        == 0
    )
    assert (received[0].expected_start, received[0].expected_end, received[0].expected_rows) == (
        "2019-01-01T00:00:00",
        "2024-10-31T23:00:00",
        51_144,
    )


def test_ar_commands_require_explicit_hashes_and_diagnose_never_auto_approves(capsys):
    diagnose_arguments = [
        "diagnose-ar",
        "--residuals",
        "residuals.parquet",
        "--output",
        "proposal.json",
        "--residual-sha256",
        "residual",
        "--config-sha256",
        "config",
        "--event-sha256",
        "events",
        "--through",
        "2023",
        "--model",
        "lightgbm",
        "--feature-set",
        "B1",
    ]
    received = []
    assert cli.main(diagnose_arguments, handlers={"diagnose-ar": received.append}) == 0
    assert received[0].command == "diagnose-ar"
    assert cli.main(diagnose_arguments) == 2
    assert "residuals.parquet" in capsys.readouterr().err

    assert cli.main(["approve-ar-calibration", "--proposal", "proposal.json"]) != 0
    assert "residual-sha256" in capsys.readouterr().err


def test_diagnose_ar_selects_one_context_from_the_combined_residual_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    timestamp = datetime(2023, 1, 1)
    rows = []
    for model, seed in (("lightgbm", 7), ("transformer", 0)):
        for offset in range(2):
            rows.append(
                {
                    "target_timestamp": timestamp + timedelta(hours=offset),
                    "occurrence_id": "seollal-2023",
                    "tau_days": offset / 24.0,
                    "hour": offset,
                    "standardized_residual": float(offset),
                    "model": model,
                    "feature_set": "B1",
                    "seed": seed,
                    "split_id": "oof-2023",
                }
            )
    residual_path = tmp_path / "standardized_residuals.parquet"
    pl.DataFrame(rows).write_parquet(residual_path)
    captured: list[pl.DataFrame] = []

    def diagnose(frame: pl.DataFrame) -> tuple[SimpleNamespace, ...]:
        captured.append(frame)
        return (SimpleNamespace(occurrence_id="seollal-2023", phi=0.5),)

    monkeypatch.setattr(cli, "diagnose_event_residuals", diagnose)
    monkeypatch.setattr(cli, "calibrate_beta_prior", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "write_ar_diagnostics", lambda *args, **kwargs: None)

    assert (
        cli.main(
            [
                "diagnose-ar",
                "--residuals",
                str(residual_path),
                "--output",
                str(tmp_path / "proposal.json"),
                "--residual-sha256",
                file_sha256(residual_path),
                "--config-sha256",
                "config",
                "--event-sha256",
                "events",
                "--through",
                "2023",
                "--model",
                "transformer",
                "--feature-set",
                "B1",
            ]
        )
        == 0
    )
    assert len(captured) == 1
    assert set(captured[0]["model"]) == {"transformer"}
    assert set(captured[0]["seed"]) == {0}
