"""CLI routing checks that do not load real forecasting data."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hqrc_v3 import cli


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
                "--data",
                "input.parquet",
                "--config",
                "experiment.toml",
                "--frozen-model-config",
                "model.toml",
                "--event-registry",
                "events.csv",
                "--holiday-calendar",
                "holiday-calendar.csv",
                "--temporary-holiday-availability",
                "temporary-holidays.csv",
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
        (
            "run-loeo-primary",
            [
                "--source-run-dir",
                "run",
                "--config",
                "experiment.toml",
                "--output-root",
                "products",
            ],
        ),
        (
            "run-hqt-reviewer",
            [
                "--data",
                "input.parquet",
                "--config",
                "experiment.toml",
                "--frozen-model-config",
                "model.toml",
                "--frozen-model-hash",
                "abc",
                "--event-registry",
                "events.csv",
                "--holiday-calendar",
                "holiday-calendar.csv",
                "--temporary-holiday-availability",
                "temporary-holidays.csv",
                "--run-dir",
                "source",
                "--cache-dir",
                "cache",
                "--output-root",
                "products",
                "--profile",
                "smoke",
                "--model",
                "xgboost",
                "--draws",
                "5",
                "--tune",
                "5",
                "--chains",
                "2",
                "--cores",
                "2",
                "--smoke-boosting-rounds",
                "2",
            ],
        ),
        (
            "run-paper",
            [
                "--data",
                "input.parquet",
                "--config",
                "experiment.toml",
                "--frozen-model-config",
                "model.toml",
                "--frozen-model-hash",
                "abc",
                "--event-registry",
                "events.csv",
                "--holiday-calendar",
                "holiday-calendar.csv",
                "--temporary-holiday-availability",
                "temporary-holidays.csv",
                "--run-dir",
                "run",
                "--output-root",
                "products",
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


def test_reviewer_handler_maps_fixed_b1w_pipeline_scope(monkeypatch, capsys):
    captured = []
    monkeypatch.setattr(
        cli,
        "_paper_stage_inputs",
        lambda arguments: {
            "matrices": {"B0": object(), "B1W": object()},
            "config": object(),
            "artifact_hashes": {"data_sha256": "a" * 64},
            "cache_dir": Path("shared-cache"),
        },
    )

    def run(options, *, progress):
        captured.append((options, progress))
        return SimpleNamespace(
            baseline_fit_count=3,
            baseline_cache_hit_count=4,
            hqt_fit_count=5,
        )

    monkeypatch.setattr(cli, "run_hqt_reviewer_pipeline", run)

    result = cli.main(
        [
            "run-hqt-reviewer",
            "--data",
            "input.parquet",
            "--config",
            "experiment.toml",
            "--frozen-model-config",
            "model.toml",
            "--frozen-model-hash",
            "a" * 64,
            "--event-registry",
            "events.csv",
            "--holiday-calendar",
            "holiday-calendar.csv",
            "--temporary-holiday-availability",
            "temporary-holidays.csv",
            "--run-dir",
            "source",
            "--cache-dir",
            "shared-cache",
            "--output-root",
            "products",
            "--profile",
            "smoke",
            "--model",
            "xgboost",
            "--draws",
            "5",
            "--tune",
            "5",
            "--chains",
            "2",
            "--cores",
            "2",
            "--smoke-boosting-rounds",
            "2",
        ]
    )

    assert result == 0
    options, progress = captured[0]
    assert set(options.matrices) == {"B0", "B1W"}
    assert options.models == ("xgboost",)
    assert options.profile == "smoke"
    assert options.draws == options.tune == 5
    assert options.chains == options.cores == 2
    assert options.smoke_boosting_rounds == 2
    assert progress is print
    assert "baseline_fits=3 baseline_cache_hits=4 hqt_fits=5" in capsys.readouterr().out


def test_ar_commands_require_canonical_run_inputs_and_diagnose_never_auto_approves(capsys):
    diagnose_arguments = [
        "diagnose-ar",
        "--run-dir",
        "run",
        "--config",
        "experiment.toml",
        "--event-registry",
        "events.csv",
        "--output",
        "proposal.json",
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
    assert "standardized_residuals_manifest" in capsys.readouterr().err

    assert (
        cli.main(
            [
                "diagnose-ar",
                "--residuals",
                "arbitrary.parquet",
                "--residual-sha256",
                "fake",
                "--config-sha256",
                "fake",
                "--event-sha256",
                "fake",
                "--output",
                "proposal.json",
                "--through",
                "2023",
                "--model",
                "lightgbm",
                "--feature-set",
                "B1",
            ]
        )
        != 0
    )

    assert cli.main(["approve-ar-calibration", "--proposal", "proposal.json"]) != 0
    assert "residual-sha256" in capsys.readouterr().err
