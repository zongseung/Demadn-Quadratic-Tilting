"""CLI routing checks that do not load real forecasting data."""

from __future__ import annotations

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


def test_missing_default_handler_fails_clearly(capsys):
    assert cli.main(["audit-data", "--data", "input.parquet"]) != 0
    assert "handler" in capsys.readouterr().err


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
