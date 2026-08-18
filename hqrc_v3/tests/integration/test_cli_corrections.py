from __future__ import annotations

from pathlib import Path

from hqrc_v3 import cli
from hqrc_v3.bayes.samplers import SamplingError

PAPER_DRAWS_HELP = "retained draws (smoke: required; paper: optional, minimum/default 1000)"
PAPER_TUNE_HELP = "warm-up draws (smoke: required; paper: optional, minimum/default 1000)"
PAPER_CHAINS_HELP = "chains (smoke: required; paper: optional/default 4 and must equal 4)"


def test_causal_cli_routes_rng_seed_and_explicit_smoke_limits(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "fit_causal_2024_correction", lambda **kwargs: received.append(kwargs))

    result = cli.main(
        [
            "fit-corrections",
            "--run-dir",
            "run",
            "--config",
            "experiment.toml",
            "--approved-ar",
            "approved.json",
            "--evaluation",
            "causal-2024",
            "--seed",
            "19",
            "--profile",
            "smoke",
            "--draws",
            "12",
            "--tune",
            "13",
            "--chains",
            "2",
        ]
    )

    assert result == 0
    assert received == [
        {
            "run_dir": Path("run"),
            "config_path": Path("experiment.toml"),
            "approved_ar_path": Path("approved.json"),
            "sampler_seed": 19,
            "profile": "smoke",
            "draws": 12,
            "tune": 13,
            "chains": 2,
            "cores": None,
        }
    ]


def test_causal_cli_forwards_explicit_cores(monkeypatch):
    received = []
    monkeypatch.setattr(cli, "fit_causal_2024_correction", lambda **kwargs: received.append(kwargs))

    result = cli.main(
        [
            "fit-corrections", "--run-dir", "run", "--config", "experiment.toml",
            "--approved-ar", "approved.json", "--evaluation", "causal-2024", "--seed", "19",
            "--profile", "smoke", "--draws", "12", "--tune", "13", "--chains", "4",
            "--cores", "4",
        ]
    )

    assert result == 0
    assert received[0]["cores"] == 4


def test_correction_cli_help_describes_smoke_and_paper_sampler_overrides():
    parser = cli.build_parser()
    command_action = next(action for action in parser._actions if action.dest == "command")
    correction_parser = command_action.choices["fit-corrections"]
    help_by_option = {
        option: action.help
        for action in correction_parser._actions
        for option in action.option_strings
    }

    assert help_by_option["--draws"] == PAPER_DRAWS_HELP
    assert help_by_option["--tune"] == PAPER_TUNE_HELP
    assert help_by_option["--chains"] == PAPER_CHAINS_HELP


def test_loeo_cli_fails_before_loading_causal_approval(capsys, monkeypatch):
    called = []
    monkeypatch.setattr(cli, "fit_causal_2024_correction", lambda **kwargs: called.append(kwargs))

    result = cli.main(
        [
            "fit-corrections",
            "--run-dir",
            "run",
            "--config",
            "experiment.toml",
            "--approved-ar",
            "causal-approved.json",
            "--evaluation",
            "loeo",
            "--seed",
            "19",
            "--profile",
            "paper",
        ]
    )

    assert result == 2 and not called
    assert "fold-specific approved calibrations" in capsys.readouterr().err


def test_sampling_diagnostic_failure_uses_cli_exit_two(capsys, monkeypatch):
    def fail_sampling(**_kwargs):
        raise SamplingError("posterior diagnostics failed")

    monkeypatch.setattr(cli, "fit_causal_2024_correction", fail_sampling)
    result = cli.main(
        [
            "fit-corrections",
            "--run-dir",
            "run",
            "--config",
            "experiment.toml",
            "--approved-ar",
            "approved.json",
            "--evaluation",
            "causal-2024",
            "--seed",
            "19",
            "--profile",
            "paper",
        ]
    )

    assert result == 2
    assert capsys.readouterr().err == "hqrc: posterior diagnostics failed\n"
