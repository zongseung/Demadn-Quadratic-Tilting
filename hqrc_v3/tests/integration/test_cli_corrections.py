from __future__ import annotations

from pathlib import Path

from hqrc_v3 import cli


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
        }
    ]


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
