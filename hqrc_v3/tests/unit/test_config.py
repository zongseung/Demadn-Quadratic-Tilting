from pathlib import Path

from hqrc_v3.config import load_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_default_config_fixes_expanding_years_and_zero_gap(tmp_path):
    config = load_config(PROJECT_ROOT / "configs/experiment.toml")

    assert config.data.gap_days == 0
    assert config.split.oof_years == (2020, 2021, 2022, 2023)
    assert config.split.final_year == 2024
