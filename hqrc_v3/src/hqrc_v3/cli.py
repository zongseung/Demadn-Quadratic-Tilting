"""Thin, staged command line entry points for the HQRC v3 pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import NoReturn

import numpy as np
import polars as pl

from hqrc_v3.config import ConfigError
from hqrc_v3.contracts import DataContractError
from hqrc_v3.data import audit_hourly_data, read_hourly_data
from hqrc_v3.diagnostics.ar import (
    approve_calibration,
    calibrate_beta_prior,
    diagnose_event_residuals,
    write_ar_diagnostics,
)
from hqrc_v3.provenance import file_sha256

StageHandler = Callable[[argparse.Namespace], object]
_EXPECTED_START = "2019-01-01T00:00:00"
_EXPECTED_END = "2024-10-31T23:00:00"
_EXPECTED_ROWS = 51_144


class StageInputError(ValueError):
    """Raised when a command lacks the handler or inputs needed for real work."""


def _unavailable_handler(stage: str) -> StageHandler:
    def handler(_: argparse.Namespace) -> NoReturn:
        raise StageInputError(
            f"{stage} handler is not configured; provide a concrete stage handler"
        )

    return handler


def audit_data_handler(arguments: argparse.Namespace) -> object:
    expected = {}
    if arguments.fixed_expected_bounds:
        expected = {
            "expected_start": datetime.fromisoformat(_EXPECTED_START),
            "expected_end": datetime.fromisoformat(_EXPECTED_END),
            "expected_rows": _EXPECTED_ROWS,
        }
    return audit_hourly_data(read_hourly_data(Path(arguments.data)), **expected)


def tune_baselines_handler(arguments: argparse.Namespace) -> object:
    return _unavailable_handler("tune-baselines")(arguments)


def generate_oof_handler(arguments: argparse.Namespace) -> object:
    return _unavailable_handler("generate-oof")(arguments)


def fit_final_baselines_handler(arguments: argparse.Namespace) -> object:
    return _unavailable_handler("fit-final-baselines")(arguments)


def diagnose_ar_handler(arguments: argparse.Namespace) -> object:
    residual_path = Path(arguments.residuals)
    if file_sha256(residual_path) != arguments.residual_sha256:
        raise StageInputError("diagnose-ar residual-sha256 does not match the residual artifact")
    residuals = pl.read_parquet(residual_path)
    diagnostics = diagnose_event_residuals(residuals)
    if any(split.split("-", 1)[-1] > str(arguments.through) for split in residuals["split_id"]):
        raise StageInputError("diagnose-ar residuals include rows later than --through")
    calibration = calibrate_beta_prior(
        np.asarray([item.phi for item in diagnostics]),
        event_ids=tuple(item.occurrence_id for item in diagnostics),
    )
    return write_ar_diagnostics(
        Path(arguments.output),
        diagnostics,
        calibration,
        residual_sha256=arguments.residual_sha256,
        config_sha256=arguments.config_sha256,
        event_sha256=arguments.event_sha256,
        context={
            "model": str(residuals["model"].item(0)),
            "feature_set": str(residuals["feature_set"].item(0)),
            "seed": int(residuals["seed"].item(0)),
            "split_ids": tuple(sorted(residuals["split_id"].unique().to_list())),
        },
    )


def approve_ar_calibration_handler(arguments: argparse.Namespace) -> object:
    """Perform only the explicit approval gate; it never derives a new proposal."""

    return approve_calibration(
        arguments.proposal,
        arguments.output,
        current_residual_sha256=arguments.residual_sha256,
        current_config_sha256=arguments.config_sha256,
        current_event_sha256=arguments.event_sha256,
    )


def report_handler(arguments: argparse.Namespace) -> object:
    """Build a concrete report only from an already materialized run directory."""

    from hqrc_v3.evaluation.reports import build_report

    return build_report(Path(arguments.run_dir), profile=arguments.profile)


def _add_data_inputs(parser: argparse.ArgumentParser, *, config: bool = True) -> None:
    parser.add_argument("--data", required=True, help="hourly source data path")
    if config:
        parser.add_argument("--config", required=True, help="experiment TOML path")


def _add_frozen_model_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--frozen-model-config", required=True, help="selected immutable model config"
    )
    parser.add_argument(
        "--frozen-model-hash", required=True, help="hash of the frozen model config"
    )
    parser.add_argument("--feature-set", choices=("B0", "B1"), required=True)
    parser.add_argument("--seed", type=int, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hqrc", description="HQRC v3 staged forecasting pipeline")
    subcommands = parser.add_subparsers(dest="command", required=True)

    audit = subcommands.add_parser(
        "audit-data", help="audit hourly data against the requested bounds"
    )
    _add_data_inputs(audit, config=False)
    audit.add_argument(
        "--fixed-expected-bounds",
        "--fixed-bounds",
        dest="fixed_expected_bounds",
        action="store_true",
        help="supply the fixed 2019-01-01 through 2024-10-31 / 51,144-row contract",
    )

    tune = subcommands.add_parser("tune-baselines", help="run one explicit selection stage")
    _add_data_inputs(tune)

    oof = subcommands.add_parser(
        "generate-oof", help="generate cache-aware frozen-config OOF forecasts"
    )
    _add_data_inputs(oof)
    _add_frozen_model_inputs(oof)
    oof.add_argument("--cache-dir", required=True, help="immutable prediction cache directory")

    final = subcommands.add_parser("fit-final-baselines", help="refit frozen config for 2024")
    _add_data_inputs(final)
    _add_frozen_model_inputs(final)

    diagnose = subcommands.add_parser(
        "diagnose-ar", help="generate an unapproved event-reset AR diagnostic proposal"
    )
    diagnose.add_argument("--residuals", required=True, help="standardized event residual artifact")
    diagnose.add_argument("--output", required=True, help="unapproved AR diagnostic JSON path")
    diagnose.add_argument("--residual-sha256", required=True, help="hash of the residual artifact")
    diagnose.add_argument("--config-sha256", required=True, help="hash of the experiment config")
    diagnose.add_argument("--event-sha256", required=True, help="hash of the event registry")
    diagnose.add_argument("--through", type=int, required=True, help="latest OOF year included")

    approve = subcommands.add_parser(
        "approve-ar-calibration", help="freeze a reviewed AR calibration proposal"
    )
    approve.add_argument("proposal_path", nargs="?", help="unapproved AR diagnostic JSON path")
    approve.add_argument("--proposal", help="unapproved AR diagnostic JSON path")
    approve.add_argument("--output", required=True, help="approved AR calibration JSON path")
    approve.add_argument("--residual-sha256", required=True, help="current residual artifact hash")
    approve.add_argument("--config-sha256", required=True, help="current experiment config hash")
    approve.add_argument("--event-sha256", required=True, help="current event registry hash")

    corrections = subcommands.add_parser(
        "fit-corrections", help="fit a correction only from an approved AR artifact"
    )
    corrections.add_argument("--run-dir", required=True)
    corrections.add_argument("--config", required=True)
    corrections.add_argument("--approved-ar", required=True)
    corrections.add_argument("--evaluation", choices=("causal-2024", "loeo"), required=True)
    corrections.add_argument("--seed", type=int, required=True)
    corrections.add_argument("--profile", choices=("smoke", "paper"), required=True)

    ablations = subcommands.add_parser("run-ablations", help="run declared H0--H5 ablations")
    ablations.add_argument("--run-dir", required=True)
    ablations.add_argument("--config", required=True)
    ablations.add_argument("--approved-ar", required=True)
    ablations.add_argument("--seed", type=int, required=True)
    ablations.add_argument("--profile", choices=("smoke", "paper"), required=True)

    benchmark = subcommands.add_parser(
        "benchmark-samplers", help="benchmark PyMC and optional nutpie under identical inputs"
    )
    benchmark.add_argument("--run-dir", required=True)
    benchmark.add_argument("--config", required=True)
    benchmark.add_argument("--approved-ar", required=True)
    benchmark.add_argument("--seed", type=int, required=True)
    benchmark.add_argument("--draws", type=int, required=True)
    benchmark.add_argument("--tune", type=int, required=True)
    benchmark.add_argument("--chains", type=int, required=True)
    benchmark.add_argument("--profile", choices=("smoke", "paper"), required=True)

    report = subcommands.add_parser(
        "report", help="validate a completed run and emit normalized outputs"
    )
    report.add_argument("--run-dir", required=True)
    report.add_argument("--profile", choices=("smoke", "paper"), required=True)
    return parser


def _default_handlers() -> dict[str, StageHandler]:
    return {
        "audit-data": audit_data_handler,
        "tune-baselines": tune_baselines_handler,
        "generate-oof": generate_oof_handler,
        "fit-final-baselines": fit_final_baselines_handler,
        "diagnose-ar": diagnose_ar_handler,
        "approve-ar-calibration": approve_ar_calibration_handler,
        "fit-corrections": _unavailable_handler("fit-corrections"),
        "run-ablations": _unavailable_handler("run-ablations"),
        "benchmark-samplers": _unavailable_handler("benchmark-samplers"),
        "report": report_handler,
    }


def main(
    argv: Sequence[str] | None = None, *, handlers: Mapping[str, StageHandler] | None = None
) -> int:
    """Parse one staged command and run its concrete, injected handler.

    OOF and final stages deliberately do not dispatch through tuning: their
    frozen model path and hash are explicit required arguments.
    """

    parser = build_parser()
    try:
        arguments = parser.parse_args(argv)
        if arguments.command == "approve-ar-calibration":
            if arguments.proposal is None:
                arguments.proposal = arguments.proposal_path
            elif arguments.proposal_path is not None:
                raise StageInputError(
                    "provide the AR proposal once, either positionally or with --proposal"
                )
            if arguments.proposal is None:
                raise StageInputError("approve-ar-calibration requires a proposal artifact")
        if arguments.command == "audit-data" and arguments.fixed_expected_bounds:
            arguments.expected_start = _EXPECTED_START
            arguments.expected_end = _EXPECTED_END
            arguments.expected_rows = _EXPECTED_ROWS
        else:
            arguments.expected_start = None
            arguments.expected_end = None
            arguments.expected_rows = None
        resolved = _default_handlers()
        if handlers is not None:
            resolved.update(handlers)
        handler = resolved.get(arguments.command)
        if handler is None:
            raise StageInputError(f"no handler configured for {arguments.command}")
        handler(arguments)
    except SystemExit as error:
        if error.code == 0:
            raise
        return int(error.code)
    except (ConfigError, DataContractError, OSError, StageInputError, ValueError) as error:
        print(f"hqrc: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the console entry point
    raise SystemExit(main())
