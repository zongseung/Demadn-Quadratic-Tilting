"""Thin, staged command line entry points for the HQRC v3 pipeline."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Mapping, Sequence
from typing import NoReturn

from hqrc_v3.config import ConfigError
from hqrc_v3.contracts import DataContractError

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
    return _unavailable_handler("audit-data")(arguments)


def tune_baselines_handler(arguments: argparse.Namespace) -> object:
    return _unavailable_handler("tune-baselines")(arguments)


def generate_oof_handler(arguments: argparse.Namespace) -> object:
    return _unavailable_handler("generate-oof")(arguments)


def fit_final_baselines_handler(arguments: argparse.Namespace) -> object:
    return _unavailable_handler("fit-final-baselines")(arguments)


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
    return parser


def _default_handlers() -> dict[str, StageHandler]:
    return {
        "audit-data": audit_data_handler,
        "tune-baselines": tune_baselines_handler,
        "generate-oof": generate_oof_handler,
        "fit-final-baselines": fit_final_baselines_handler,
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
