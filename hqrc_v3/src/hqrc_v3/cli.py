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

from hqrc_v3.baselines.config import MODEL_NAMES, load_paper_baselines
from hqrc_v3.baselines.paper import run_paper_final_stage, run_paper_oof_stage
from hqrc_v3.bayes.samplers import SamplingError
from hqrc_v3.config import ConfigError, load_config
from hqrc_v3.contracts import DataContractError
from hqrc_v3.correction_stage import fit_causal_2024_correction
from hqrc_v3.data import (
    FIXED_PUBLIC_HOLIDAY_DATES,
    FIXED_SUBSTITUTE_OR_TEMPORARY_DATES,
    audit_hourly_data,
    load_temporary_holiday_availability,
    read_hourly_data,
)
from hqrc_v3.diagnostics.ar import (
    approve_calibration,
    calibrate_beta_prior,
    diagnose_event_residuals,
    write_ar_diagnostics,
)
from hqrc_v3.events import load_event_registry, load_holiday_calendar
from hqrc_v3.features import attach_calendar_features, build_daily_forecast_matrix
from hqrc_v3.paper_pipeline import run_paper_loeo_pipeline
from hqrc_v3.provenance import file_sha256
from hqrc_v3.residual_stage import (
    load_standardized_residual_manifest,
    prepare_standardized_residual_artifact,
    select_diagnostic_residual_context,
)

StageHandler = Callable[[argparse.Namespace], object]
_EXPECTED_START = "2019-01-01T00:00:00"
_EXPECTED_END = "2024-10-31T23:00:00"
_EXPECTED_ROWS = 51_144
_DEFAULT_TEMPORARY_AVAILABILITY = (
    Path(__file__).resolve().parents[2] / "configs/temporary_holiday_availability.csv"
)


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
            "expected_public_holiday_dates": FIXED_PUBLIC_HOLIDAY_DATES,
            "expected_substitute_or_temporary_dates": (FIXED_SUBSTITUTE_OR_TEMPORARY_DATES),
            "temporary_holiday_availability": load_temporary_holiday_availability(
                Path(arguments.temporary_holiday_availability)
            ),
        }
    return audit_hourly_data(read_hourly_data(Path(arguments.data)), **expected)


def tune_baselines_handler(arguments: argparse.Namespace) -> object:
    return _unavailable_handler("tune-baselines")(arguments)


def generate_oof_handler(arguments: argparse.Namespace) -> object:
    inputs = _paper_stage_inputs(arguments)
    return run_paper_oof_stage(
        **inputs,
        oof_years=None if arguments.oof_years is None else tuple(arguments.oof_years),
    )


def fit_final_baselines_handler(arguments: argparse.Namespace) -> object:
    return run_paper_final_stage(**_paper_stage_inputs(arguments))


def prepare_residuals_handler(arguments: argparse.Namespace) -> object:
    """Publish all OOF point streams as one fold-standardized event artifact."""

    return prepare_standardized_residual_artifact(
        run_dir=Path(arguments.run_dir),
        data_path=Path(arguments.data),
        config_path=Path(arguments.config),
        model_config_path=Path(arguments.frozen_model_config),
        event_registry_path=Path(arguments.event_registry),
        holiday_calendar_path=Path(arguments.holiday_calendar),
        temporary_holiday_availability_path=Path(arguments.temporary_holiday_availability),
        profile=arguments.profile,
    )


def _paper_stage_inputs(arguments: argparse.Namespace) -> dict[str, object]:
    """Load, validate, feature, and hash every concrete baseline-stage input."""

    if not arguments.run_dir:
        raise StageInputError("baseline stages require --run-dir for immutable publication")
    data_path = Path(arguments.data)
    experiment_path = Path(arguments.config)
    model_path = Path(arguments.frozen_model_config)
    config_directory = experiment_path.parent
    event_path = Path(arguments.event_registry or config_directory / "events.csv")
    holiday_path = Path(arguments.holiday_calendar or config_directory / "holiday_calendar.csv")
    availability_path = Path(
        arguments.temporary_holiday_availability
        or config_directory / "temporary_holiday_availability.csv"
    )
    load_config(experiment_path)
    load_event_registry(event_path)
    calendar = load_holiday_calendar(holiday_path)
    temporary_availability = load_temporary_holiday_availability(availability_path)
    baseline_config = load_paper_baselines(
        model_path,
        expected_sha256=arguments.frozen_model_hash,
    )
    paper_bounds = (
        {
            "expected_start": datetime.fromisoformat(_EXPECTED_START),
            "expected_end": datetime.fromisoformat(_EXPECTED_END),
            "expected_rows": _EXPECTED_ROWS,
            "expected_public_holiday_dates": FIXED_PUBLIC_HOLIDAY_DATES,
            "expected_substitute_or_temporary_dates": (FIXED_SUBSTITUTE_OR_TEMPORARY_DATES),
            "temporary_holiday_availability": temporary_availability,
        }
        if arguments.profile == "paper"
        else {
            "expected_start": None,
            "expected_end": None,
            "expected_rows": None,
            "temporary_holiday_availability": temporary_availability,
        }
    )
    audited = audit_hourly_data(read_hourly_data(data_path), **paper_bounds)
    featured = attach_calendar_features(audited, calendar)
    selected_features = ("B0", "B1") if arguments.feature_set == "all" else (arguments.feature_set,)
    matrices = {
        feature_set: build_daily_forecast_matrix(featured, feature_set=feature_set)
        for feature_set in selected_features
    }
    selected_models = baseline_config.models if arguments.model == "all" else (arguments.model,)
    run_dir = Path(arguments.run_dir)
    cache_dir = Path(arguments.cache_dir or run_dir / "prediction-stream-cache")
    return {
        "matrices": matrices,
        "config": baseline_config,
        "run_dir": run_dir,
        "cache_dir": cache_dir,
        "artifact_hashes": {
            "data_sha256": file_sha256(data_path),
            "experiment_sha256": file_sha256(experiment_path),
            "model_config_sha256": file_sha256(model_path),
            "event_registry_sha256": file_sha256(event_path),
            "holiday_calendar_sha256": file_sha256(holiday_path),
            "temporary_holiday_availability_sha256": file_sha256(availability_path),
        },
        "classical_seed": arguments.seed,
        "models": selected_models,
        "feature_sets": selected_features,
        "profile": arguments.profile,
        "smoke_boosting_rounds": arguments.smoke_boosting_rounds,
    }


def diagnose_ar_handler(arguments: argparse.Namespace) -> object:
    run_dir = Path(arguments.run_dir)
    manifest_path = run_dir / "inputs/standardized_residuals_manifest.json"
    if not manifest_path.is_file():
        raise StageInputError(
            "diagnose-ar requires the canonical standardized_residuals_manifest.json"
        )
    config_path = Path(arguments.config)
    event_path = Path(arguments.event_registry)
    config_sha = file_sha256(config_path)
    event_sha = file_sha256(event_path)
    manifest = load_standardized_residual_manifest(
        manifest_path,
        run_dir=run_dir,
        config_sha256=config_sha,
        event_sha256=event_sha,
    )
    residual_path = run_dir / manifest["outputs"]["standardized_residuals"]["path"]
    residuals = select_diagnostic_residual_context(
        pl.read_parquet(residual_path),
        manifest,
        events=load_event_registry(event_path),
        model=arguments.model,
        feature_set=arguments.feature_set,
        seed=arguments.seed,
        through=arguments.through,
    )
    diagnostics = diagnose_event_residuals(residuals)
    calibration = calibrate_beta_prior(
        np.asarray([item.phi for item in diagnostics]),
        event_ids=tuple(item.occurrence_id for item in diagnostics),
    )
    return write_ar_diagnostics(
        Path(arguments.output),
        diagnostics,
        calibration,
        residual_sha256=file_sha256(residual_path),
        config_sha256=config_sha,
        event_sha256=event_sha,
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


def fit_corrections_handler(arguments: argparse.Namespace) -> object:
    """Fit only the bounded causal-2024 H3 stage from one approved context."""

    if arguments.evaluation == "loeo":
        raise StageInputError(
            "loeo requires fold-specific approved calibrations; that stage is not yet implemented"
        )
    return fit_causal_2024_correction(
        run_dir=Path(arguments.run_dir),
        config_path=Path(arguments.config),
        approved_ar_path=Path(arguments.approved_ar),
        sampler_seed=arguments.seed,
        profile=arguments.profile,
        draws=arguments.draws,
        tune=arguments.tune,
        chains=arguments.chains,
        cores=arguments.cores,
    )


def report_handler(arguments: argparse.Namespace) -> object:
    """Build a concrete report only from an already materialized run directory."""

    from hqrc_v3.evaluation.reports import build_report

    return build_report(Path(arguments.run_dir), profile=arguments.profile)


def _pipeline_models(value: str) -> tuple[str, ...]:
    return MODEL_NAMES if value == "all" else (value,)


def _pipeline_feature_sets(value: str) -> tuple[str, ...]:
    return ("B0", "B1") if value == "all" else (value,)


def _print_pipeline_result(result: object) -> None:
    """Emit one compact, machine-independent completion line per HQRC context."""

    for item in result:
        label = f"{item.context.model}/{item.context.feature_set}/seed-{item.context.seed}"
        if item.status == "COMPLETE":
            outputs = ",".join(f"{variant}={path}" for variant, path in item.variant_output_dirs)
            print(
                f"[HQRC] {label} COMPLETE fits={item.sampler_fit_count} "
                f"reused={item.reused} outputs={outputs}"
            )
        else:
            print(
                f"[HQRC] {label} {item.status} proposal={item.proposal_set_sha256} "
                f"ar_dir={item.ar_dir}"
            )


def run_loeo_primary_handler(arguments: argparse.Namespace) -> object:
    """Run one or all model contexts in the fixed manuscript order."""

    result = run_paper_loeo_pipeline(
        source_run_dir=Path(arguments.source_run_dir),
        config_path=Path(arguments.config),
        output_root=Path(arguments.output_root),
        models=_pipeline_models(arguments.model),
        feature_sets=_pipeline_feature_sets(arguments.feature_set),
        variants=tuple(arguments.variants),
        root_seed=arguments.root_seed,
        profile=arguments.profile,
        draws=arguments.draws,
        tune=arguments.tune,
        chains=arguments.chains,
        cores=arguments.cores,
        init=arguments.init,
        target_accept=arguments.target_accept,
        diagnostic_attempts=arguments.diagnostic_attempts,
        approve_derived_ar=arguments.approve_derived_ar,
        progress=print,
    )
    _print_pipeline_result(result)
    return result


def run_paper_handler(arguments: argparse.Namespace) -> object:
    """Build/reuse the common baseline source, then finish HQRC contexts in order."""

    # A paper-profile residual publication must cover every manuscript baseline
    # and both feature sets.  The subsequent HQRC selection can be a subset.
    arguments.seed = arguments.baseline_seed
    arguments.model = "all"
    arguments.feature_set = "all"
    arguments.oof_years = None
    print("[baseline] OOF: models in manuscript order, B0 then B1")
    generate_oof_handler(arguments)
    print("[baseline] final 2019--2023 fit and 2024 evaluation forecast")
    fit_final_baselines_handler(arguments)
    print("[residuals] fold-standardized OOF residual publication")
    prepare_residuals_handler(arguments)

    # The baseline stage reuses --model/--feature-set names, so the HQRC scope
    # is intentionally held in distinct parsed fields and cannot be overwritten.
    result = run_paper_loeo_pipeline(
        source_run_dir=Path(arguments.run_dir),
        config_path=Path(arguments.config),
        output_root=Path(arguments.output_root),
        models=_pipeline_models(arguments.hqrc_model),
        feature_sets=_pipeline_feature_sets(arguments.hqrc_feature_set),
        variants=tuple(arguments.hqrc_variants),
        root_seed=arguments.root_seed,
        profile=arguments.profile,
        draws=arguments.draws,
        tune=arguments.tune,
        chains=arguments.chains,
        cores=arguments.cores,
        init=arguments.init,
        target_accept=arguments.target_accept,
        diagnostic_attempts=arguments.diagnostic_attempts,
        approve_derived_ar=arguments.approve_derived_ar,
        progress=print,
    )
    _print_pipeline_result(result)
    return result


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
    parser.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    parser.add_argument("--feature-set", choices=("B0", "B1", "all"), default="all")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile", choices=("paper", "smoke"), default="paper")
    parser.add_argument(
        "--smoke-boosting-rounds",
        type=int,
        help="explicit non-paper XGBoost/LightGBM round cap",
    )
    parser.add_argument("--run-dir", help="run root containing predictions/")
    parser.add_argument("--event-registry", help="event registry CSV; defaults beside --config")
    parser.add_argument(
        "--holiday-calendar", help="holiday feature calendar CSV; defaults beside --config"
    )
    parser.add_argument(
        "--temporary-holiday-availability",
        help="temporary-holiday availability CSV; defaults beside --config",
    )


def _add_loeo_pipeline_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-root",
        required=True,
        help="portable artifact root for LOEO, AR, posterior, and aggregate products",
    )
    parser.add_argument(
        "--root-seed",
        type=int,
        default=20260813,
        help="root seed from which all fold sampler/predictive seeds are derived",
    )
    parser.add_argument("--draws", type=int, help="retained draws per chain")
    parser.add_argument("--tune", type=int, help="warm-up draws per chain")
    parser.add_argument("--chains", type=int, help="number of NUTS chains")
    parser.add_argument("--cores", type=int, help="parallel PyMC chain worker count")
    parser.add_argument(
        "--init",
        choices=("adapt_diag", "jitter+adapt_diag"),
        default="jitter+adapt_diag",
        help="explicit NUTS initialization contract",
    )
    parser.add_argument(
        "--target-accept",
        type=float,
        default=0.99,
        help="NUTS target acceptance probability",
    )
    parser.add_argument(
        "--diagnostic-attempts",
        type=int,
        default=1,
        help=(
            "maximum deterministic sampler attempts per H1/H2 fold; rejected diagnostics "
            "are recorded and retries use separately derived seeds"
        ),
    )
    parser.add_argument(
        "--approve-derived-ar",
        action="store_true",
        help=(
            "freeze the generated ACF/PACF-derived proposal digest and continue to H3 sampling; "
            "without this flag the run stops after AR diagnostics"
        ),
    )


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
    audit.add_argument(
        "--temporary-holiday-availability",
        default=str(_DEFAULT_TEMPORARY_AVAILABILITY),
        help="versioned exceptional-holiday availability CSV",
    )

    tune = subcommands.add_parser("tune-baselines", help="run one explicit selection stage")
    _add_data_inputs(tune)

    oof = subcommands.add_parser(
        "generate-oof", help="generate cache-aware frozen-config OOF forecasts"
    )
    _add_data_inputs(oof)
    _add_frozen_model_inputs(oof)
    oof.add_argument("--cache-dir", required=True, help="immutable prediction cache directory")
    oof.add_argument(
        "--oof-years",
        nargs="+",
        type=int,
        help="smoke-only ordered OOF-year subset; paper always uses 2020 2021 2022 2023",
    )

    final = subcommands.add_parser("fit-final-baselines", help="refit frozen config for 2024")
    _add_data_inputs(final)
    _add_frozen_model_inputs(final)
    final.add_argument("--cache-dir", help="immutable prediction cache directory")

    residuals = subcommands.add_parser(
        "prepare-residuals",
        help="standardize OOF event residuals with fold-local non-event scales",
    )
    residuals.add_argument("--run-dir", required=True, help="run root containing predictions/")
    residuals.add_argument("--data", required=True, help="hourly source data path")
    residuals.add_argument("--config", required=True, help="experiment TOML path")
    residuals.add_argument(
        "--frozen-model-config", required=True, help="selected immutable model config"
    )
    residuals.add_argument("--event-registry", required=True, help="fixed correction registry")
    residuals.add_argument("--holiday-calendar", required=True, help="holiday feature calendar")
    residuals.add_argument(
        "--temporary-holiday-availability",
        required=True,
        help="versioned exceptional-holiday availability CSV",
    )
    residuals.add_argument("--profile", choices=("paper", "smoke"), default="paper")

    diagnose = subcommands.add_parser(
        "diagnose-ar", help="generate an unapproved event-reset AR diagnostic proposal"
    )
    diagnose.add_argument("--run-dir", required=True, help="canonical completed run root")
    diagnose.add_argument("--config", required=True, help="experiment TOML path")
    diagnose.add_argument("--event-registry", required=True, help="fixed correction registry")
    diagnose.add_argument("--output", required=True, help="unapproved AR diagnostic JSON path")
    diagnose.add_argument("--through", type=int, required=True, help="latest OOF year included")
    diagnose.add_argument("--model", choices=MODEL_NAMES, required=True)
    diagnose.add_argument("--feature-set", choices=("B0", "B1"), required=True)
    diagnose.add_argument(
        "--seed",
        type=int,
        help="optional point-stream seed assertion; otherwise inferred after context selection",
    )

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
    corrections.add_argument(
        "--seed",
        type=int,
        required=True,
        help="sampler and posterior-predictive RNG seed; baseline seed comes from approval",
    )
    corrections.add_argument(
        "--draws",
        type=int,
        help="retained draws (smoke: required; paper: optional, minimum/default 1000)",
    )
    corrections.add_argument(
        "--tune",
        type=int,
        help="warm-up draws (smoke: required; paper: optional, minimum/default 1000)",
    )
    corrections.add_argument(
        "--chains",
        type=int,
        help="chains (smoke: required; paper: optional/default 4 and must equal 4)",
    )
    corrections.add_argument(
        "--cores",
        type=int,
        help="PyMC worker cores (optional/default 1; must not exceed chains)",
    )
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

    loeo_primary = subcommands.add_parser(
        "run-loeo-primary",
        help="sequentially run ACF/AR/H3 LOEO for one or more existing baseline contexts",
    )
    loeo_primary.add_argument(
        "--source-run-dir", required=True, help="completed local baseline/residual publication"
    )
    loeo_primary.add_argument("--config", required=True, help="experiment TOML bound to source")
    loeo_primary.add_argument("--model", choices=(*MODEL_NAMES, "all"), default="all")
    loeo_primary.add_argument("--feature-set", choices=("B0", "B1", "all"), default="B1")
    loeo_primary.add_argument(
        "--variants",
        nargs="+",
        choices=("H1", "H2", "H3"),
        default=("H1", "H2", "H3"),
        help="ordered correction variants (default: H1 H2 H3)",
    )
    loeo_primary.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    _add_loeo_pipeline_options(loeo_primary)

    paper = subcommands.add_parser(
        "run-paper",
        help="build all baseline sources then process HQRC models sequentially",
    )
    _add_data_inputs(paper)
    paper.add_argument("--frozen-model-config", required=True, help="frozen manuscript model TOML")
    paper.add_argument("--frozen-model-hash", required=True, help="SHA-256 of frozen model TOML")
    paper.add_argument("--event-registry", required=True, help="fixed correction registry")
    paper.add_argument("--holiday-calendar", required=True, help="holiday feature calendar")
    paper.add_argument(
        "--temporary-holiday-availability",
        required=True,
        help="known-at-origin temporary-holiday registry",
    )
    paper.add_argument("--run-dir", required=True, help="local immutable baseline/residual root")
    paper.add_argument(
        "--cache-dir", help="baseline prediction-stream cache; defaults below run-dir"
    )
    paper.add_argument("--baseline-seed", type=int, default=7, help="classical baseline RNG seed")
    paper.add_argument("--profile", choices=("smoke", "paper"), default="paper")
    paper.add_argument(
        "--smoke-boosting-rounds", type=int, help="explicit reduced-round smoke override"
    )
    paper.add_argument(
        "--hqrc-model", choices=(*MODEL_NAMES, "all"), default="all", help="HQRC model scope"
    )
    paper.add_argument(
        "--hqrc-feature-set",
        choices=("B0", "B1", "all"),
        default="B1",
        help="HQRC feature-set scope after the common baseline source is complete",
    )
    paper.add_argument(
        "--hqrc-variants",
        nargs="+",
        choices=("H1", "H2", "H3"),
        default=("H1", "H2", "H3"),
        help="ordered correction variants (default: H1 H2 H3)",
    )
    _add_loeo_pipeline_options(paper)

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
        "prepare-residuals": prepare_residuals_handler,
        "diagnose-ar": diagnose_ar_handler,
        "approve-ar-calibration": approve_ar_calibration_handler,
        "fit-corrections": fit_corrections_handler,
        "run-ablations": _unavailable_handler("run-ablations"),
        "benchmark-samplers": _unavailable_handler("benchmark-samplers"),
        "run-loeo-primary": run_loeo_primary_handler,
        "run-paper": run_paper_handler,
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
    except (
        ConfigError,
        DataContractError,
        OSError,
        SamplingError,
        StageInputError,
        ValueError,
    ) as error:
        print(f"hqrc: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised by the console entry point
    raise SystemExit(main())
