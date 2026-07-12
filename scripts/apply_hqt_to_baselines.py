"""Apply Hierarchical Quadratic Tilting to saved baseline forecasts."""

from __future__ import annotations

import argparse
from pathlib import Path

from demand_quadratic_tilting.hqt_experiment import (
    BASELINE_MODEL_NAMES,
    run_hqt_baseline_experiment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit one HQT posterior per baseline and compare MW errors."
    )
    parser.add_argument(
        "--predictions",
        type=Path,
        default=Path("artifacts/baseline_168_24/predictions.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hqt_baselines"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=BASELINE_MODEL_NAMES,
        default=list(BASELINE_MODEL_NAMES),
    )
    parser.add_argument("--fit-splits", nargs="+", default=["train", "validation"])
    parser.add_argument("--evaluation-splits", nargs="+", default=["test"])
    parser.add_argument(
        "--sampler", choices=("nuts", "numpyro", "advi"), default="nuts"
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--target-accept", type=float, default=0.99)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--tilt-mode", choices=("hybrid", "event", "type"), default="hybrid"
    )
    parser.add_argument("--ci", type=float, default=0.95)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--threshold-k", type=float, default=0.5)
    parser.add_argument("--gate-scale-k", type=float, default=0.3)
    parser.add_argument("--pre-pad-days", type=int, default=1)
    parser.add_argument("--post-pad-days", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use 2 chains × 100 draws for a pipeline smoke test only.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.quick:
        chains, draws, tune, target_accept = 2, 100, 100, 0.9
    else:
        chains, draws, tune, target_accept = (
            args.chains,
            args.draws,
            args.tune,
            args.target_accept,
        )
    run_hqt_baseline_experiment(
        args.predictions,
        args.output_dir,
        model_names=args.models,
        fit_splits=args.fit_splits,
        evaluation_splits=args.evaluation_splits,
        sampler=args.sampler,
        chains=chains,
        draws=draws,
        tune=tune,
        target_accept=target_accept,
        random_seed=args.seed,
        tilt_mode=args.tilt_mode,
        ci=args.ci,
        gate=args.gate,
        threshold_k=args.threshold_k,
        gate_scale_k=args.gate_scale_k,
        pre_pad_days=args.pre_pad_days,
        post_pad_days=args.post_pad_days,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
