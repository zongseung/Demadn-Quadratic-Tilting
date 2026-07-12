"""Command-line interface for the 168-to-24 baseline experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import ForecastConfig
from .experiment import ALL_MODEL_NAMES, run_baseline_experiment
from .seq2seq import Seq2SeqTrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train XGBoost, LightGBM, Random Forest, SVR, Seq2Seq-LSTM, "
            "and Seq2Seq-GRU "
            "on 168-hour histories to forecast the next 24 hours."
        )
    )
    parser.add_argument("--data", type=Path, default=Path("power_demand_final.csv"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/baseline_168_24"),
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=ALL_MODEL_NAMES,
        default=list(ALL_MODEL_NAMES),
    )
    parser.add_argument("--history-hours", type=int, default=168)
    parser.add_argument("--horizon-hours", type=int, default=24)
    parser.add_argument("--stride-hours", type=int, default=24)
    parser.add_argument("--train-end", default="2022-12-31 23:00:00")
    parser.add_argument("--validation-end", default="2023-12-31 23:00:00")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--ml-n-jobs", type=int, default=None)
    parser.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "mps", "cuda")
    )
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--teacher-forcing-ratio", type=float, default=0.5)
    parser.add_argument(
        "--decoder-mode",
        choices=("autoregressive", "context"),
        default="autoregressive",
    )
    parser.add_argument(
        "--known-covariates",
        nargs="+",
        default=["spring", "summer", "autoum", "winter"],
        help=(
            "Columns known for the forecast horizon. To add the generic "
            "holiday indicator, append is_holiday_dummies."
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use tiny model budgets for a pipeline smoke test.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Keep existing model columns and replace only the requested models.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ForecastConfig(
        known_covariate_cols=tuple(args.known_covariates),
        history_hours=args.history_hours,
        horizon_hours=args.horizon_hours,
        stride_hours=args.stride_hours,
        train_end=args.train_end,
        validation_end=args.validation_end,
        random_seed=args.seed,
    )
    if args.quick:
        training = Seq2SeqTrainingConfig(
            hidden_size=16,
            num_layers=1,
            dropout=0.0,
            learning_rate=args.learning_rate,
            batch_size=max(args.batch_size, 128),
            max_epochs=min(args.max_epochs, 2),
            patience=2,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            decoder_mode=args.decoder_mode,
        )
    else:
        training = Seq2SeqTrainingConfig(
            hidden_size=args.hidden_size,
            num_layers=args.num_layers,
            dropout=args.dropout,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            teacher_forcing_ratio=args.teacher_forcing_ratio,
            decoder_mode=args.decoder_mode,
        )
    run_baseline_experiment(
        args.data,
        args.output_dir,
        config=config,
        models=args.models,
        seq2seq_training=training,
        quick=args.quick,
        ml_n_jobs=args.ml_n_jobs,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
