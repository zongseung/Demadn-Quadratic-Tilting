"""End-to-end baseline training and artifact generation."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable

import joblib
import numpy as np
import polars as pl
import torch

from .config import ForecastConfig
from .data import (
    WindowedDataset,
    build_windowed_dataset,
    fit_scalers,
    flatten_model_inputs,
    load_hourly_data,
    scale_split,
)
from .evaluation import build_prediction_frame, metric_rows
from .ml import ML_MODEL_NAMES, fit_direct_multi_horizon, model_summary
from .seq2seq import (
    Seq2SeqTrainingConfig,
    checkpoint_payload,
    fit_seq2seq,
    predict_seq2seq,
    resolve_device,
    resolve_seq2seq_device,
)


ALL_MODEL_NAMES = (*ML_MODEL_NAMES, "seq2seq_lstm", "seq2seq_gru")


def _inverse_target(values: np.ndarray, scaler: object) -> np.ndarray:
    shape = values.shape
    return (
        scaler.inverse_transform(values.reshape(-1, 1))
        .reshape(shape)
        .astype(np.float32)
    )


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _load_existing_predictions(
    output: Path,
    dataset: WindowedDataset,
    replacing: tuple[str, ...],
) -> dict[str, dict[str, np.ndarray]]:
    path = output / "predictions.csv"
    if not path.exists():
        return {}
    frame = pl.read_csv(path, try_parse_dates=True).sort("datetime")
    existing: dict[str, dict[str, np.ndarray]] = {}
    for model_name in ALL_MODEL_NAMES:
        if model_name not in frame.columns or model_name in replacing:
            continue
        model_splits: dict[str, np.ndarray] = {}
        for split_name in ("train", "validation", "test"):
            split_frame = frame.filter(pl.col("split") == split_name)
            expected = dataset.splits[split_name]
            observed_dates = split_frame["datetime"].to_numpy().astype("datetime64[us]")
            if not np.array_equal(
                observed_dates, expected.target_datetimes.reshape(-1)
            ):
                raise ValueError(
                    "existing prediction timestamps do not match this experiment"
                )
            model_splits[split_name] = (
                split_frame[model_name]
                .to_numpy()
                .reshape(expected.target.shape)
                .astype(np.float32)
            )
        existing[model_name] = model_splits
    return existing


def run_baseline_experiment(
    data_path: str | Path,
    output_dir: str | Path,
    *,
    config: ForecastConfig | None = None,
    models: Iterable[str] = ALL_MODEL_NAMES,
    seq2seq_training: Seq2SeqTrainingConfig | None = None,
    quick: bool = False,
    ml_n_jobs: int | None = None,
    device: str = "auto",
    resume: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Train requested baselines and write predictions, metrics, and models."""

    config = config or ForecastConfig()
    config.validate()
    selected = tuple(dict.fromkeys(models))
    unknown = sorted(set(selected) - set(ALL_MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown models: {unknown}; choices are {ALL_MODEL_NAMES}")
    if not selected:
        raise ValueError("at least one model must be selected")

    output = Path(output_dir)
    model_dir = output / "models"
    output.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    progress(f"Loading hourly data from {data_path}")
    frame = load_hourly_data(data_path, config)
    dataset = build_windowed_dataset(frame, config)
    scalers = fit_scalers(frame, config)
    scaled = {
        name: scale_split(split, scalers, config)
        for name, split in dataset.splits.items()
    }
    split_counts = {name: len(split) for name, split in dataset.splits.items()}
    progress(f"Window counts: {split_counts}")

    joblib.dump(scalers, output / "scalers.joblib")
    ml_inputs: dict[str, np.ndarray] = {}
    if any(name in ML_MODEL_NAMES for name in selected):
        ml_inputs = {
            name: flatten_model_inputs(split) for name, split in scaled.items()
        }

    training = seq2seq_training or Seq2SeqTrainingConfig()
    all_predictions = (
        _load_existing_predictions(output, dataset, selected) if resume else {}
    )
    metrics_path = output / "metrics.csv"
    if resume and metrics_path.exists():
        metrics = (
            pl.read_csv(metrics_path)
            .filter(~pl.col("model").is_in(selected))
            .to_dicts()
        )
    else:
        metrics: list[dict[str, object]] = []

    summary_path = output / "training_summary.json"
    if resume and summary_path.exists():
        summaries = json.loads(summary_path.read_text(encoding="utf-8"))
        summaries["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        summaries["device"] = str(resolve_device(device))
        summaries["split_windows"] = split_counts
        summaries.setdefault("models", {})
        for model_name in selected:
            summaries["models"].pop(model_name, None)
    else:
        summaries: dict[str, object] = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "device": str(resolve_device(device)),
            "split_windows": split_counts,
            "models": {},
        }

    configured_models = list(dict.fromkeys([*all_predictions, *selected]))
    _write_json(
        output / "config.json",
        {
            **config.to_dict(),
            "data_path": str(Path(data_path)),
            "models": configured_models,
            "quick": quick,
        },
    )

    for model_name in selected:
        progress(f"Training {model_name}")
        started = perf_counter()
        scaled_predictions: dict[str, np.ndarray]

        if model_name in ML_MODEL_NAMES:
            fitted = fit_direct_multi_horizon(
                model_name,
                ml_inputs["train"],
                scaled["train"].target,
                ml_inputs["validation"],
                scaled["validation"].target,
                random_seed=config.random_seed,
                n_jobs=ml_n_jobs,
                quick=quick,
            )
            joblib.dump(fitted, model_dir / f"{model_name}.joblib")
            scaled_predictions = {
                split_name: fitted.predict(ml_inputs[split_name])
                for split_name in ("train", "validation", "test")
            }
            detail: dict[str, object] = model_summary(fitted)
        else:
            kind = "lstm" if model_name == "seq2seq_lstm" else "gru"
            model_device = resolve_seq2seq_device(kind, device)
            if model_device.type != resolve_device(device).type:
                progress(
                    f"Using {model_device} for {model_name} to preserve "
                    "checkpoint round-trip reproducibility"
                )
            fitted, records = fit_seq2seq(
                kind,
                scaled["train"].history,
                scaled["train"].future_known,
                scaled["train"].target,
                scaled["validation"].history,
                scaled["validation"].future_known,
                scaled["validation"].target,
                training=training,
                random_seed=config.random_seed,
                device=device,
                progress=lambda message: progress(f"{model_name}: {message}"),
            )
            torch.save(
                checkpoint_payload(fitted, training, records),
                model_dir / f"{model_name}.pt",
            )
            scaled_predictions = {
                split_name: predict_seq2seq(
                    fitted,
                    scaled[split_name].history,
                    scaled[split_name].future_known,
                    batch_size=max(256, training.batch_size),
                    device=device,
                )
                for split_name in ("train", "validation", "test")
            }
            best_record = min(records, key=lambda row: row["validation_mse_scaled"])
            detail = {
                "kind": kind,
                "device": str(model_device),
                "epochs_run": len(records),
                "best_epoch": int(best_record["epoch"]),
                "best_validation_mse_scaled": best_record["validation_mse_scaled"],
                "training_config": asdict(training),
            }

        predictions = {
            split_name: _inverse_target(values, scalers.target)
            for split_name, values in scaled_predictions.items()
        }
        all_predictions[model_name] = predictions
        elapsed = perf_counter() - started
        detail["elapsed_seconds"] = float(elapsed)
        summaries["models"][model_name] = detail

        for split_name in ("train", "validation", "test"):
            metrics.extend(
                metric_rows(
                    model_name,
                    split_name,
                    dataset.splits[split_name].target,
                    predictions[split_name],
                )
            )

        build_prediction_frame(dataset, all_predictions).write_csv(
            output / "predictions.csv"
        )
        pl.DataFrame(metrics).write_csv(output / "metrics.csv")
        _write_json(output / "training_summary.json", summaries)
        progress(f"Finished {model_name} in {elapsed:.1f}s")

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    overall_test = [
        row for row in metrics if row["split"] == "test" and row["scope"] == "overall"
    ]
    summaries["test_overall"] = overall_test
    _write_json(output / "training_summary.json", summaries)
    return {
        "config": config,
        "dataset": dataset,
        "scalers": scalers,
        "predictions": all_predictions,
        "metrics": metrics,
        "summary": summaries,
        "output_dir": output,
    }
