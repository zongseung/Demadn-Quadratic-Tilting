"""Apply the same HQT specification to multiple baseline forecasts."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Callable, Iterable, Mapping

import joblib
import numpy as np
import polars as pl

from .constants import CHUSEOK_LABELS, SEOLLAL_LABELS
from .model import compute_sigma_and_residuals, fit_hqt_pymc_lkj
from .tilt import apply_tilt, tilt_from_posterior
from .windows import build_holiday_windows_and_tau


BASELINE_MODEL_NAMES = (
    "xgboost",
    "lightgbm",
    "random_forest",
    "svr",
    "seq2seq_lstm",
    "seq2seq_gru",
)


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _metric_row(
    model_name: str,
    split_name: str,
    scope: str,
    y_true: np.ndarray,
    baseline: np.ndarray,
    tilted: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    e_tilt: np.ndarray,
    mask: np.ndarray,
    sigma_resid_mw: float,
) -> dict[str, object]:
    y = y_true[mask]
    base = baseline[mask]
    adjusted = tilted[mask]
    lo = lower[mask]
    hi = upper[mask]
    tilt = e_tilt[mask]
    if not y.size:
        raise ValueError(f"scope {scope!r} has no observations")

    base_error = base - y
    tilted_error = adjusted - y
    baseline_mae = float(np.mean(np.abs(base_error)))
    tilted_mae = float(np.mean(np.abs(tilted_error)))
    baseline_rmse = float(np.sqrt(np.mean(base_error**2)))
    tilted_rmse = float(np.sqrt(np.mean(tilted_error**2)))
    interval_active = bool(np.all(hi > lo))
    return {
        "model": model_name,
        "split": split_name,
        "scope": scope,
        "n": int(y.size),
        "sigma_resid_mw": sigma_resid_mw,
        "baseline_mae_mw": baseline_mae,
        "tilted_mae_mw": tilted_mae,
        "mae_improvement_mw": baseline_mae - tilted_mae,
        "mae_improvement_pct": 100 * (baseline_mae - tilted_mae) / baseline_mae,
        "baseline_rmse_mw": baseline_rmse,
        "tilted_rmse_mw": tilted_rmse,
        "rmse_improvement_mw": baseline_rmse - tilted_rmse,
        "rmse_improvement_pct": (100 * (baseline_rmse - tilted_rmse) / baseline_rmse),
        "baseline_bias_mw": float(np.mean(base_error)),
        "tilted_bias_mw": float(np.mean(tilted_error)),
        "mean_tilt_mw": float(np.mean(tilt)),
        "picp_95_pct": (
            float(100 * np.mean((y >= lo) & (y <= hi))) if interval_active else None
        ),
        "aiw_mw": float(np.mean(hi - lo)) if interval_active else None,
    }


def _evaluate_model(
    model_name: str,
    split_frame: pl.DataFrame,
    tilted_frame: pl.DataFrame,
    event_id_of_date: Mapping[object, str],
    type_of_date: Mapping[object, str],
    sigma_resid_mw: float,
    y_col: str,
    holiday_name_col: str,
    datetime_col: str,
) -> list[dict[str, object]]:
    dates = split_frame[datetime_col].to_list()
    y = split_frame[y_col].cast(pl.Float64).to_numpy()
    baseline = split_frame[model_name].cast(pl.Float64).to_numpy()
    tilted = tilted_frame["tilted_pred"].to_numpy()
    lower = tilted_frame["lower_t"].to_numpy()
    upper = tilted_frame["upper_t"].to_numpy()
    e_tilt = tilted_frame["e_tilt"].to_numpy()
    holiday_names = split_frame[holiday_name_col].to_list()

    masks = {
        "overall": np.ones(len(dates), dtype=bool),
        "holiday_window": np.array(
            [date in event_id_of_date for date in dates], dtype=bool
        ),
        "official_major_holiday": np.array(
            [name in (CHUSEOK_LABELS | SEOLLAL_LABELS) for name in holiday_names],
            dtype=bool,
        ),
        "chuseok_window": np.array(
            [type_of_date.get(date) == "Chuseok" for date in dates], dtype=bool
        ),
        "seollal_window": np.array(
            [type_of_date.get(date) == "Seollal" for date in dates], dtype=bool
        ),
    }
    split_name = str(split_frame["split"][0])
    return [
        _metric_row(
            model_name,
            split_name,
            scope,
            y,
            baseline,
            tilted,
            lower,
            upper,
            e_tilt,
            mask,
            sigma_resid_mw,
        )
        for scope, mask in masks.items()
        if mask.any()
    ]


def run_hqt_baseline_experiment(
    predictions_path: str | Path,
    output_dir: str | Path,
    *,
    model_names: Iterable[str] = BASELINE_MODEL_NAMES,
    fit_splits: Iterable[str] = ("train", "validation"),
    evaluation_splits: Iterable[str] = ("test",),
    y_col: str = "power demand(MW)",
    holiday_name_col: str = "holiday_name",
    datetime_col: str = "datetime",
    sampler: str = "nuts",
    chains: int = 4,
    draws: int = 1000,
    tune: int = 1000,
    target_accept: float = 0.99,
    random_seed: int = 2025,
    tilt_mode: str = "hybrid",
    ci: float = 0.95,
    gate: bool = False,
    threshold_k: float = 0.5,
    gate_scale_k: float = 0.3,
    pre_pad_days: int = 1,
    post_pad_days: int = 1,
    tau_scale_hours: float = 24.0,
    resume: bool = False,
    progress: Callable[[str], None] = print,
) -> dict[str, object]:
    """Fit HQT per baseline and evaluate the same held-out event windows."""

    selected = tuple(dict.fromkeys(model_names))
    unknown = sorted(set(selected) - set(BASELINE_MODEL_NAMES))
    if unknown:
        raise ValueError(f"unknown baseline models: {unknown}")
    fit_splits = tuple(fit_splits)
    evaluation_splits = tuple(evaluation_splits)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    frame = pl.read_csv(predictions_path, try_parse_dates=True).sort(datetime_col)
    required = {
        datetime_col,
        y_col,
        holiday_name_col,
        "split",
        *selected,
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"prediction file is missing columns: {missing}")
    if frame.select(list(required)).null_count().to_numpy().sum():
        raise ValueError("HQT input columns contain null values")

    fit_frame = frame.filter(pl.col("split").is_in(fit_splits))
    if fit_frame.is_empty():
        raise ValueError("HQT fitting split is empty")
    evaluation_frames = {
        split_name: frame.filter(pl.col("split") == split_name)
        for split_name in evaluation_splits
    }
    if any(split.is_empty() for split in evaluation_frames.values()):
        raise ValueError("one or more HQT evaluation splits are empty")

    windows, tau_map, event_id_of_date, type_of_date, tau_unit_hours = (
        build_holiday_windows_and_tau(
            frame.select([datetime_col, holiday_name_col]),
            holiday_name_col=holiday_name_col,
            datetime_col=datetime_col,
            tau_unit="1h",
            pre_pad_days=pre_pad_days,
            post_pad_days=post_pad_days,
        )
    )
    fit_event_ids = sorted(
        {
            event_id_of_date[date]
            for date in fit_frame[datetime_col].to_list()
            if date in event_id_of_date
        }
    )

    config_path = output / "config.json"
    existing_models: list[str] = []
    if config_path.exists():
        existing_config = json.loads(config_path.read_text(encoding="utf-8"))
        existing_models = list(existing_config.get("models", []))
    configured_models = list(dict.fromkeys([*existing_models, *selected]))
    config = {
        "predictions_path": str(predictions_path),
        "models": configured_models,
        "fit_splits": list(fit_splits),
        "evaluation_splits": list(evaluation_splits),
        "y_col": y_col,
        "sampler": sampler,
        "chains": chains,
        "draws": draws,
        "tune": tune,
        "target_accept": target_accept,
        "random_seed": random_seed,
        "tilt_mode": tilt_mode,
        "ci": ci,
        "gate": gate,
        "threshold_k": threshold_k,
        "gate_scale_k": gate_scale_k,
        "pre_pad_days": pre_pad_days,
        "post_pad_days": post_pad_days,
        "tau_scale_hours": tau_scale_hours,
        "fit_event_ids": fit_event_ids,
    }
    _write_json(config_path, config)

    for model_name in selected:
        model_dir = output / model_name
        metrics_path = model_dir / "metrics.json"
        if resume and metrics_path.exists():
            progress(f"Skipping completed HQT model {model_name}")
            continue
        model_dir.mkdir(parents=True, exist_ok=True)
        progress(f"Fitting HQT for {model_name}")
        started = perf_counter()

        sigma_resid, residuals = compute_sigma_and_residuals(
            fit_frame,
            y_col,
            model_name,
            holiday_name_col,
        )
        hqt = fit_hqt_pymc_lkj(
            df_train=fit_frame,
            sigma_resid=sigma_resid,
            residuals=residuals,
            tau_map=tau_map,
            event_id_of_date=event_id_of_date,
            type_of_date=type_of_date,
            datetime_col=datetime_col,
            tau_scale_hours=tau_scale_hours,
            tau_unit_hours=tau_unit_hours,
            chains=chains,
            draws=draws,
            tune=tune,
            target_accept=target_accept,
            sampler=sampler,
            random_seed=random_seed,
        )
        joblib.dump(hqt, model_dir / "posterior.joblib", compress=3)

        rows: list[dict[str, object]] = []
        for split_name, split_frame in evaluation_frames.items():
            tilt_df, _ = tilt_from_posterior(
                hqt,
                sigma_resid,
                split_frame[datetime_col].to_list(),
                tilt_mode=tilt_mode,
                ci=ci,
                rng_seed=random_seed,
                gate=gate,
                threshold_k=threshold_k,
                gate_scale_k=gate_scale_k,
            )
            tilted_frame = apply_tilt(
                split_frame,
                tilt_df,
                model_name,
                datetime_col,
            )
            event_ids = [
                event_id_of_date.get(date)
                for date in split_frame[datetime_col].to_list()
            ]
            event_types = [
                type_of_date.get(date) for date in split_frame[datetime_col].to_list()
            ]
            detailed = tilted_frame.select(
                [
                    datetime_col,
                    y_col,
                    "split",
                    holiday_name_col,
                    pl.col(model_name).alias("baseline_pred"),
                    "e_tilt",
                    "tilted_pred",
                    "half_width_t",
                    "lower_t",
                    "upper_t",
                ]
            ).with_columns(
                pl.Series("event_id", event_ids, dtype=pl.String),
                pl.Series("event_type", event_types, dtype=pl.String),
            )
            detailed.write_csv(model_dir / f"{split_name}_predictions.csv")
            rows.extend(
                _evaluate_model(
                    model_name,
                    split_frame,
                    tilted_frame,
                    event_id_of_date,
                    type_of_date,
                    sigma_resid,
                    y_col,
                    holiday_name_col,
                    datetime_col,
                )
            )

        elapsed = perf_counter() - started
        result = {
            "model": model_name,
            "sigma_resid_mw": sigma_resid,
            "fit_event_ids": hqt.event_ids,
            "sampling": {
                "sampler": sampler,
                "chains": chains,
                "draws": draws,
                "tune": tune,
                "target_accept": target_accept,
                "random_seed": random_seed,
            },
            "diagnostics": hqt.diagnostics,
            "elapsed_seconds": elapsed,
            "metrics": rows,
        }
        _write_json(metrics_path, result)
        progress(
            f"Finished HQT for {model_name} in {elapsed:.1f}s; "
            f"diagnostics={hqt.diagnostics}"
        )

    model_results = []
    metric_rows = []
    for metrics_path in sorted(output.glob("*/metrics.json")):
        value = json.loads(metrics_path.read_text(encoding="utf-8"))
        sampling = value.get(
            "sampling",
            {
                "sampler": "nuts",
                "chains": 4,
                "draws": 1000,
                "tune": 1000,
                "target_accept": 0.99,
                "random_seed": 2025,
            },
        )
        if "sampling" not in value:
            value["sampling"] = sampling
            _write_json(metrics_path, value)
        model_results.append(
            {
                "model": value["model"],
                "sigma_resid_mw": value["sigma_resid_mw"],
                "fit_event_ids": value["fit_event_ids"],
                "sampling": sampling,
                "diagnostics": value["diagnostics"],
                "elapsed_seconds": value["elapsed_seconds"],
            }
        )
        metric_rows.extend(value["metrics"])
    if metric_rows:
        pl.DataFrame(metric_rows).write_csv(output / "metrics.csv")
    summary = {"config": config, "models": model_results}
    _write_json(output / "summary.json", summary)
    return {"summary": summary, "metrics": metric_rows, "windows": windows}
