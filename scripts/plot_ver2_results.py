"""Generate the publication-ready figures referenced by the ver2 README."""

from __future__ import annotations

import argparse
from datetime import timedelta
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter


MODEL_ORDER = (
    "lightgbm",
    "xgboost",
    "svr",
    "seq2seq_lstm",
    "random_forest",
    "seq2seq_gru",
)
MODEL_LABELS = {
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "random_forest": "Random Forest",
    "svr": "SVR",
    "seq2seq_lstm": "Seq2Seq-LSTM",
    "seq2seq_gru": "Seq2Seq-GRU",
}

ACTUAL_COLOR = "#111111"
BASELINE_COLOR = "#E69F00"
TILTED_COLOR = "#0072B2"
RMSE_COLOR = "#009E73"
INTERVAL_COLOR = "#56B4E9"
HOLIDAY_COLOR = "#D9D9D9"


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.titleweight": "semibold",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "xtick.color": "#333333",
            "ytick.color": "#333333",
            "grid.color": "#D0D0D0",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def _read_csv(path: Path, required: set[str]) -> pl.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"required plotting input does not exist: {path}")
    frame = pl.read_csv(path, try_parse_dates=True)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing plotting columns: {missing}")
    return frame


def _save(fig: plt.Figure, path: Path, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def _annotate_horizontal_bars(
    ax: plt.Axes,
    bars: object,
    values: np.ndarray,
    *,
    suffix: str = "",
    decimals: int = 0,
) -> None:
    max_value = float(np.max(np.abs(values))) if values.size else 1.0
    offset = max_value * 0.018
    for bar, value in zip(bars, values, strict=True):
        ax.text(
            float(value) + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.{decimals}f}{suffix}",
            va="center",
            ha="left",
            fontsize=8.5,
            color="#222222",
        )


def plot_baseline_performance(metrics_path: Path, output: Path, dpi: int) -> Path:
    """Plot held-out 2024 MAE and RMSE for the six baseline models."""

    frame = _read_csv(metrics_path, {"model", "n", "mae_mw", "rmse_mw"}).sort("mae_mw")
    models = frame["model"].to_list()
    if set(models) != set(MODEL_ORDER):
        raise ValueError(
            "baseline metric file must contain exactly the six ver2 models"
        )

    labels = [MODEL_LABELS[model] for model in models]
    mae = frame["mae_mw"].to_numpy()
    rmse = frame["rmse_mw"].to_numpy()
    y_pos = np.arange(len(models))
    height = 0.36

    fig, ax = plt.subplots(figsize=(10.5, 5.8), layout="constrained")
    mae_bars = ax.barh(
        y_pos - height / 2,
        mae,
        height,
        color=TILTED_COLOR,
        label="MAE",
    )
    rmse_bars = ax.barh(
        y_pos + height / 2,
        rmse,
        height,
        color=RMSE_COLOR,
        label="RMSE",
    )
    _annotate_horizontal_bars(ax, mae_bars, mae)
    _annotate_horizontal_bars(ax, rmse_bars, rmse)

    ax.set_yticks(y_pos, labels)
    ax.invert_yaxis()
    ax.set_xlim(0, float(rmse.max()) * 1.18)
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax.set_xlabel("Forecast error (MW)")
    ax.set_title("Baseline accuracy on the 2024 hold-out period (7,320 hours)")
    ax.grid(axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    return _save(fig, output, dpi)


def plot_holiday_improvement(metrics_path: Path, output: Path, dpi: int) -> Path:
    """Plot the main model-agnostic HQT result on official major holidays."""

    required = {
        "model",
        "baseline_mae_mw",
        "tilted_mae_mw",
        "mae_improvement_pct",
        "rmse_improvement_pct",
    }
    frame = _read_csv(metrics_path, required).sort(
        "mae_improvement_pct", descending=True
    )
    models = frame["model"].to_list()
    if set(models) != set(MODEL_ORDER):
        raise ValueError("HQT metric file must contain exactly the six ver2 models")

    labels = [MODEL_LABELS[model] for model in models]
    baseline_mae = frame["baseline_mae_mw"].to_numpy()
    tilted_mae = frame["tilted_mae_mw"].to_numpy()
    mae_gain = frame["mae_improvement_pct"].to_numpy()
    rmse_gain = frame["rmse_improvement_pct"].to_numpy()
    y_pos = np.arange(len(models))
    height = 0.36

    fig, (ax_mae, ax_gain) = plt.subplots(
        1,
        2,
        figsize=(14, 6.2),
        gridspec_kw={"width_ratios": (1.3, 1.0)},
        layout="constrained",
    )
    base_bars = ax_mae.barh(
        y_pos - height / 2,
        baseline_mae,
        height,
        color=BASELINE_COLOR,
        label="Baseline",
    )
    tilted_bars = ax_mae.barh(
        y_pos + height / 2,
        tilted_mae,
        height,
        color=TILTED_COLOR,
        label="HQT-adjusted",
    )
    _annotate_horizontal_bars(ax_mae, base_bars, baseline_mae)
    _annotate_horizontal_bars(ax_mae, tilted_bars, tilted_mae)
    ax_mae.set_yticks(y_pos, labels)
    ax_mae.invert_yaxis()
    ax_mae.set_xlim(0, float(baseline_mae.max()) * 1.2)
    ax_mae.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    ax_mae.set_xlabel("MAE (MW)")
    ax_mae.set_title("Forecast error before and after HQT")
    ax_mae.grid(axis="x")
    ax_mae.spines[["top", "right"]].set_visible(False)
    ax_mae.legend(frameon=False, loc="lower right")

    mae_bars = ax_gain.barh(
        y_pos - height / 2,
        mae_gain,
        height,
        color=TILTED_COLOR,
        label="MAE",
    )
    rmse_bars = ax_gain.barh(
        y_pos + height / 2,
        rmse_gain,
        height,
        color=RMSE_COLOR,
        label="RMSE",
    )
    _annotate_horizontal_bars(
        ax_gain,
        mae_bars,
        mae_gain,
        suffix="%",
        decimals=1,
    )
    _annotate_horizontal_bars(
        ax_gain,
        rmse_bars,
        rmse_gain,
        suffix="%",
        decimals=1,
    )
    ax_gain.set_yticks(y_pos, labels)
    ax_gain.invert_yaxis()
    ax_gain.set_xlim(0, max(float(mae_gain.max()), float(rmse_gain.max())) * 1.27)
    ax_gain.set_xlabel("Error reduction (%)")
    ax_gain.set_title("Relative HQT improvement")
    ax_gain.grid(axis="x")
    ax_gain.spines[["top", "right"]].set_visible(False)
    ax_gain.legend(frameon=False, loc="lower right")

    fig.suptitle(
        "HQT correction on the 2024 Seollal and Chuseok holidays (168 hours)",
        fontsize=15,
        fontweight="bold",
    )
    return _save(fig, output, dpi)


def _official_holiday_spans(
    dates: list[object],
    holiday_names: list[str],
    official_label: str,
) -> list[tuple[object, object]]:
    mask = np.array([name == official_label for name in holiday_names], dtype=bool)
    indices = np.flatnonzero(mask)
    if not indices.size:
        return []

    breaks = np.flatnonzero(np.diff(indices) > 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    stops = np.r_[indices[breaks], indices[-1]]
    return [
        (dates[start], dates[stop] + timedelta(hours=1))
        for start, stop in zip(starts, stops, strict=True)
    ]


def _load_event_predictions(
    predictions_dir: Path,
    event_id: str,
) -> dict[str, pl.DataFrame]:
    required = {
        "datetime",
        "power demand(MW)",
        "holiday_name",
        "baseline_pred",
        "tilted_pred",
        "lower_t",
        "upper_t",
        "event_id",
    }
    frames: dict[str, pl.DataFrame] = {}
    reference_dates: np.ndarray | None = None
    reference_actual: np.ndarray | None = None
    for model in MODEL_ORDER:
        path = predictions_dir / model / "test_predictions.csv"
        frame = _read_csv(path, required).filter(pl.col("event_id") == event_id)
        if frame.is_empty():
            raise ValueError(f"{path} has no rows for event {event_id}")
        frame = frame.sort("datetime")
        dates = frame["datetime"].to_numpy()
        actual = frame["power demand(MW)"].to_numpy()
        if reference_dates is None:
            reference_dates = dates
            reference_actual = actual
        elif not np.array_equal(dates, reference_dates) or not np.array_equal(
            actual, reference_actual
        ):
            raise ValueError(
                "model prediction files are not aligned on event timestamps"
            )
        frames[model] = frame
    return frames


def plot_event_forecasts(
    predictions_dir: Path,
    output: Path,
    *,
    event_id: str,
    event_title: str,
    official_label: str,
    dpi: int,
) -> Path:
    """Plot actual, baseline, and HQT forecasts for every baseline model."""

    frames = _load_event_predictions(predictions_dir, event_id)
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15, 11),
        sharex=True,
        sharey=True,
    )
    axes_flat = axes.ravel()

    for ax, model in zip(axes_flat, MODEL_ORDER, strict=True):
        frame = frames[model]
        dates = frame["datetime"].to_list()
        actual = frame["power demand(MW)"].to_numpy()
        baseline = frame["baseline_pred"].to_numpy()
        tilted = frame["tilted_pred"].to_numpy()
        lower = frame["lower_t"].to_numpy()
        upper = frame["upper_t"].to_numpy()

        for start, stop in _official_holiday_spans(
            dates,
            frame["holiday_name"].to_list(),
            official_label,
        ):
            ax.axvspan(start, stop, color=HOLIDAY_COLOR, alpha=0.45, zorder=0)

        ax.fill_between(
            dates,
            lower,
            upper,
            color=INTERVAL_COLOR,
            alpha=0.2,
            linewidth=0,
            zorder=1,
        )
        ax.plot(dates, actual, color=ACTUAL_COLOR, linewidth=1.8, zorder=4)
        ax.plot(
            dates,
            baseline,
            color=BASELINE_COLOR,
            linewidth=1.35,
            alpha=0.95,
            zorder=2,
        )
        ax.plot(
            dates,
            tilted,
            color=TILTED_COLOR,
            linewidth=1.55,
            alpha=0.98,
            zorder=3,
        )

        baseline_mae = float(np.mean(np.abs(baseline - actual)))
        tilted_mae = float(np.mean(np.abs(tilted - actual)))
        improvement = 100 * (baseline_mae - tilted_mae) / baseline_mae
        ax.set_title(
            f"{MODEL_LABELS[model]}  |  MAE {baseline_mae:,.0f} → "
            f"{tilted_mae:,.0f} MW ({improvement:+.1f}%)"
        )
        ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.grid(axis="both")
        ax.spines[["top", "right"]].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("Power demand (MW)")
    for ax in axes[-1, :]:
        ax.set_xlabel("2024 event-window date")

    legend_handles = [
        Line2D([0], [0], color=ACTUAL_COLOR, linewidth=1.8, label="Actual"),
        Line2D([0], [0], color=BASELINE_COLOR, linewidth=1.5, label="Baseline"),
        Line2D([0], [0], color=TILTED_COLOR, linewidth=1.6, label="HQT-adjusted"),
        Patch(facecolor=INTERVAL_COLOR, alpha=0.2, label="HQT 95% interval"),
        Patch(facecolor=HOLIDAY_COLOR, alpha=0.45, label="Official holiday"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        f"{event_title}: baseline and HQT-adjusted forecasts",
        fontsize=15,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return _save(fig, output, dpi)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("ver2/results/baseline_test.csv"),
    )
    parser.add_argument(
        "--hqt-metrics",
        type=Path,
        default=Path("ver2/results/hqt_official_major_holiday.csv"),
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        default=Path("artifacts/hqt_baselines"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ver2/figures"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.dpi <= 0:
        raise ValueError("dpi must be positive")
    _configure_style()
    outputs = [
        plot_baseline_performance(
            args.baseline_metrics,
            args.output_dir / "baseline_test_performance.png",
            args.dpi,
        ),
        plot_holiday_improvement(
            args.hqt_metrics,
            args.output_dir / "hqt_holiday_improvement.png",
            args.dpi,
        ),
        plot_event_forecasts(
            args.predictions_dir,
            args.output_dir / "seollal_2024_forecasts.png",
            event_id="2024_Seollal",
            event_title="2024 Seollal",
            official_label="Korean New Year",
            dpi=args.dpi,
        ),
        plot_event_forecasts(
            args.predictions_dir,
            args.output_dir / "chuseok_2024_forecasts.png",
            event_id="2024_Chuseok",
            event_title="2024 Chuseok",
            official_label="Chuseok",
            dpi=args.dpi,
        ),
    ]
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
