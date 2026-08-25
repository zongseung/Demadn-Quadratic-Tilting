"""Normalized point-metric tables and figures for the HQT reviewer response."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import polars as pl

from hqrc_v3._loeo_contract import derive_loeo_seed
from hqrc_v3.baselines.config import MODEL_NAMES
from hqrc_v3.evaluation.inference import bootstrap_event_median, wilcoxon_event_test
from hqrc_v3.evaluation.metrics import point_metric_frame
from hqrc_v3.provenance import file_sha256

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

_METRICS = ("rmse", "mae", "mape", "smape", "r2")
_MAIN_CORRECTIONS = ("H0", "H1", "H2")
_EXPECTED_EVENTS = frozenset(
    f"{holiday}-{year}" for year in range(2020, 2025) for holiday in ("seollal", "chuseok")
)
_EXPECTED_CAUSAL_EVENTS = frozenset(("seollal-2024", "chuseok-2024"))
_TABLE_NAMES = (
    "overall_2024",
    "loeo_event",
    "loeo_pooled",
    "loeo_inference",
    "causal_2024",
    "scale_stability",
)
_FIGURE_NAMES = ("fig_per_event.png", "fig_correction_2024.png")


class ReviewerReportError(ValueError):
    """Raised when reviewer inputs cannot form the frozen point-metric report."""


@dataclass(frozen=True, slots=True)
class ReviewerReportInputs:
    """Explicit completed artifacts consumed by the reviewer report stage."""

    final_predictions_path: Path
    loeo_context_dirs: tuple[Path, ...]
    causal_context_dirs: tuple[Path, ...]
    root_seed: int


@dataclass(frozen=True, slots=True)
class ReviewerReportResult:
    """Published reviewer report directories and manifest."""

    output_root: Path
    tables_dir: Path
    figures_dir: Path
    manifest_path: Path


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise ReviewerReportError("reviewer manifest is not canonical JSON") from error


def _write_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(payload)
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_table(frame: pl.DataFrame, tables_dir: Path, name: str) -> None:
    parquet = tables_dir / f"{name}.parquet"
    csv = tables_dir / f"{name}.csv"
    with tempfile.NamedTemporaryFile(
        prefix=f".{name}.", suffix=".parquet", dir=tables_dir, delete=False
    ) as temporary:
        parquet_temporary = Path(temporary.name)
    with tempfile.NamedTemporaryFile(
        prefix=f".{name}.", suffix=".csv", dir=tables_dir, delete=False
    ) as temporary:
        csv_temporary = Path(temporary.name)
    try:
        frame.write_parquet(parquet_temporary)
        frame.write_csv(csv_temporary)
        os.replace(parquet_temporary, parquet)
        os.replace(csv_temporary, csv)
    finally:
        parquet_temporary.unlink(missing_ok=True)
        csv_temporary.unlink(missing_ok=True)


def _read_parquet(path: Path, description: str) -> pl.DataFrame:
    if not path.is_file():
        raise ReviewerReportError(f"{description} is missing")
    try:
        frame = pl.read_parquet(path)
    except (OSError, pl.exceptions.PolarsError) as error:
        raise ReviewerReportError(f"{description} is unreadable") from error
    if frame.is_empty():
        raise ReviewerReportError(f"{description} is empty")
    return frame


def _require_columns(frame: pl.DataFrame, columns: Iterable[str], description: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ReviewerReportError(f"{description} is missing columns {sorted(missing)}")


def _require_finite_metrics(frame: pl.DataFrame, description: str) -> None:
    _require_columns(frame, ("n_timestamps", *_METRICS), description)
    if frame["n_timestamps"].null_count() or not frame["n_timestamps"].gt(0).all():
        raise ReviewerReportError(f"{description} has invalid timestamp counts")
    for metric in _METRICS:
        column = frame[metric]
        if column.null_count() or not column.is_finite().all():
            raise ReviewerReportError(f"{description} has non-finite {metric}")


def _context_identity(context_dir: Path) -> tuple[str, str, int, pl.DataFrame]:
    scale = _read_parquet(context_dir / "scale_stability.parquet", "LOEO scale stability")
    _require_columns(
        scale,
        (
            "model",
            "feature_set",
            "seed",
            "sigma_fit_oof_2023_mw",
            "sigma_2024_non_event_mw",
            "scale_ratio",
            "n_2024_non_event_hours",
        ),
        "LOEO scale stability",
    )
    if scale.height != 1:
        raise ReviewerReportError("LOEO scale stability must identify exactly one context")
    model = scale["model"].item()
    feature_set = scale["feature_set"].item()
    seed = scale["seed"].item()
    if (
        not isinstance(model, str)
        or model not in MODEL_NAMES
        or feature_set != "B1W"
        or isinstance(seed, bool)
        or not isinstance(seed, int)
    ):
        raise ReviewerReportError("LOEO context must identify a valid B1W model and seed")
    for column in ("sigma_fit_oof_2023_mw", "sigma_2024_non_event_mw", "scale_ratio"):
        if (
            scale[column].null_count()
            or not scale[column].is_finite().all()
            or not scale[column].gt(0).all()
        ):
            raise ReviewerReportError(f"LOEO scale stability has invalid {column}")
    if (
        scale["n_2024_non_event_hours"].null_count()
        or not scale["n_2024_non_event_hours"].gt(0).all()
    ):
        raise ReviewerReportError("LOEO scale stability has invalid non-event hour count")
    return model, feature_set, seed, scale


def _with_context(
    frame: pl.DataFrame, *, model: str, feature_set: str, seed: int, description: str
) -> pl.DataFrame:
    expected: Mapping[str, object] = {
        "model": model,
        "feature_set": feature_set,
        "seed": seed,
    }
    for column, value in expected.items():
        if column in frame.columns and frame[column].unique().to_list() != [value]:
            raise ReviewerReportError(f"{description} context differs from its directory")
    return frame.with_columns(
        pl.lit(model).alias("model"),
        pl.lit(feature_set).alias("feature_set"),
        pl.lit(seed).alias("seed"),
    )


def _improvements(frame: pl.DataFrame, *, group_column: str, description: str) -> pl.DataFrame:
    keys = ["model", "feature_set", "seed", group_column]
    if frame.select(keys + ["correction"]).is_duplicated().any():
        raise ReviewerReportError(f"{description} has duplicate correction rows")
    baseline = frame.filter(pl.col("correction") == "H0").select(
        *keys, pl.col("rmse").alias("baseline_rmse")
    )
    if baseline.height != frame.select(keys).unique().height:
        raise ReviewerReportError(f"{description} does not have one H0 row per scope")
    result = frame.join(baseline, on=keys, how="left", validate="m:1")
    if result["baseline_rmse"].null_count() or not result["baseline_rmse"].gt(0).all():
        raise ReviewerReportError(f"{description} has invalid H0 RMSE")
    return result.with_columns(
        pl.col("rmse").alias("corrected_rmse"),
        (100.0 * (1.0 - pl.col("rmse") / pl.col("baseline_rmse"))).alias("rmse_improvement_pct"),
    )


def _overall_table(final: pl.DataFrame) -> pl.DataFrame:
    required = (
        "origin",
        "target_timestamp",
        "horizon",
        "observed_mw",
        "predicted_mw",
        "model",
        "feature_set",
        "seed",
        "split_id",
    )
    _require_columns(final, required, "final-2024 predictions")
    if set(final["feature_set"]) != {"B0", "B1W"}:
        raise ReviewerReportError("final-2024 predictions must contain exactly B0 and B1W")
    if set(final["split_id"]) != {"final-2024"}:
        raise ReviewerReportError("final predictions must contain only final-2024 rows")
    if not final["observed_mw"].is_finite().all() or not final["predicted_mw"].is_finite().all():
        raise ReviewerReportError("final-2024 predictions must be finite")
    rows: list[dict[str, Any]] = []
    keys = ["origin", "target_timestamp", "horizon", "split_id"]
    for model in sorted(final["model"].unique().to_list()):
        selected = final.filter(pl.col("model") == model)
        contexts = selected.select("feature_set", "seed").unique().sort("feature_set")
        if contexts.height != 2 or contexts["seed"].n_unique() != 1:
            raise ReviewerReportError(f"{model} final predictions must have one aligned seed")
        reference = selected.filter(pl.col("feature_set") == "B0").sort(keys)
        candidate = selected.filter(pl.col("feature_set") == "B1W").sort(keys)
        if (
            reference.height == 0
            or reference.height != candidate.height
            or reference.select(keys).is_duplicated().any()
            or candidate.select(keys).is_duplicated().any()
            or not reference.select(keys).equals(candidate.select(keys))
            or not np.array_equal(
                reference["observed_mw"].to_numpy(), candidate["observed_mw"].to_numpy()
            )
        ):
            raise ReviewerReportError(f"{model} B0/B1W final prediction coverage differs")
        reference_metrics = point_metric_frame(
            "all", reference["observed_mw"].to_numpy(), reference["predicted_mw"].to_numpy()
        ).row(0, named=True)
        candidate_metrics = point_metric_frame(
            "all", candidate["observed_mw"].to_numpy(), candidate["predicted_mw"].to_numpy()
        ).row(0, named=True)
        if reference_metrics["rmse"] <= 0:
            raise ReviewerReportError(f"{model} B0 final RMSE must be positive")
        rows.append(
            {
                "model": model,
                "seed": contexts["seed"].item(0),
                "scope": "all-2024",
                "comparison": "B0-H0_to_B1W-H0",
                "reference_feature_set": "B0",
                "reference_correction": "H0",
                "candidate_feature_set": "B1W",
                "candidate_correction": "H0",
                "n_timestamps": reference.height,
                **{f"reference_{metric}": reference_metrics[metric] for metric in _METRICS},
                **{f"candidate_{metric}": candidate_metrics[metric] for metric in _METRICS},
                "rmse_improvement_pct": 100.0
                * (1.0 - candidate_metrics["rmse"] / reference_metrics["rmse"]),
            }
        )
    if not rows:
        raise ReviewerReportError("final-2024 predictions contain no model contexts")
    return pl.DataFrame(rows).sort("model")


def _loeo_tables(
    context_dirs: tuple[Path, ...], *, root_seed: int
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, dict[str, int]]:
    if not context_dirs:
        raise ReviewerReportError("at least one LOEO context is required")
    event_frames: list[pl.DataFrame] = []
    pooled_frames: list[pl.DataFrame] = []
    scale_frames: list[pl.DataFrame] = []
    for context_dir in context_dirs:
        model, feature_set, seed, scale = _context_identity(context_dir)
        event = _read_parquet(context_dir / "event_metrics.parquet", "LOEO event metrics")
        pooled = _read_parquet(context_dir / "pooled_metrics.parquet", "LOEO pooled metrics")
        _require_columns(event, ("event_id", "holiday_type", "correction"), "LOEO event metrics")
        _require_columns(pooled, ("scope", "correction"), "LOEO pooled metrics")
        _require_finite_metrics(event, "LOEO event metrics")
        _require_finite_metrics(pooled, "LOEO pooled metrics")
        event = _with_context(
            event, model=model, feature_set=feature_set, seed=seed, description="LOEO event metrics"
        ).filter(pl.col("correction").is_in(_MAIN_CORRECTIONS))
        pooled = _with_context(
            pooled,
            model=model,
            feature_set=feature_set,
            seed=seed,
            description="LOEO pooled metrics",
        ).filter(pl.col("correction").is_in(_MAIN_CORRECTIONS))
        if set(event["event_id"]) != _EXPECTED_EVENTS:
            raise ReviewerReportError("LOEO inference requires exactly 10 physical events")
        if not event.select(
            (
                pl.col("holiday_type")
                == pl.col("event_id").str.split_exact("-", 1).struct.field("field_0")
            ).all()
        ).item():
            raise ReviewerReportError("LOEO event holiday labels differ from occurrence ids")
        if event.height != 10 * len(_MAIN_CORRECTIONS):
            raise ReviewerReportError("LOEO inference requires exactly 10 physical events")
        if set(pooled["scope"]) != {"all", "seollal", "chuseok"} or pooled.height != 9:
            raise ReviewerReportError("LOEO pooled metrics require three scopes and H0/H1/H2")
        event_frames.append(event)
        pooled_frames.append(pooled)
        scale_frames.append(scale)
    event_all = pl.concat(event_frames, how="diagonal_relaxed")
    contexts = event_all.select("model", "feature_set", "seed").unique()
    if contexts.height != len(context_dirs):
        raise ReviewerReportError("LOEO context directories must be unique")
    event_output = (
        _improvements(event_all, group_column="event_id", description="LOEO event metrics")
        .select(
            "model",
            "feature_set",
            "seed",
            "event_id",
            "holiday_type",
            "correction",
            "n_timestamps",
            *_METRICS,
            "baseline_rmse",
            "corrected_rmse",
            "rmse_improvement_pct",
        )
        .sort("model", "event_id", "correction")
    )
    pooled_output = (
        _improvements(
            pl.concat(pooled_frames, how="diagonal_relaxed"),
            group_column="scope",
            description="LOEO pooled metrics",
        )
        .select(
            "model",
            "feature_set",
            "seed",
            "scope",
            "correction",
            "n_timestamps",
            *_METRICS,
            "baseline_rmse",
            "corrected_rmse",
            "rmse_improvement_pct",
        )
        .sort("model", "scope", "correction")
    )
    inference_rows = []
    bootstrap_seeds: dict[str, int] = {}
    for context in contexts.sort("model").iter_rows(named=True):
        selected = event_output.filter(
            (pl.col("model") == context["model"])
            & (pl.col("feature_set") == context["feature_set"])
            & (pl.col("seed") == context["seed"])
        )
        h0 = selected.filter(pl.col("correction") == "H0").sort("event_id")
        h2 = selected.filter(pl.col("correction") == "H2").sort("event_id")
        if h0.height != 10 or h2.height != 10:
            raise ReviewerReportError("LOEO inference requires exactly 10 physical events")
        event_ids = h0["event_id"].to_list()
        if event_ids != h2["event_id"].to_list():
            raise ReviewerReportError("LOEO H0/H2 event pairs are not aligned")
        reference = h0["rmse"].to_numpy()
        candidate = h2["rmse"].to_numpy()
        values = 100.0 * (1.0 - candidate / reference)
        label = f"reviewer:B1W-H0_to_B1W-H2:{context['model']}:{context['seed']}"
        bootstrap_seed = derive_loeo_seed(root_seed, label)
        bootstrap = bootstrap_event_median(
            {event_id: np.asarray([value]) for event_id, value in zip(event_ids, values)},
            draws=10_000,
            seed=bootstrap_seed,
        )
        wilcoxon = wilcoxon_event_test(reference, candidate, alternative="greater")
        bootstrap_seeds[str(context["model"])] = bootstrap_seed
        inference_rows.append(
            {
                **context,
                "comparison": "B1W-H0_to_B1W-H2",
                "n_events": wilcoxon.n_events,
                "improved_events": int(np.count_nonzero(values > 0)),
                "degraded_events": int(np.count_nonzero(values < 0)),
                "median_rmse_improvement_pct": bootstrap.median,
                "bootstrap_lower_95_pct": bootstrap.lower,
                "bootstrap_upper_95_pct": bootstrap.upper,
                "bootstrap_draws": 10_000,
                "bootstrap_seed": bootstrap_seed,
                "resampled_units": bootstrap.resampled_units,
                "wilcoxon_statistic": wilcoxon.statistic,
                "wilcoxon_p_value": wilcoxon.p_value,
                "alternative": "greater",
                "worst_rmse_improvement_pct": float(min(0.0, values.min())),
                "max_degradation_pct": float(max(0.0, -values.min())),
            }
        )
    scale_output = pl.concat(scale_frames, how="diagonal_relaxed").sort(
        "model", "feature_set", "seed"
    )
    return (
        event_output,
        pooled_output,
        pl.DataFrame(inference_rows).sort("model"),
        scale_output,
        bootstrap_seeds,
    )


def _causal_tables(
    context_dirs: tuple[Path, ...],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not context_dirs:
        raise ReviewerReportError("at least one causal-2024 context is required")
    table_frames: list[pl.DataFrame] = []
    hourly_frames: list[pl.DataFrame] = []
    identities: set[tuple[str, str, int]] = set()
    for context_dir in context_dirs:
        event = _read_parquet(context_dir / "event_metrics.parquet", "causal event metrics")
        pooled = _read_parquet(context_dir / "pooled_metrics.parquet", "causal pooled metrics")
        hourly = _read_parquet(
            context_dir / "hourly_predictions.parquet", "causal hourly predictions"
        )
        _require_columns(
            event,
            ("model", "feature_set", "seed", "event_id", "holiday_type", "correction"),
            "causal event metrics",
        )
        _require_columns(
            pooled,
            ("model", "feature_set", "seed", "scope", "correction"),
            "causal pooled metrics",
        )
        _require_columns(
            hourly,
            (
                "model",
                "feature_set",
                "seed",
                "occurrence_id",
                "holiday_type",
                "target_timestamp",
                "observed_mw",
                "H0_mw",
                "H2_mw",
            ),
            "causal hourly predictions",
        )
        _require_finite_metrics(event, "causal event metrics")
        _require_finite_metrics(pooled, "causal pooled metrics")
        identity_frame = event.select("model", "feature_set", "seed").unique()
        if identity_frame.height != 1:
            raise ReviewerReportError("causal artifacts must identify exactly one context")
        identity = identity_frame.row(0)
        if identity[0] not in MODEL_NAMES or identity[1] != "B1W" or identity in identities:
            raise ReviewerReportError("causal contexts must be unique valid B1W model contexts")
        identities.add(identity)
        for frame, description in ((pooled, "causal pooled"), (hourly, "causal hourly")):
            frame_identity = frame.select("model", "feature_set", "seed").unique()
            if frame_identity.height != 1 or frame_identity.row(0) != identity:
                raise ReviewerReportError(f"{description} context differs from event metrics")
        if set(event["event_id"]) != _EXPECTED_CAUSAL_EVENTS:
            raise ReviewerReportError("causal event metrics require both 2024 physical events")
        event_main = event.filter(pl.col("correction").is_in(_MAIN_CORRECTIONS))
        pooled_main = pooled.filter(pl.col("correction").is_in(_MAIN_CORRECTIONS))
        if event_main.height != 2 * len(_MAIN_CORRECTIONS):
            raise ReviewerReportError("causal event metrics have incomplete H0/H1/H2 rows")
        if set(hourly["occurrence_id"]) != _EXPECTED_CAUSAL_EVENTS:
            raise ReviewerReportError("causal hourly predictions require both 2024 events")
        if hourly.select("model", "occurrence_id", "target_timestamp").is_duplicated().any():
            raise ReviewerReportError("causal hourly predictions contain duplicate timestamps")
        for column in ("observed_mw", "H0_mw", "H2_mw"):
            if hourly[column].null_count() or not hourly[column].is_finite().all():
                raise ReviewerReportError(f"causal hourly predictions have invalid {column}")
        event_output = _improvements(
            event_main, group_column="event_id", description="causal event metrics"
        ).with_columns(
            pl.lit("event").alias("aggregation"),
            pl.col("event_id").alias("scope"),
        )
        pooled_output = _improvements(
            pooled_main, group_column="scope", description="causal pooled metrics"
        ).with_columns(
            pl.lit("pooled").alias("aggregation"),
            pl.lit(None, dtype=pl.String).alias("holiday_type"),
        )
        table_frames.extend((event_output, pooled_output))
        hourly_frames.append(hourly)
    columns = (
        "model",
        "feature_set",
        "seed",
        "aggregation",
        "scope",
        "holiday_type",
        "correction",
        "n_timestamps",
        *_METRICS,
        "baseline_rmse",
        "corrected_rmse",
        "rmse_improvement_pct",
    )
    return (
        pl.concat(table_frames, how="diagonal_relaxed")
        .select(columns)
        .sort("model", "aggregation", "scope", "correction"),
        pl.concat(hourly_frames, how="diagonal_relaxed").sort(
            "model", "occurrence_id", "target_timestamp"
        ),
    )


def _save_figure(figure: plt.Figure, path: Path) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.stem}.", suffix=".png", dir=path.parent
    )
    os.close(descriptor)
    try:
        figure.savefig(
            temporary,
            dpi=180,
            bbox_inches="tight",
            metadata={"Software": "hqrc_v3"},
        )
        os.replace(temporary, path)
    finally:
        plt.close(figure)
        if os.path.exists(temporary):
            os.unlink(temporary)


def _plot_per_event(event: pl.DataFrame, path: Path) -> None:
    selected = event.filter(pl.col("correction") == "H2").sort("event_id", "model")
    models = sorted(selected["model"].unique().to_list())
    event_ids = sorted(selected["event_id"].unique().to_list())
    figure, axis = plt.subplots(figsize=(13, 5.5))
    width = 0.8 / len(models)
    positions = np.arange(len(event_ids), dtype=float)
    for index, model in enumerate(models):
        values = (
            selected.filter(pl.col("model") == model)
            .sort("event_id")["rmse_improvement_pct"]
            .to_numpy()
        )
        if values.size != len(event_ids):
            raise ReviewerReportError("per-event figure has incomplete model/event coverage")
        axis.bar(
            positions - 0.4 + width / 2 + index * width,
            values,
            width=width,
            label=model,
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(positions, event_ids, rotation=45, ha="right")
    axis.set_ylabel("H2 RMSE improvement over H0 (%)")
    axis.set_title("Retrospective LOEO improvement by physical event")
    axis.legend(ncol=min(3, len(models)), fontsize="small")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    _save_figure(figure, path)


def _plot_causal_corrections(hourly: pl.DataFrame, path: Path) -> None:
    models = sorted(hourly["model"].unique().to_list())
    holidays = ("seollal", "chuseok")
    figure, axes = plt.subplots(
        len(models), len(holidays), figsize=(13, max(4.0, 3.2 * len(models))), squeeze=False
    )
    for row, model in enumerate(models):
        for column, holiday in enumerate(holidays):
            axis = axes[row][column]
            selected = hourly.filter(
                (pl.col("model") == model) & (pl.col("holiday_type") == holiday)
            ).sort("target_timestamp")
            if selected.is_empty():
                raise ReviewerReportError("causal correction figure lacks a holiday panel")
            timestamps = selected["target_timestamp"].to_list()
            axis.plot(timestamps, selected["observed_mw"], label="Observed", linewidth=1.5)
            axis.plot(timestamps, selected["H0_mw"], label="H0", linewidth=1.1)
            axis.plot(timestamps, selected["H2_mw"], label="H2", linewidth=1.1)
            axis.set_title(f"{model} — {holiday.title()} 2024")
            axis.tick_params(axis="x", labelrotation=25)
            axis.grid(alpha=0.25)
            if column == 0:
                axis.set_ylabel("Demand (MW)")
            if row == 0 and column == 0:
                axis.legend(fontsize="small")
    figure.suptitle("Causal-2024 observed demand and HQT correction")
    figure.tight_layout()
    _save_figure(figure, path)


def _input_records(inputs: ReviewerReportInputs) -> list[dict[str, str]]:
    paths = [inputs.final_predictions_path]
    for directory in (*inputs.loeo_context_dirs, *inputs.causal_context_dirs):
        paths.extend(sorted(path for path in directory.iterdir() if path.is_file()))
    records = []
    for path in sorted(set(paths), key=lambda item: str(item.resolve())):
        records.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return records


def build_reviewer_report(
    inputs: ReviewerReportInputs, *, output_root: Path
) -> ReviewerReportResult:
    """Build reviewer-only point tables and figures from completed baseline/HQT artifacts."""

    if not isinstance(inputs, ReviewerReportInputs):
        raise TypeError("inputs must be ReviewerReportInputs")
    if (
        isinstance(inputs.root_seed, bool)
        or not isinstance(inputs.root_seed, int)
        or inputs.root_seed < 0
    ):
        raise ReviewerReportError("reviewer root seed must be a non-negative integer")
    output = Path(output_root).expanduser().resolve()
    tables_dir = output / "tables"
    figures_dir = output / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    final = _read_parquet(inputs.final_predictions_path, "final-2024 predictions")
    overall = _overall_table(final)
    loeo_event, loeo_pooled, inference, scale, bootstrap_seeds = _loeo_tables(
        tuple(Path(path) for path in inputs.loeo_context_dirs), root_seed=inputs.root_seed
    )
    causal, causal_hourly = _causal_tables(tuple(Path(path) for path in inputs.causal_context_dirs))
    overall_contexts = set(overall.select("model", "seed").iter_rows())
    loeo_contexts = set(loeo_event.select("model", "seed").unique().iter_rows())
    causal_contexts = set(causal.select("model", "seed").unique().iter_rows())
    if overall_contexts != loeo_contexts or overall_contexts != causal_contexts:
        raise ReviewerReportError("baseline, LOEO, and causal report contexts must match")
    tables = {
        "overall_2024": overall,
        "loeo_event": loeo_event,
        "loeo_pooled": loeo_pooled,
        "loeo_inference": inference,
        "causal_2024": causal,
        "scale_stability": scale,
    }
    for name in _TABLE_NAMES:
        _write_table(tables[name], tables_dir, name)
    _plot_per_event(loeo_event, figures_dir / _FIGURE_NAMES[0])
    _plot_causal_corrections(causal_hourly, figures_dir / _FIGURE_NAMES[1])

    output_paths = [
        *(
            tables_dir / f"{name}.{suffix}"
            for name in _TABLE_NAMES
            for suffix in ("parquet", "csv")
        ),
        *(figures_dir / name for name in _FIGURE_NAMES),
    ]
    manifest = {
        "schema_version": 1,
        "root_seed": inputs.root_seed,
        "inputs": _input_records(inputs),
        "inference": {
            "comparison": "B1W-H0_to_B1W-H2",
            "event_count_per_context": 10,
            "bootstrap_draws": 10_000,
            "bootstrap_seeds": bootstrap_seeds,
            "resampled_units": "event",
            "wilcoxon_alternative": "greater",
        },
        "outputs": {
            str(path.relative_to(output)): file_sha256(path)
            for path in sorted(output_paths, key=lambda item: str(item.relative_to(output)))
        },
    }
    manifest_path = output / "manifest.json"
    _write_atomic(manifest_path, _canonical_json(manifest))
    return ReviewerReportResult(output, tables_dir, figures_dir, manifest_path)


__all__ = [
    "ReviewerReportError",
    "ReviewerReportInputs",
    "ReviewerReportResult",
    "build_reviewer_report",
]
