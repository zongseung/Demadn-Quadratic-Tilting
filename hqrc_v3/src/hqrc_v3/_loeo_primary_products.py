"""Deterministic aggregation for the primary H3 LOEO matrix."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from hqrc_v3._loeo_contract import canonical_json, sha_json
from hqrc_v3._loeo_primary_types import LOEOPrimaryError, LOEOPrimaryProducts
from hqrc_v3._loeo_types import LOEOFoldMaterial
from hqrc_v3.corrections import psis_loo_summary
from hqrc_v3.evaluation.metrics import point_metric_frame

_PROBABILISTIC_METRICS = (
    "crps",
    "coverage_50",
    "coverage_90",
    *(f"pinball_{level:02d}" for level in range(5, 100, 5)),
    "pinball_mean",
)


def _manifest_sha256(manifest: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(dict(manifest)) + b"\n").hexdigest()


def _fold_reference(material: LOEOFoldMaterial) -> dict[str, object]:
    held = material.inputs.held_out_occurrence_id
    return {
        "held_out_occurrence_id": held,
        "fold_identity_sha256": sha_json(material.identity),
        "fold_manifest_sha256": _manifest_sha256(material.manifest),
        "fold_output_dir": material.result.output_dir.resolve().as_posix(),
        "fold_state": "COMPLETE_REVALIDATED",
    }


def _weighted_metric(rows: pl.DataFrame, name: str) -> float:
    values = rows[name].to_numpy().astype(float)
    weights = rows["held_out_row_count"].to_numpy().astype(float)
    if not np.isfinite(values).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise LOEOPrimaryError("LOEO aggregate probabilistic metrics must be finite")
    return float(np.average(values, weights=weights))


def _aggregate_metrics(hourly: pl.DataFrame, per_event: pl.DataFrame) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    groups: list[tuple[str, pl.DataFrame, pl.DataFrame]] = [("pooled", hourly, per_event)]
    for holiday in ("seollal", "chuseok"):
        selected_hourly = hourly.filter(pl.col("holiday_type") == holiday)
        if selected_hourly.height:
            groups.append(
                (
                    holiday,
                    selected_hourly,
                    per_event.filter(pl.col("holiday_type") == holiday),
                )
            )
    for name, hourly_group, event_group in groups:
        point = point_metric_frame(
            name,
            hourly_group["observed_mw"].to_numpy(),
            hourly_group["corrected_point_mw"].to_numpy(),
        ).to_dicts()[0]
        row: dict[str, object] = {
            "group": name,
            "event_count": event_group.height,
            "hour_count": hourly_group.height,
            **{
                key: value
                for key, value in point.items()
                if key not in {"event_id", "n_timestamps"}
            },
        }
        row.update(
            {metric: _weighted_metric(event_group, metric) for metric in _PROBABILISTIC_METRICS}
        )
        row["causal"] = False
        rows.append(row)
    return pl.DataFrame(rows)


def _psis_row(material: LOEOFoldMaterial, reference: Mapping[str, object]) -> dict[str, object]:
    try:
        summary = psis_loo_summary(material.inference_data, var_name="event")
    except (TypeError, ValueError) as error:
        raise LOEOPrimaryError("LOEO training PSIS-LOO computation failed") from error
    pareto = np.asarray(summary.pareto_k, dtype=float).reshape(-1)
    if (
        pareto.size != 9
        or not np.isfinite(pareto).all()
        or not math.isfinite(summary.elpd_loo)
        or not math.isfinite(summary.standard_error)
    ):
        raise LOEOPrimaryError("LOEO training PSIS-LOO must contain nine finite event values")
    return {
        **reference,
        "diagnostic_label": "training_nine_event_psis_loo",
        "training_event_count": 9,
        "elpd_loo": summary.elpd_loo,
        "se": summary.standard_error,
        "pareto_k": pareto.tolist(),
        "pareto_k_max": float(pareto.max()),
        "pareto_k_above_0_5": int((pareto > 0.5).sum()),
        "pareto_k_above_0_7": int((pareto > 0.7).sum()),
        "pareto_k_above_1_0": int((pareto > 1.0).sum()),
        "causal": False,
    }


def generate_loeo_primary_products(
    materials: Sequence[LOEOFoldMaterial],
    *,
    selected_occurrence_ids: tuple[str, ...],
) -> LOEOPrimaryProducts:
    """Aggregate only fully revalidated fold material in canonical selected order."""

    if not materials or len(materials) != len(selected_occurrence_ids):
        raise LOEOPrimaryError("LOEO primary matrix requires one material per selected fold")
    if tuple(item.inputs.held_out_occurrence_id for item in materials) != selected_occurrence_ids:
        raise LOEOPrimaryError("LOEO primary fold material order differs")
    hourly_frames: list[pl.DataFrame] = []
    metric_frames: list[pl.DataFrame] = []
    posterior_rows: list[dict[str, object]] = []
    psis_rows: list[dict[str, object]] = []
    for material in materials:
        reference = _fold_reference(material)
        held = str(reference["held_out_occurrence_id"])
        hourly = material.products.hourly_predictions
        metrics = material.products.metrics
        if (
            hourly.is_empty()
            or metrics.height != 1
            or hourly["occurrence_id"].unique().to_list() != [held]
            or metrics["occurrence_id"].to_list() != [held]
            or hourly["causal"].unique().to_list() != [False]
            or metrics["causal"].to_list() != [False]
        ):
            raise LOEOPrimaryError("LOEO primary fold products differ from the selected event")
        reference_columns = [pl.lit(value).alias(key) for key, value in reference.items()]
        hourly_frames.append(hourly.with_columns(*reference_columns))
        holiday = str(hourly["holiday_type"].item(0))
        year_text = held.rsplit("-", 1)[-1]
        if holiday not in {"seollal", "chuseok"} or not year_text.isdigit():
            raise LOEOPrimaryError("LOEO primary held-out metadata differs")
        metric_frames.append(
            metrics.with_columns(
                pl.lit(holiday).alias("holiday_type"),
                pl.lit(int(year_text)).alias("year"),
                pl.lit(hourly.height).alias("held_out_row_count"),
                *reference_columns,
            )
        )
        summary = dict(material.products.posterior_summary)
        phi = summary.get("phi")
        if (
            summary.get("held_out_occurrence_id") != held
            or summary.get("causal") is not False
            or not isinstance(phi, dict)
        ):
            raise LOEOPrimaryError("LOEO primary posterior summary differs")
        posterior_rows.append(
            {
                **reference,
                "parameterization": phi.get("parameterization"),
                "sample_count": phi.get("sample_count"),
                "phi_mean": phi.get("mean"),
                "phi_sd": phi.get("sd"),
                "phi_interval_2_5": phi.get("interval_2_5"),
                "phi_interval_97_5": phi.get("interval_97_5"),
                "prior_a": phi.get("prior_a"),
                "prior_b": phi.get("prior_b"),
                "causal": False,
            }
        )
        psis_rows.append(_psis_row(material, reference))
    hourly_all = pl.concat(hourly_frames, how="vertical")
    per_event = pl.concat(metric_frames, how="vertical")
    if (
        tuple(hourly_all["occurrence_id"].unique(maintain_order=True).to_list())
        != selected_occurrence_ids
    ):
        raise LOEOPrimaryError("LOEO primary hourly occurrence order differs")
    return LOEOPrimaryProducts(
        hourly_predictions=hourly_all,
        per_event_metrics=per_event,
        aggregate_metrics=_aggregate_metrics(hourly_all, per_event),
        posterior_summaries=pl.DataFrame(posterior_rows),
        training_psis_loo=pl.DataFrame(psis_rows),
    )


__all__ = ["generate_loeo_primary_products"]
