"""Resumable H1/H2 LOEO ablations using the reviewed H3 data and AR contracts."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import arviz as az
import numpy as np
import polars as pl
from scipy import stats

import hqrc_v3._loeo_publication as fold_contract
from hqrc_v3._loeo_contract import MODEL_OPTIONS, derive_loeo_seed, sha_json
from hqrc_v3._loeo_products import generate_loeo_fold_products
from hqrc_v3._loeo_types import LOEOFoldError
from hqrc_v3.bayes.samplers import PYMC_INITIALIZATION, SamplingError, sample_hqrc
from hqrc_v3.correction_source import ValidatedCorrectionSource
from hqrc_v3.diagnostics.loeo import LOEOPublication
from hqrc_v3.diagnostics.loeo_ar import ApprovedLOEOARSet
from hqrc_v3.evaluation.inference import hac_dm_test
from hqrc_v3.evaluation.metrics import point_metric_frame
from hqrc_v3.loeo_stage import prepare_loeo_fold_inputs, validate_loeo_fold_sources
from hqrc_v3.provenance import file_sha256

AblationVariant = Literal["H1", "H2"]
_FOLD_FILES = (
    "posterior.nc",
    "hourly_predictions.parquet",
    "metrics.parquet",
    "posterior_summary.json",
)
_PRIMARY_FILES = (
    "hourly_predictions.parquet",
    "per_event_metrics.parquet",
    "aggregate_metrics.parquet",
)


class LOEOAblationError(ValueError):
    """Raised when an H1/H2 ablation cannot be fitted or safely reused."""


@dataclass(frozen=True, slots=True)
class LOEOAblationResult:
    """One complete ten-event H1 or H2 publication."""

    variant: AblationVariant
    output_dir: Path
    hourly_predictions_path: Path
    per_event_metrics_path: Path
    aggregate_metrics_path: Path
    sampler_fit_count: int
    reused: bool


def _variant(value: str) -> AblationVariant:
    if value not in {"H1", "H2"}:
        raise LOEOAblationError("ablation variant must be H1 or H2")
    return value  # type: ignore[return-value]


def _read_json(path: Path, description: str) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise LOEOAblationError(f"{description} is unreadable") from error
    if not isinstance(value, dict):
        raise LOEOAblationError(f"{description} must be a JSON object")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _output_records(directory: Path, files: tuple[str, ...]) -> dict[str, dict[str, str]]:
    return {name: {"path": name, "sha256": file_sha256(directory / name)} for name in files}


def _validate_complete(
    directory: Path,
    *,
    identity: Mapping[str, object],
    files: tuple[str, ...],
) -> None:
    expected = {*files, "manifest.json", "COMPLETE"}
    try:
        actual = {entry.name for entry in directory.iterdir()}
    except OSError as error:
        raise LOEOAblationError("completed ablation publication is unreadable") from error
    if actual != expected:
        raise LOEOAblationError("completed ablation publication is partial or unknown")
    manifest = _read_json(directory / "manifest.json", "ablation manifest")
    if manifest.get("identity") != dict(identity):
        raise LOEOAblationError("ablation publication identity differs")
    retry_failure = manifest.get("retry_failure")
    if files == _FOLD_FILES:
        sampler = identity.get("sampler")
        if not isinstance(sampler, Mapping):
            raise LOEOAblationError("ablation sampler identity differs")
        try:
            if "retry" in sampler:
                fold_contract.validate_retry_failure(sampler, retry_failure)
            elif "retry_failure" in manifest:
                raise LOEOFoldError("base ablation publication cannot record retry failure")
        except LOEOFoldError as error:
            raise LOEOAblationError("ablation retry provenance differs") from error
    elif "retry_failure" in manifest:
        raise LOEOAblationError("aggregate ablation publication cannot record retry failure")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, dict) or set(outputs) != set(files):
        raise LOEOAblationError("ablation publication output registry differs")
    for name in files:
        record = outputs.get(name)
        if (
            not isinstance(record, dict)
            or record.get("path") != name
            or record.get("sha256") != file_sha256(directory / name)
        ):
            raise LOEOAblationError(f"ablation artifact {name} differs")
    unsigned = {key: value for key, value in manifest.items() if key != "manifest_digest"}
    if manifest.get("manifest_digest") != sha_json(unsigned):
        raise LOEOAblationError("ablation manifest digest differs")
    complete = _read_json(directory / "COMPLETE", "ablation completion marker")
    if complete != {
        "manifest_sha256": file_sha256(directory / "manifest.json"),
        "state": "COMPLETE",
    }:
        raise LOEOAblationError("ablation completion marker differs")


def _publish_directory(
    final: Path,
    *,
    identity: Mapping[str, object],
    files: tuple[str, ...],
    writer,
    retry_failure: Mapping[str, object] | None = None,
) -> None:
    if files == _FOLD_FILES:
        sampler = identity.get("sampler")
        if not isinstance(sampler, Mapping):
            raise LOEOAblationError("ablation sampler identity differs")
        try:
            if "retry" in sampler:
                fold_contract.validate_retry_failure(sampler, retry_failure)
            elif retry_failure is not None:
                raise LOEOFoldError("base ablation publication cannot record retry failure")
        except LOEOFoldError as error:
            raise LOEOAblationError("ablation retry provenance differs") from error
    elif retry_failure is not None:
        raise LOEOAblationError("aggregate ablation publication cannot record retry failure")
    final.parent.mkdir(parents=True, exist_ok=True)
    if final.exists():
        _validate_complete(final, identity=identity, files=files)
        return
    with tempfile.TemporaryDirectory(prefix=f".{final.name}.tmp-", dir=final.parent) as raw:
        temporary = Path(raw)
        writer(temporary)
        unsigned: dict[str, object] = {
            "schema_version": 1,
            "state": "COMPLETE",
            "causal": False,
            "identity": dict(identity),
            "outputs": _output_records(temporary, files),
        }
        if retry_failure is not None:
            unsigned["retry_failure"] = dict(retry_failure)
        _write_json(
            temporary / "manifest.json",
            {**unsigned, "manifest_digest": sha_json(unsigned)},
        )
        _write_json(
            temporary / "COMPLETE",
            {
                "manifest_sha256": file_sha256(temporary / "manifest.json"),
                "state": "COMPLETE",
            },
        )
        try:
            os.rename(temporary, final)
        except FileExistsError:
            _validate_complete(final, identity=identity, files=files)
    _validate_complete(final, identity=identity, files=files)


def _fold_identity(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    approved_set: ApprovedLOEOARSet,
    *,
    held_out: str,
    variant: AblationVariant,
    sampler: Mapping[str, object],
) -> dict[str, object]:
    context = publication.context
    approved = approved_set.calibration_for(held_out)
    return {
        "schema_version": 1,
        "evaluation": "retrospective-loeo-ablation-fold",
        "causal": False,
        "variant": variant,
        "context": {
            "model": context.model,
            "feature_set": context.feature_set,
            "seed": context.seed,
            "split_ids": list(context.split_ids),
        },
        "source": {
            "run_dir": source.run_dir.resolve().as_posix(),
            "residual_sha256": source.residual_sha256,
            "source_hashes": dict(sorted(source.source_hashes.items())),
        },
        "loeo": {
            "universe_sha256": publication.universe_sha256,
            "held_out_occurrence_id": held_out,
        },
        "approved_ar": {
            "approved_set_sha256": approved_set.approved_set_sha256,
            "artifact_digest": approved.artifact_digest,
            "a": approved.a,
            "b": approved.b,
        },
        "model": {
            "variant": variant,
            "pooling": "partial",
            "options": asdict(MODEL_OPTIONS),
        },
        "sampler": dict(sampler),
        "predictive_seed": derive_loeo_seed(
            int(sampler["root_seed"]), f"{variant}-fold-predictive:{held_out}"
        ),
    }


def _validate_posterior_metadata(idata, inputs, sampler, *, variant: AblationVariant) -> None:
    try:
        fold_contract.validate_posterior_provenance(
            idata,
            inputs,
            sampler,
            variant=variant,
        )
    except LOEOFoldError as error:
        raise LOEOAblationError("ablation posterior metadata differs") from error


def _fold_directory(
    output_root: Path,
    publication: LOEOPublication,
    *,
    held_out: str,
    variant: AblationVariant,
    profile: str,
    identity: Mapping[str, object],
) -> Path:
    context = publication.context
    return (
        output_root
        / f"loeo-ablation-{variant.lower()}"
        / context.model
        / context.feature_set
        / f"seed-{context.seed}"
        / held_out
        / profile
        / f"identity-{sha_json(identity)}"
    )


def _fit_fold_attempt(
    inputs,
    *,
    variant: AblationVariant,
    sampler: Mapping[str, object],
    identity: Mapping[str, object],
    directory: Path,
    retry_failure: Mapping[str, object] | None = None,
    reuse_only: bool = False,
) -> tuple[Path, int, bool]:
    if directory.exists():
        _validate_complete(directory, identity=identity, files=_FOLD_FILES)
        try:
            idata = az.from_netcdf(directory / "posterior.nc")
        except (OSError, ValueError) as error:
            raise LOEOAblationError("ablation posterior metadata is unreadable") from error
        try:
            _validate_posterior_metadata(idata, inputs, sampler, variant=variant)
        finally:
            idata.close()
        return directory, 0, True

    if reuse_only:
        raise LOEOAblationError(
            "reuse-only retry publication disappeared before it could be loaded"
        )

    idata = sample_hqrc(
        inputs.hqrc_data,
        inputs.approved,
        variant=variant,
        pooling="partial",
        options=MODEL_OPTIONS,
        draws=int(sampler["draws"]),
        tune=int(sampler["tune"]),
        chains=int(sampler["chains"]),
        cores=int(sampler["cores"]),
        seed=int(sampler["seed"]),
        init=(str(sampler["init"]) if sampler["backend"] == "pymc" else PYMC_INITIALIZATION),
        target_accept=float(sampler["target_accept"]),
        full_mass=bool(sampler.get("full_mass", False)),
        backend=str(sampler["backend"]),
        device=str(sampler.get("logical_device", "cpu")),
        paper_profile=sampler["profile"] == "paper",
    )
    try:
        idata.attrs["hqrc_causal"] = "false"
        _validate_posterior_metadata(idata, inputs, sampler, variant=variant)
        products = generate_loeo_fold_products(
            inputs,
            idata,
            variant=variant,
            predictive_seed=int(identity["predictive_seed"]),
            predictive_draws=int(sampler["draws"]) * int(sampler["chains"]),
        )

        def write(temporary: Path) -> None:
            az.to_netcdf(idata, temporary / "posterior.nc")
            products.hourly_predictions.with_columns(
                pl.lit(variant).alias("variant")
            ).write_parquet(temporary / "hourly_predictions.parquet")
            products.metrics.with_columns(pl.lit(variant).alias("variant")).write_parquet(
                temporary / "metrics.parquet"
            )
            _write_json(temporary / "posterior_summary.json", products.posterior_summary)

        _publish_directory(
            directory,
            identity=identity,
            files=_FOLD_FILES,
            writer=write,
            retry_failure=retry_failure,
        )
    finally:
        close = getattr(idata, "close", None)
        if callable(close):
            close()
    return directory, 1, False


def _fit_fold(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    approved_set: ApprovedLOEOARSet,
    *,
    held_out: str,
    variant: AblationVariant,
    root_seed: int,
    profile: str,
    draws: int | None,
    tune: int | None,
    chains: int | None,
    cores: int | None,
    init: str | None,
    target_accept: float | None,
    backend: str,
    device: str,
    output_root: Path,
) -> tuple[Path, int, bool]:
    inputs = prepare_loeo_fold_inputs(
        source, publication, approved_set, held_out_occurrence_id=held_out
    )
    sampler = fold_contract.sampler_contract(
        profile,
        root_seed=root_seed,
        held_out_occurrence_id=held_out,
        variant=variant,
        draws=draws,
        tune=tune,
        chains=chains,
        cores=cores,
        init=init,
        target_accept=target_accept,
        backend=backend,
        device=device,
    )
    if profile == "paper" and source.source_profile != "paper":
        raise LOEOAblationError("paper ablation requires a paper-profile residual source")
    identity = _fold_identity(
        source,
        publication,
        approved_set,
        held_out=held_out,
        variant=variant,
        sampler=sampler,
    )
    directory = _fold_directory(
        output_root,
        publication,
        held_out=held_out,
        variant=variant,
        profile=profile,
        identity=identity,
    )
    retry_sampler: dict[str, object] | None = None
    retry_identity: dict[str, object] | None = None
    retry_directory: Path | None = None
    legacy_retry: tuple[dict[str, object], dict[str, object], Path] | None = None
    if profile == "paper":
        retry_sampler = fold_contract.retry_sampler_contract(
            sampler, variant=variant, held_out_occurrence_id=held_out
        )
        retry_identity = _fold_identity(
            source,
            publication,
            approved_set,
            held_out=held_out,
            variant=variant,
            sampler=retry_sampler,
        )
        retry_directory = _fold_directory(
            output_root,
            publication,
            held_out=held_out,
            variant=variant,
            profile=profile,
            identity=retry_identity,
        )
        legacy_sampler = fold_contract.legacy_retry_sampler_contract(
            sampler, variant=variant, held_out_occurrence_id=held_out
        )
        legacy_identity = _fold_identity(
            source,
            publication,
            approved_set,
            held_out=held_out,
            variant=variant,
            sampler=legacy_sampler,
        )
        legacy_retry = (
            legacy_sampler,
            legacy_identity,
            _fold_directory(
                output_root,
                publication,
                held_out=held_out,
                variant=variant,
                profile=profile,
                identity=legacy_identity,
            ),
        )
    if directory.exists():
        return _fit_fold_attempt(
            inputs, variant=variant, sampler=sampler, identity=identity, directory=directory
        )
    if retry_directory is not None and retry_directory.exists():
        return _fit_fold_attempt(
            inputs,
            variant=variant,
            sampler=retry_sampler,
            identity=retry_identity,
            directory=retry_directory,
            reuse_only=True,
        )
    if legacy_retry is not None and legacy_retry[2].exists():
        return _fit_fold_attempt(
            inputs,
            variant=variant,
            sampler=legacy_retry[0],
            identity=legacy_retry[1],
            directory=legacy_retry[2],
            reuse_only=True,
        )
    try:
        return _fit_fold_attempt(
            inputs, variant=variant, sampler=sampler, identity=identity, directory=directory
        )
    except SamplingError as error:
        diagnostics = fold_contract.retryable_diagnostics(error)
        if (
            diagnostics is None
            or retry_sampler is None
            or retry_identity is None
            or retry_directory is None
        ):
            raise
        retry_failure = fold_contract.retry_failure_payload(retry_sampler, diagnostics)
        retry_path, retry_fitted, retry_reused = _fit_fold_attempt(
            inputs,
            variant=variant,
            sampler=retry_sampler,
            identity=retry_identity,
            directory=retry_directory,
            retry_failure=retry_failure,
        )
        return retry_path, retry_fitted + 1, retry_reused


def _point_row(event_id: str, observed: np.ndarray, forecast: np.ndarray) -> dict[str, object]:
    return point_metric_frame(event_id, observed, forecast).to_dicts()[0]


def _aggregate_products(
    fold_dirs: tuple[Path, ...],
    occurrence_ids: tuple[str, ...],
    *,
    variant: AblationVariant,
    root_seed: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    hourly = pl.concat(
        [pl.read_parquet(path / "hourly_predictions.parquet") for path in fold_dirs],
        how="vertical",
    )
    if tuple(hourly["occurrence_id"].unique(maintain_order=True).to_list()) != occurrence_ids:
        raise LOEOAblationError("ablation fold order differs from the canonical LOEO order")
    event_rows: list[dict[str, object]] = []
    for occurrence_id in occurrence_ids:
        event = hourly.filter(pl.col("occurrence_id") == occurrence_id)
        observed = event["observed_mw"].to_numpy().astype(float)
        baseline = event["baseline_mw"].to_numpy().astype(float)
        corrected = event["corrected_point_mw"].to_numpy().astype(float)
        base = _point_row(occurrence_id, observed, baseline)
        candidate = _point_row(occurrence_id, observed, corrected)
        event_rows.append(
            {
                "occurrence_id": occurrence_id,
                "year": int(occurrence_id.rsplit("-", 1)[1]),
                "holiday_type": str(event["holiday_type"].item(0)),
                "variant": variant,
                **{f"baseline_{key}": value for key, value in base.items() if key != "event_id"},
                **{
                    f"corrected_{key}": value
                    for key, value in candidate.items()
                    if key != "event_id"
                },
                "delta_rmse_percent": 100.0
                * (1.0 - float(candidate["rmse"]) / float(base["rmse"])),
            }
        )
    per_event = pl.DataFrame(event_rows)
    aggregate_rows: list[dict[str, object]] = []
    groups = ("pooled", "seollal", "chuseok")
    for group in groups:
        selected = hourly if group == "pooled" else hourly.filter(pl.col("holiday_type") == group)
        selected_events = (
            per_event if group == "pooled" else per_event.filter(pl.col("holiday_type") == group)
        )
        observed = selected["observed_mw"].to_numpy().astype(float)
        baseline = selected["baseline_mw"].to_numpy().astype(float)
        corrected = selected["corrected_point_mw"].to_numpy().astype(float)
        baseline_point = _point_row(group, observed, baseline)
        corrected_point = _point_row(group, observed, corrected)
        baseline_event_rmse = selected_events["baseline_rmse"].to_numpy().astype(float)
        corrected_event_rmse = selected_events["corrected_rmse"].to_numpy().astype(float)
        improvements = selected_events["delta_rmse_percent"].to_numpy().astype(float)
        wilcoxon = stats.wilcoxon(
            baseline_event_rmse,
            corrected_event_rmse,
            alternative="greater",
            method="auto",
        )
        rng = np.random.default_rng(
            derive_loeo_seed(root_seed, f"{variant}-{group}-event-bootstrap")
        )
        bootstrap = np.median(
            improvements[rng.integers(0, improvements.size, size=(10_000, improvements.size))],
            axis=1,
        )
        dm = hac_dm_test(
            (baseline - observed) ** 2,
            (corrected - observed) ** 2,
            event_ids=selected["occurrence_id"].to_list(),
            bandwidth=24,
            event_block_seed=derive_loeo_seed(root_seed, f"{variant}-{group}-dm"),
        )
        common = {
            "group": group,
            "event_count": selected_events.height,
            "hour_count": selected.height,
        }
        aggregate_rows.append(
            {
                **common,
                "variant": "H0",
                **{
                    key: value
                    for key, value in baseline_point.items()
                    if key not in {"event_id", "n_timestamps"}
                },
                "delta_rmse_percent": 0.0,
                "event_median_delta_rmse_percent": 0.0,
                "bootstrap_ci_lower": None,
                "bootstrap_ci_upper": None,
                "wilcoxon_p_one_sided": None,
                "dm_p_event_block": None,
            }
        )
        aggregate_rows.append(
            {
                **common,
                "variant": variant,
                **{
                    key: value
                    for key, value in corrected_point.items()
                    if key not in {"event_id", "n_timestamps"}
                },
                "delta_rmse_percent": 100.0
                * (1.0 - float(corrected_point["rmse"]) / float(baseline_point["rmse"])),
                "event_median_delta_rmse_percent": float(np.median(improvements)),
                "bootstrap_ci_lower": float(np.quantile(bootstrap, 0.025)),
                "bootstrap_ci_upper": float(np.quantile(bootstrap, 0.975)),
                "wilcoxon_p_one_sided": float(wilcoxon.pvalue),
                "dm_p_event_block": dm.p_value,
            }
        )
    return hourly, per_event, pl.DataFrame(aggregate_rows)


def _aggregate_fold_reference(directory: Path, held_out: str) -> dict[str, object]:
    manifest = _read_json(directory / "manifest.json", "ablation fold manifest")
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping):
        raise LOEOAblationError("ablation fold identity differs")
    _validate_complete(directory, identity=identity, files=_FOLD_FILES)
    loeo = identity.get("loeo")
    sampler = identity.get("sampler")
    if (
        not isinstance(loeo, Mapping)
        or loeo.get("held_out_occurrence_id") != held_out
        or not isinstance(sampler, Mapping)
    ):
        raise LOEOAblationError("ablation fold identity differs")
    return {
        "held_out_occurrence_id": held_out,
        "output_dir": directory.resolve().as_posix(),
        "identity_sha256": sha_json(identity),
        "manifest_sha256": file_sha256(directory / "manifest.json"),
        "manifest_digest": manifest["manifest_digest"],
        "sampler": dict(sampler),
    }


def fit_loeo_ablation(
    source: ValidatedCorrectionSource,
    publication: LOEOPublication,
    approved_set: ApprovedLOEOARSet,
    *,
    variant: str,
    held_out_occurrence_ids: tuple[str, ...],
    root_seed: int,
    profile: str,
    draws: int | None = None,
    tune: int | None = None,
    chains: int | None = None,
    cores: int | None = None,
    init: str | None = None,
    target_accept: float | None = None,
    backend: str = "pymc",
    device: str = "cpu",
    output_root: Path,
) -> LOEOAblationResult:
    """Fit/reuse all selected H1 or H2 folds and publish table-ready metrics."""

    selected_variant = _variant(variant)
    if held_out_occurrence_ids != publication.occurrence_ids:
        raise LOEOAblationError("paper ablation requires all canonical LOEO occurrences")
    if len(held_out_occurrence_ids) != 10:
        raise LOEOAblationError("paper ablation requires exactly ten held-out occurrences")
    if not Path(output_root).is_absolute():
        raise LOEOAblationError("ablation output root must be absolute")
    validate_loeo_fold_sources(source, publication, approved_set)
    fold_dirs: list[Path] = []
    fit_count = 0
    all_reused = True
    for held_out in held_out_occurrence_ids:
        directory, fitted, reused = _fit_fold(
            source,
            publication,
            approved_set,
            held_out=held_out,
            variant=selected_variant,
            root_seed=root_seed,
            profile=profile,
            draws=draws,
            tune=tune,
            chains=chains,
            cores=cores,
            init=init,
            target_accept=target_accept,
            backend=backend,
            device=device,
            output_root=Path(output_root),
        )
        fold_dirs.append(directory)
        fit_count += fitted
        all_reused = all_reused and reused
    context = publication.context
    fold_references = [
        _aggregate_fold_reference(path, held_out)
        for path, held_out in zip(fold_dirs, held_out_occurrence_ids, strict=True)
    ]
    fold_manifest_hashes = [reference["manifest_sha256"] for reference in fold_references]
    identity: dict[str, object] = {
        "schema_version": 1,
        "evaluation": "retrospective-loeo-ablation-primary",
        "causal": False,
        "variant": selected_variant,
        "context": {
            "model": context.model,
            "feature_set": context.feature_set,
            "seed": context.seed,
        },
        "selected_occurrence_ids": list(held_out_occurrence_ids),
        "root_seed": root_seed,
        "fold_manifest_sha256": fold_manifest_hashes,
    }
    if any("retry" in reference["sampler"] for reference in fold_references):
        identity["folds"] = fold_references
    primary = (
        Path(output_root)
        / f"loeo-ablation-primary-{selected_variant.lower()}"
        / context.model
        / context.feature_set
        / f"seed-{context.seed}"
        / profile
        / f"identity-{sha_json(identity)}"
    )

    def write(temporary: Path) -> None:
        hourly, per_event, aggregate = _aggregate_products(
            tuple(fold_dirs),
            held_out_occurrence_ids,
            variant=selected_variant,
            root_seed=root_seed,
        )
        hourly.write_parquet(temporary / "hourly_predictions.parquet")
        per_event.write_parquet(temporary / "per_event_metrics.parquet")
        aggregate.write_parquet(temporary / "aggregate_metrics.parquet")

    primary_existed = primary.exists()
    _publish_directory(primary, identity=identity, files=_PRIMARY_FILES, writer=write)
    return LOEOAblationResult(
        variant=selected_variant,
        output_dir=primary,
        hourly_predictions_path=primary / "hourly_predictions.parquet",
        per_event_metrics_path=primary / "per_event_metrics.parquet",
        aggregate_metrics_path=primary / "aggregate_metrics.parquet",
        sampler_fit_count=fit_count,
        reused=primary_existed and all_reused,
    )


__all__ = [
    "AblationVariant",
    "LOEOAblationError",
    "LOEOAblationResult",
    "fit_loeo_ablation",
]
