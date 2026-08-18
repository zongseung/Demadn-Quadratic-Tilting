"""Fresh-process entry point for one real HQRC sampler measurement."""

from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
import time
from hashlib import sha256
from pathlib import Path

import numpy as np

from hqrc_v3.bayes.artifacts import load_hqrc_data
from hqrc_v3.bayes.benchmark import (
    SamplerWorkerError,
    _atomic_json,
    _canonical,
    load_sampler_request,
)
from hqrc_v3.bayes.model import HQRCModelOptions
from hqrc_v3.bayes.samplers import sample_hqrc, validate_inference_data
from hqrc_v3.diagnostics.ar import load_approved_calibration
from hqrc_v3.peak_rss import peak_rss_mb


def _rss_mb() -> float:
    return peak_rss_mb()


def _posterior_summary(idata) -> dict[str, dict[str, object]]:
    result = {}
    for name in sorted(idata.posterior.data_vars):
        values = np.asarray(idata.posterior[name], dtype=float)
        if values.ndim < 2 or not np.isfinite(values).all():
            raise SamplerWorkerError("posterior contains invalid sampler dimensions or values")
        flattened = values.reshape(values.shape[0] * values.shape[1], *values.shape[2:])
        means = np.mean(flattened, axis=0)
        sds = np.std(flattened, axis=0)
        result[name] = {
            "shape": list(means.shape),
            "means": np.asarray(means).reshape(-1).astype(float).tolist(),
            "sds": np.asarray(sds).reshape(-1).astype(float).tolist(),
        }
    if not result:
        raise SamplerWorkerError("posterior contains no parameters")
    return result


def execute(request_path: Path, result_path: Path) -> Path:
    request = load_sampler_request(request_path)
    inputs, hashes = request["inputs"], request["hashes"]
    data, _ = load_hqrc_data(
        Path(inputs["hqrc_npz"]["path"]), Path(inputs["hqrc_metadata"]["path"])
    )
    calibration = load_approved_calibration(
        Path(inputs["approved_ar"]["path"]),
        current_residual_sha256=hashes["residual_sha256"],
        current_config_sha256=hashes["config_sha256"],
        current_event_sha256=hashes["event_sha256"],
    )
    model, sampler = request["model"], request["sampler"]
    try:
        options = HQRCModelOptions(**model["options"])
    except TypeError as error:
        raise SamplerWorkerError("sampler request model options are invalid") from error
    started = time.perf_counter()
    idata = sample_hqrc(
        data,
        calibration,
        variant=model["variant"],
        pooling=model["pooling"],
        options=options,
        backend=sampler["backend"],
        seed=sampler["seed"],
        draws=sampler["draws"],
        tune=sampler["tune"],
        chains=sampler["chains"],
        cores=sampler["cores"],
        paper_profile=sampler["profile"] == "paper",
    )
    wall = time.perf_counter() - started
    diagnostics = validate_inference_data(idata, paper_profile=sampler["profile"] == "paper")
    payload = {
        "schema_version": 1,
        "request_digest": request["request_digest"],
        "backend": sampler["backend"],
        "pid": os.getpid(),
        "wall_seconds": wall,
        "peak_rss_mb": _rss_mb(),
        "diagnostics": {
            "max_rhat": diagnostics.max_rhat,
            "min_bulk_ess": diagnostics.min_bulk_ess,
            "min_tail_ess": diagnostics.min_tail_ess,
            "divergences": diagnostics.divergences,
        },
        "posterior": _posterior_summary(idata),
        "versions": {
            "python": platform.python_version(),
            "pymc": importlib.metadata.version("pymc"),
            "arviz": importlib.metadata.version("arviz"),
        },
    }
    if sampler["backend"] == "nutpie":
        payload["versions"]["nutpie"] = importlib.metadata.version("nutpie")
    payload["result_digest"] = sha256(_canonical(payload)).hexdigest()
    return _atomic_json(Path(result_path), payload)


def main(argv: list[str] | None = None) -> int:
    arguments = sys.argv[1:] if argv is None else argv
    if len(arguments) != 2:
        print("usage: python -m hqrc_v3.bayes.sampler_worker REQUEST RESULT", file=sys.stderr)
        return 2
    try:
        execute(Path(arguments[0]), Path(arguments[1]))
    except Exception as error:
        print(f"sampler-worker: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
