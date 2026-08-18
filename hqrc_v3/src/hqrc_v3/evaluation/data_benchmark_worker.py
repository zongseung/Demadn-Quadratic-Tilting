"""Fresh-process Polars benchmark worker.

This module deliberately imports Polars only inside execution, after the parent
process has fixed ``POLARS_MAX_THREADS`` in this child's environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_REQUEST_KEYS = {
    "schema_version",
    "input",
    "columns",
    "repetitions",
    "workers",
    "seed",
    "workload",
    "request_digest",
}


class DataBenchmarkError(RuntimeError):
    """Raised when the isolated worker cannot honor its measurement contract."""


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_request(path: Path) -> dict[str, Any]:
    try:
        raw = path.read_bytes()
        payload = json.loads(raw)
    except (OSError, json.JSONDecodeError) as error:
        raise DataBenchmarkError("data benchmark request is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != _REQUEST_KEYS:
        raise DataBenchmarkError("data benchmark request schema differs")
    if raw != _canonical(payload) + b"\n":
        raise DataBenchmarkError("data benchmark request is not canonical JSON")
    request_digest = payload.pop("request_digest")
    if not isinstance(request_digest, str) or request_digest != _digest(payload):
        raise DataBenchmarkError("data benchmark request digest differs")
    payload["request_digest"] = request_digest
    if type(payload["schema_version"]) is not int or payload["schema_version"] != 1:
        raise DataBenchmarkError("data benchmark request version differs")
    entry = payload["input"]
    if not isinstance(entry, dict) or set(entry) != {"path", "sha256"}:
        raise DataBenchmarkError("data benchmark request input schema differs")
    artifact = Path(entry["path"]) if isinstance(entry["path"], str) else Path()
    if not artifact.is_absolute() or not artifact.is_file():
        raise DataBenchmarkError("data benchmark request input path differs")
    if entry["sha256"] != _file_sha256(artifact):
        raise DataBenchmarkError("data benchmark request input digest differs")
    columns = payload["columns"]
    if (
        not isinstance(columns, list)
        or not columns
        or any(not isinstance(item, str) or not item for item in columns)
        or len(columns) != len(set(columns))
    ):
        raise DataBenchmarkError("data benchmark request columns differ")
    repetitions = payload["repetitions"]
    workers = payload["workers"]
    if isinstance(repetitions, bool) or not isinstance(repetitions, int) or repetitions < 3:
        raise DataBenchmarkError("data benchmark request repetitions differ")
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 2:
        raise DataBenchmarkError("data benchmark request workers differ")
    seed = payload["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise DataBenchmarkError("data benchmark request seed differs")
    if payload["workload"] != "polars-numeric-summary-v1":
        raise DataBenchmarkError("data benchmark request workload differs")
    return payload


def _write_json(path: Path, value: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return path


def _array_checksum(columns: list[str], arrays: list[Any]) -> str:
    digest = hashlib.sha256()
    for name, array in zip(columns, arrays, strict=True):
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _elapsed(start: float) -> float:
    return max(time.perf_counter() - start, sys.float_info.epsilon)


def _peak_rss_mb() -> float:
    raw = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return max(raw / divisor, sys.float_info.epsilon)


def _canonical_aggregate_value(value: object) -> dict[str, object]:
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean", "value": value}
    if isinstance(value, int):
        return {"type": "integer", "value": str(value)}
    if isinstance(value, float):
        if math.isnan(value):
            encoded = "nan"
        elif math.isinf(value):
            encoded = "positive-infinity" if value > 0 else "negative-infinity"
        else:
            encoded = value.hex()
        return {"type": "float", "value": encoded}
    if isinstance(value, str):
        return {"type": "string", "value": value}
    raise DataBenchmarkError(
        f"workload aggregate value type is unsupported: {type(value).__name__}"
    )


def _aggregate_checksum(
    *, identity: str, seed: int, columns: list[str], rows: list[tuple[object, ...]]
) -> str:
    if any(len(row) != len(columns) for row in rows):
        raise DataBenchmarkError("workload aggregate shape differs")
    values = [
        [_canonical_aggregate_value(value) for value in row]
        for row in rows
    ]
    return _digest(
        {
            "identity": identity,
            "seed": seed,
            "columns": columns,
            "aggregate_values": values,
        }
    )


def _workload(pl: Any, request: dict[str, Any]) -> str:
    expressions = []
    for position, name in enumerate(request["columns"], start=1):
        scaled = pl.col(name) * float(position)
        expressions.extend(
            (
                scaled.min().alias(f"{position}_minimum"),
                scaled.max().alias(f"{position}_maximum"),
                pl.col(name).null_count().alias(f"{position}_null_count"),
                pl.col(name).len().alias(f"{position}_row_count"),
            )
        )
    result = (
        pl.scan_parquet(request["input"]["path"])
        .select(expressions)
        .collect()
    )
    return _aggregate_checksum(
        identity=request["workload"],
        seed=request["seed"],
        columns=result.columns,
        rows=result.rows(),
    )


def execute(request_path: Path, result_path: Path, *, requested_threads: int) -> Path:
    """Execute all measurements and write their canonical result."""

    request = _load_request(request_path)
    if requested_threads not in {1, request["workers"]}:
        raise DataBenchmarkError("worker threads must be serial or the requested parallel count")
    expected_environment = str(requested_threads)
    if os.environ.get("POLARS_MAX_THREADS") != expected_environment:
        raise DataBenchmarkError("POLARS_MAX_THREADS was not fixed before worker import")

    import numpy as np
    import polars as pl

    actual_threads = pl.thread_pool_size()
    frame = pl.read_parquet(request["input"]["path"], columns=request["columns"]).rechunk()
    rows = frame.height
    if rows <= 0:
        raise DataBenchmarkError("zero-copy benchmark requires at least one row")
    series = [frame.get_column(name).rechunk() for name in request["columns"]]
    if any(item.dtype != pl.Float64 for item in series):
        raise DataBenchmarkError("zero-copy benchmark requires Float64 columns")
    if any(not item.is_finite().all() for item in series):
        raise DataBenchmarkError("zero-copy benchmark requires finite Float64 values")
    if any(item.n_chunks() != 1 for item in series):
        raise DataBenchmarkError("zero-copy benchmark requires rechunked Series")

    zero_timings: list[float] = []
    arrays: list[Any] = []
    for _ in range(request["repetitions"]):
        start = time.perf_counter()
        try:
            arrays = [item.to_numpy(allow_copy=False) for item in series]
        except RuntimeError as error:
            raise DataBenchmarkError(
                "zero-copy benchmark failed because Polars must copy"
            ) from error
        zero_timings.append(_elapsed(start))
        if any(array.flags.owndata or array.flags.writeable for array in arrays):
            raise DataBenchmarkError("zero-copy result must be non-owning and read-only")
    zero_checksum = _array_checksum(request["columns"], arrays)

    copy_timings: list[float] = []
    copied: list[Any] = []
    for _ in range(request["repetitions"]):
        start = time.perf_counter()
        copied = [np.array(array, copy=True) for array in arrays]
        copy_timings.append(_elapsed(start))
        if any(not array.flags.owndata or not array.flags.writeable for array in copied):
            raise DataBenchmarkError("explicit NumPy copy must own writeable memory")
    copy_checksum = _array_checksum(request["columns"], copied)
    if zero_checksum != copy_checksum:
        raise DataBenchmarkError("zero-copy and explicit-copy checksums differ")
    allocation_bytes = rows * len(series) * 8

    workload_timings: list[float] = []
    workload_checksums: list[str] = []
    for _ in range(request["repetitions"]):
        start = time.perf_counter()
        workload_checksums.append(_workload(pl, request))
        workload_timings.append(_elapsed(start))
    if len(set(workload_checksums)) != 1:
        raise DataBenchmarkError("repeated workload checksums differ")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "request_digest": request["request_digest"],
        "pid": os.getpid(),
        "threads": {"requested": requested_threads, "actual": actual_threads},
        "dimensions": {"rows": rows, "columns": request["columns"]},
        "zero_copy": {
            "timings_seconds": zero_timings,
            "median_seconds": statistics.median(zero_timings),
            "owns_data": False,
            "writeable": False,
            "owned_bytes_per_repetition": 0,
            "shape": [rows, len(series)],
            "checksum": zero_checksum,
        },
        "explicit_copy": {
            "timings_seconds": copy_timings,
            "median_seconds": statistics.median(copy_timings),
            "owns_data": True,
            "writeable": True,
            "allocation_bytes_per_repetition": allocation_bytes,
            "allocation_bytes_total": allocation_bytes * request["repetitions"],
            "shape": [rows, len(series)],
            "checksum": copy_checksum,
        },
        "workload": {
            "identity": request["workload"],
            "seed": request["seed"],
            "timings_seconds": workload_timings,
            "median_seconds": statistics.median(workload_timings),
            "checksum": workload_checksums[0],
        },
        "peak_rss_mb": _peak_rss_mb(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
        },
    }
    payload["result_digest"] = _digest(payload)
    return _write_json(Path(result_path), payload)


def main() -> int:
    if len(sys.argv) != 4:
        raise SystemExit("usage: data_benchmark_worker REQUEST RESULT REQUESTED_THREADS")
    request_path, result_path, requested_threads = sys.argv[1:]
    try:
        parsed_threads = int(requested_threads)
    except ValueError as error:
        raise SystemExit("REQUESTED_THREADS must be an integer") from error
    execute(Path(request_path), Path(result_path), requested_threads=parsed_threads)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
