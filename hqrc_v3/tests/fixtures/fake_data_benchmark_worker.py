from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


mode, request_path, result_path, requested_threads = sys.argv[1:]
if mode == "nonzero":
    raise SystemExit(7)
if mode == "sleep":
    time.sleep(10)
if mode == "malformed":
    Path(result_path).write_text("not-json")
    raise SystemExit(0)
request = json.loads(Path(request_path).read_text())
rows = 4
columns = request["columns"]
repetitions = request["repetitions"]
checksum = "b" * 64
workload_checksum = "d" * 64 if mode == "checksum-mismatch" else "c" * 64
timings = [0.001 + index * 0.0001 for index in range(repetitions)]
payload = {
    "schema_version": 1,
    "request_digest": request["request_digest"],
    "pid": os.getpid(),
    "threads": {"requested": int(requested_threads), "actual": int(requested_threads)},
    "dimensions": {"rows": rows, "columns": columns},
    "zero_copy": {
        "timings_seconds": timings,
        "median_seconds": sorted(timings)[len(timings) // 2],
        "owns_data": False,
        "writeable": False,
        "owned_bytes_per_repetition": 0,
        "shape": [rows, len(columns)],
        "checksum": checksum,
    },
    "explicit_copy": {
        "timings_seconds": timings,
        "median_seconds": sorted(timings)[len(timings) // 2],
        "owns_data": True,
        "writeable": True,
        "allocation_bytes_per_repetition": rows * len(columns) * 8,
        "allocation_bytes_total": rows * len(columns) * 8 * repetitions,
        "shape": [rows, len(columns)],
        "checksum": checksum,
    },
    "workload": {
        "identity": request["workload"],
        "seed": request["seed"],
        "timings_seconds": timings,
        "median_seconds": sorted(timings)[len(timings) // 2],
        "checksum": workload_checksum,
    },
    "peak_rss_mb": 10.0,
    "versions": {"python": "fake", "numpy": "fake", "polars": "fake"},
}
if mode == "extra-key":
    payload["unexpected"] = True
payload["result_digest"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
if mode == "bad-digest":
    payload["request_digest"] = "0" * 64
if mode == "bad-pid":
    payload["pid"] = os.getppid()
    payload.pop("result_digest")
    payload["result_digest"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
Path(result_path).write_text(canonical(payload) + "\n")
