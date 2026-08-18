from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


mode, request_path, result_path = sys.argv[1:]
if mode == "nonzero":
    raise SystemExit(7)
if mode == "sleep":
    time.sleep(10)
if mode == "malformed":
    Path(result_path).write_text("not-json")
    raise SystemExit(0)
request = json.loads(Path(request_path).read_text())
payload = {
    "schema_version": 1,
    "request_digest": request["request_digest"],
    "backend": request["sampler"]["backend"],
    "pid": os.getpid(),
    "wall_seconds": 2.0,
    "peak_rss_mb": 10.0,
    "diagnostics": {
        "max_rhat": 1.0,
        "min_bulk_ess": 100.0,
        "min_tail_ess": 80.0,
        "divergences": 0,
    },
    "posterior": {
        "phi": {"shape": [2], "means": [0.2, 0.4], "sds": [0.1, 0.2]},
    },
    "versions": {"python": "fake", "pymc": "fake", "arviz": "fake"},
}
payload["result_digest"] = hashlib.sha256(canonical(payload).encode()).hexdigest()
if mode == "bad-digest":
    payload["request_digest"] = "0" * 64
Path(result_path).write_text(canonical(payload) + "\n")
