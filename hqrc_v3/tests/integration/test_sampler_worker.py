from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import pytest
from hqrc_v3.bayes.benchmark import (
    SamplerWorkerError,
    load_sampler_request,
    load_sampler_result,
    maximum_posterior_mean_distance,
    run_sampler_worker,
    write_sampler_request,
)
from hqrc_v3.provenance import file_sha256

FAKE = Path(__file__).parents[1] / "fixtures" / "fake_sampler_worker.py"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _request(tmp_path, *, backend="pymc"):
    files = {}
    for name in ("data.npz", "data.json", "approved.json"):
        path = tmp_path / name
        path.write_text(name)
        files[name] = path
    return write_sampler_request(
        tmp_path / "request.json",
        hqrc_npz=files["data.npz"],
        hqrc_metadata=files["data.json"],
        approved_ar=files["approved.json"],
        residual_sha256="a" * 64,
        config_sha256="b" * 64,
        event_sha256="c" * 64,
        variant="H3",
        pooling="partial",
        options={"covariance": "full", "include_restriction": True},
        backend=backend,
        seed=7,
        draws=20,
        tune=20,
        chains=2,
        profile="smoke",
        bound_hashes={path.name: file_sha256(path) for path in files.values()},
    )


def test_parent_launches_separate_worker_and_verifies_pid_and_digest(tmp_path):
    request = _request(tmp_path)
    result = run_sampler_worker(
        request,
        tmp_path / "result.json",
        timeout_seconds=5,
        worker_command=(sys.executable, str(FAKE), "ok"),
    )
    assert result.pid != os.getpid()
    assert result.request_digest
    assert result.min_bulk_ess_per_second == pytest.approx(50.0)


def test_sampler_request_rejects_boolean_version_after_redigest(tmp_path):
    request = _request(tmp_path)
    payload = json.loads(request.read_bytes())
    payload.pop("request_digest")
    payload["schema_version"] = True
    payload["request_digest"] = _digest(payload)
    request.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(SamplerWorkerError, match="request version"):
        load_sampler_request(request)


def test_sampler_result_rejects_boolean_version_after_redigest(tmp_path):
    request_path = _request(tmp_path)
    request = load_sampler_request(request_path)
    result_path = tmp_path / "result.json"
    run_sampler_worker(
        request_path,
        result_path,
        timeout_seconds=5,
        worker_command=(sys.executable, str(FAKE), "ok"),
    )
    payload = json.loads(result_path.read_bytes())
    payload.pop("result_digest")
    payload["schema_version"] = True
    payload["result_digest"] = _digest(payload)
    result_path.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(SamplerWorkerError, match="result version"):
        load_sampler_result(result_path, request=request)


@pytest.mark.parametrize(
    ("mode", "match"),
    [("bad-digest", "digest"), ("nonzero", "exit"), ("malformed", "JSON"), ("sleep", "timeout")],
)
def test_parent_fails_closed_for_worker_errors(tmp_path, mode, match):
    with pytest.raises(SamplerWorkerError, match=match):
        run_sampler_worker(
            _request(tmp_path),
            tmp_path / "result.json",
            timeout_seconds=0.2 if mode == "sleep" else 5,
            worker_command=(sys.executable, str(FAKE), mode),
        )


def test_posterior_distance_uses_each_element_and_zero_sd_requires_equality():
    left = {"x": {"shape": [2], "means": [1.0, 2.0], "sds": [1.0, 0.0]}}
    right = {"x": {"shape": [2], "means": [1.1, 2.0], "sds": [1.0, 0.0]}}
    distance, audit = maximum_posterior_mean_distance(left, right)
    assert distance == pytest.approx(0.1)
    assert audit["x"][0]["distance_sd"] == pytest.approx(0.1)
    right["x"]["means"][1] = 2.1
    with pytest.raises(SamplerWorkerError, match="zero pooled SD"):
        maximum_posterior_mean_distance(left, right)
