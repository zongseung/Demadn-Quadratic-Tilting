from __future__ import annotations

import fcntl
import json
import multiprocessing
import threading
from hashlib import sha256

import pytest
from hqrc_v3.diagnostics.ar import (
    ARCalibration,
    ARCalibrationError,
    approve_calibration,
    calibrate_beta_prior,
    load_approved_calibration,
    write_ar_diagnostics,
)
from hqrc_v3.provenance import ArtifactMismatch


@pytest.fixture
def calibration() -> ARCalibration:
    return calibrate_beta_prior([0.4, 0.5, 0.6], event_ids=("event-a", "event-b", "event-c"))


def _write(tmp_path, calibration):
    return write_ar_diagnostics(
        tmp_path / "proposed.json",
        (),
        calibration,
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="events",
        context={
            "model": "baseline",
            "feature_set": "B1",
            "seed": 1,
            "split_ids": ("oof-2020", "oof-2021", "oof-2022"),
        },
    )


def test_json_round_trip_approval_and_unapproved_loader_rejection(tmp_path, calibration):
    proposed = _write(tmp_path, calibration)
    with pytest.raises(ARCalibrationError, match="not approved"):
        load_approved_calibration(
            proposed,
            current_residual_sha256="residual",
            current_config_sha256="config",
            current_event_sha256="events",
        )

    approved = approve_calibration(
        proposed,
        tmp_path / "approved.json",
        current_residual_sha256="residual",
        current_config_sha256="config",
        current_event_sha256="events",
    )

    loaded = load_approved_calibration(
        approved,
        current_residual_sha256="residual",
        current_config_sha256="config",
        current_event_sha256="events",
    )
    assert loaded == calibration
    payload = json.loads(approved.read_text(encoding="utf-8"))
    assert payload["approved"] is True
    assert (
        payload["approved_proposal_digest"]
        == json.loads(proposed.read_text(encoding="utf-8"))["proposal_digest"]
    )


def test_approval_rejects_changed_hashes_tampering_and_incompatible_overwrite(
    tmp_path, calibration
):
    proposed = _write(tmp_path, calibration)
    with pytest.raises(ArtifactMismatch, match="residual"):
        approve_calibration(
            proposed,
            tmp_path / "approved.json",
            current_residual_sha256="changed",
            current_config_sha256="config",
            current_event_sha256="events",
        )
    with pytest.raises(ArtifactMismatch, match="config"):
        approve_calibration(
            proposed,
            tmp_path / "approved.json",
            current_residual_sha256="residual",
            current_config_sha256="changed",
            current_event_sha256="events",
        )
    tampered = json.loads(proposed.read_text(encoding="utf-8"))
    tampered["calibration"]["a"] = 13.0
    proposed.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ArtifactMismatch, match="digest"):
        approve_calibration(
            proposed,
            tmp_path / "approved.json",
            current_residual_sha256="residual",
            current_config_sha256="config",
            current_event_sha256="events",
        )

    original = write_ar_diagnostics(
        tmp_path / "original.json",
        (),
        calibration,
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="events",
        context={
            "model": "baseline",
            "feature_set": "B1",
            "seed": 1,
            "split_ids": ("oof-2020", "oof-2021", "oof-2022"),
        },
    )
    changed = calibrate_beta_prior([0.3, 0.4, 0.5], event_ids=("event-a", "event-b", "event-c"))
    with pytest.raises(ArtifactMismatch, match="overwrite"):
        write_ar_diagnostics(
            original,
            (),
            changed,
            residual_sha256="residual",
            config_sha256="config",
            event_sha256="events",
            context={
                "model": "baseline",
                "feature_set": "B1",
                "seed": 1,
                "split_ids": ("oof-2020", "oof-2021", "oof-2022"),
            },
        )


def test_atomic_write_preserves_existing_file_when_publish_fails(
    tmp_path, calibration, monkeypatch
):
    destination = _write(tmp_path, calibration)
    before = destination.read_bytes()

    def fail_replace(*_args):
        raise OSError("simulated replace failure")

    monkeypatch.setattr("hqrc_v3.diagnostics.ar.os.replace", fail_replace)
    with pytest.raises(OSError, match="simulated"):
        write_ar_diagnostics(
            tmp_path / "other.json",
            (),
            calibration,
            residual_sha256="other",
            config_sha256="config",
            event_sha256="events",
            context={
                "model": "baseline",
                "feature_set": "B1",
                "seed": 1,
                "split_ids": ("oof-2020", "oof-2021", "oof-2022"),
            },
        )
    assert destination.read_bytes() == before
    assert not list(tmp_path.glob(".*.tmp"))


def _reseal(payload: dict[str, object]) -> None:
    proposal = dict(payload)
    proposal.pop("artifact_digest", None)
    proposal.pop("proposal_digest", None)
    payload["proposal_digest"] = sha256(
        json.dumps(proposal, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    artifact = dict(payload)
    artifact.pop("artifact_digest", None)
    payload["artifact_digest"] = sha256(
        json.dumps(artifact, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("formula_version", "manual"),
        ("phi_estimates", [0.4, 2.0, 0.6]),
        ("event_ids", ["event-a", "event-a", "event-c"]),
        ("a", 1.0),
    ],
)
def test_resealed_handcrafted_invalid_calibration_is_rejected(tmp_path, calibration, field, value):
    proposal = _write(tmp_path, calibration)
    payload = json.loads(proposal.read_text(encoding="utf-8"))
    payload["calibration"][field] = value
    _reseal(payload)
    proposal.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )

    with pytest.raises((ARCalibrationError, ArtifactMismatch)):
        approve_calibration(
            proposal,
            tmp_path / "approved.json",
            current_residual_sha256="residual",
            current_config_sha256="config",
            current_event_sha256="events",
        )


def _hold_artifact_lock(lock_path, acquired, release) -> None:
    with open(lock_path, "a+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        acquired.set()
        release.wait()
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def test_concurrent_incompatible_writers_leave_one_immutable_winner(tmp_path, calibration):
    destination = tmp_path / "shared.json"
    lock_path = tmp_path / ".shared.json.lock"
    context = multiprocessing.get_context("spawn")
    acquired, release = context.Event(), context.Event()
    holder = context.Process(target=_hold_artifact_lock, args=(str(lock_path), acquired, release))
    holder.start()
    assert acquired.wait(timeout=10)
    with open(lock_path, "a+", encoding="utf-8") as probe:
        with pytest.raises(BlockingIOError):
            fcntl.flock(probe.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    changed = calibrate_beta_prior([0.3, 0.4, 0.5], event_ids=("event-a", "event-b", "event-c"))
    results: list[object] = []

    def writer(candidate):
        try:
            results.append(
                write_ar_diagnostics(
                    destination,
                    (),
                    candidate,
                    residual_sha256="residual",
                    config_sha256="config",
                    event_sha256="events",
                    context={
                        "model": "baseline",
                        "feature_set": "B1",
                        "seed": 1,
                        "split_ids": ("oof-2020", "oof-2021", "oof-2022"),
                    },
                )
            )
        except Exception as error:  # assertion below verifies the exact losing outcome
            results.append(error)

    first, second = (
        threading.Thread(target=writer, args=(calibration,)),
        threading.Thread(target=writer, args=(changed,)),
    )
    first.start()
    second.start()
    try:
        release.set()
    finally:
        first.join(timeout=10)
        second.join(timeout=10)
        holder.join(timeout=10)
        if holder.is_alive():
            holder.terminate()
            holder.join()
    assert not first.is_alive() and not second.is_alive() and holder.exitcode == 0
    assert sum(isinstance(result, type(destination)) for result in results) == 1
    assert sum(isinstance(result, ArtifactMismatch) for result in results) == 1
