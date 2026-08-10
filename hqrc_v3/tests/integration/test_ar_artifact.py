from __future__ import annotations

import json

import pytest
from hqrc_v3.diagnostics.ar import (
    ARCalibration,
    ARCalibrationError,
    approve_calibration,
    load_approved_calibration,
    write_ar_diagnostics,
)
from hqrc_v3.provenance import ArtifactMismatch


@pytest.fixture
def calibration() -> ARCalibration:
    return ARCalibration(
        a=12.0,
        b=4.0,
        phi_center=0.5,
        event_count=3,
        event_ids=("event-a", "event-b", "event-c"),
        phi_estimates=(0.4, 0.5, 0.6),
        formula_version="robust-mad-beta-v1",
    )


def _write(tmp_path, calibration):
    return write_ar_diagnostics(
        tmp_path / "proposed.json",
        (),
        calibration,
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="events",
        context={"model": "baseline", "feature_set": "B1", "seed": 1},
    )


def test_json_round_trip_approval_and_unapproved_loader_rejection(tmp_path, calibration):
    proposed = _write(tmp_path, calibration)
    with pytest.raises(ARCalibrationError, match="not approved"):
        load_approved_calibration(proposed, current_residual_sha256="residual")

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
        )
    with pytest.raises(ArtifactMismatch, match="config"):
        approve_calibration(
            proposed,
            tmp_path / "approved.json",
            current_residual_sha256="residual",
            current_config_sha256="changed",
        )
    proposed.write_text(proposed.read_text(encoding="utf-8").replace('"a":12.0', '"a":13.0'))
    with pytest.raises(ArtifactMismatch, match="digest"):
        approve_calibration(
            proposed,
            tmp_path / "approved.json",
            current_residual_sha256="residual",
        )

    original = write_ar_diagnostics(
        tmp_path / "original.json",
        (),
        calibration,
        residual_sha256="residual",
        config_sha256="config",
        event_sha256="events",
        context={"model": "baseline", "feature_set": "B1", "seed": 1},
    )
    changed = ARCalibration(
        a=11.0,
        b=5.0,
        phi_center=0.4,
        event_count=3,
        event_ids=("event-a", "event-b", "event-c"),
        phi_estimates=(0.3, 0.4, 0.5),
        formula_version="robust-mad-beta-v1",
    )
    with pytest.raises(ArtifactMismatch, match="overwrite"):
        write_ar_diagnostics(
            original,
            (),
            changed,
            residual_sha256="residual",
            config_sha256="config",
            event_sha256="events",
            context={"model": "baseline", "feature_set": "B1", "seed": 1},
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
            context={"model": "baseline", "feature_set": "B1", "seed": 1},
        )
    assert destination.read_bytes() == before
