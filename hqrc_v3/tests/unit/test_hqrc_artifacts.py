import json

import numpy as np
import pytest
from hqrc_v3.bayes.artifacts import HQRCArtifactError, load_hqrc_data, write_hqrc_data
from hqrc_v3.bayes.model import HQRCData


def _data():
    return HQRCData(
        observations=np.arange(4, dtype=float),
        occurrence_index=np.array([0, 0, 1, 1]),
        holiday_type_index=np.array([0, 0, 1, 1]),
        tau_days=np.array([0.0, 1 / 24, 0.0, 1 / 24]),
        hour=np.array([0, 1, 0, 1]),
        restriction=np.array([0, 0, 1, 1]),
        occurrence_ids=("a", "b"),
    )


def test_hqrc_data_artifact_round_trip_and_detects_metadata_tamper(tmp_path):
    npz, metadata = write_hqrc_data(tmp_path / "hqrc.npz", _data(), settings={"variant": "H3"})
    loaded, settings = load_hqrc_data(npz, metadata)
    assert loaded.occurrence_ids == ("a", "b")
    assert settings == {"variant": "H3"}
    payload = json.loads(metadata.read_text())
    payload["arrays"]["observations"] = "other"
    metadata.write_text(json.dumps(payload))
    with pytest.raises(HQRCArtifactError, match="digest|schema"):
        load_hqrc_data(npz, metadata)


def test_hqrc_data_artifact_rejects_unknown_npz_array(tmp_path):
    npz, metadata = write_hqrc_data(tmp_path / "hqrc.npz", _data(), settings={})
    np.savez_compressed(npz, observations=np.ones(4), extra=np.ones(4))
    with pytest.raises(HQRCArtifactError, match="digest"):
        load_hqrc_data(npz, metadata)
