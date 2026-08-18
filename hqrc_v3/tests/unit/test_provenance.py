import pytest

from hqrc_v3.provenance import (
    ArtifactMismatch,
    RunManifest,
    assert_artifact_compatible,
)


def test_manifest_rejects_changed_data_hash(tmp_path):
    manifest = RunManifest(data_sha256="aaa", config_sha256="bbb", event_sha256="ccc")

    with pytest.raises(ArtifactMismatch, match="data_sha256"):
        assert_artifact_compatible(
            manifest,
            data_sha256="changed",
            config_sha256="bbb",
            event_sha256="ccc",
        )
