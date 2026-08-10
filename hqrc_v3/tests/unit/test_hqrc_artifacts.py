import json
import threading
from pathlib import Path

import numpy as np
import pytest
from hqrc_v3.bayes import artifacts as artifacts_module
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


def test_concurrent_pair_publication_never_exposes_cross_generation(tmp_path, monkeypatch):
    destination = tmp_path / "hqrc.npz"
    first = _data()
    second = HQRCData(
        observations=first.observations + 100.0,
        occurrence_index=first.occurrence_index,
        holiday_type_index=first.holiday_type_index,
        tau_days=first.tau_days,
        hour=first.hour,
        restriction=first.restriction,
        occurrence_ids=("c", "d"),
    )
    npz_replaced = threading.Event()
    second_lock_attempted = threading.Event()
    reader_lock_attempted = threading.Event()
    release_first_writer = threading.Event()
    errors: list[BaseException] = []
    observed: list[tuple[HQRCData, dict[str, object]]] = []
    real_flock = artifacts_module.fcntl.flock
    real_replace = artifacts_module.os.replace

    def instrumented_flock(descriptor, operation):
        name = threading.current_thread().name
        if name == "writer-b" and operation == artifacts_module.fcntl.LOCK_EX:
            second_lock_attempted.set()
        if name == "reader" and operation == artifacts_module.fcntl.LOCK_SH:
            reader_lock_attempted.set()
        return real_flock(descriptor, operation)

    def interleaved_replace(source, target):
        real_replace(source, target)
        if threading.current_thread().name == "writer-a" and Path(target) == destination:
            npz_replaced.set()
            assert release_first_writer.wait(15), "writer-a release timed out"

    monkeypatch.setattr(artifacts_module.fcntl, "flock", instrumented_flock)
    monkeypatch.setattr(artifacts_module.os, "replace", interleaved_replace)

    def write(data, generation):
        try:
            write_hqrc_data(destination, data, settings={"generation": generation})
        except BaseException as error:
            errors.append(error)

    def read():
        try:
            observed.append(load_hqrc_data(destination))
        except BaseException as error:
            errors.append(error)

    writer_a = threading.Thread(
        target=write, args=(first, "a"), name="writer-a", daemon=True
    )
    writer_b = threading.Thread(
        target=write, args=(second, "b"), name="writer-b", daemon=True
    )
    reader = threading.Thread(target=read, name="reader", daemon=True)
    started: list[threading.Thread] = []
    saw_npz = saw_second_lock = saw_reader_lock = False
    try:
        writer_a.start()
        started.append(writer_a)
        saw_npz = npz_replaced.wait(5)
        if saw_npz:
            writer_b.start()
            reader.start()
            started.extend((writer_b, reader))
            saw_second_lock = second_lock_attempted.wait(5)
            saw_reader_lock = reader_lock_attempted.wait(5)
    finally:
        release_first_writer.set()
        for thread in started:
            thread.join(5)

    assert saw_npz, "writer-a did not reach the forced NPZ interleaving"
    assert saw_second_lock, "writer-b did not attempt the exclusive pair lock"
    assert saw_reader_lock, "reader did not attempt the shared pair lock"
    assert not [thread.name for thread in started if thread.is_alive()]

    assert not errors
    assert len(observed) == 1
    observed_data, observed_settings = observed[0]
    if observed_settings["generation"] == "a":
        assert observed_data.occurrence_ids == ("a", "b")
        assert observed_data.observations[0] == 0.0
    else:
        assert observed_settings["generation"] == "b"
        assert observed_data.occurrence_ids == ("c", "d")
        assert observed_data.observations[0] == 100.0
    final_data, final_settings = load_hqrc_data(destination)
    assert final_settings == {"generation": "b"}
    assert final_data.occurrence_ids == ("c", "d")
    assert final_data.observations[0] == 100.0
