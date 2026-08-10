import hashlib
import json
import threading

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


def _second_data():
    first = _data()
    return HQRCData(
        observations=first.observations + 100.0,
        occurrence_index=first.occurrence_index,
        holiday_type_index=first.holiday_type_index,
        tau_days=first.tau_days,
        hour=first.hour,
        restriction=first.restriction,
        occurrence_ids=("c", "d"),
    )


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _current_pointer(destination):
    return destination.with_name(f"{destination.stem}.current.json")


def test_hqrc_data_artifact_round_trip_and_detects_metadata_tamper(tmp_path):
    npz, metadata = write_hqrc_data(tmp_path / "hqrc.npz", _data(), settings={"variant": "H3"})
    loaded, settings = load_hqrc_data(npz, metadata)
    assert loaded.occurrence_ids == ("a", "b")
    assert settings == {"variant": "H3"}
    assert npz.read_bytes().startswith(b"PK")
    payload = json.loads(metadata.read_text())
    assert payload["artifact_kind"] == "hqrc-data-generation"
    assert payload["generation"] == npz.stem == metadata.stem
    payload["arrays"]["observations"] = "other"
    metadata.write_text(json.dumps(payload))
    with pytest.raises(HQRCArtifactError, match="digest|schema"):
        load_hqrc_data(npz, metadata)


def test_hqrc_data_artifact_rejects_unknown_npz_array(tmp_path):
    npz, metadata = write_hqrc_data(tmp_path / "hqrc.npz", _data(), settings={})
    np.savez_compressed(npz, observations=np.ones(4), extra=np.ones(4))
    with pytest.raises(HQRCArtifactError, match="digest"):
        load_hqrc_data(npz, metadata)


def test_immutable_generation_rejects_boolean_version_after_redigest(tmp_path):
    npz, metadata = write_hqrc_data(tmp_path / "hqrc.npz", _data(), settings={})
    payload = json.loads(metadata.read_bytes())
    payload.pop("artifact_sha256")
    payload["schema_version"] = True
    payload["artifact_sha256"] = hashlib.sha256(_canonical(payload)).hexdigest()
    metadata.write_bytes(_canonical(payload) + b"\n")

    with pytest.raises(HQRCArtifactError, match="metadata version"):
        load_hqrc_data(npz, metadata)


def test_writer_rejects_generation_namespace_symlink_without_touching_outside(tmp_path):
    parent = tmp_path / "artifacts"
    outside = tmp_path / "outside"
    parent.mkdir()
    outside.mkdir()
    (parent / "sentinel.txt").write_bytes(b"parent")
    (outside / "sentinel.txt").write_bytes(b"outside")
    destination = parent / "hqrc.npz"
    namespace = parent / ".hqrc.generations"
    namespace.symlink_to(outside, target_is_directory=True)
    parent_before = sorted(path.name for path in parent.iterdir())
    outside_before = {
        path.name: path.read_bytes() for path in outside.iterdir() if path.is_file()
    }

    with pytest.raises(HQRCArtifactError, match="generation namespace"):
        write_hqrc_data(destination, _data(), settings={})

    assert sorted(path.name for path in parent.iterdir()) == parent_before
    assert {
        path.name: path.read_bytes() for path in outside.iterdir() if path.is_file()
    } == outside_before
    assert namespace.is_symlink()


def test_writer_rejects_regular_file_generation_namespace_without_mutation(tmp_path):
    parent = tmp_path / "artifacts"
    parent.mkdir()
    (parent / "sentinel.txt").write_bytes(b"parent")
    destination = parent / "hqrc.npz"
    namespace = parent / ".hqrc.generations"
    namespace.write_bytes(b"not-a-directory")
    parent_before = {
        path.name: path.read_bytes() for path in parent.iterdir() if path.is_file()
    }

    with pytest.raises(HQRCArtifactError, match="generation namespace"):
        write_hqrc_data(destination, _data(), settings={})

    assert {
        path.name: path.read_bytes() for path in parent.iterdir() if path.is_file()
    } == parent_before
    assert sorted(path.name for path in parent.iterdir()) == sorted(parent_before)


def test_returned_generation_tuple_stays_valid_after_current_pointer_advances(tmp_path):
    destination = tmp_path / "hqrc.npz"
    first_npz, first_metadata = write_hqrc_data(
        destination, _data(), settings={"generation": "a"}
    )
    second_npz, second_metadata = write_hqrc_data(
        destination, _second_data(), settings={"generation": "b"}
    )

    first_data, first_settings = load_hqrc_data(first_npz, first_metadata)
    second_data, second_settings = load_hqrc_data(second_npz, second_metadata)
    current_data, current_settings = load_hqrc_data(destination)
    assert first_settings == {"generation": "a"}
    assert first_data.occurrence_ids == ("a", "b")
    assert second_settings == current_settings == {"generation": "b"}
    assert second_data.occurrence_ids == current_data.occurrence_ids == ("c", "d")


@pytest.mark.parametrize(
    ("boundary", "expected_generation"),
    [
        ("before-generation-npz-publish", "a"),
        ("generation-npz-published", "a"),
        ("generation-metadata-published", "a"),
        ("generation-directory-synced", "a"),
        ("before-pointer-swap", "a"),
        ("pointer-swapped", "b"),
        ("parent-directory-synced", "b"),
    ],
)
def test_generation_pointer_failure_is_crash_consistent(
    tmp_path, monkeypatch, boundary, expected_generation
):
    destination = tmp_path / "hqrc.npz"
    write_hqrc_data(destination, _data(), settings={"generation": "a"})
    pointer_path = _current_pointer(destination)
    previous_pointer = pointer_path.read_bytes()

    def fail_at_publication_boundary(name):
        if name == boundary:
            raise RuntimeError(f"injected failure at {name}")

    monkeypatch.setattr(
        artifacts_module,
        "_publication_boundary",
        fail_at_publication_boundary,
        raising=False,
    )
    with pytest.raises(RuntimeError, match="injected failure"):
        write_hqrc_data(destination, _second_data(), settings={"generation": "b"})

    loaded, settings = load_hqrc_data(destination)
    assert settings == {"generation": expected_generation}
    expected_ids = ("a", "b") if expected_generation == "a" else ("c", "d")
    assert loaded.occurrence_ids == expected_ids
    if expected_generation == "a":
        assert pointer_path.read_bytes() == previous_pointer
    else:
        assert pointer_path.read_bytes() != previous_pointer


def test_generation_pointer_rejects_path_traversal_even_with_recomputed_digest(tmp_path):
    destination = tmp_path / "hqrc.npz"
    write_hqrc_data(destination, _data(), settings={"generation": "a"})
    pointer_path = _current_pointer(destination)
    pointer = json.loads(pointer_path.read_bytes())
    pointer.pop("pointer_digest")
    pointer["npz"] = "../escape.npz"
    pointer["pointer_digest"] = hashlib.sha256(_canonical(pointer)).hexdigest()
    pointer_path.write_bytes(_canonical(pointer) + b"\n")

    with pytest.raises(HQRCArtifactError, match="unsafe|pointer"):
        load_hqrc_data(destination)


def test_generation_pointer_rejects_boolean_version_even_with_recomputed_digest(tmp_path):
    destination = tmp_path / "hqrc.npz"
    write_hqrc_data(destination, _data(), settings={"generation": "a"})
    pointer_path = _current_pointer(destination)
    pointer = json.loads(pointer_path.read_bytes())
    pointer.pop("pointer_digest")
    pointer["schema_version"] = True
    pointer["pointer_digest"] = hashlib.sha256(_canonical(pointer)).hexdigest()
    pointer_path.write_bytes(_canonical(pointer) + b"\n")

    with pytest.raises(HQRCArtifactError, match="pointer identity"):
        load_hqrc_data(destination)


def test_logical_generation_lookup_fails_closed_when_current_pointer_is_missing(tmp_path):
    destination = tmp_path / "hqrc.npz"
    write_hqrc_data(destination, _data(), settings={"generation": "a"})
    _current_pointer(destination).unlink()

    with pytest.raises(HQRCArtifactError, match="pointer is missing"):
        load_hqrc_data(destination)


def test_concurrent_pointer_reader_sees_complete_generation_and_writer_terminates(
    tmp_path, monkeypatch
):
    destination = tmp_path / "hqrc.npz"
    write_hqrc_data(destination, _data(), settings={"generation": "a"})
    before_pointer_swap = threading.Event()
    release_writer = threading.Event()
    errors: list[BaseException] = []
    observed: list[tuple[HQRCData, dict[str, object]]] = []

    def pause_before_pointer(name):
        if name == "before-pointer-swap" and threading.current_thread().name == "writer":
            before_pointer_swap.set()
            assert release_writer.wait(15), "writer release timed out"

    monkeypatch.setattr(
        artifacts_module, "_publication_boundary", pause_before_pointer, raising=False
    )

    def write():
        try:
            write_hqrc_data(destination, _second_data(), settings={"generation": "b"})
        except BaseException as error:
            errors.append(error)

    def read():
        try:
            observed.append(load_hqrc_data(destination))
        except BaseException as error:
            errors.append(error)

    writer = threading.Thread(target=write, name="writer", daemon=True)
    reader = threading.Thread(target=read, name="reader", daemon=True)
    started: list[threading.Thread] = []
    reached_swap = reader_completed = False
    try:
        writer.start()
        started.append(writer)
        reached_swap = before_pointer_swap.wait(5)
        if reached_swap:
            reader.start()
            started.append(reader)
            reader.join(5)
            reader_completed = not reader.is_alive()
    finally:
        release_writer.set()
        for thread in started:
            thread.join(5)

    assert reached_swap, "writer did not reach the pointer commit boundary"
    assert reader_completed, "reader blocked on an unpublished generation"
    assert not [thread.name for thread in started if thread.is_alive()]
    assert not errors
    assert len(observed) == 1
    observed_data, observed_settings = observed[0]
    assert observed_settings == {"generation": "a"}
    assert observed_data.occurrence_ids == ("a", "b")
    final_data, final_settings = load_hqrc_data(destination)
    assert final_settings == {"generation": "b"}
    assert final_data.occurrence_ids == ("c", "d")
    assert final_data.observations[0] == 100.0
