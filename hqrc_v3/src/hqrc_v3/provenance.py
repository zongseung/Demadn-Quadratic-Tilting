"""Hash and compatibility contracts for HQRC v3 artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path


class ArtifactMismatch(ValueError):
    """Raised when an artifact was produced from different immutable inputs."""


def file_sha256(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file's raw bytes."""

    digest = sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class RunManifest:
    """The content identities required to reuse a generated artifact."""

    data_sha256: str
    config_sha256: str
    event_sha256: str


def assert_artifact_compatible(
    manifest: RunManifest,
    *,
    data_sha256: str,
    config_sha256: str,
    event_sha256: str,
) -> None:
    """Raise if an artifact manifest does not match its candidate inputs."""

    for field, expected, actual in (
        ("data_sha256", manifest.data_sha256, data_sha256),
        ("config_sha256", manifest.config_sha256, config_sha256),
        ("event_sha256", manifest.event_sha256, event_sha256),
    ):
        if expected != actual:
            raise ArtifactMismatch(f"{field} differs from the artifact manifest")
