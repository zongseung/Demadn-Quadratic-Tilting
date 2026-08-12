"""Canonical fixed options and stable identities for one-fold LOEO."""

from __future__ import annotations

import hashlib
import json

from hqrc_v3._loeo_types import LOEOFoldError
from hqrc_v3.bayes.model import HQRCModelOptions

MODEL_OPTIONS = HQRCModelOptions(
    covariance="full",
    include_restriction=True,
    innovation="normal_ar1",
)


def canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    except (TypeError, ValueError) as error:
        raise LOEOFoldError("LOEO fold metadata is not canonical JSON") from error


def sha_json(value: object) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def derive_loeo_seed(root_seed: int, label: str) -> int:
    """Derive a stable positive RNG seed from a root seed and canonical SHA-256 label."""

    if isinstance(root_seed, bool) or not isinstance(root_seed, int) or root_seed < 0:
        raise LOEOFoldError("LOEO root seed must be a non-negative integer")
    if not isinstance(label, str) or not label.strip():
        raise LOEOFoldError("LOEO seed label must be a nonblank string")
    digest = hashlib.sha256(
        canonical_json({"label": label, "root_seed": root_seed, "version": 1})
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1) + 1


__all__ = ["MODEL_OPTIONS", "canonical_json", "derive_loeo_seed", "sha_json"]
