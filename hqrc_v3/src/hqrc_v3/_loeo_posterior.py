"""Central semantic validation for one-fold H3 posterior checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from hqrc_v3._loeo_types import LOEOFoldError, LOEOFoldInputs

_TRAILING_SHAPES = {
    "mu": (2, 3),
    "delta": (2, 3),
    "between_cholesky": (2, 3, 3),
    "between_scale": (2, 3),
    "between_cov_0": (6,),
    "between_cov_1": (6,),
    "beta_offset": (9, 3),
    "beta": (9, 3),
    "sigma_gamma": (2,),
    "gamma_innovation_raw": (2, 23),
    "gamma_innovation": (2, 23),
    "gamma": (2, 24),
    "u_phi": (),
    "phi": (),
    "sigma_r": (),
    "event_log_likelihood": (9,),
}


def posterior_mapping(idata: object) -> Mapping[str, object]:
    posterior = getattr(idata, "posterior", None)
    if posterior is None or not hasattr(posterior, "data_vars"):
        raise LOEOFoldError("LOEO posterior InferenceData is missing")
    return posterior


def validate_h3_posterior(idata: object, inputs: LOEOFoldInputs) -> Mapping[str, object]:
    """Fail closed unless a posterior is the exact finite H3 nine-event contract."""

    posterior = posterior_mapping(idata)
    if len(inputs.hqrc_data.occurrence_ids) != 9:
        raise LOEOFoldError("LOEO posterior requires exactly nine training events")
    try:
        chains = int(posterior.sizes["chain"])
        draws = int(posterior.sizes["draw"])
    except (KeyError, TypeError, ValueError) as error:
        raise LOEOFoldError("LOEO posterior sample dimensions differ") from error
    if chains <= 0 or draws <= 0 or not set(_TRAILING_SHAPES).issubset(posterior):
        raise LOEOFoldError("LOEO H3 posterior variables are incomplete")
    for name, trailing in _TRAILING_SHAPES.items():
        values = np.asarray(posterior[name])
        if values.shape != (chains, draws, *trailing) or not np.isfinite(values).all():
            raise LOEOFoldError(f"LOEO H3 posterior variable {name} differs")
    u_phi = np.asarray(posterior["u_phi"], dtype=float)
    phi = np.asarray(posterior["phi"], dtype=float)
    if (
        (u_phi <= 0).any()
        or (u_phi >= 1).any()
        or (np.abs(phi) >= 1).any()
        or not np.allclose(phi, 2.0 * u_phi - 1.0, rtol=1e-12, atol=1e-12)
    ):
        raise LOEOFoldError("LOEO posterior phi transform differs")
    if (np.asarray(posterior["sigma_r"]) <= 0).any() or (
        np.asarray(posterior["sigma_gamma"]) <= 0
    ).any():
        raise LOEOFoldError("LOEO posterior scale draws differ")
    cholesky = np.asarray(posterior["between_cholesky"])
    if (np.diagonal(cholesky, axis1=-2, axis2=-1) <= 0).any() or not np.allclose(
        cholesky, np.tril(cholesky), rtol=0.0, atol=1e-12
    ):
        raise LOEOFoldError("LOEO posterior full covariance draws differ")
    return posterior


__all__ = ["posterior_mapping", "validate_h3_posterior"]
