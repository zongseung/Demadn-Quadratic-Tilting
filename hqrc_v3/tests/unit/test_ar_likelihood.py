from __future__ import annotations

import numpy as np
import pytest
from hqrc_v3.bayes.model import stationary_ar1_logp_numpy
from scipy import stats


def test_stationary_ar1_logp_resets_at_segments():
    segments = [np.array([1.0, 0.5]), np.array([2.0, 1.0])]
    got = stationary_ar1_logp_numpy(segments, phi=0.5, sigma=1.0)
    stationary_sd = 1.0 / np.sqrt(1.0 - 0.25)
    expected = sum(
        stats.norm.logpdf(segment[0], 0.0, stationary_sd)
        + stats.norm.logpdf(segment[1] - 0.5 * segment[0], 0.0, 1.0)
        for segment in segments
    )
    assert got == pytest.approx(expected)


@pytest.mark.parametrize(("phi", "sigma"), [(1.0, 1.0), (-1.0, 1.0), (0.5, 0.0)])
def test_stationary_ar1_logp_rejects_nonstationary_or_invalid_scale(phi, sigma):
    with pytest.raises(ValueError):
        stationary_ar1_logp_numpy([np.array([1.0, 0.5])], phi=phi, sigma=sigma)
