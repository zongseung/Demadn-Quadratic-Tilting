"""Interfaces shared by all forecasting baseline implementations."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from hqrc_v3.contracts import ForecastMatrix


class BaselineFactory(Protocol):
    """Creates a fitted baseline from a fixed, externally selected configuration."""

    name: str

    def fit(
        self, train: ForecastMatrix, validation: ForecastMatrix | None, seed: int
    ) -> FittedBaseline:
        """Fit only on ``train`` and return a model ready for evaluation."""


class FittedBaseline(Protocol):
    """A baseline fitted for one split, feature set, and seed."""

    model_name: str

    def predict(self, batch: ForecastMatrix) -> np.ndarray:
        """Return finite predictions shaped ``(n_samples, 24)``."""
