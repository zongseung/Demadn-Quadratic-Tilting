"""Shared data contracts for the hourly forecasting pipeline."""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


class DataContractError(ValueError):
    """Raised when hourly input or its derived forecast samples are invalid."""


@dataclass(frozen=True)
class ForecastMatrix:
    """Daily samples issued at 00:00 with 168 observed and 24 target hours."""

    origins: np.ndarray
    target_times: np.ndarray
    history: np.ndarray
    future: np.ndarray
    target: np.ndarray
    history_columns: tuple[str, ...]
    future_columns: tuple[str, ...]

    def take(self, indices: np.ndarray) -> ForecastMatrix:
        """Return a sample-aligned subset without changing feature contracts."""

        return replace(
            self,
            origins=self.origins[indices],
            target_times=self.target_times[indices],
            history=self.history[indices],
            future=self.future[indices],
            target=self.target[indices],
        )
