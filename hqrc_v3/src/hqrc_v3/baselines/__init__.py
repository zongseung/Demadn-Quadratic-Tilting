"""Fixed-configuration forecasting baselines."""

from hqrc_v3.baselines.classical import (
    FixedBaselineSelection,
    HorizonRegressor,
    make_classical_baseline,
    predictions_to_frame,
    select_fixed_baseline_config,
)
from hqrc_v3.baselines.protocol import BaselineFactory, FittedBaseline

__all__ = [
    "BaselineFactory",
    "FittedBaseline",
    "FixedBaselineSelection",
    "HorizonRegressor",
    "make_classical_baseline",
    "predictions_to_frame",
    "select_fixed_baseline_config",
]
