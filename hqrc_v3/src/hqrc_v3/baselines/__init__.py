"""Fixed-configuration forecasting baselines."""

from hqrc_v3.baselines.classical import (
    CandidateScore,
    FixedBaselineSelection,
    HorizonRegressor,
    make_classical_baseline,
    predictions_to_frame,
    select_fixed_baseline_config,
)
from hqrc_v3.baselines.config import PaperBaselineConfig, load_paper_baselines
from hqrc_v3.baselines.paper import (
    make_paper_factory,
    run_paper_final_stage,
    run_paper_oof_stage,
)
from hqrc_v3.baselines.protocol import BaselineFactory, FittedBaseline

__all__ = [
    "BaselineFactory",
    "CandidateScore",
    "FittedBaseline",
    "FixedBaselineSelection",
    "HorizonRegressor",
    "PaperBaselineConfig",
    "make_classical_baseline",
    "make_paper_factory",
    "predictions_to_frame",
    "load_paper_baselines",
    "run_paper_final_stage",
    "run_paper_oof_stage",
    "select_fixed_baseline_config",
]
