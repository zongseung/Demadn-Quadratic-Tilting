"""demand-quadratic-tilting: 한국 명절 수요 보정 패키지.

workalendar 기반 공휴일 전처리 + Hierarchical Quadratic Tilting (HQT).
"""

from .constants import CHUSEOK_LABELS, SEOLLAL_LABELS
from .metrics import evaluate_split, mae, rmse
from .model import HQTResult, compute_sigma_and_residuals, fit_hqt_pymc_lkj
from .pipeline import run_hqt_pipeline
from .preprocessing import annotate_holidays, filter_major_holidays
from .tilt import apply_tilt, tilt_from_posterior
from .windows import build_holiday_windows_and_tau

__all__ = [
    "CHUSEOK_LABELS",
    "SEOLLAL_LABELS",
    "HQTResult",
    "annotate_holidays",
    "apply_tilt",
    "build_holiday_windows_and_tau",
    "compute_sigma_and_residuals",
    "evaluate_split",
    "filter_major_holidays",
    "fit_hqt_pymc_lkj",
    "mae",
    "rmse",
    "run_hqt_pipeline",
    "tilt_from_posterior",
]
