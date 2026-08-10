"""Event-level metrics and inference utilities."""

from hqrc_v3.evaluation.inference import (
    EventBootstrapResult,
    HACDMResult,
    InferenceContractError,
    WilcoxonEventResult,
    bootstrap_event_median,
    hac_dm_test,
    holm_adjust,
    wilcoxon_event_test,
)
from hqrc_v3.evaluation.metrics import (
    MetricContractError,
    empirical_crps,
    event_metric_frame,
    point_metric_frame,
    probabilistic_metric_frame,
)
from hqrc_v3.evaluation.reports import (
    ReportContractError,
    SamplerBenchmark,
    build_report,
    run_synthetic_pipeline,
    sampler_eligible_default,
    write_benchmark,
)

__all__ = [
    "EventBootstrapResult",
    "HACDMResult",
    "InferenceContractError",
    "MetricContractError",
    "ReportContractError",
    "SamplerBenchmark",
    "WilcoxonEventResult",
    "bootstrap_event_median",
    "build_report",
    "empirical_crps",
    "event_metric_frame",
    "hac_dm_test",
    "holm_adjust",
    "point_metric_frame",
    "probabilistic_metric_frame",
    "run_synthetic_pipeline",
    "sampler_eligible_default",
    "wilcoxon_event_test",
    "write_benchmark",
]
