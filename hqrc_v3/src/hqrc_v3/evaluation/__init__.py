"""Event-level metrics and inference utilities."""

from hqrc_v3.evaluation.data_benchmark import (
    DataBenchmarkError,
    benchmark_polars_data,
    load_data_benchmark,
    load_data_benchmark_request,
    load_data_benchmark_worker_result,
    run_data_benchmark_worker,
    write_data_benchmark_request,
)
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
    "DataBenchmarkError",
    "EventBootstrapResult",
    "HACDMResult",
    "InferenceContractError",
    "MetricContractError",
    "ReportContractError",
    "SamplerBenchmark",
    "WilcoxonEventResult",
    "bootstrap_event_median",
    "benchmark_polars_data",
    "build_report",
    "empirical_crps",
    "event_metric_frame",
    "hac_dm_test",
    "holm_adjust",
    "load_data_benchmark",
    "load_data_benchmark_request",
    "load_data_benchmark_worker_result",
    "point_metric_frame",
    "probabilistic_metric_frame",
    "run_synthetic_pipeline",
    "run_data_benchmark_worker",
    "sampler_eligible_default",
    "wilcoxon_event_test",
    "write_benchmark",
    "write_data_benchmark_request",
]
