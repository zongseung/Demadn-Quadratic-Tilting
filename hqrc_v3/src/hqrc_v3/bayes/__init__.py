"""Hierarchical Bayesian HQRC model and sampling contracts."""

from hqrc_v3.bayes.artifacts import HQRCArtifactError, load_hqrc_data, write_hqrc_data
from hqrc_v3.bayes.benchmark import (
    SamplerWorkerError,
    SamplerWorkerResult,
    benchmark_sampler_processes,
    maximum_posterior_mean_distance,
    run_sampler_worker,
    write_sampler_request,
)
from hqrc_v3.bayes.model import (
    HQRCData,
    HQRCModelOptions,
    build_hqrc_model,
    stationary_ar1_logp_numpy,
)
from hqrc_v3.bayes.predictive import (
    PredictiveShapeError,
    baseline_bootstrap_draws,
    corrected_predictive_draws,
    draw_new_event_correction,
    simulate_stationary_ar1,
    simulate_stationary_ar2,
    simulate_student_t_ar1,
)
from hqrc_v3.bayes.samplers import (
    SamplingDiagnostics,
    SamplingError,
    sample_hqrc,
    validate_inference_data,
)

__all__ = [
    "HQRCData",
    "HQRCArtifactError",
    "HQRCModelOptions",
    "PredictiveShapeError",
    "SamplingDiagnostics",
    "SamplingError",
    "SamplerWorkerError",
    "SamplerWorkerResult",
    "baseline_bootstrap_draws",
    "benchmark_sampler_processes",
    "build_hqrc_model",
    "corrected_predictive_draws",
    "draw_new_event_correction",
    "load_hqrc_data",
    "maximum_posterior_mean_distance",
    "run_sampler_worker",
    "sample_hqrc",
    "simulate_stationary_ar1",
    "simulate_stationary_ar2",
    "simulate_student_t_ar1",
    "stationary_ar1_logp_numpy",
    "validate_inference_data",
    "write_hqrc_data",
    "write_sampler_request",
]
