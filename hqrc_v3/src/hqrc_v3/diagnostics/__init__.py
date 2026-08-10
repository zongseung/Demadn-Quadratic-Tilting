"""Diagnostics contracts for residual-based HQRC calibration."""

from hqrc_v3.diagnostics.ar import (
    ApprovedARCalibration,
    ARCalibration,
    ARCalibrationError,
    EventARDiagnostic,
    EventResidualContext,
    approve_calibration,
    calibrate_beta_prior,
    detrend_event_residuals,
    diagnose_event_residuals,
    estimate_event_phi,
    estimate_event_phis,
    load_approved_calibration,
    require_approved_calibration,
    validate_event_residual_context,
    write_ar_diagnostics,
)

__all__ = [
    "ARCalibration",
    "ARCalibrationError",
    "ApprovedARCalibration",
    "EventARDiagnostic",
    "EventResidualContext",
    "approve_calibration",
    "calibrate_beta_prior",
    "detrend_event_residuals",
    "diagnose_event_residuals",
    "estimate_event_phi",
    "estimate_event_phis",
    "load_approved_calibration",
    "require_approved_calibration",
    "validate_event_residual_context",
    "write_ar_diagnostics",
]
