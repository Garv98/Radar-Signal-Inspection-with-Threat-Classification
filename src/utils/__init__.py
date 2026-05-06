"""
Utility Modules

Production-grade utilities for:
- Structured logging
- Experiment tracking (MLflow)
- Uncertainty estimation
- Configuration validation
"""

from .logging_utils import (
    setup_logging,
    PerformanceMonitor,
    AuditLogger,
    timed
)
from .experiment_tracking import (
    ExperimentTracker,
    ModelRegistry,
    init_tracking
)
from .uncertainty import (
    MCDropoutWrapper,
    TemperatureScaling,
    EnsembleClassifier,
    UncertaintyEstimator,
    compute_expected_calibration_error
)

__all__ = [
    # Logging
    'setup_logging',
    'PerformanceMonitor',
    'AuditLogger',
    'timed',
    # Tracking
    'ExperimentTracker',
    'ModelRegistry',
    'init_tracking',
    # Uncertainty
    'MCDropoutWrapper',
    'TemperatureScaling',
    'EnsembleClassifier',
    'UncertaintyEstimator',
    'compute_expected_calibration_error'
]
