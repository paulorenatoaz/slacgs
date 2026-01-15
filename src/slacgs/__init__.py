"""
SLACGS - Simulator for Loss Analysis of Linear Classifiers on Gaussian Samples

A scientific Python package for evaluating the trade-off between sample size
and feature dimensionality in classification problems on Gaussian-distributed data.
"""

# Import from subpackages for backward compatibility
from slacgs.core import Model, Simulator
from slacgs.core.enumtypes import DictionaryType, LossType
from slacgs.reporting import Report, create_scenario_report

# Configuration module
from slacgs.config import (
    load_config,
    validate_config,
    get_output_dir,
    get_reports_dir,
    get_data_dir,
    get_log_file,
    init_project_config,
    init_user_config,
    ConfigError,
    DEFAULT_CONFIG,
)

# Logging module
from slacgs.logging_config import (
    setup_logging,
    setup_logging_from_config,
    get_logger,
    reset_logging,
    is_logging_configured,
)

# Utils module
from slacgs.utils import init_report_service_conf

# Public API (backward compatible)
__all__ = [
    # Core simulation classes
    'Model',
    'Simulator',
    'DictionaryType',
    'LossType',
    # Reporting
    'Report',
    'create_scenario_report',
    # Configuration
    'load_config',
    'validate_config',
    'get_output_dir',
    'get_reports_dir',
    'get_data_dir',
    'get_log_file',
    'init_project_config',
    'init_user_config',
    'ConfigError',
    'DEFAULT_CONFIG',
    # Logging
    'setup_logging',
    'setup_logging_from_config',
    'get_logger',
    'reset_logging',
    'is_logging_configured',
    # Utils
    'init_report_service_conf',
]

__version__ = '0.2.0'
