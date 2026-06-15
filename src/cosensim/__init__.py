"""
CoSenSim - A Simulator for Evaluating Cooperative Advantage in Sensor Networks

A scientific Python package for evaluating when combinations of sensor or
measurement channels become more useful than isolated channels or smaller
subsets. CoSenSim is an academic evolution of the SLACGS project.
"""

# Import from subpackages for backward compatibility
from cosensim.core import Model, Simulator
from cosensim.core.enumtypes import DictionaryType, LossType
from cosensim.reporting import Report, create_scenario_report

# Configuration module
from cosensim.config import (
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
from cosensim.logging_config import (
    setup_logging,
    setup_logging_from_config,
    get_logger,
    reset_logging,
    is_logging_configured,
)

# Utils module
from cosensim.utils import init_report_service_conf

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

__version__ = '0.1.0'
