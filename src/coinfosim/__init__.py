"""
CoInfoSim - A Simulator for Cooperative Classification from Multiple Information Channels

A scientific Python package for evaluating when cooperation among information
channels improves supervised classification. CoInfoSim compares isolated
channels, channel pairs, and larger channel subsets through Monte Carlo
simulation of the average classification loss. It is an academic evolution of
the SLACGS and CoSenSim lines of work.
"""

# Import from subpackages for backward compatibility
from coinfosim.core import Model, Simulator
from coinfosim.core.enumtypes import DictionaryType, LossType
from coinfosim.reporting import Report, create_scenario_report

# Configuration module
from coinfosim.config import (
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
from coinfosim.logging_config import (
    setup_logging,
    setup_logging_from_config,
    get_logger,
    reset_logging,
    is_logging_configured,
)

# Utils module
from coinfosim.utils import init_report_service_conf

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
