"""
Configuration management for SLACGS.

This module provides a layered configuration system optimized for scientific
reproducibility. Configuration sources are merged in priority order:

1. Command-line arguments (highest priority)
2. Environment variables
3. Project-local config file (./slacgs.toml)
4. User config file (~/.config/slacgs/config.toml)
5. Built-in defaults (lowest priority)

Design Principles:
- Config files are OPTIONAL - defaults provide sane behavior
- Project-local configs (./slacgs.toml) for reproducible experiments
- Environment variables for HPC/cluster environments
- Lazy directory creation (no side effects on import)
- Clear error messages for validation failures

Example:
    >>> from slacgs.config import load_config, get_output_dir
    >>> config = load_config()
    >>> output_dir = get_output_dir(config)
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional
from copy import deepcopy

# Use tomllib (Python 3.11+) or tomli (Python 3.6-3.10)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore


# Platform-specific config directory
try:
    from platformdirs import user_config_dir
except ImportError:
    # Fallback if platformdirs not installed
    def user_config_dir(appname: str, **kwargs) -> str:
        """Simple fallback for config directory."""
        if sys.platform == "win32":
            return os.path.join(os.environ.get("APPDATA", "~"), appname)
        elif sys.platform == "darwin":
            return os.path.expanduser(f"~/Library/Application Support/{appname}")
        else:  # Linux/Unix
            return os.path.expanduser(f"~/.config/{appname}")


# Default configuration
DEFAULT_CONFIG: Dict[str, Any] = {
    "paths": {
        "output_dir": None,  # None means use ~/slacgs/output
        "reports_dir": None,  # None means <output_dir>/reports
        "data_dir": None,  # None means <output_dir>/data
    },
    "experiment": {
        "seed": None,  # None means random seed
        "n_jobs": -1,  # -1 means use all cores
        "verbose": True,
    },
    "logging": {
        "level": "INFO",
        "file": "slacgs.log",  # Relative to <output_dir>/logs/
        "quiet": False,  # Suppress console output
        "max_age_days": 30,  # Auto-delete logs older than 30 days
        "levels": {},  # Module-specific log levels: {"slacgs.core.simulator": "DEBUG"}
    },
    "publishing": {
        "enabled": False,
        "target_dir": "../slacgs-reports-pages",
        "auto_push": False,
    },
}


class ConfigError(Exception):
    """Raised when configuration is invalid or cannot be loaded."""
    pass


def _load_toml_file(path: Path) -> Dict[str, Any]:
    """
    Load a TOML file and return its contents as a dictionary.
    
    Args:
        path: Path to the TOML file
        
    Returns:
        Dictionary with config data
        
    Raises:
        ConfigError: If file cannot be parsed
    """
    if tomllib is None:
        raise ConfigError(
            "TOML support not available. Install tomli for Python < 3.11: "
            "pip install tomli"
        )
    
    try:
        with open(path, "rb") as f:
            return tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML syntax in {path}: {e}")
    except Exception as e:
        raise ConfigError(f"Failed to read config file {path}: {e}")


def _merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge override config into base config.
    
    Args:
        base: Base configuration dictionary
        override: Override configuration dictionary
        
    Returns:
        Merged configuration
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dicts
            result[key] = _merge_config(result[key], value)
        else:
            # Override value
            result[key] = value
    
    return result


def _get_user_config_path() -> Path:
    """Get path to user config file (~/.config/slacgs/config.toml)."""
    config_dir = user_config_dir("slacgs", roaming=True)
    return Path(config_dir) / "config.toml"


def _get_project_config_path() -> Path:
    """Get path to project-local config file (./slacgs.toml)."""
    return Path.cwd() / "slacgs.toml"


def _apply_env_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply environment variable overrides to config.
    
    Supported environment variables:
    - SLACGS_OUTPUT_DIR: Override paths.output_dir
    - SLACGS_LOG_LEVEL: Override logging.level
    - SLACGS_SEED: Override experiment.seed
    - SLACGS_N_JOBS: Override experiment.n_jobs
    
    Args:
        config: Base configuration
        
    Returns:
        Configuration with environment overrides applied
    """
    result = config.copy()
    
    if "SLACGS_OUTPUT_DIR" in os.environ:
        result["paths"]["output_dir"] = os.environ["SLACGS_OUTPUT_DIR"]
    
    if "SLACGS_LOG_LEVEL" in os.environ:
        result["logging"]["level"] = os.environ["SLACGS_LOG_LEVEL"]
    
    if "SLACGS_SEED" in os.environ:
        try:
            result["experiment"]["seed"] = int(os.environ["SLACGS_SEED"])
        except ValueError:
            raise ConfigError(
                f"Invalid SLACGS_SEED: {os.environ['SLACGS_SEED']} "
                "(must be an integer)"
            )
    
    if "SLACGS_N_JOBS" in os.environ:
        try:
            result["experiment"]["n_jobs"] = int(os.environ["SLACGS_N_JOBS"])
        except ValueError:
            raise ConfigError(
                f"Invalid SLACGS_N_JOBS: {os.environ['SLACGS_N_JOBS']} "
                "(must be an integer)"
            )
    
    return result


def load_config(
    config_file: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Load configuration from multiple sources with proper precedence.
    
    Priority order (highest to lowest):
    1. cli_overrides parameter
    2. Environment variables (SLACGS_*)
    3. Explicit config_file (if provided)
    4. Project config (./slacgs.toml)
    5. User config (~/.config/slacgs/config.toml)
    6. DEFAULT_CONFIG
    
    Args:
        config_file: Explicit path to config file (skips auto-discovery)
        cli_overrides: Dictionary with CLI argument overrides
        
    Returns:
        Merged configuration dictionary
        
    Raises:
        ConfigError: If config cannot be loaded or is invalid
        
    Example:
        >>> config = load_config()
        >>> config = load_config(config_file="experiment.toml")
        >>> config = load_config(cli_overrides={"paths": {"output_dir": "./out"}})
    """
    # Start with defaults (deep copy to avoid mutation)
    config = deepcopy(DEFAULT_CONFIG)
    
    # Layer 1: User config (~/.config/slacgs/config.toml)
    user_config_path = _get_user_config_path()
    if user_config_path.exists():
        user_config = _load_toml_file(user_config_path)
        config = _merge_config(config, user_config)
    
    # Layer 2: Project config (./slacgs.toml) or explicit file
    if config_file:
        # Explicit config file provided
        explicit_path = Path(config_file)
        if not explicit_path.exists():
            raise ConfigError(f"Config file not found: {config_file}")
        project_config = _load_toml_file(explicit_path)
        config = _merge_config(config, project_config)
    else:
        # Auto-discover project config
        project_config_path = _get_project_config_path()
        if project_config_path.exists():
            project_config = _load_toml_file(project_config_path)
            config = _merge_config(config, project_config)
    
    # Layer 3: Environment variables
    config = _apply_env_overrides(config)
    
    # Layer 4: CLI overrides (highest priority)
    if cli_overrides:
        config = _merge_config(config, cli_overrides)
    
    # Validate and return
    return validate_config(config)


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and ensure all required fields exist.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Validated configuration
        
    Raises:
        ConfigError: If configuration is invalid
    """
    # Check required sections
    required_sections = ["paths", "experiment", "logging", "publishing"]
    for section in required_sections:
        if section not in config:
            raise ConfigError(f"Missing required config section: [{section}]")
    
    # Validate logging level
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    log_level = config["logging"]["level"].upper()
    if log_level not in valid_levels:
        raise ConfigError(
            f"Invalid logging level: {config['logging']['level']}. "
            f"Must be one of: {', '.join(valid_levels)}"
        )
    config["logging"]["level"] = log_level
    
    # Validate n_jobs
    n_jobs = config["experiment"]["n_jobs"]
    if not isinstance(n_jobs, int) or n_jobs == 0:
        raise ConfigError(
            f"Invalid n_jobs: {n_jobs}. Must be -1 (all cores) or positive integer."
        )
    
    # Validate seed (if provided)
    seed = config["experiment"]["seed"]
    if seed is not None and (not isinstance(seed, int) or seed < 0):
        raise ConfigError(f"Invalid seed: {seed}. Must be None or non-negative integer.")
    
    return config


def get_output_dir(config: Optional[Dict[str, Any]] = None, create: bool = False) -> Path:
    """
    Get the output directory path.
    
    Args:
        config: Configuration dictionary (loads default if None)
        create: Create directory if it doesn't exist
        
    Returns:
        Path object for output directory
        
    Example:
        >>> output_dir = get_output_dir(create=True)
        >>> print(output_dir)
        PosixPath('/home/user/slacgs/output')
    """
    if config is None:
        config = load_config()
    
    output_dir_value = config["paths"]["output_dir"]
    
    # If None, use ~/slacgs/output (user home folder for visibility)
    if output_dir_value is None:
        output_dir = Path.home() / "slacgs" / "output"
    else:
        output_dir = Path(output_dir_value).resolve()
    
    if create and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    return output_dir


def get_reports_dir(config: Optional[Dict[str, Any]] = None, create: bool = False) -> Path:
    """
    Get the reports directory path.
    
    Defaults to <output_dir>/reports if not explicitly configured.
    
    Args:
        config: Configuration dictionary (loads default if None)
        create: Create directory if it doesn't exist
        
    Returns:
        Path object for reports directory
    """
    if config is None:
        config = load_config()
    
    reports_dir = config["paths"]["reports_dir"]
    
    if reports_dir is None:
        # Default to <output_dir>/reports
        reports_dir = get_output_dir(config) / "reports"
    else:
        reports_dir = Path(reports_dir).resolve()
    
    if create and not reports_dir.exists():
        reports_dir.mkdir(parents=True, exist_ok=True)
    
    return reports_dir


def get_data_dir(config: Optional[Dict[str, Any]] = None, create: bool = False) -> Path:
    """
    Get the data directory path.
    
    Defaults to <output_dir>/data if not explicitly configured.
    
    Args:
        config: Configuration dictionary (loads default if None)
        create: Create directory if it doesn't exist
        
    Returns:
        Path object for data directory
    """
    if config is None:
        config = load_config()
    
    data_dir = config["paths"]["data_dir"]
    
    if data_dir is None:
        # Default to <output_dir>/data
        data_dir = get_output_dir(config) / "data"
    else:
        data_dir = Path(data_dir).resolve()
    
    if create and not data_dir.exists():
        data_dir.mkdir(parents=True, exist_ok=True)
    
    return data_dir


def get_log_file(config: Optional[Dict[str, Any]] = None, create_dir: bool = True) -> Optional[Path]:
    """
    Get the log file path.
    
    Defaults to <output_dir>/logs/slacgs.log if not explicitly configured.
    
    Args:
        config: Configuration dictionary (loads default if None)
        create_dir: If True, create the log directory if it doesn't exist
        
    Returns:
        Path object for log file, or None if logging to file is disabled
        
    Examples:
        >>> log_file = get_log_file()  # Uses default config
        >>> # Returns: <output_dir>/logs/slacgs.log
        
        >>> config = {"logging": {"file": "/var/log/slacgs.log"}}
        >>> log_file = get_log_file(config)
        >>> # Returns: /var/log/slacgs.log (absolute path)
        
        >>> config = {"logging": {"file": False}}
        >>> log_file = get_log_file(config)
        >>> # Returns: None (file logging disabled)
    """
    if config is None:
        config = load_config()
    
    log_file = config["logging"]["file"]
    
    if log_file is None:
        # Default to <output_dir>/logs/slacgs.log
        log_path = get_output_dir(config) / "logs" / "slacgs.log"
    elif log_file is False:
        # Explicitly disabled
        return None
    else:
        log_path = Path(log_file)
        # If relative path, place in <output_dir>/logs/
        if not log_path.is_absolute():
            log_path = get_output_dir(config) / "logs" / log_file
        else:
            log_path = log_path.resolve()
    
    # Create log directory if requested
    if create_dir and log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    return log_path


def init_user_config(force: bool = False) -> Path:
    """
    Create a user config file template at ~/.config/slacgs/config.toml.
    
    Args:
        force: Overwrite existing file if True
        
    Returns:
        Path to created config file
        
    Raises:
        ConfigError: If file exists and force=False
    """
    config_path = _get_user_config_path()
    
    if config_path.exists() and not force:
        raise ConfigError(
            f"Config file already exists: {config_path}\n"
            "Use force=True to overwrite."
        )
    
    # Create directory
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write template
    template = """# SLACGS User Configuration
# This file provides default settings for all SLACGS projects.
# Project-specific settings should go in ./slacgs.toml

[paths]
# Base output directory (can be relative or absolute)
output_dir = "./output"

# Optional: Separate directories for reports and data
# If not set, defaults to <output_dir>/reports and <output_dir>/data
# reports_dir = "./reports"
# data_dir = "./data"

[experiment]
# Random seed for reproducibility (null = random)
seed = null

# Number of parallel jobs (-1 = all cores, 1 = sequential)
n_jobs = -1

# Verbose output during experiments
verbose = true

[logging]
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
level = "INFO"

# Log file path
# Uncomment to customize, or leave commented for default (<output_dir>/slacgs.log)
# file = "./custom.log"
# file = false  # Disable file logging

# Suppress console output (only log to file)
quiet = false

[publishing]
# Enable automatic publishing to GitHub Pages
enabled = false

# Target directory for published reports
target_dir = "../slacgs-reports-pages"

# Automatically git push after publishing
auto_push = false
"""
    
    config_path.write_text(template)
    return config_path


def init_project_config(force: bool = False) -> Path:
    """
    Create a project config file template at ./slacgs.toml.
    
    Args:
        force: Overwrite existing file if True
        
    Returns:
        Path to created config file
        
    Raises:
        ConfigError: If file exists and force=False
    """
    config_path = _get_project_config_path()
    
    if config_path.exists() and not force:
        raise ConfigError(
            f"Config file already exists: {config_path}\n"
            "Use force=True to overwrite."
        )
    
    # Write template
    template = """# SLACGS Project Configuration
# This file should be version-controlled with your project for reproducibility.
# It overrides user config (~/.config/slacgs/config.toml).

[paths]
output_dir = "./output"

[experiment]
# Set a fixed seed for reproducible results
seed = 42

# Use all available cores
n_jobs = -1

verbose = true

[logging]
level = "INFO"
# Uncomment to set custom log file:
# file = "./custom.log"
quiet = false

[publishing]
enabled = false
target_dir = "../slacgs-reports-pages"
auto_push = false
"""
    
    config_path.write_text(template)
    return config_path
