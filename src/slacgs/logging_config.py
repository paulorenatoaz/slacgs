"""
Logging configuration for SLACGS.

Provides structured logging with:
- Rotating file handler (10MB max, 5 backups)
- Rich console handler with colors
- Configurable log levels (global and per-module)
- Integration with config system
- Support for quiet mode and log file customization
- Session metadata logging
- Exception tracking

Example:
    >>> from slacgs.logging_config import setup_logging, get_logger
    >>> setup_logging(level="INFO", log_file="./output/logs/slacgs.log")
    >>> logger = get_logger(__name__)
    >>> logger.info("Simulation started")
    
    >>> # With context manager for timing
    >>> from slacgs.logging_config import log_time
    >>> with log_time("simulation"):
    >>>     run_simulation()
"""

import logging
import sys
import time
from pathlib import Path
from logging.handlers import RotatingFileHandler
from typing import Optional, Union
from contextlib import contextmanager

try:
    from rich.console import Console
    from rich.logging import RichHandler
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


# Global state to track if logging has been configured
_LOGGING_CONFIGURED = False


def setup_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[Union[str, Path]] = None,
    quiet: bool = False,
    no_color: bool = False,
    force_reconfigure: bool = False,
) -> None:
    """
    Configure logging for SLACGS.

    This should be called once at the start of the application (e.g., in CLI entry point).
    Multiple calls are safe - will only configure once unless force_reconfigure=True.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL) or int (10-50)
        log_file: Path to log file. If provided, enables file logging with rotation.
        quiet: If True, suppresses console output (file logging still works)
        no_color: If True, disables colored console output
        force_reconfigure: If True, reconfigures even if already configured

    Example:
        >>> setup_logging(level="DEBUG", log_file="./output/logs/slacgs.log")
        >>> setup_logging(level="WARNING", quiet=True)  # Only errors to console
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED and not force_reconfigure:
        return

    # Normalize log level
    if isinstance(level, str):
        level = level.upper()
        level = getattr(logging, level, logging.INFO)

    # Get root logger
    root_logger = logging.getLogger("slacgs")
    root_logger.setLevel(level)
    root_logger.handlers.clear()  # Remove existing handlers

    # Define log format
    # File format: timestamp, level, module, message
    file_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    file_formatter = logging.Formatter(file_format, datefmt="%Y-%m-%d %H:%M:%S")

    # Console format: simpler, rich handles timestamp/level if available
    console_format = "%(message)s"
    console_formatter = logging.Formatter(console_format)

    # Add console handler (unless quiet mode)
    if not quiet:
        if RICH_AVAILABLE and not no_color:
            # Use Rich for beautiful colored output
            console_handler = RichHandler(
                console=Console(stderr=True),
                show_time=True,
                show_path=False,
                rich_tracebacks=True,
                tracebacks_show_locals=False,
                markup=True,
            )
            console_handler.setFormatter(logging.Formatter("%(message)s"))
        else:
            # Fallback to standard StreamHandler
            console_handler = logging.StreamHandler(sys.stderr)
            console_format_full = "%(asctime)s | %(levelname)-8s | %(message)s"
            console_handler.setFormatter(
                logging.Formatter(console_format_full, datefmt="%H:%M:%S")
            )

        console_handler.setLevel(level)
        root_logger.addHandler(console_handler)

    # Add file handler with rotation (if log_file specified)
    if log_file:
        log_path = Path(log_file)
        
        # Create log directory if it doesn't exist
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler: 10MB max, 5 backups
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    # Prevent propagation to root logger to avoid duplicate logs
    root_logger.propagate = False

    _LOGGING_CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a specific module.

    Args:
        name: Logger name (typically __name__ of the calling module)

    Returns:
        Logger instance under the 'slacgs' namespace

    Example:
        >>> logger = get_logger(__name__)
        >>> logger.info("Starting simulation")
    """
    # Ensure logger is under 'slacgs' namespace
    if not name.startswith("slacgs"):
        name = f"slacgs.{name}"
    return logging.getLogger(name)


def setup_logging_from_config(
    config: Optional[dict] = None,
    cli_overrides: Optional[dict] = None,
    force_reconfigure: bool = False,
) -> None:
    """
    Configure logging using config dict and CLI overrides.

    Priority order (highest to lowest):
    1. CLI arguments (cli_overrides)
    2. Config dict (from config.toml)
    3. Environment variables (SLACGS_LOG_LEVEL)
    4. Defaults

    Args:
        config: Configuration dictionary (from load_config())
        cli_overrides: CLI arguments dict with keys: log_level, log_file, quiet, no_color

    Example:
        >>> from slacgs.config import load_config
        >>> config = load_config()
        >>> setup_logging_from_config(config, {"log_level": "DEBUG", "quiet": False})
    """
    import os
    from slacgs.config import get_log_file

    # Default values
    level = "INFO"
    log_file = None
    quiet = False
    no_color = False
    module_levels = {}

    # 1. Get from config dict
    if config:
        logging_config = config.get("logging", {})
        level = logging_config.get("level", level)
        
        # Get log file path using helper (Phase 1)
        log_file_path = get_log_file(config, create_dir=True)
        if log_file_path:
            log_file = str(log_file_path)
        
        quiet = logging_config.get("quiet", quiet)
        no_color = logging_config.get("no_color", no_color)
        
        # Phase 3: Get module-specific log levels
        module_levels = logging_config.get("levels", {})

    # 2. Check environment variables
    env_level = os.environ.get("SLACGS_LOG_LEVEL")
    if env_level:
        level = env_level

    # 3. Apply CLI overrides (highest priority)
    if cli_overrides:
        if "log_level" in cli_overrides and cli_overrides["log_level"]:
            level = cli_overrides["log_level"]
        if "log_file" in cli_overrides and cli_overrides["log_file"]:
            log_file = cli_overrides["log_file"]
        if "quiet" in cli_overrides:
            quiet = cli_overrides["quiet"]
        if "no_color" in cli_overrides:
            no_color = cli_overrides["no_color"]

    # Setup logging with final values
    setup_logging(
        level=level,
        log_file=log_file,
        quiet=quiet,
        no_color=no_color,
        force_reconfigure=force_reconfigure,
    )
    
    # Phase 3: Apply module-specific log levels
    if module_levels:
        for module_name, module_level in module_levels.items():
            module_logger = logging.getLogger(module_name)
            if isinstance(module_level, str):
                module_level = getattr(logging, module_level.upper(), logging.INFO)
            module_logger.setLevel(module_level)
    
    # Phase 4: Setup exception logging
    setup_exception_logging()


def reset_logging() -> None:
    """
    Reset logging configuration.

    Useful for testing or when you need to reconfigure logging.
    """
    global _LOGGING_CONFIGURED
    
    root_logger = logging.getLogger("slacgs")
    root_logger.handlers.clear()
    root_logger.setLevel(logging.WARNING)
    
    _LOGGING_CONFIGURED = False


def is_logging_configured() -> bool:
    """
    Check if logging has been configured.

    Returns:
        True if setup_logging() has been called, False otherwise
    """
    return _LOGGING_CONFIGURED


# Phase 2: Structured logging helpers
def log_session_start(command: str, version: str, output_dir: str, **kwargs) -> None:
    """
    Log session start metadata.
    
    Args:
        command: Command being executed (e.g., 'run-experiment')
        version: SLACGS version
        output_dir: Output directory path
        **kwargs: Additional metadata to log
    """
    logger = get_logger("slacgs")
    logger.info("=" * 70)
    logger.info(f"SLACGS v{version} - Session started")
    logger.info(f"Command: {command}")
    logger.info(f"Output directory: {output_dir}")
    
    for key, value in kwargs.items():
        logger.info(f"{key.replace('_', ' ').title()}: {value}")
    
    logger.info("=" * 70)


def log_session_end(elapsed_time: float = None) -> None:
    """
    Log session end.
    
    Args:
        elapsed_time: Total elapsed time in seconds
    """
    logger = get_logger("slacgs")
    logger.info("=" * 70)
    if elapsed_time:
        logger.info(f"Session completed in {elapsed_time:.2f}s")
    else:
        logger.info("Session completed")
    logger.info("=" * 70)


# Phase 4: Context manager for timing operations
@contextmanager
def log_time(operation: str, level: str = "INFO"):
    """
    Context manager for logging operation timing.
    
    Args:
        operation: Description of the operation being timed
        level: Log level for the timing message (DEBUG, INFO, WARNING, etc.)
        
    Example:
        >>> with log_time("data loading"):
        >>>     data = load_large_dataset()
        >>> # Logs: "Starting: data loading" and "Completed: data loading (12.34s)"
    """
    logger = get_logger("slacgs")
    log_func = getattr(logger, level.lower(), logger.info)
    
    start = time.time()
    log_func(f"Starting: {operation}")
    
    try:
        yield
    except Exception as e:
        elapsed = time.time() - start
        logger.error(f"Failed: {operation} ({elapsed:.2f}s) - {e}")
        raise
    else:
        elapsed = time.time() - start
        log_func(f"Completed: {operation} ({elapsed:.2f}s)")


# Phase 4: Exception tracking
def setup_exception_logging() -> None:
    """
    Setup global exception handler to log uncaught exceptions.
    
    This captures exceptions that would otherwise only print to stderr.
    """
    def exception_handler(exc_type, exc_value, exc_traceback):
        # Ignore KeyboardInterrupt so Ctrl+C works normally
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        
        logger = get_logger("slacgs")
        logger.critical(
            "Uncaught exception",
            exc_info=(exc_type, exc_value, exc_traceback)
        )
    
    sys.excepthook = exception_handler


# Convenience functions for quick logging without logger instances
def log_info(message: str) -> None:
    """Quick info log."""
    get_logger("slacgs").info(message)


def log_warning(message: str) -> None:
    """Quick warning log."""
    get_logger("slacgs").warning(message)


def log_error(message: str) -> None:
    """Quick error log."""
    get_logger("slacgs").error(message)


def log_debug(message: str) -> None:
    """Quick debug log."""
    get_logger("slacgs").debug(message)


def log_error(message: str) -> None:
    """Quick error log."""
    get_logger("slacgs").error(message)


def log_debug(message: str) -> None:
    """Quick debug log."""
    get_logger("slacgs").debug(message)
