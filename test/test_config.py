"""
Tests for the configuration module.
"""

import os
import tempfile
from pathlib import Path
import pytest

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


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def clean_env(monkeypatch):
    """Remove SLACGS environment variables."""
    env_vars = [
        "SLACGS_OUTPUT_DIR",
        "SLACGS_LOG_LEVEL",
        "SLACGS_SEED",
        "SLACGS_N_JOBS",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)


class TestDefaultConfig:
    """Test default configuration."""
    
    def test_default_config_structure(self):
        """Default config should have all required sections."""
        assert "paths" in DEFAULT_CONFIG
        assert "experiment" in DEFAULT_CONFIG
        assert "logging" in DEFAULT_CONFIG
        assert "publishing" in DEFAULT_CONFIG
    
    def test_default_paths(self):
        """Test default path configuration."""
        assert DEFAULT_CONFIG["paths"]["output_dir"] == "./output"
        assert DEFAULT_CONFIG["paths"]["reports_dir"] is None
        assert DEFAULT_CONFIG["paths"]["data_dir"] is None
    
    def test_default_experiment(self):
        """Test default experiment configuration."""
        assert DEFAULT_CONFIG["experiment"]["seed"] is None
        assert DEFAULT_CONFIG["experiment"]["n_jobs"] == -1
        assert DEFAULT_CONFIG["experiment"]["verbose"] is True
    
    def test_default_logging(self):
        """Test default logging configuration."""
        assert DEFAULT_CONFIG["logging"]["level"] == "INFO"
        assert DEFAULT_CONFIG["logging"]["file"] is None
        assert DEFAULT_CONFIG["logging"]["quiet"] is False


class TestLoadConfig:
    """Test config loading with precedence."""
    
    def test_load_default_config(self, clean_env, temp_dir, monkeypatch):
        """Should load defaults when no config files exist."""
        monkeypatch.chdir(temp_dir)
        config = load_config()
        
        assert config["paths"]["output_dir"] == "./output"
        assert config["experiment"]["seed"] is None
        assert config["logging"]["level"] == "INFO"
    
    def test_load_project_config(self, clean_env, temp_dir, monkeypatch):
        """Should load project config (./slacgs.toml)."""
        monkeypatch.chdir(temp_dir)
        
        # Create project config
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text("""
[paths]
output_dir = "./custom_output"

[experiment]
seed = 42
""")
        
        config = load_config()
        assert config["paths"]["output_dir"] == "./custom_output"
        assert config["experiment"]["seed"] == 42
    
    def test_load_explicit_config(self, clean_env, temp_dir):
        """Should load explicitly specified config file."""
        config_file = temp_dir / "my_config.toml"
        config_file.write_text("""
[paths]
output_dir = "./explicit_output"
""")
        
        config = load_config(config_file=str(config_file))
        assert config["paths"]["output_dir"] == "./explicit_output"
    
    def test_env_override(self, temp_dir, monkeypatch):
        """Environment variables should override config files."""
        monkeypatch.chdir(temp_dir)
        
        # Create project config
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text("""
[paths]
output_dir = "./config_output"

[experiment]
seed = 42
n_jobs = 4
""")
        
        # Set environment variables
        monkeypatch.setenv("SLACGS_OUTPUT_DIR", "/env/output")
        monkeypatch.setenv("SLACGS_SEED", "99")
        monkeypatch.setenv("SLACGS_N_JOBS", "8")
        monkeypatch.setenv("SLACGS_LOG_LEVEL", "DEBUG")
        
        config = load_config()
        assert config["paths"]["output_dir"] == "/env/output"
        assert config["experiment"]["seed"] == 99
        assert config["experiment"]["n_jobs"] == 8
        assert config["logging"]["level"] == "DEBUG"
    
    def test_cli_override(self, clean_env, temp_dir, monkeypatch):
        """CLI overrides should have highest priority."""
        monkeypatch.chdir(temp_dir)
        
        # Create project config
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text("""
[paths]
output_dir = "./config_output"
""")
        
        # Set environment variable
        monkeypatch.setenv("SLACGS_OUTPUT_DIR", "/env/output")
        
        # CLI override
        cli_overrides = {
            "paths": {"output_dir": "/cli/output"},
            "experiment": {"seed": 123},
        }
        
        config = load_config(cli_overrides=cli_overrides)
        assert config["paths"]["output_dir"] == "/cli/output"
        assert config["experiment"]["seed"] == 123
    
    def test_config_file_not_found(self, clean_env):
        """Should raise error if explicit config file not found."""
        with pytest.raises(ConfigError, match="Config file not found"):
            load_config(config_file="/nonexistent/config.toml")
    
    def test_invalid_toml_syntax(self, temp_dir):
        """Should raise error for invalid TOML syntax."""
        config_file = temp_dir / "bad.toml"
        config_file.write_text("[paths\ninvalid toml")
        
        with pytest.raises(ConfigError, match="Invalid TOML syntax"):
            load_config(config_file=str(config_file))


class TestValidateConfig:
    """Test config validation."""
    
    def test_valid_config(self):
        """Should accept valid configuration."""
        from copy import deepcopy
        config = deepcopy(DEFAULT_CONFIG)
        validated = validate_config(config)
        assert validated == config
    
    def test_missing_section(self):
        """Should raise error for missing required section."""
        from copy import deepcopy
        config = deepcopy(DEFAULT_CONFIG)
        del config["paths"]
        
        with pytest.raises(ConfigError, match="Missing required config section"):
            validate_config(config)
    
    def test_invalid_log_level(self):
        """Should raise error for invalid log level."""
        from copy import deepcopy
        config = deepcopy(DEFAULT_CONFIG)
        config["logging"]["level"] = "INVALID"
        
        with pytest.raises(ConfigError, match="Invalid logging level"):
            validate_config(config)
    
    def test_log_level_case_insensitive(self):
        """Should normalize log level to uppercase."""
        from copy import deepcopy
        config = deepcopy(DEFAULT_CONFIG)
        config["logging"]["level"] = "debug"
        
        validated = validate_config(config)
        assert validated["logging"]["level"] == "DEBUG"
    
    def test_invalid_n_jobs(self):
        """Should raise error for invalid n_jobs."""
        from copy import deepcopy
        config = deepcopy(DEFAULT_CONFIG)
        config["experiment"]["n_jobs"] = 0
        
        with pytest.raises(ConfigError, match="Invalid n_jobs"):
            validate_config(config)
        
        config = deepcopy(DEFAULT_CONFIG)
        config["experiment"]["n_jobs"] = "invalid"
        with pytest.raises(ConfigError, match="Invalid n_jobs"):
            validate_config(config)
    
    def test_invalid_seed(self):
        """Should raise error for invalid seed."""
        from copy import deepcopy
        
        config = deepcopy(DEFAULT_CONFIG)
        config["experiment"]["seed"] = -1
        
        with pytest.raises(ConfigError, match="Invalid seed"):
            validate_config(config)
        
        config = deepcopy(DEFAULT_CONFIG)
        config["experiment"]["seed"] = "invalid"
        with pytest.raises(ConfigError, match="Invalid seed"):
            validate_config(config)
    
    def test_valid_seed_none(self):
        """Should accept None as valid seed."""
        from copy import deepcopy
        
        config = deepcopy(DEFAULT_CONFIG)
        config["experiment"]["seed"] = None
        
        validated = validate_config(config)
        assert validated["experiment"]["seed"] is None


class TestPathHelpers:
    """Test path helper functions."""
    
    def test_get_output_dir_default(self, clean_env, temp_dir, monkeypatch):
        """Should return default output dir."""
        monkeypatch.chdir(temp_dir)
        output_dir = get_output_dir()
        assert output_dir == (temp_dir / "output").resolve()
    
    def test_get_output_dir_custom(self, clean_env, temp_dir, monkeypatch):
        """Should return custom output dir from config."""
        monkeypatch.chdir(temp_dir)
        
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text('[paths]\noutput_dir = "./my_output"')
        
        output_dir = get_output_dir()
        assert output_dir == (temp_dir / "my_output").resolve()
    
    def test_get_output_dir_create(self, clean_env, temp_dir, monkeypatch):
        """Should create output dir when create=True."""
        monkeypatch.chdir(temp_dir)
        output_dir = get_output_dir(create=True)
        
        assert output_dir.exists()
        assert output_dir.is_dir()
    
    def test_get_reports_dir_default(self, clean_env, temp_dir, monkeypatch):
        """Should default to <output_dir>/reports."""
        monkeypatch.chdir(temp_dir)
        reports_dir = get_reports_dir()
        assert reports_dir == (temp_dir / "output" / "reports").resolve()
    
    def test_get_reports_dir_custom(self, clean_env, temp_dir, monkeypatch):
        """Should use custom reports dir from config."""
        monkeypatch.chdir(temp_dir)
        
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text('[paths]\nreports_dir = "./custom_reports"')
        
        reports_dir = get_reports_dir()
        assert reports_dir == (temp_dir / "custom_reports").resolve()
    
    def test_get_data_dir_default(self, clean_env, temp_dir, monkeypatch):
        """Should default to <output_dir>/data."""
        monkeypatch.chdir(temp_dir)
        data_dir = get_data_dir()
        assert data_dir == (temp_dir / "output" / "data").resolve()
    
    def test_get_data_dir_custom(self, clean_env, temp_dir, monkeypatch):
        """Should use custom data dir from config."""
        monkeypatch.chdir(temp_dir)
        
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text('[paths]\ndata_dir = "./custom_data"')
        
        data_dir = get_data_dir()
        assert data_dir == (temp_dir / "custom_data").resolve()
    
    def test_get_log_file_default(self, clean_env, temp_dir, monkeypatch):
        """Should default to <output_dir>/slacgs.log."""
        monkeypatch.chdir(temp_dir)
        log_file = get_log_file()
        assert log_file == (temp_dir / "output" / "slacgs.log").resolve()
    
    def test_get_log_file_custom(self, clean_env, temp_dir, monkeypatch):
        """Should use custom log file from config."""
        monkeypatch.chdir(temp_dir)
        
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text('[logging]\nfile = "./custom.log"')
        
        log_file = get_log_file()
        assert log_file == (temp_dir / "custom.log").resolve()
    
    def test_get_log_file_disabled(self, clean_env, temp_dir, monkeypatch):
        """Should return None when file logging disabled."""
        monkeypatch.chdir(temp_dir)
        
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text('[logging]\nfile = false')
        
        log_file = get_log_file()
        assert log_file is None


class TestConfigInit:
    """Test config file initialization."""
    
    def test_init_project_config(self, temp_dir, monkeypatch):
        """Should create project config template."""
        monkeypatch.chdir(temp_dir)
        
        config_path = init_project_config()
        assert config_path == temp_dir / "slacgs.toml"
        assert config_path.exists()
        
        content = config_path.read_text()
        assert "[paths]" in content
        assert "[experiment]" in content
        assert "[logging]" in content
        assert "[publishing]" in content
    
    def test_init_project_config_exists(self, temp_dir, monkeypatch):
        """Should raise error if project config exists."""
        monkeypatch.chdir(temp_dir)
        
        # Create existing config
        (temp_dir / "slacgs.toml").write_text("[paths]\noutput_dir = './old'")
        
        with pytest.raises(ConfigError, match="already exists"):
            init_project_config()
    
    def test_init_project_config_force(self, temp_dir, monkeypatch):
        """Should overwrite existing config with force=True."""
        monkeypatch.chdir(temp_dir)
        
        # Create existing config
        (temp_dir / "slacgs.toml").write_text("[paths]\noutput_dir = './old'")
        
        config_path = init_project_config(force=True)
        content = config_path.read_text()
        assert "seed = 42" in content  # New template content
    
    def test_init_user_config(self, temp_dir, monkeypatch):
        """Should create user config template."""
        # Mock user config directory
        def mock_user_config_dir(appname, **kwargs):
            return str(temp_dir / ".config" / appname)
        
        monkeypatch.setattr("slacgs.config.user_config_dir", mock_user_config_dir)
        
        config_path = init_user_config()
        assert config_path == temp_dir / ".config" / "slacgs" / "config.toml"
        assert config_path.exists()
        
        content = config_path.read_text()
        assert "SLACGS User Configuration" in content


class TestEnvironmentVariables:
    """Test environment variable handling."""
    
    def test_invalid_seed_env(self, clean_env, temp_dir, monkeypatch):
        """Should raise error for invalid SLACGS_SEED."""
        monkeypatch.chdir(temp_dir)
        monkeypatch.setenv("SLACGS_SEED", "not_a_number")
        
        with pytest.raises(ConfigError, match="Invalid SLACGS_SEED"):
            load_config()
    
    def test_invalid_n_jobs_env(self, clean_env, temp_dir, monkeypatch):
        """Should raise error for invalid SLACGS_N_JOBS."""
        monkeypatch.chdir(temp_dir)
        monkeypatch.setenv("SLACGS_N_JOBS", "not_a_number")
        
        with pytest.raises(ConfigError, match="Invalid SLACGS_N_JOBS"):
            load_config()


class TestConfigMerging:
    """Test configuration merging behavior."""
    
    def test_nested_merge(self, clean_env, temp_dir, monkeypatch):
        """Should merge nested dictionaries correctly."""
        monkeypatch.chdir(temp_dir)
        
        # Project config only sets some paths
        project_config = temp_dir / "slacgs.toml"
        project_config.write_text("""
[paths]
output_dir = "./custom"
# reports_dir and data_dir not specified
""")
        
        config = load_config()
        
        # Custom value from project config
        assert config["paths"]["output_dir"] == "./custom"
        
        # Default values still present
        assert config["paths"]["reports_dir"] is None
        assert config["paths"]["data_dir"] is None
        
        # Other sections untouched
        assert config["experiment"]["n_jobs"] == -1
        assert config["logging"]["level"] == "INFO"
