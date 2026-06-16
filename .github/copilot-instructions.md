# CoInfoSim - AI Agent Instructions

**CoInfoSim** is a scientific Python package for simulating Loss Analysis of Linear Classifiers on Gaussian Samples. It evaluates the trade-off between sample size (n) and feature dimensionality (d) in classification problems.

## Architecture Overview

### Core Components
- **`Model`** (`src/coinfosim/core/model.py`): Defines Gaussian distributions with parameters `[σ₁, σ₂, ..., σₐ, ρ₁₂, ρ₁₃, ...]`
  - Standard deviations (σ) and correlations (ρ) between features
  - Validates covariance matrix is positive definite and symmetric
  - Minimum 3 params for 2D: `[σ₁, σ₂, ρ₁₂]`
- **`Simulator`** (`src/coinfosim/core/simulator.py`): Runs experiments across cardinalities N = [2, 4, 8, 16, ..., 1024+]
  - Computes three loss types: `THEORETICAL`, `EMPIRICAL_TRAIN`, `EMPIRICAL_TEST`
  - `test_mode=True` reduces computation by 10x (for rapid testing)
- **`Report`** (`src/coinfosim/reporting/report.py`): Generates HTML reports and JSON data files
  - Stores results in `~/coinfosim/output/` (configurable via `config.py`)

### Package Structure (src/coinfosim/)
```
core/           # Model, Simulator, enumtypes (DictionaryType, LossType)
reporting/      # Report generation (HTML, JSON, visualizations)
publish/        # GitHub Pages publishing automation
legacy/         # Deprecated Google Drive/Sheets code (DO NOT USE)
config.py       # TOML-based configuration (optional, defaults work)
demo.py         # Legacy demo functions (being replaced by CLI)
utils.py        # Helpers (report_service_conf, path utilities)
```

## Critical Workflows

### Running Simulations
1. **Single Simulation** (`demo_scripts/run_simulation.py`):
   ```python
   from coinfosim import Model, Simulator
   model = Model([1, 4, 0.6])  # σ₁=1, σ₂=4, ρ₁₂=0.6
   sim = Simulator(model, test_mode=False)
   sim.run()
   sim.report.write_to_json()
   sim.report.create_html_report()
   ```

2. **Batch Experiments** (`demo_scripts/run_experiment.py`):
   - Iterates through `SCENARIOS` (7 predefined parameter sets in `demo.py`)
   - Uses `is_param_in_simulation_reports()` to resume from previous runs (JSON-based checkpointing)
   - Auto-skips already-simulated parameters

3. **Publishing to GitHub Pages** (`scripts/publish_output_to_pages.sh` or `coinfosim.publish.publisher`):
   - Copies `output/reports/*.html` and `output/data/*.json` to `reports-pages` branch
   - Uses git worktree to avoid checkout conflicts

### Configuration System (Task 024 Complete)
- **Priority**: CLI args > Env vars > `./coinfosim.toml` > `~/.config/coinfosim/config.toml` > defaults
- **Key functions** (all in `config.py`):
  - `load_config()` - Merges all config sources
  - `get_output_dir(config, create=True)` - Returns Path object with optional lazy creation
  - `init_project_config(path)` - Creates `coinfosim.toml` template
- **Environment variables**: `COINFOSIM_OUTPUT_DIR`, `COINFOSIM_LOG_LEVEL`
- **Config is OPTIONAL** - defaults to `./output/` with sane settings
- **Example**: See `coinfosim.toml.example` for comprehensive documentation

### Testing
- Run tests: `pytest test/` (use `-v` for verbose)
- Test mode simulations: Pass `test_mode=True` to Simulator (10x faster, reduced precision)
- Config tests: 35 tests in `test/test_config.py` cover all precedence scenarios

## Project-Specific Conventions

### Development Setup Pattern
All `demo_scripts/*.py` use this idiom for local development:
```python
import os, sys
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)
```
**Why**: Allows running scripts without installing package (`pip install -e .` is preferred for AI agents)

### Parameter Format
- **2D**: `[σ₁, σ₂, ρ₁₂]` - e.g., `[1, 2, 0.5]`
- **3D**: `[σ₁, σ₂, σ₃, ρ₁₂, ρ₁₃, ρ₂₃]` - e.g., `[1, 1, 2, 0, 0.3, 0.3]`
- **Constraint**: `|ρ₁₃| < sqrt((1 + ρ₁₂) / 2)` - validated in Model.__init__
- **Test identifier**: Reports use `[test]` suffix in filenames when `test_mode=True`

### Import Guidelines
- **DO**: `from coinfosim import Model, Simulator, Report, load_config`
- **DON'T**: Import from `coinfosim.legacy.*` (deprecated Google Drive code)
- **DON'T**: Use wildcard imports (Task 021 - cleaned up)
- **Public API**: See `__all__` in `src/coinfosim/__init__.py` (16 exported symbols)

### Output Organization
```
output/
  reports/        # HTML report files (scenario_N_report.html, sim_report_id[params].html)
  data/           # JSON files (simulation_reports.json, simulation_reports_test.json)
    tables/       # CSV tables with loss values
```

## Active Development (See TASKS.md)

### Recently Completed (Tasks 020-024)
- ✅ Removed duplicate `__all__` and wildcard imports
- ✅ Added configuration system with TOML support (tomli/tomllib)
- ✅ Updated `setup.py` with CLI dependencies (typer, platformdirs)
- ✅ Split dependencies: core vs `extras_require['legacy']` vs `extras_require['dev']`

### In Progress (Tasks 025-027)
- 🚧 CLI implementation with `typer` (`src/coinfosim/cli.py` - not yet created)
  - Planned commands: `coinfosim run-experiment`, `coinfosim run-simulation`, `coinfosim make-report`, `coinfosim publish`, `coinfosim config`
- 🚧 Structured logging with rotation (`src/coinfosim/logging_config.py` - not yet created)

### Known Issues
- `demo.py` still has legacy Google Drive code (Task 090 - to be deprecated)
- Report class stores entire Simulator object (Task 031 - should use ReportData payload)

## Key Dependencies
- **Scientific**: numpy, scipy, scikit-learn, matplotlib, plotly, pandas
- **CLI/Config**: typer[all], tomli (Python <3.11), platformdirs
- **Legacy** (DO NOT USE in new code): pygsheets, google-api-python-client

## Multi-Workspace Context
This repo has a companion **`coinfosim-reports-pages`** repo (sibling directory) for GitHub Pages hosting. Publishing workflow syncs `output/` to that repo's `reports-pages` branch.

## When Editing Code
1. **Parameter validation**: Model constructor has strict checks - preserve them
2. **Test mode awareness**: Check `self.test_mode` flag reduces computational load
3. **Output paths**: Always use `report_service_conf` or new `config.py` functions, never hardcode paths
4. **JSON format**: Simulation results stored as JSON arrays in `output/data/simulation_reports.json`
5. **Reproducibility**: Configuration files (`coinfosim.toml`) are meant for version control

## Command Cheat Sheet
```bash
# Install package in development mode
pip install -e .[dev]

# Run single simulation
python demo_scripts/run_simulation.py

# Run batch experiment (resumes from JSON)
python demo_scripts/run_experiment.py

# Generate scenario reports from existing data
python demo_scripts/run_make_scenario_reports.py

# Publish to GitHub Pages
bash scripts/publish_output_to_pages.sh

# Run tests
pytest test/ -v

# Initialize config file
python -c "from coinfosim.config import init_project_config; init_project_config('.')"
```
