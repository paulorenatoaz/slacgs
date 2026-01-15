# SLACGS Configuration Design

**Design Philosophy:** Reproducibility > Convenience

Scientific research requires reproducible experiments. Configuration should be:
- Explicit and version-controllable
- Optional (sensible defaults always work)
- Simple and flat (no complex inheritance)
- CLI-friendly (scriptable and batch-processable)

---

## Configuration Priority Order

1. **CLI arguments** (highest) - One-off experiments
   ```bash
   slacgs run-experiment --output-dir ./results --log-level DEBUG
   ```

2. **Environment variables** - CI/CD and cluster configs
   ```bash
   export SLACGS_OUTPUT_DIR=/scratch/user/exp_001
   export SLACGS_LOG_LEVEL=DEBUG
   slacgs run-experiment
   ```

3. **Project config** - Version-controlled, reproducible
   ```bash
   # ./slacgs.toml committed to git
   slacgs run-experiment  # Uses ./slacgs.toml
   ```

4. **User config** - Personal defaults (optional)
   ```bash
   # ~/.config/slacgs/config.toml
   slacgs run-experiment  # Falls back to user config
   ```

5. **Built-in defaults** (lowest) - Always available
   ```bash
   slacgs run-experiment  # Works without any config!
   ```

---

## Config File Format: `slacgs.toml`

```toml
# Example slacgs.toml - Simple, flat, readable
# Place in project root for reproducible experiments

[paths]
output_dir = "./slacgs_output"  # Relative to project (reproducible)
reports_dir = "reports"          # Subdirectory of output_dir
data_dir = "data"                # Subdirectory of output_dir
images_dir = "images"            # Subdirectory of output_dir

[experiment]
# Default simulation parameters (can override via CLI)
n_samples = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
dimensions = [2, 3, 4]
loss_types = ["THEORETICAL", "EMPIRICAL_TRAIN", "EMPIRICAL_TEST"]
max_iterations = 10000
test_mode = false

[logging]
level = "INFO"               # DEBUG, INFO, WARNING, ERROR
file = "slacgs.log"          # Log file name (in output_dir)
format = "json"              # json or pretty
console = true               # Show logs in console
colors = true                # Colored console output

[publishing]
auto_push = false            # Auto-push to GitHub Pages
remote_branch = "reports-pages"
title = "SLACGS Reports"
```

---

## Usage Examples

### Minimal - Just Works
```bash
# No config needed, uses defaults
$ slacgs run-experiment
✓ Using default configuration
✓ Output: ./slacgs_output/
```

### Reproducible Research Project
```bash
# Step 1: Initialize project config
$ slacgs config init
✓ Created: slacgs.toml

# Step 2: Edit slacgs.toml for your experiment
$ nano slacgs.toml

# Step 3: Commit to git
$ git add slacgs.toml
$ git commit -m "Add experiment configuration"

# Step 4: Run (anyone can reproduce with same config)
$ slacgs run-experiment
✓ Using config: ./slacgs.toml
✓ Output: ./slacgs_output/
```

### Quick Exploration
```bash
# Override specific settings without editing config
$ slacgs run-experiment \
    --output-dir ~/experiments/quick_test \
    --dimensions "2,3" \
    --log-level DEBUG
```

### Cluster/HPC Usage
```bash
# Set output to scratch storage via environment
export SLACGS_OUTPUT_DIR=/scratch/$USER/experiment_001
export SLACGS_LOG_LEVEL=INFO

# Run batch jobs
sbatch run_experiment.sh  # Uses env vars
```

---

## Config Discovery Logic

```python
def find_config_file():
    """Find config file in priority order."""
    # 1. Project-local (best for reproducibility)
    if Path("slacgs.toml").exists():
        return Path("slacgs.toml")
    
    # 2. User config (personal defaults)
    user_config = platformdirs.user_config_path("slacgs") / "config.toml"
    if user_config.exists():
        return user_config
    
    # 3. No config found - use defaults
    return None


def load_config(cli_args=None):
    """Load config with proper precedence."""
    # Start with defaults
    config = DEFAULT_CONFIG.copy()
    
    # Layer 1: Load from file (if exists)
    config_file = find_config_file()
    if config_file:
        file_config = parse_toml(config_file)
        config.update(file_config)
    
    # Layer 2: Override with environment variables
    if os.getenv("SLACGS_OUTPUT_DIR"):
        config["paths"]["output_dir"] = os.getenv("SLACGS_OUTPUT_DIR")
    if os.getenv("SLACGS_LOG_LEVEL"):
        config["logging"]["level"] = os.getenv("SLACGS_LOG_LEVEL")
    
    # Layer 3: Override with CLI arguments (highest priority)
    if cli_args:
        if cli_args.output_dir:
            config["paths"]["output_dir"] = cli_args.output_dir
        if cli_args.log_level:
            config["logging"]["level"] = cli_args.log_level
    
    return config
```

---

## Directory Creation Strategy

**Lazy Creation (Not on Import):**

```python
# BAD: Creates directories on import
import slacgs  # Suddenly ~/slacgs/ exists!

# GOOD: Only create when needed
import slacgs
slacgs.run_experiment()  # Creates ./slacgs_output/ now
```

**Implementation:**
```python
def ensure_output_dirs(config):
    """Create output directories only when running simulation."""
    output_dir = Path(config["paths"]["output_dir"])
    (output_dir / config["paths"]["reports_dir"]).mkdir(parents=True, exist_ok=True)
    (output_dir / config["paths"]["data_dir"]).mkdir(parents=True, exist_ok=True)
    (output_dir / config["paths"]["images_dir"]).mkdir(parents=True, exist_ok=True)
```

---

## Benefits for Scientific Research

✅ **Reproducibility**
- Config files can be shared with papers
- Exact experiment parameters are documented
- Results can be reproduced by others

✅ **Flexibility**
- Quick experiments via CLI overrides
- Batch processing via environment variables
- Default behavior always works

✅ **Simplicity**
- Flat, readable TOML format
- No complex inheritance or magic
- Clear precedence rules

✅ **Version Control Friendly**
- Text-based format (TOML)
- Can diff configs in git
- Track experiment evolution over time

---

## Anti-Patterns Avoided

❌ **Required config files**
```bash
Error: No config.toml found!
```

❌ **Hidden magic**
```python
import slacgs  # Creates ~/.slacgs/ silently
```

❌ **Complex hierarchies**
```yaml
base: &defaults
  nested:
    deep:
      settings: value
```

❌ **Mutating globals**
```python
slacgs.config.OUTPUT_DIR = "/tmp"  # Side effects!
```

---

## Implementation Checklist (Task 024)

- [ ] Create `src/slacgs/config.py`
- [ ] Implement `load_config()` with proper precedence
- [ ] Implement `get_output_dir()` helper
- [ ] Implement `validate_config()` for type checking
- [ ] Create `DEFAULT_CONFIG` with sensible defaults
- [ ] Support tomllib (3.11+) with tomli fallback
- [ ] Add `slacgs config init` command to generate template
- [ ] Write tests for config loading and precedence
- [ ] Document in README.md
- [ ] Add example `slacgs.toml` to repository

---

**Next Steps:**
1. Implement `config.py` module (Task 024)
2. Implement `logging_config.py` (Task 025)
3. Create CLI with typer (Task 026+)
4. Test full workflow end-to-end
