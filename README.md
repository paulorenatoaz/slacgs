# SLACGS [![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://slacgs.netlify.app/)

SLACGS is a scientific Python package for simulating loss/error behavior of linear classifiers on Gaussian samples, with a focus on the trade-off between sample size ($n$) and feature dimensionality ($d$).

SLACGS supports arbitrary dimensionality ($d \ge 2$). The model is defined by a single parameter vector containing:

- $d$ standard deviations: $[\sigma_1,\ldots,\sigma_d]$
- $d(d-1)/2$ correlations in upper-triangular order: $[\rho_{12},\rho_{13},\ldots,\rho_{(d-1)d}]$

This package supported contributions to:

- the undergraduate thesis ["SLACGS: Simulator for Loss Analysis of Classifiers using Gaussian Samples"](./slacgs.pdf) by Paulo Azevedo (advisor: Daniel Menasché; co-advisor: João Pinheiro)
- the work ["Learning with Few Features and Samples"](./learning_with_few_features_and_samples.pdf) by João Pinheiro, Y.Z. Janice Chen, Paulo Azevedo, Daniel Menasché, and Don Towsley

## Updated SLACGS (current system)

The current SLACGS system is centered around a CLI + a clean data/reporting pipeline:

- CLI entrypoint: `slacgs` (or `python -m slacgs`) with commands to run simulations/experiments and generate reports.
- Configuration: optional TOML config with precedence **CLI args > env vars > ./slacgs.toml > ~/.config/slacgs/config.toml > defaults**.
- Outputs: simulation results are persisted as JSON and exported as HTML reports plus images/tables.
- Logging: structured logs with rotation; `cleanup-logs` command for retention.
- Reporting: `ReportData` data object decouples `Simulator` from `Report` (no circular dependency).

By default, output is written under `~/slacgs/output/` (configurable).

## Legacy note

Older Google Drive / Google Sheets integration is considered legacy and is not part of the recommended workflow. Legacy dependencies are optional via `slacgs[legacy]`.


# Experiment Description Available in the PDF

[Download Experiment PDF](./slacgs.pdf)


# Demo

1. Download and Install
2. Configure output/logging (optional)
3. Experiment Scenarios
4. Demo workflows (CLI + Python API)


## 1. Download And Install

```bash
pip install slacgs
```


Quick sanity check:

```bash
slacgs --help
# or
python -m slacgs --help
```


## 2. Configure Output & Logging (optional)

Configuration is optional; defaults work out of the box.

Create a project config template:

```bash
slacgs config init --project
slacgs config show
slacgs config validate
```

Common environment variables:

```bash
export SLACGS_OUTPUT_DIR="/path/to/output"
export SLACGS_LOG_LEVEL="INFO"
```


## 3. Predefined Experiment Scenarios

SLACGS ships with predefined scenarios (see `slacgs.demo.SCENARIOS`). You can run all scenarios or select a subset.

```bash
# Run all predefined scenarios
slacgs run-experiment

# Run scenarios 1, 2, 3
slacgs run-experiment --scenarios 1,2,3

# Fast exploratory run
slacgs run-experiment --scenarios 1 --test-mode
```


## 4. Workflows 

### 4.1 Run a single simulation

```bash
# 2D: [sigma1, sigma2, rho12]
slacgs run-simulation --params "[1,4,0.6]"

# 3D: [sigma1, sigma2, sigma3, rho12, rho13, rho23]
slacgs run-simulation --params "[1,1,2,0,0,0]" --test-mode

# 4D: [sigma1, sigma2, sigma3, sigma4, rho12, rho13, rho14, rho23, rho24, rho34]
slacgs run-simulation --params "[1,1,1,2,0,0,0,0,0,-0.1]" --test-mode
```

### 4.2 Run custom experiments

```bash
# Inline parameter sets
slacgs run-experiment --custom-params "[[1,1,-0.4], [1,1,0.4]]" --test-mode

# Load parameter sets from JSON
slacgs run-experiment --params-file my_scenario.json

# Organize an experiment under ~/slacgs/experiments/{tag}/
slacgs run-experiment --params-file my_scenario.json --tag custom_experiment_1
```

### 4.3 Generate reports from existing JSON

```bash
slacgs make-report --scenario 1
slacgs make-report --params "[1,4,0.6]"
```

### 4.4 Publishing (GitHub Pages)

```bash
slacgs publish
slacgs publish --auto-push
```

### 4.5 Log cleanup

```bash
slacgs cleanup-logs --older-than 30 --dry-run
slacgs cleanup-logs --older-than 30
```

### 4.6 Python API

```python
from slacgs import Model, Simulator

model = Model([1, 4, 0.6])
sim = Simulator(model, test_mode=True)
sim.run()

sim.report.save_graphs_png_images_files()
sim.report.create_report_tables()
sim.report.write_to_json()
sim.report.create_html_report()
```

### Output structure

By default, outputs go to `~/slacgs/output/`:

```text
~/slacgs/output/
  data/
    simulation_reports.json
    simulation_reports_test.json
    tables/
      sim_tables_id[...]/
  reports/
    sim_report_id[...].html
    scenario_1_report.html
    images/
      graphs/
      visualizations/
  logs/
    slacgs.log
```


