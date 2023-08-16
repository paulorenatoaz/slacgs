# SLACGS [![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://slacgs.netlify.app/)

A Simulator for Loss Analysis of Linear Classifiers on Gaussian Samples in order to evaluate Trade Off Between Samples and Features sizes in Classification Problems on gaussian Samples.

Documentation: https://slacgs.netlify.app/


* Reports with results will be stored in different Google Spreadsheet:  
    - Experiment Scenario
    - Custom Scenario
    - Custom Simulation.
* The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'                         
  owned by slacgs' google service account and shared with the user's Google Drive account.
* Also, images with data visualization will be exported to a local folder inside project's root folder ( slacgs/images/ )

* Reports Exported (Google Spreadsheets):
    - Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
    - Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
    - Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

* Images Exported (<user>/slacgs/images/ or /content/slacgs/images (for Gcolab) ):
    - Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
    - Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
    - Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

* Loss Functions:
    - Theoretical Loss: estimated using probability theory
    - Empirical Loss with Train Data: estimated using empirical approach with train data
    - Empirical Loss with Test Data: estimated using empirical approach with test data

# Experiment

[Download Experiment PDF](./experiment.pdf)

# Demo

1. Download and Install
2. Set/Start Report Service
3. Experiment Scenarios
4. Demo Functions:
    * Run an Experiment Simulation
      * run a simulation for one of the experiment scenarios and return True if there are still parameters to be simulated and False otherwise
    * Add a Simulation to an Experiment Scenario
      * add simulation results to one of the experiment scenario spreadsheets
    * Run a Custom Scenario
      * run a custom scenario and write the results to a Google Spreadsheet shared with the user
    * Add a Simulation to a Custom Scenario
      * add a simulation to a custom scenario spreadsheet
    * Run a Custom Simulation
      * run a custom simulation for any dimensionality and cardinality
    * Run All Experiment Simulations
      * run all simulations in all experiment scenarios


## 1. Download And Install

```bash
git clone https://github.com/paulorenatoaz/slacgs

pip install slacgs/
```

## 2. Set Report Service

```python
from slacgs.demo import *

## opt-1: set report service configuration with your own google cloud service account key file
path_to_google_cloud_service_account_api_key = 'path/to/key.json'
set_report_service_conf(path_to_google_cloud_service_account_api_key)

# opt-2 set report service configuration to use slacgs' server if you have the access password
set_report_service_conf()

```

## 3. Experiment Scenarios

```python
from slacgs.demo import print_experiment_scenarios

print_experiment_scenarios()
```

## 4. Demo Functions

```python
from slacgs.demo import *

## 1. Run an Experiment Simulation ##
run_experiment_simulation()
  
## 2. Add a Simulation to an Experiment Scenario Spreadsheet ##
### Scenario 1
scenario_number = 1
params = [1, 1, 2.1, 0, 0, 0]
add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

### Scenario 2
scenario_number = 2
params = [1, 1, 2, -0.15, 0, 0]
add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

### Scenario 3
scenario_number = 3
params = [1, 1, 2, 0, 0.15, 0.15]
add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

### Scenario 4
scenario_number = 4
params = [1, 1, 2, -0.1, 0.15, 0.15]
add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

## 3. Run a Custom Scenario ##
scenario_list = [[1,1,3,round(0.1*rho,1),0,0] for rho in range(-5,6)]
scenario_number = 5
run_custom_scenario(scenario_list, scenario_number)
  
## 4. Add a Simulation to a Custom Scenario Spreadsheet ##
params = (1, 1, 3, -0.7, 0, 0)
scenario_number = 5
add_simulation_to_custom_scenario_spreadsheet(params, scenario_number)

## 5. Run a Custom Simulation ##
### 2 features
params = [1, 2, 0.4]
run_custom_simulation(params)

### 3 features
params = [1, 1, 4, -0.2, 0.1, 0.1]
run_custom_simulation(params)

### 4 features
params = [1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
run_custom_simulation(params)

### 5 features
params = [1, 1, 2, 2, 2, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.2, 0, 0, 0]
dims_to_compare = (2, 5)
run_custom_simulation(params, dims_to_compare)

### 6 features
params = [1, 2, 3, 4, 5, 6, -0.3, -0.3, -0.2, -0.2, -0.1, -0.1, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4]
run_custom_simulation(params)

## 6. Run All Experiment Simulations ##
run_experiment()

```

### 4.2 Demo Test Functions (simulations running at 1% of its default number of iterations)

```python
from slacgs.demo import *


## 1. Run an Experiment Simulation ##
run_experiment_simulation_test()
  
## 2. Add a Simulation to an Experiment Scenario Spreadsheet ##
### scenario 1
scenario_number = 1
params = [1, 1, 2.1, 0, 0, 0]
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)

### scenario 2
scenario_number = 2
params = [1, 1, 2, -0.15, 0, 0]
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)

### scenario 3
scenario_number = 3
params = [1, 1, 2, 0, 0.15, 0.15]
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)

### scenario 4
scenario_number = 4
params = [1, 1, 2, -0.1, 0.15, 0.15]
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)
  
## 3. Run a Custom Scenario ##
scenario_list = [[1,1,3,round(0.1*rho,1),0,0] for rho in range(-5,6)]
scenario_number = 5
run_custom_scenario_test(scenario_list, scenario_number)
  
## 4. Add a Simulation to a Custom Scenario Spreadsheet ##
params = (1, 1, 3, -0.7, 0, 0)
scenario_number = 5
add_simulation_to_custom_scenario_spreadsheet_test(params, scenario_number)

## 5. Run a Custom Simulation ##
### 2 features
params = [1, 2, 0.4]
run_custom_simulation_test(params)

### 3 features
params = [1, 1, 4, -0.2, 0.1, 0.1]
run_custom_simulation_test(params)

### 4 features
params = [1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
run_custom_simulation_test(params)

### 5 features
params = [1, 1, 2, 2, 2, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.2, 0, 0, 0]
dims_to_compare = _test(2, 5)
run_custom_simulation_test(params, dims_to_compare)

### 6 features
params = [1, 2, 3, 4, 5, 6, -0.3, -0.3, -0.2, -0.2, -0.1, -0.1, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4]
run_custom_simulation_test(params)


## 6. Run All Experiment Simulations ##
run_experiment_test()

```
