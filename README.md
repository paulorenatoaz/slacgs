# SLACGS [![Documentation](https://img.shields.io/badge/docs-available-brightgreen)](https://slacgs.netlify.app/)

A Simulator for Loss Analysis of Linear Classifiers on Gaussian Samples in order to evaluate Trade Off Between Samples and Features sizes in Classification Problems on gaussian Samples.

This is a Python package developed for research purposses. This Simulator supported contributions to: 

 - the undergraduate thesis ["SLACGS: Simulator for Loss Analysis of Classifiers using Gaussian Samples"](./slacgs.pdf) by: Paulo Azevedo, supervised by: Daniel Menasché, and co-supervised by: Joao Pinheiro
 - the work ["Learning with Few Features and Samples"](./learning_with_few_features_and_samples.pdf), by: Joao Pinheiro, Y.Z. Janice Chen, Paulo Azevedo, Daniel Menasché, and Don Towsley, Life Fellow, IEEE


Updated version of the SLACGS package with new features:

* Simulation Reports and Scenario Reports are now: 
  - stored in json files
  - exported to HTML files
  - Reports stored in user's local folder (<user_home>/slacgs/output/reports/ or /content/slacgs/output/reports/ for G-colab)

* execute scripts from demo_scripts folder to:
  - Run a single Simulation and create a Simulation Report: run_simulation.py
  - Run a set of simulations grouped by Experiment Scenarios: run_experiment.py
  - Make Scenario Reports from Simulation Report data saved in json: run_make_scenario_report.py
  - Make Simulation Reports from Simulation Report data saved in json: run_make_simulation_report.py


Outdated README for the SLACGS package starting here:

Documentation: https://slacgs.netlify.app/

* Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
* The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
* Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

* Reports Exported:
  - Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
  - Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
  - Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

* Images Exported (<user_home>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
  - Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
  - Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
  - Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

* Loss Functions:
  - Theoretical Loss: estimated using probability theory
  - Empirical Loss with Train Data: estimated using empirical approach with train data
  - Empirical Loss with Test Data: estimated using empirical approach with test data


# Experiment Description Available in the PDF

[Download Experiment PDF](./slacgs.pdf)

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
pip install slacgs
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

## 4 Demo Functions

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


