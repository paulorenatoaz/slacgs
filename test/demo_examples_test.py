from slacgs.demo import *

## You can opt to build your own Report Service Configuration
path_to_google_cloud_service_account_api_key = 'path/to/key.json'
set_report_service_conf(slacgs_password=path_to_google_cloud_service_account_api_key)

## Or you can use our Report Service Configuration if you have the password
set_report_service_conf()


## 1. Run an Experiment Simulation ##
run_experiment_simulation_test()

## 2. Add a Simulation to an Experiment Scenario Spreadsheet ##
scenario_number = 1
params = [1, 1, 2.1, 0, 0, 0]
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)

scenario_number = 2
params = [1, 1, 2, -0.15, 0, 0]
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)

scenario_number = 3
params = [1, 1, 2, 0, 0.15, 0.15]
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)

scenario_number = 4
params = [1, 1, 2, -0.1, 0.15, 0.15]
add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number)

## 3. Run a Custom Scenario ##
scenario_list = [[1, 1, 3, round(0.1 * rho, 1), 0, 0] for rho in range(-5, 6)]
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
dims_to_compare = (2, 5)
run_custom_simulation_test(params, dims_to_compare)

### 6 features
params = [1, 2, 3, 4, 5, 6, -0.3, -0.3, -0.2, -0.2, -0.1, -0.1, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4]
run_custom_simulation_test(params)

## 6. Run All Experiment Simulations ##
run_experiment_test()
