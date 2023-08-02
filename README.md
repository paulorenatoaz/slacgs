# SLACGS

A Simulator for Loss Analysis of Linear Classifiers on Gaussian Samples in order to evaluate Trade Off Between Samples and Features sizes in Classification Problems on gaussian Samples.

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

```latex

\documentclass{article}
\usepackage{amsmath}
\usepackage{amsfonts}
\usepackage{bm}

\begin{document}

The General 3D Classification Problem
Assuming that the conditional covariance matrices are equal, the 3D classification problem can be formulated as follows:

\[
(X_1, X_2, X_3) \sim
\begin{cases}
N ((\mu_1^+, \mu_2^+, \mu_3^+), \Sigma), & \text{if } Y = +1 \\
N ((\mu_1^-, \mu_2^-, \mu_3^-), \Sigma), & \text{if } Y = -1
\end{cases}
\]

where

\[
\Sigma =
\begin{bmatrix}
\sigma_1^2 & \rho_{12}\sigma_1\sigma_2 & \rho_{13}\sigma_1\sigma_3 \\
\rho_{12}\sigma_1\sigma_2 & \sigma_2^2 & \rho_{23}\sigma_2\sigma_3 \\
\rho_{13}\sigma_1\sigma_3 & \rho_{23}\sigma_2\sigma_3 & \sigma_3^2
\end{bmatrix}
\]

We also assume:

\[
P(Y = +1) = P(Y = -1) = \frac{1}{2}
\]

and:

\[
\mu_+ = (1, 1, 1), \quad \mu_- = (-1, -1, -1)
\]

Now, suppose the equation of the optimal separation plane is:

\[
X_3 = d^* + e^*X_1 + f^*X_2
\]

To minimize \(P(\text{Error})\) with respect to \(d^*\), \(e^*\), and \(f^*\), we must set their partial derivatives with respect to these coefficients to zero. Then, it can be shown that:

\begin{itemize}
\item \(d^* = 0\) (due to general symmetry with respect to the origin \((0, 0, 0)\))
\item Both \(e^*\) and \(f^*\) depend on the six parameters that define the matrix \(\Sigma\), namely: \(\sigma_1\), \(\sigma_2\), \(\sigma_3\), \(\rho_{12}\), \(\rho_{13}\), \(\rho_{23}\).
\item The minimum probability of error (Bayes risk) is \(P(\text{Error}) = 1 - \Phi\left|\frac{1 - e^* - f^*}{\sqrt{\Delta}}\right|\), where:
\begin{itemize}
\item \(\Delta = A^2 + B^2 + \lambda^2_{33}\), with: \(A = e^*\lambda_{11} + f^*\lambda_{21} - \lambda_{31}\), \(B = f^*\lambda_{22} - \lambda_{32}\)
\item \(\lambda_{11} = \sigma_1\), \(\lambda_{

```

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

## You can opts to build your own Report Service Configuration
path_to_google_cloud_service_account_api_key = 'path/to/key.json'
set_report_service_conf(path_to_google_cloud_service_account_api_key)

## Or you can use our Report Service Configuration if you have the password
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
scenario_number = 1
params = [1, 1, 2.1, 0, 0, 0]
add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

scenario_number = 2
params = [1, 1, 2, -0.15, 0, 0]
add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

scenario_number = 3
params = [1, 1, 2, 0, 0.15, 0.15]
add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

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
