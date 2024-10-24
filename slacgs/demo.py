import math
import contextlib
import io
import os
import numpy as np

from PIL import Image
from matplotlib import pyplot as plt
from tabulate import tabulate

from .model import Model
from .simulator import Simulator
from .gspread_client import GspreadClient
from .gdrive_client import GdriveClient
from .utils import report_service_conf, set_report_service_conf, get_grandparent_folder_path

"""
This module provides functions and classes that demonstrate the functionality of the slacgs package. 

Attributes:
	SCENARIO1 (list): list of parameters for scenario 1
	SCENARIO2 (list): list of parameters for scenario 2
	SCENARIO3 (list): list of parameters for scenario 3
	SCENARIO4 (list): list of parameters for scenario 4
	SCENARIOS (list): list of scenarios for simulation
	GDC (GdriveClient): Google Drive Client for reports service

Functions:
	start_google_drive_service(): starts Google Drive service
	run_experiment_simulation( scenario, model, simulator, gdc ): runs experiment simulation
				
	
"""



## define list of parameters for scenario 1
SCENARIO1 = [[1, 1, round(1 + 0.1 * sigma3, 2), 0, 0, 0] for sigma3 in range(3, 10)]
SCENARIO1 += [[1, 1, sigma3 / 2, 0, 0, 0] for sigma3 in range(4, 11, 1)]
SCENARIO1 += [[1, 1, sigma3, 0, 0, 0] for sigma3 in range(6, 13, 1)]

## define list of parameters for scenario 2
SCENARIO2 = [[1, 1, 2, round(rho12 * 0.1, 1), 0, 0] for rho12 in range(-8, 9)]

## define list of parameters for scenario 3
RHO_12 = 0
SCENARIO3 = []
for r in range(-8, 8):
	if abs(round(0.1 * r, 1)) < math.sqrt((1 + RHO_12) / 2):
		SCENARIO3 += [[1, 1, 2, RHO_12, round(0.1 * r, 1), round(0.1 * r, 1)]]

## define list of parameters for scenario 4
RHO_12 = -0.1
SCENARIO4 = []
for r in range(-8, 8):
	if abs(round(0.1 * r, 1)) < math.sqrt((1 + RHO_12) / 2):
		SCENARIO4 += [[1, 1, 1, RHO_12, round(0.1 * r, 1), round(0.1 * r, 1)]]

## define list of parameters for scenario 0
SCENARIO5 = [[1, sigma2, 0] for sigma2 in range(1, 8)]

## define list of parameters for scenario 0
SCENARIO6 = [[1, 1, round(r*0.1, 1)] for r in range(-8, 8)] + [[1, 1, round(r*0.1, 1)] for r in range(1, 9)]

## define list of parameters for scenario 0
SCENARIO7 = [[1, 4, round(r*0.2, 1)] for r in range(-4, 4)] + [[1, 1, round(r*0.1, 1)] for r in range(1, 9)]



## create list of scenarios
SCENARIOS = [SCENARIO1, SCENARIO2, SCENARIO3, SCENARIO4, SCENARIO5, SCENARIO6, SCENARIO7]
SCENARIOS = [SCENARIO7]

## free global variables
r = RHO_12 = None

# initialize Google Drive Client global variable for demo reports service
GDC = None


def start_google_drive_service(password=None, user_email=None):
	"""start Google Drive Client for demo report service

	Parameters:
		password (str): password for slacgs report service
		user_email (str): email for Google Drive account to be used for report service

	Notes:
		if password and user_email are None, the report_service_conf dictionary will be used to get the password and user_email
	"""

	## create GdriveClient object and connect to Google Drive for reports service
	if report_service_conf['drive_service'] is None:
		set_report_service_conf(user_google_account_email=user_email, slacgs_password=password)

	global GDC
	if GDC is None:
		GDC = GdriveClient(report_service_conf['drive_service'], report_service_conf['spreadsheet_service'],
		                   report_service_conf['user_email'])

	if GDC.gdrive_account_email:
		if not GDC.folder_exists('slacgs.demo.' + GDC.gdrive_account_email):
			folder_id = GDC.create_folder('slacgs.demo.' + GDC.gdrive_account_email)
		else:
			folder_id = GDC.get_folder_id_by_name('slacgs.demo.' + GDC.gdrive_account_email)

	# GDC.share_folder_with_gdrive_account(folder_id)


def run_experiment_simulation(start_scenario=1, verbose=True):
	""" run a simulation for one of the experiment scenarios and return True if there are still parameters to be simulated and False otherwise.

	- Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
	- The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
	- Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

	Reports Exported:
		- Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		- Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		- Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported (<user>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
		- Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		- Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		- Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

	Loss Functions:
		- Theoretical Loss: estimated using probability theory
		- Empirical Loss with Train Data: estimated using empirical approach with train data
		- Empirical Loss with Test Data: estimated using empirical approach with test data

	Dimensions simulated:
		- 1D: 1 feature
		- 2D: 2 features
		- 3D: 3 features

	Dimensions compared:
		- 2D vs 3D: 2 features vs 3 features

	Cardinalities simulated:
		- N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		start_scenario (int): scenario to start the simulation (default is 1)
		verbose (bool): if True, print simulation progress to stdout (default is True)

	Returns:
		bool: True if there are still parameters to be simulated and False otherwise

	Raises:
		ValueError: if start_scenario is not between 1 and 4
		TypeError: if start_scenario is not an int


	Example:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> run_experiment_simulation()

	"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if not 1 <= start_scenario <= 4:
		raise ValueError("start_scenario must be between 1 and 4")

	start_google_drive_service()


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	SPREADSHEET_TITLE = 'scenario1'
	## create spreadsheet for the first simulation if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)
		gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])
		PARAM = SCENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(start_scenario - 1, len(SCENARIOS)):
			SPREADSHEET_TITLE = 'scenario' + str(i + 1)

			## create spreadsheet if it doesn't exist
			if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
				folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
				GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

			gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for j in range(len(SCENARIOS[i])):
				params = SCENARIOS[i][j]
				if gsc.param_not_in_scenario(params):
					PARAM = params
					break

			## if all parameters are in the spreadsheet report home, then go to the next spreadsheet
			if PARAM:
				break

	if not PARAM:
		print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
		return False

	## create model object
	model = Model(PARAM)

	## create simulator object
	slacgs = Simulator(model, verbose=verbose)

	## run simulation
	slacgs.run()

	## upload png images to drive
	slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number, verbose=True):
	""" add simulation results to one of the experiment scenario spreadsheets

	Parameters:
		params (list[float|int] or tuple[float|int]): a list containnong Sigmas and Rhos
		scenario_number (int): scenario number
		verbose (bool): if True, print simulation progress (default: True)

	Returns:
		None

	Raises:
		TypeError:
			if scenario is not an int;
			if params is not a list[int|float] or tuple[int|float]
		ValueError:
			if params is not a valid parameter list;
			if scenario is not between 1 and 4

	See Also:
		:func:`slacgs.demo.run_experiment_simulation`

	Example:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> scenario_number = 1
		>>> params = [1, 1, 2.1, 0, 0, 0]
		>>> add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

		>>> scenario_number = 2
		>>> params = [1, 1, 2, -0.15, 0, 0]
		>>> add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

		>>> scenario_number = 3
		>>> params = [1, 1, 2, 0, 0.15, 0.15]
		>>> add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)

		>>> scenario_number = 4
		>>> params = [1, 1, 2, -0.1, 0.15, 0.15]
		>>> add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number)


	"""

	if not isinstance(scenario_number, int):
		raise TypeError("scenario must be an int")

	if not isinstance(params, (list, tuple)):
		raise TypeError("params must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in params):
		raise TypeError("params must be a list or tuple of int or float")

	if scenario_number < 1 or scenario_number > 4:
		raise ValueError("scenario must be between 1 and 4")

	if len(params) != 6:
		raise ValueError("params must be a list or tuple of 6 elements for this experiment")

	if scenario_number == 1:
		if params[0] != 1 or params[1] != 1 or params[3] != 0 or params[4] != 0 or params[5] != 0:
			raise ValueError(
				"for scenario 1, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[3] = 0, params[4] = 0 and params[5] = 0")

	elif scenario_number == 2:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[4] != 0 or params[5] != 0:
			raise ValueError(
				"for scenario 2, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[2] = 2, params[4] = 0 and params[5] = 0")
		if params[3] < -0.8 or params[3] > 0.8:
			raise ValueError("for scenario 2, params[3] must be between -0.8 and 0.8")

	elif scenario_number == 3:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[3] != 0:
			raise ValueError(
				"for scenario 3, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[2] = 2 and params[3] = 0")
		if params[4] < -0.7 or params[4] > 0.7 or params[4] != params[5]:
			raise ValueError(
				"for scenario 3, params[4] must be between -0.7 and 0.7 and params[4] must be equal to params[5]")

	elif scenario_number == 4:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[3] != -0.1:
			raise ValueError(
				"for scenario 4, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[2] = 2 and params[3] = -0.1")
		if params[4] < -0.6 or params[4] > 0.6 or params[4] != params[5]:
			raise ValueError(
				"for scenario 4, params[4] must be between -0.6 and 0.6 and params[4] must be equal to params[5]")



	start_google_drive_service()


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'scenario' + str(scenario_number)

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

	## create model object
	model = Model(params)

	## create simulator object
	slacgs = Simulator(model, verbose=verbose)

	## run simulation
	slacgs.run()

	## upload png images to drive
	slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)


def run_custom_scenario(scenario_list, scenario_number, dims_to_simulate=None, dims_to_compare=None, verbose=True):
	""" run a custom scenario and write the results to a Google Spreadsheet shared with the user.	A Scenario is a list with params to simulate.

	- Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
	- The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
	- Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

	Reports Exported:
		- Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		- Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		- Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported (<user>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
		- Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		- Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		- Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

	Loss Functions:
		- Theoretical Loss: estimated using probability theory
		- Empirical Loss with Train Data: estimated using empirical approach with train data
		- Empirical Loss with Test Data: estimated using empirical approach with test data

	Cardinalities simulated:
		- N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		scenario_list (list[list[float|int]] or tuple[list[float|int]]): scenario list
		scenario_number (int): scenario number
		dims_to_simulate (list[int] or tuple[int]): dimensionalities to simulate (if None, all possible dimensionalities will be simulated) (default: None)
		dims_to_compare (list[int] or tuple[int]): dimensionalities to compare (if None, the last two dimensionalities of dims_to_simulate will be compared) (default: None)
		verbose (bool): whether to print messages to stdout or not

	Returns:
		bool: True if simulation was successful

	Raises:
		TypeError:
			if scenario_list is not a list[list[float|int]] or tuple[list[float|int]];
			if scenario_number is not an int;
			if dims_to_simulate is not a list[int] or tuple[int];
			if dims_to_compare is not a list[int] or tuple[int];

		ValueError:
			if scenario_number is not a valid scenario number;
			if dims_to_compare is not a subset of dims_to_simulate;
			if dims_to_compare length is not 2;


	Example:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> scenario_list = [[1,1,3,round(0.1*rho,1),0,0] for rho in range(-1,2)]
		>>> scenario_number = 5
		>>> run_custom_scenario(scenario_list, scenario_number)
	"""

	if not isinstance(scenario_list, (list, tuple)):
		raise TypeError("scenario must be a list or tuple")

	if not all(isinstance(x, (list, tuple)) for x in scenario_list):
		raise TypeError("scenario must be a list or tuple of list or tuple")

	if not all(isinstance(x, (int, float)) for y in scenario_list for x in y):
		raise TypeError("scenario must be a list or tuple of list or tuple of int or float")

	if not isinstance(scenario_number, int):
		raise TypeError("scenario_number must be an int")

	if dims_to_simulate and not isinstance(dims_to_simulate, (list, tuple)):
		raise TypeError("dims_to_simulate must be a list or tuple")

	if dims_to_simulate and not all(isinstance(x, int) for x in dims_to_simulate):
		raise TypeError("dims_to_simulate must be a list or tuple of int")

	if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if dims_to_compare and not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	if scenario_number < 5:
		raise ValueError("Custom scenario_number must be >= 5")

	start_google_drive_service()

	## create Model objects to test each parameter set before continuing
	models = []
	for params in scenario_list:
		models.append(Model(params))

	## create Simulator objects to test each parameter set before continuing
	simulators = []
	for model in models:
		simulators.append(Simulator(model, dims=dims_to_simulate, dims_to_compare=dims_to_compare, verbose=verbose))


	if dims_to_compare and not all(dim in dims_to_simulate for dim in dims_to_compare):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")

	if dims_to_compare and len(dims_to_compare) != 2:
		raise ValueError("dims_to_compare length must be 2")

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_scenario' + str(scenario_number)

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

	for slacgs in simulators:
		if dims_to_compare == (2, 3) or dims_to_compare == [2, 3]:
			if gsc.param_not_in_scenario(slacgs.model.params):
				## run simulation
				slacgs.run()

				## upload png images to drive
				slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

				## write results to spreadsheet
				slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)
			else:
				continue

		else:
			## run simulation
			slacgs.run()

			## upload png images to drive
			slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

			## write results to spreadsheet
			slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def add_simulation_to_custom_scenario_spreadsheet(params, scenario_number, dims_to_simulate=None, dims_to_compare=None,
                                                  verbose=True):
	""" add a simulation to a custom  scenario spreadsheet

	Parameters:
		params (list[float|int] or tuple[float|int]): a list containnong Sigmas and Rhos
		scenario_number (int): scenario number
		dims_to_simulate (list[int] or tuple[int]): dimensionalities to simulate (if None, all dimensionalities will be simulated) (default: None)
		dims_to_compare (list[int] or tuple[int]): dimensionalities to compare (if None, the last two dimensionalities of dims_to_simulate will be compared) (default: None)
		verbose (bool): whether to print progress to console output (default: True)

	Returns:
		bool: True if successful, False otherwise

	Raises:
		TypeError:
			if params is not a list[float|int] or tuple[float|int];
			if scenario_number is not an int;
			if dims_to_simulate is not a list[int] or tuple[int];
			if dims_to_compare is not a list[int] or tuple[int]

		ValueError:
			if scenario_number is < 5;
			if dims_to_compare is not a subset of dims_to_simulate;
			if dims_to_compare length is not 2


	Example:

		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> params = (1, 1, 3, -0.2, 0, 0)
		>>> scenario_number = 5
		>>> add_simulation_to_custom_scenario_spreadsheet(params, scenario_number)


	"""

	if not isinstance(params, (list, tuple)):
		raise TypeError("params must be a list or tuple")

	if not isinstance(scenario_number, int):
		raise TypeError("scenario_number must be an int")

	if dims_to_simulate and not isinstance(dims_to_simulate, (list, tuple)):
		raise TypeError("dims_to_simulate must be a list or tuple")

	if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if scenario_number < 5:
		raise ValueError("scenario_number must be >= 5 for a Custom Scenario")

	if dims_to_compare and dims_to_simulate and not set(dims_to_compare).issubset(set(dims_to_simulate)):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")

	if dims_to_compare and len(dims_to_compare) != 2:
		raise ValueError("dims_to_compare length must be 2")


	start_google_drive_service()

	## create Model object to test parameter set before continuing
	model = Model(params)

	## create Simulator object to test parameters before continuing
	slacgs = Simulator(model, dims=dims_to_simulate, dims_to_compare=dims_to_compare, verbose=verbose)

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_scenario' + str(scenario_number)

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

	## run simulation
	slacgs.run()

	## upload png images to drive
	slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def run_custom_simulation(params, dims_to_simulate=None, dims_to_compare=None, verbose=True):
	""" run a custom simulation for any dimensionality and cardinality

	- Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
	- The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
	- Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

	Reports Exported:
		- Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		- Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		- Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported (<user>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
		- Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		- Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		- Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

	Loss Functions:
		- Theoretical Loss: estimated using probability theory
		- Empirical Loss with Train Data: estimated using empirical approach with train data
		- Empirical Loss with Test Data: estimated using empirical approach with test data

	Cardinalities simulated:
		- N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		params (list[float|int] or tuple[float|int]): a list containnong Sigmas and Rhos
		dims_to_simulate (list[int] or tuple[int]): dimensionalities to simulate (if None, all dimensionalities will be simulated) (default: None)
		dims_to_compare (list[int] or tuple[int]): dimensionalities to compare (if None, the last two dimensionalities of dims_to_simulate will be compared) (default: None)
		verbose (bool): whether to print progress to console output (default: True)

	Returns:
		bool: True if successful, False otherwise

	Raises:
		TypeError:
			if params is not a list[float|int] or tuple[float|int];
			if dims_to_compare is not a list[int] or tuple[int]

		ValueError:
			if dims_to_compare length is not 2;
			if dims_to_compare is not a subset of dims_to_simulate


	Examples:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> ## 2 features
		>>> params = [1, 2, 0.4]
		>>> run_custom_simulation(params)

		>>> ## 3 features
		>>> params = [1, 1, 4, -0.2, 0.1, 0.1]
		>>> run_custom_simulation(params)

		>>> ## 4 features
		>>> params = [1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
		>>> run_custom_simulation(params)

		>>> ## 5 features
		>>> params = [1, 1, 2, 2, 2, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.2, 0, 0, 0]
		>>> dims_to_compare = (2, 5)
		>>> run_custom_simulation(params, dims_to_compare)

		>>> ## 6 features
		>>> params = [1, 2, 3, 4, 5, 6, -0.3, -0.3, -0.2, -0.2, -0.1, -0.1, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4]
		>>> run_custom_simulation(params)

	"""

	if not isinstance(params, (list, tuple)):
		raise TypeError("params must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in params):
		raise TypeError("params must be a list or tuple of int or float")

	if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if dims_to_compare and not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	if dims_to_compare and len(dims_to_compare) != 2:
		raise ValueError("dims_to_compare must be a list or tuple of length 2")

	if dims_to_compare and dims_to_simulate and not set(dims_to_compare).issubset(set(dims_to_simulate)):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")

	## initialize gdrive client if it hasn't been initialized yet
	start_google_drive_service()

	## create model object
	model = Model(params)

	## create simulator object
	slacgs = Simulator(model, dims=dims_to_simulate, dims_to_compare=dims_to_compare, verbose=verbose)

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_simulations'

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

	## run simulation
	slacgs.run()

	## upload png images to drive
	slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, dims_to_compare=dims_to_compare, verbose=verbose)

	return True


def run_experiment(start_scenario=1, verbose=True):
	""" run all simulations in all experiment scenarios

	- Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
	- The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
	- Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

	Reports Exported:
		- Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		- Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		- Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported (<user>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
		- Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		- Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		- Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

	Loss Functions:
		- Theoretical Loss: estimated using probability theory
		- Empirical Loss with Train Data: estimated using empirical approach with train data
		- Empirical Loss with Test Data: estimated using empirical approach with test data

	Dimensions simulated:
		- 1D: 1 feature
		- 2D: 2 features
		- 3D: 3 features

	Dimensions compared:
		- 2D vs 3D: 2 features vs 3 features

	Cardinalities simulated:
		- N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		start_scenario: scenario to start the experiment
		verbose: if True, prints messages to console output

	Returns:
	 		0 if all parameters have been simulated

	Raises:
		ValueError: if start_scenario is not between 1 and 4
		TypeError: if start_scenario is not an int


	Examples:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> run_experiment()

	"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if start_scenario < 1 or start_scenario > 4:
		raise ValueError("start_scenario must be between 1 and 4")

	try:
		file_path = save_scenario_data_plots_animation_as_gif(start_scenario)
	except Exception as e:
		print("An error occurred while saving scenario data plots animation as gif: ", e)
	else:
		start_google_drive_service()
		drive_images_folder_id = GDC.create_folder('images', GDC.get_folder_id_by_name('slacgs.demo.' + GDC.gdrive_account_email), verbose=verbose)
		GDC.upload_file_to_drive(file_path, drive_images_folder_id, verbose=verbose)

	while run_experiment_simulation(start_scenario):
		continue

	if verbose:
		print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
	return 0


def doctest_next_parameter():
	""" return the next parameter to be simulated on doctests, and also the adequate spreadsheet title

	:returns: PARAM, SPREADSHEET_TITLE
	:rtype: tuple

	:Example:
	>>> from slacgs.demo import *

	>>> # opt-1. set report service configuration with your own google cloud service account key file
	>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
	>>> set_report_service_conf(PATH_TO_KEY_FILE)

	>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
	>>> set_report_service_conf()

	>>> params, spreadsheet_title = doctest_next_parameter()

	"""
	start_google_drive_service()

	REPORT_FOLDER_NAME = 'slacgs.doctest'
	SPREADSHEET_TITLE = 'scenario1.doctest'
	## create spreadsheet for the first simulation if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		with contextlib.redirect_stdout(io.StringIO()):
			## do operations without printing to stdout
			spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
			folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
			GDC.move_file_to_folder(spreadsheet_id, folder_id)
		PARAM = SCENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(len(SCENARIOS)):
			SPREADSHEET_TITLE = 'scenario' + str(i + 1) + '.doctest'

			## create spreadsheet if it doesn't exist
			if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
				folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
				GDC.move_file_to_folder(spreadsheet_id, folder_id)

			gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for params in SCENARIOS[i]:
				if gsc.param_not_in_scenario(params):
					PARAM = params
					break

			## if all parameters are in the spreadsheet report home, then go to the next spreadsheet
			if PARAM:
				break

	return PARAM, SPREADSHEET_TITLE


def run_experiment_simulation_test(start_scenario=1, power=0.1, verbose=True):
	""" run a simulation test for one of the experiment scenarios and return True if there are still parameters to be simulated and False otherwise.

	- Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
	- The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
	- Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

	Reports Exported:
		- Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		- Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		- Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported (<user>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
		- Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		- Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		- Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

	Loss Functions:
		- Theoretical Loss: estimated using probability theory
		- Empirical Loss with Train Data: estimated using empirical approach with train data
		- Empirical Loss with Test Data: estimated using empirical approach with test data

	Dimensions simulated:
		- 1D: 1 feature
		- 2D: 2 features
		- 3D: 3 features

	Dimensions compared:
		- 2D vs 3D: 2 features vs 3 features

	Cardinalities simulated:
		- N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		start_scenario (int): scenario to start the simulation test
		verbose (bool): if True, print simulation progress to stdout (default is True)

	Returns:
		bool: True if there are still parameters to be simulated and False otherwise

	Raises:
		ValueError: if start_scenario is not between 1 and 4
		TypeError: if start_scenario is not an int


	Example:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> run_experiment_simulation_test(verbose=False)

	"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if not 1 <= start_scenario <= 4:
		raise ValueError("start_scenario must be between 1 and 4")

	start_google_drive_service()

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	SPREADSHEET_TITLE = 'scenario1.test'
	## create spreadsheet for the first simulation if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)
		gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])
		PARAM = SCENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(start_scenario - 1, len(SCENARIOS)):
			SPREADSHEET_TITLE = 'scenario' + str(i + 1) + '.test'

			## create spreadsheet if it doesn't exist
			if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
				folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
				GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

			gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for j in range(len(SCENARIOS[i])):
				params = SCENARIOS[i][j]
				if gsc.param_not_in_scenario(params):
					PARAM = params
					break

			## if all parameters are in the spreadsheet report home, then go to the next spreadsheet
			if PARAM:
				break

	if not PARAM:
		print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
		return False

	## create model object
	model = Model(PARAM)

	## create simulator object
	slacgs = Simulator(model, step_size=5, max_steps=int(200*power), min_steps=int(100*power), precision=1e-4, augmentation_until_n=1024, verbose=verbose)

	## run simulation
	slacgs.run()

	## upload png images to drive
	slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number, power=0.1, verbose=True):
	""" add simulation results to one of the experiment scenario spreadsheets

	Parameters:
		params (list[float|int] or tuple[float|int]): a list containnong Sigmas and Rhos
		scenario_number (int): scenario number
		verbose (bool): if True, print simulation progress (default: True)

	Returns:
		None

	Raises:
		TypeError:
			if scenario is not an int;
			if params is not a list[int|float] or tuple[int|float]
		ValueError:
			if params is not a valid parameter list;
			if scenario is not between 1 and 4

	See Also:
		:func:`slacgs.demo.run_experiment_simulation_test`

	Example:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> scenario_number = 1
		>>> params = [1, 1, 2.1, 0, 0, 0]
		>>> add_simulation_to_experiment_scenario_spreadsheet_test(params,scenario_number)

		>>> scenario_number = 2
		>>> params = [1, 1, 2, -0.15, 0, 0]
		>>> add_simulation_to_experiment_scenario_spreadsheet_test(params,scenario_number)

		>>> scenario_number = 3
		>>> params = [1, 1, 2, 0, 0.15, 0.15]
		>>> add_simulation_to_experiment_scenario_spreadsheet_test(params,scenario_number)

		>>> scenario_number = 4
		>>> params = [1, 1, 2, -0.1, 0.15, 0.15]
		>>> add_simulation_to_experiment_scenario_spreadsheet_test(params,scenario_number)


	"""

	if not isinstance(scenario_number, int):
		raise TypeError("scenario must be an int")

	if not isinstance(params, (list, tuple)):
		raise TypeError("params must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in params):
		raise TypeError("params must be a list or tuple of int or float")

	if scenario_number < 1 or scenario_number > 4:
		raise ValueError("scenario must be between 1 and 4")

	if len(params) != 6:
		raise ValueError("params must be a list or tuple of 6 elements for this experiment")

	if scenario_number == 1:
		if params[0] != 1 or params[1] != 1 or params[3] != 0 or params[4] != 0 or params[5] != 0:
			raise ValueError(
				"for scenario 1, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[3] = 0, params[4] = 0 and params[5] = 0")

	elif scenario_number == 2:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[4] != 0 or params[5] != 0:
			raise ValueError(
				"for scenario 2, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[2] = 2, params[4] = 0 and params[5] = 0")
		if params[3] < -0.8 or params[3] > 0.8:
			raise ValueError("for scenario 2, params[3] must be between -0.8 and 0.8")

	elif scenario_number == 3:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[3] != 0:
			raise ValueError(
				"for scenario 3, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[2] = 2 and params[3] = 0")
		if params[4] < -0.7 or params[4] > 0.7 or params[4] != params[5]:
			raise ValueError(
				"for scenario 3, params[4] must be between -0.7 and 0.7 and params[4] must be equal to params[5]")

	elif scenario_number == 4:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[3] != -0.1:
			raise ValueError(
				"for scenario 4, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[2] = 2 and params[3] = -0.1")
		if params[4] < -0.6 or params[4] > 0.6 or params[4] != params[5]:
			raise ValueError(
				"for scenario 4, params[4] must be between -0.6 and 0.6 and params[4] must be equal to params[5]")

	start_google_drive_service()

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'scenario' + str(scenario_number) + '.test'

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

	## create model object
	model = Model(params)

	## create simulator object
	slacgs = Simulator(model, step_size=1, max_steps=int(200*power), min_steps=int(100*power), precision=1e-4, augmentation_until_n=1024,
	                   verbose=verbose)

	## run simulation
	slacgs.run()

	## upload png images to drive
	slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)


def run_custom_scenario_test(scenario_list, scenario_number, dims_to_simulate=None, dims_to_compare=None, power=0.1,
                             verbose=True):
	""" run a custom test scenario and write the results to a Google Spreadsheet shared with the user. A Scenario is a list with params to simulate.

	- Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
	- The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
	- Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

	Reports Exported:
		- Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		- Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		- Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported (<user>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
		- Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		- Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		- Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

	Loss Functions:
		- Theoretical Loss: estimated using probability theory
		- Empirical Loss with Train Data: estimated using empirical approach with train data
		- Empirical Loss with Test Data: estimated using empirical approach with test data

	Cardinalities simulated:
		- N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		scenario_list (list[list[float|int]] or tuple[list[float|int]]): scenario list
		scenario_number (int): scenario number
		dims_to_simulate (list[int] or tuple[int]): dimensionalities to simulate (if None, all possible dimensionalities will be simulated) (default: None)
		dims_to_compare (list[int] or tuple[int]): dimensionalities to compare (if None, the last two dimensionalities of dims_to_simulate will be compared) (default: None)
		verbose (bool): whether to print messages to stdout or not

	Returns:
		bool: True if simulation was successful

	Raises:
		TypeError:
			if scenario_list is not a list[list[float|int]] or tuple[list[float|int]];
			if scenario_number is not an int;
			if dims_to_simulate is not a list[int] or tuple[int];
			if dims_to_compare is not a list[int] or tuple[int];

		ValueError:
			if scenario_number is not a valid scenario number;
			if dims_to_compare is not a subset of dims_to_simulate;
			if dims_to_compare length is not 2;


	Example:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> scenario_list = [[1,1,3,round(0.1*rho,1),0,0] for rho in range(-1,2)]
		>>> scenario_number = 5
		>>> run_custom_scenario_test(scenario_list,scenario_number)
	"""

	if not isinstance(scenario_list, (list, tuple)):
		raise TypeError("scenario must be a list or tuple")

	if not all(isinstance(x, (list, tuple)) for x in scenario_list):
		raise TypeError("scenario must be a list or tuple of list or tuple")

	if not all(isinstance(x, (int, float)) for y in scenario_list for x in y):
		raise TypeError("scenario must be a list or tuple of list or tuple of int or float")

	if not isinstance(scenario_number, int):
		raise TypeError("scenario_number must be an int")

	if dims_to_simulate and not isinstance(dims_to_simulate, (list, tuple)):
		raise TypeError("dims_to_simulate must be a list or tuple")

	if dims_to_simulate and not all(isinstance(x, int) for x in dims_to_simulate):
		raise TypeError("dims_to_simulate must be a list or tuple of int")

	if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if dims_to_compare and not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	if scenario_number < 5:
		raise ValueError("Custom scenario_number must be >= 5")

	start_google_drive_service()

	## create Model objects to test each parameter set before continuing
	models = []
	for params in scenario_list:
		models.append(Model(params))

	## create Simulator objects to test each parameter set before continuing
	simulators = []
	for model in models:
		simulators.append(
			Simulator(model, dims=dims_to_simulate, dims_to_compare=dims_to_compare, step_size=1, max_steps=int(200*power),
			          min_steps=int(100*power), precision=1e-4,
			          augmentation_until_n=1024, verbose=verbose))


	if dims_to_compare and not all(dim in dims_to_simulate for dim in dims_to_compare):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")

	if dims_to_compare and len(dims_to_compare) != 2:
		raise ValueError("dims_to_compare length must be 2")

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_scenario' + str(scenario_number) + '.test'

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

	for slacgs in simulators:
		if dims_to_compare == (2, 3) or dims_to_compare == [2, 3]:
			if gsc.param_not_in_scenario(slacgs.model.params):
				## run simulation
				slacgs.run()

				## upload png images to drive
				slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

				## write results to spreadsheet
				slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)
			else:
				continue

		else:
			## run simulation
			slacgs.run()

			## upload png images to drive
			slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

			## write results to spreadsheet
			slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def add_simulation_to_custom_scenario_spreadsheet_test(params, scenario_number, dims_to_simulate=None,
                                                       dims_to_compare=None, power=0.1, verbose=True):
	""" add a simulation to a custom test scenario

	Parameters:
		params (list[float|int] or tuple[float|int]): a list containnong Sigmas and Rhos
		scenario_number (int): scenario number
		dims_to_simulate (list[int] or tuple[int]): dimensionalities to simulate (if None, all dimensionalities will be simulated) (default: None)
		dims_to_compare (list[int] or tuple[int]): dimensionalities to compare (if None, the last two dimensionalities of dims_to_simulate will be compared) (default: None)
		verbose (bool): whether to print progress to console output (default: True)

	Returns:
		bool: True if successful, False otherwise

	Raises:
		TypeError:
			if params is not a list[float|int] or tuple[float|int];
			if scenario_number is not an int;
			if dims_to_simulate is not a list[int] or tuple[int];
			if dims_to_compare is not a list[int] or tuple[int]

		ValueError:
			if scenario_number is < 5;
			if dims_to_compare is not a subset of dims_to_simulate;
			if dims_to_compare length is not 2


	Example:

		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> params = (1, 1, 3, -0.2, 0, 0)
		>>> scenario_number = 5
		>>> add_simulation_to_custom_scenario_spreadsheet_test(params,scenario_number)

	"""

	if not isinstance(params, (list, tuple)):
		raise TypeError("params must be a list or tuple")

	if not isinstance(scenario_number, int):
		raise TypeError("scenario_number must be an int")

	if dims_to_simulate and not isinstance(dims_to_simulate, (list, tuple)):
		raise TypeError("dims_to_simulate must be a list or tuple")

	if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if scenario_number < 5:
		raise ValueError("scenario_number must be >= 5 for a Custom Scenario")

	if dims_to_compare and dims_to_simulate and not set(dims_to_compare).issubset(set(dims_to_simulate)):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")

	if dims_to_compare and len(dims_to_compare) != 2:
		raise ValueError("dims_to_compare length must be 2")

	start_google_drive_service()

	## create Model object to test parameter set before continuing
	model = Model(params)

	## create Simulator object to test parameters before continuing
	slacgs = Simulator(model, dims=dims_to_simulate, dims_to_compare=dims_to_compare, step_size=1, max_steps=int(200*power),
	                   min_steps=int(100*power), precision=1e-4,
	                   augmentation_until_n=1024, verbose=verbose)

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_scenario' + str(scenario_number) + '.test'

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(SPREADSHEET_TITLE, report_service_conf['pygsheets_service'])

	## run simulation
	slacgs.run()

	## upload png images to drive
	slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def run_custom_simulation_test(params, dims_to_simulate=None, dims_to_compare=None, power=0.1, verbose=True):
	""" run a custom simulation for any dimensionality and cardinality

	- Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
	- The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
	- Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

	Reports Exported:
		- Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		- Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		- Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported (<user>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
		- Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		- Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		- Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

	Loss Functions:
		- Theoretical Loss: estimated using probability theory
		- Empirical Loss with Train Data: estimated using empirical approach with train data
		- Empirical Loss with Test Data: estimated using empirical approach with test data

	Cardinalities simulated:
		- N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		params (list[float|int] or tuple[float|int]): a list containnong Sigmas and Rhos
		dims_to_simulate (list[int] or tuple[int]): dimensionalities to simulate (if None, all dimensionalities will be simulated) (default: None)
		dims_to_compare (list[int] or tuple[int]): dimensionalities to compare (if None, the last two dimensionalities of dims_to_simulate will be compared) (default: None)
		verbose (bool): whether to print progress to console output (default: True)

	Returns:
		bool: True if successful, False otherwise

	Raises:
		TypeError:
			if params is not a list[float|int] or tuple[float|int];
			if dims_to_compare is not a list[int] or tuple[int]

		ValueError:
			if dims_to_compare length is not 2;
			if dims_to_compare is not a subset of dims_to_simulate


	:Example:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> ## 2 features
		>>> params = [1, 2, 0.4]
		>>> run_custom_simulation_test(params,dims_to_compare)

		>>> ## 3 features
		>>> params = [1, 1, 4, -0.2, 0.1, 0.1]
		>>> run_custom_simulation_test(params,dims_to_compare)

		>>> ## 4 features
		>>> params = [1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
		>>> dims_to_compare = (3, 4)
		>>> run_custom_simulation_test(params,dims_to_compare)

		>>> ## 5 features
		>>> params = [1, 1, 2, 2, 2, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.2, 0, 0, 0]
		>>> dims_to_compare = (2, 5)
		>>> run_custom_simulation_test(params,dims_to_compare)

		>>> params = [1, 2, 3, 4, 5, 6, -0.3, -0.3, -0.2, -0.2, -0.1, -0.1, 0, 0, 0.1, 0.1, 0.2, 0.2, 0.3, 0.3, 0.4]
		>>> run_custom_simulation_test(params)


	"""

	if not isinstance(params, (list, tuple)):
		raise TypeError("params must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in params):
		raise TypeError("params must be a list or tuple of int or float")

	if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if dims_to_compare and not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	if dims_to_compare and len(dims_to_compare) != 2:
		raise ValueError("dims_to_compare must be a list or tuple of length 2")

	if dims_to_compare and dims_to_simulate and not set(dims_to_compare).issubset(set(dims_to_simulate)):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")

	## initialize gdrive client if it hasn't been initialized yet
	start_google_drive_service()

	## create model object
	model = Model(params)

	## create simulator object
	slacgs = Simulator(model, dims=dims_to_simulate, dims_to_compare=dims_to_compare, step_size=1, max_steps=int(200*power),
	                   min_steps=int(100*power), precision=1e-4,
	                   augmentation_until_n=1024, verbose=verbose)

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.folder_exists(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_simulations.test'

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient( SPREADSHEET_TITLE)

	## run simulation
	slacgs.run()

	## upload png images to drive
	slacgs.report.upload_report_images_to_drive(GDC, verbose=verbose)

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, dims_to_compare=dims_to_compare, verbose=verbose)

	return True


def run_experiment_test(start_scenario=1, power=0.1, verbose=True):
	""" run all simulations in all experiment scenarios

	- Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario	and another one for the Custom Simulations.
	- The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service	account and shared with the user's Google Drive account.
	- Also, images with data visualization will be exported to a local folder inside user's local folder (<user>/slacgs/images/ or /content/slacgs/images (for G-colab) )

	Reports Exported:
		- Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		- Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		- Home Report (Scenario): Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported (<user>/slacgs/images/ or /content/slacgs/images [for G-colab] ):
		- Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		- Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		- Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

	Loss Functions:
		- Theoretical Loss: estimated using probability theory
		- Empirical Loss with Train Data: estimated using empirical approach with train data
		- Empirical Loss with Test Data: estimated using empirical approach with test data

	Dimensions simulated:
		- 1D: 1 feature
		- 2D: 2 features
		- 3D: 3 features

	Dimensions compared:
		- 2D vs 3D: 2 features vs 3 features

	Cardinalities simulated:
		- N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		start_scenario: scenario to start the experiment test
		verbose: if True, prints messages to console output

	Returns:
	 		0 if all parameters have been simulated

	Raises:
		ValueError: if start_scenario is not between 1 and 4
		TypeError: if start_scenario is not an int


	Examples:
		>>> from slacgs.demo import *

		>>> # opt-1. set report service configuration with your own google cloud service account key file
		>>> PATH_TO_KEY_FILE = '/path/to/your/google/cloud/service/account/key/file.json'
		>>> set_report_service_conf(PATH_TO_KEY_FILE)

		>>> # opt-2 set report service configuration to use slacgs' server if you have the access password
		>>> set_report_service_conf()

		>>> run_experiment_test()

	"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if start_scenario < 1 or start_scenario > 4:
		raise ValueError("start_scenario must be between 1 and 4")

	try:
		file_path = save_scenario_data_plots_animation_as_gif(start_scenario)
	except Exception as e:
		print("An error occurred while saving scenario data plots animation as gif: ", e)
	else:
		start_google_drive_service()
		drive_images_folder_id = GDC.create_folder('images', GDC.get_folder_id_by_name('slacgs.demo.' + GDC.gdrive_account_email), verbose=verbose)
		GDC.upload_file_to_drive(file_path, drive_images_folder_id, verbose=verbose)


	while run_experiment_simulation_test(start_scenario, power=power, verbose=verbose):
		continue

	if verbose:
		print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
	return 0


def print_experiment_scenarios():
	""" Print the experiment scenarios parameters in a table. """

	## fill lists with empty strings to make them the same length
	def fill_lists(lst, fill_value=''):
		max_len = max(len(sublist) for sublist in lst)

		# Fill the shorter lists
		for sublist in lst:
			sublist.extend([fill_value] * (max_len - len(sublist)))

		return lst

	## convert to numpy array, transpose, convert back to list
	data = np.array(fill_lists(SCENARIOS), dtype=object).T.tolist()

	## add index column
	indexed_data = [[i] + sublist for i, sublist in enumerate(data)]

	## make table and print
	table = tabulate(indexed_data, tablefmt='grid', headers=['Scenario 1', 'Scenario 2', 'Scenario 3', 'Scenario 4'])
	print(table)


def save_scenario_data_plots_animation_as_gif(scenario_number,scenario=None):
	''' save scenario data plots animation as gif

	Parameters:
		scenario_number (int): scenario number
		scenario (list[list[float|int]] or tuple[list[float|int]]): scenario list (default: None)

	Returns:
		str: export path

	'''

	print(f'Wait while Scenario {scenario_number} data plots animation is exported as a gif...')

	if scenario is None:
		scenario = SCENARIOS[scenario_number - 1]

	model_plots_png = []
	for param in scenario:
		model_plots_png.append(Model(param).data_plots_image)

	export_path = report_service_conf['images_path']
	export_path += 'scenario' + str(scenario_number) + '_data_plots.gif'


	model_plots_png[-1].save(export_path, save_all=True, append_images=model_plots_png[0:-1],
	                            duration=1000, loop=0)

	return export_path