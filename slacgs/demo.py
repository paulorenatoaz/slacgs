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

## define list of parameters for scenario 1
SCENARIO1 = [[1, 1, round(1 + 0.1 * sigma3, 2), 0, 0, 0] for sigma3 in range(3, 10)]
SCENARIO1 += [[1, 1, sigma3 / 2, 0, 0, 0] for sigma3 in range(4, 11, 1)]
SCENARIO1 += [[1, 1, sigma3, 0, 0, 0] for sigma3 in range(6, 14, 1)]

## define list of parameters for scenario 2
SCENARIO2 = [[1, 1, 2, round(rho12 * 0.1, 1), 0, 0] for rho12 in range(-8, 9)]

## define list of parameters for scenario 3
RHO_12=0
SCENARIO3 = []
for r in range(-8,8):
  if  abs(round(0.1*r,1)) < math.sqrt((1 + RHO_12) / 2) :
    SCENARIO3 += [[1, 1, 2, RHO_12, round(0.1 * r, 1), round(0.1 * r, 1)]]

## define list of parameters for scenario 4
RHO_12=-0.1
SCENARIO4 = []
for r in range(-8,8):
  if  abs(round(0.1*r,1)) < math.sqrt((1 + RHO_12) / 2) :
    SCENARIO4 += [[1, 1, 2, RHO_12, round(0.1 * r, 1), round(0.1 * r, 1)]]

## create list of scenarios
SCENARIOS = [SCENARIO1, SCENARIO2, SCENARIO3, SCENARIO4]

## free global variables
r = RHO_12 = None


# initialize Google Drive Client global variable for demo reports service
GDC = None


def start_google_drive_service(password=None, user_email=None, verbose=True):
	"""
		start Google Drive Client for demo report service

		Parameters:
			password (str): password for slacgs report service
			user_email (str): email for Google Drive account to be used for report service

		Obs: if password and user_email are None, the report_service_conf dictionary will be used to get the password and user_email
	"""

	## create GdriveClient object and connect to Google Drive for reports service
	if report_service_conf['drive_service'] is None:
		set_report_service_conf(password=password, user_google_account_email=user_email)

	global GDC
	if GDC is None:
		GDC = GdriveClient(report_service_conf['drive_service'], report_service_conf['spreadsheet_service'], report_service_conf['user_email'])

	if GDC.gdrive_account_email:
		if not GDC.check_folder_existence('slacgs.demo.' + GDC.gdrive_account_email):
			folder_id = GDC.create_folder('slacgs.demo.' + GDC.gdrive_account_email)
		else:
			folder_id = GDC.get_folder_id_by_name('slacgs.demo.' + GDC.gdrive_account_email)

		GDC.share_folder_with_gdrive_account(folder_id)


def run_experiment_simulation(start_scenario=1):
	""" run the simulation and return True if there are still parameters to be simulated and False otherwise

		:param start_scenario: scenario to start the simulation test
		:type start_scenario: int
		:returns: True if there are still parameters to be simulated and False otherwise
		:rtype: bool

		:raises ValueError: if start_scenario is not between 1 and 4
		:raises TypeError: if start_scenario is not an int

		"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if not 1 <= start_scenario <= 4:
		raise ValueError("start_scenario must be between 1 and 4")

	## start google drive client
	start_google_drive_service()

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id)  # share folder with user's google drive account

	SPREADSHEET_TITLE = 'sscenario1'
	## create spreadsheet for the first simulation if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id)
		gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)
		PARAM = SCENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(start_scenario - 1, len(SCENARIOS)):
			SPREADSHEET_TITLE = 'scenario' + str(i + 1)

			## create spreadsheet if it doesn't exist
			if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
				folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
				GDC.move_file_to_folder(spreadsheet_id, folder_id)

			gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for j in range(len(SCENARIOS[i])):
				param = SCENARIOS[i][j]
				if gsc.param_not_in_home(param):
					PARAM = param
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
	slacgs = Simulator(model)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc)

	return True


def run_custom_simulation(param, dims_to_compare):
	""" run a custom simulation

	:param param: parameters list
	:type param: list[float|int] or tuple[float|int]

	:param dims_to_compare: dimensionalities to compare
	:type dims_to_compare: list[int] or tuple[int]

	:returns: 0 if simulation was successful
	:rtype: int

	:raises TypeError:
		if param is not a list[int|float] or tuple[int|float];
		if dims_to_compare is not a list[int] or tuple[int];

	:raises ValueError:
		if param is not a valid parameter list;
		if dims_to_compare is not a valid dimensionalities list;

	"""

	if not isinstance(param, (list, tuple)):
		raise TypeError("param must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in param):
		raise TypeError("param must be a list or tuple of int or float")

	if not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	start_google_drive_service()



	## create model object
	model = Model(param)

	## create simulator object
	slacgs = Simulator(model, dims=dims_to_compare)


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_simulations'

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id)

	## create gspread client object
	gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc)


def run_custom_scenario(scenario_list, scenario_number, dims_to_simulate, dims_to_compare):
	""" run a custom scenario

	:param scenario_list: scenario list
	:type scenario_list: list[list[float|int]] or tuple[list[float|int]]

	:param dims_to_simulate: dimensionalities to simulate (if None, all dimensionalities will be simulated)
	:type dims_to_simulate: list[int] or tuple[int]

	:param dims_to_compare: dimensionalities to compare
	:type dims_to_compare: list[int] or tuple[int]

	:returns: 0 if simulation was successful
	:rtype: int

	:raises TypeError:
		if scenario is not a list[list[float|int]] or tuple[list[float|int]];
		if dims_to_simulate is not a list[int] or tuple[int];
		if dims_to_compare is not a list[int] or tuple[int];

	:raises ValueError:
		if scenario is not a valid scenario list;
		if dims_to_compare is not a subset of dims_to_simulate;

	"""

	if not isinstance(scenario_list, (list, tuple)):
		raise TypeError("scenario must be a list or tuple")

	if not all(isinstance(x, (list, tuple)) for x in scenario_list):
		raise TypeError("scenario must be a list or tuple of list or tuple")

	if not all(isinstance(x, (int, float)) for y in scenario_list for x in y):
		raise TypeError("scenario must be a list or tuple of list or tuple of int or float")

	if not isinstance(dims_to_simulate, (list, tuple)):
		raise TypeError("dims_to_simulate must be a list or tuple")

	if not all(isinstance(x, int) for x in dims_to_simulate):
		raise TypeError("dims_to_simulate must be a list or tuple of int")

	if not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	## create Model objects to test each parameter set before continuing
	models = []
	for param in scenario_list:
		models.append(Model(param))

	## create Simulator objects to test each parameter set before continuing
	simulators = []
	for model in models:
		simulators.append(Simulator(model, dims=dims_to_simulate))

	save_scenario_figures_as_gif(scenario_list, scenario_number)

	if not all(dim in dims_to_simulate for dim in dims_to_compare):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_scenario' + str(scenario_number)

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id)

	## create gspread client object
	gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

	for slacgs in simulators:
		if dims_to_compare == (2,3) or dims_to_compare == [2,3]:
			if gsc.param_not_in_home(slacgs.model.param):
				## run simulation
				slacgs.run()

				## write results to spreadsheet
				slacgs.report.write_to_spreadsheet(gsc)
			else:
				continue

		else:
			## run simulation
			slacgs.run()

			## write results to spreadsheet
			slacgs.report.write_to_spreadsheet(gsc)


def add_simulation_to_experiment_scenario_spreadsheet(scenario_number, param):
	""" add simulation results to the scenario spreadsheet on google drive

	:param scenario_number: scenario number
	:type scenario_number: int

	:param param: parameters list
	:type param: list[float|int] or tuple[float|int]

	:returns: 0 if simulation was successful
	:rtype: int

	:raises TypeError:
		if scenario is not an int;
		if param is not a list[int|float] or tuple[int|float]

	:raises ValueError:
		if param is not a valid parameter list;
		if scenario is not between 1 and 4


	"""

	if not isinstance(scenario_number, int):
		raise TypeError("scenario must be an int")

	if not isinstance(param, (list, tuple)):
		raise TypeError("param must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in param):
		raise TypeError("param must be a list or tuple of int or float")

	if scenario_number < 1 or scenario_number > 4:
		raise ValueError("scenario must be between 1 and 4")

	if len(param) != 6:
		raise ValueError("param must be a list or tuple of 6 elements for this experiment")

	if scenario_number == 1:
		if param[0] != 1 or param[1] != 1 or param[3] != 0 or param[4] != 0 or param[5] != 0:
			raise ValueError("for scenario 1, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[3] = 0, param[4] = 0 and param[5] = 0")

	elif scenario_number == 2:
		if param[0] != 1 or param[1] != 1 or param[2] != 2 or param[4] != 0 or param[5] != 0:
			raise ValueError("for scenario 2, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2, param[4] = 0 and param[5] = 0")
		if param[3] < -0.8 or param[3] > 0.8:
			raise ValueError("for scenario 2, param[3] must be between -0.8 and 0.8")

	elif scenario_number == 3:
		if param[0] != 1 or param[1] != 1 or param[2] != 2 or param[3] != 0:
			raise ValueError("for scenario 3, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2 and param[3] = 0")
		if param[4] < -0.7 or param[4] > 0.7 or param[4] != param[5]:
			raise ValueError("for scenario 3, param[4] must be between -0.7 and 0.7 and param[4] must be equal to param[5]")

	elif scenario_number == 4:
		if param[0] != 1 or param[1] != 1 or param[2] != 2 or param[3] != -0.1:
			raise ValueError("for scenario 4, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2 and param[3] = -0.1")
		if param[4] < -0.6 or param[4] > 0.6 or param[4] != param[5]:
			raise ValueError("for scenario 4, param[4] must be between -0.7 and 0.7 and param[4] must be equal to param[5]")


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'scenario' + str(scenario_number)

	## create spreadsheet if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id)

	## create gspread client object
	gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

	## create model object
	model = Model(param)

	## update scenario gif
	save_scenario_figures_as_gif([model], scenario_number)

	## create simulator object
	slacgs = Simulator(model)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc)


def run_experiment(start_scenario=1):
	""" run the experiment test for the simulator and return 0 if all parameters have been simulated

	:param start_scenario: scenario to start the experiment test
	:type start_scenario: int

	:returns: 0 if all parameters have been simulated
	:rtype: int

	:raises ValueError: if start_scenario is not between 1 and 4
	:raises TypeError: if start_scenario is not an int

	"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if start_scenario < 1 or start_scenario > 4:
		raise ValueError("start_scenario must be between 1 and 4")

	for index in range(len(SCENARIOS)):
		save_scenario_figures_as_gif(SCENARIOS[index], index + 1)

	while run_experiment_simulation(start_scenario):
		continue

	print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
	return 0


def doctest_next_parameter():
	""" return the next parameter to be simulated on doctests, and also the adequate spreadsheet title

	:returns: PARAM, SPREADSHEET_TITLE
	:rtype: tuple

	:Example:
		>>> from slacgs.demo import *
		>>> set_report_service_conf(password, user_email)
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

			gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for params in SCENARIOS[i]:
				if gsc.param_not_in_home(params):
					PARAM = params
					break

			## if all parameters are in the spreadsheet report home, then go to the next spreadsheet
			if PARAM:
				break

	return PARAM, SPREADSHEET_TITLE


def run_experiment_simulation_test(start_scenario=1, verbose=True):
	""" run a simulation test for one of the experiment scenarios and return True if there are still parameters to be simulated and False otherwise.

	Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario
	and another one for the Custom Simulations.
	The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service
	account and shared with the user's Google Drive account.
	Also, images with data visualization will be exported to a local folder inside project's root folder ( slacgs/images/ )

	Reports Exported:
		Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		Home Report: Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported:
		Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

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
		N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

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
		>>> set_report_service_conf(password, user_email)
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
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = GDC.create_folder(REPORT_FOLDER_NAME)  # create folder
		GDC.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	SPREADSHEET_TITLE = 'scenario1.test'
	## create spreadsheet for the first simulation if it doesn't exist
	if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
		GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)
		gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)
		PARAM = SCENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(start_scenario - 1, len(SCENARIOS)):
			SPREADSHEET_TITLE = 'scenario' + str(i + 1) + '.test'

			## create spreadsheet if it doesn't exist
			if not GDC.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = GDC.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
				folder_id = GDC.get_folder_id_by_name(REPORT_FOLDER_NAME)
				GDC.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

			gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for j in range(len(SCENARIOS[i])):
				params = SCENARIOS[i][j]
				if gsc.param_not_in_home(params):
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
	slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n=1024,
	                   verbose=verbose)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number, verbose=True):
	""" add simulation results to one of the experiment scenario spreadsheets

	Parameters:
		params (list[float|int] or tuple[float|int]): a list containnong Sigmas and Rhos
		scenario_number (int): scenario number
		verbose (bool): if True, print simulation progress (default: True)

	Returns:
		int: 0 if simulation was successful

	Raises:
		TypeError:
			if scenario is not an int;
			if params is not a list[int|float] or tuple[int|float]
		ValueError:
			if params is not a valid parameter list;
			if scenario is not between 1 and 4

	Example:
		>>> from slacgs.demo import *
		>>> set_report_service_conf(password, user_email)

		>>> scenario_number = 1
		>>> params = [1, 1, 2.1, 0, 0, 0]
		>>> add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number, verbose=False)

		>>> scenario_number = 2
		>>> params = [1, 1, 2, -0.15, 0, 0]
		>>> add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number, verbose=False)

		>>> scenario_number = 3
		>>> params = [1, 1, 2, 0, 0.15, 0.15]
		>>> add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number, verbose=False)

		>>> scenario_number = 4
		>>> params = [1, 1, 2, -0.1, 0.15, 0.15]
		>>> add_simulation_to_experiment_scenario_spreadsheet_test(params, scenario_number, verbose=False)


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
			raise ValueError("for scenario 3, params[4] must be between -0.7 and 0.7 and params[4] must be equal to params[5]")

	elif scenario_number == 4:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[3] != -0.1:
			raise ValueError(
				"for scenario 4, params must be a list or tuple of 6 elements where params[0] = 1, params[1] = 1, params[2] = 2 and params[3] = -0.1")
		if params[4] < -0.6 or params[4] > 0.6 or params[4] != params[5]:
			raise ValueError("for scenario 4, params[4] must be between -0.6 and 0.6 and params[4] must be equal to params[5]")

	## update scenario gif
	save_scenario_figures_as_gif([params], scenario_number, verbose=verbose)

	start_google_drive_service()

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
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
	gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

	## create model object
	model = Model(params)

	## create simulator object
	slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n=1024,
	                   verbose=verbose)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

def run_custom_scenario_test(scenario_list, scenario_number, dims_to_simulate=None, dims_to_compare=None, verbose=True):
	""" run a custom test scenario and write the results to a Google Spreadsheet shared with the user.
	A Scenario is a list with params to simulate.

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
			if scenario_list is not a valid scenario list;
			if scenario_number is not a valid scenario number;
			if dims_to_simulate is not a valid dimensionalities list;


	Example:
		>>> from slacgs.demo import *
		>>> set_report_service_conf(password, user_email)

		>>> scenario_list = [[1,1,3,round(0.1*rho,1),0,0] for rho in range(-1,2)]
		>>> scenario_number = 5
		>>> run_custom_scenario_test(scenario_list, scenario_number, verbose=False)
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
			Simulator(model, dims=dims_to_simulate, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4,
			          augmentation_until_n=1024, verbose=verbose))

	save_scenario_figures_as_gif(scenario_list, scenario_number, verbose=verbose)

	if dims_to_compare and not all(dim in dims_to_simulate for dim in dims_to_compare):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
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
	gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

	for slacgs in simulators:
		if dims_to_compare == (2, 3) or dims_to_compare == [2, 3]:
			if gsc.param_not_in_home(slacgs.model.params):
				## run simulation
				slacgs.run()

				## write results to spreadsheet
				slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)
			else:
				continue

		else:
			## run simulation
			slacgs.run()

			## write results to spreadsheet
			slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def add_simulation_to_custom_scenario_spreadsheet_test(params, scenario_number, dims_to_simulate=None, dims_to_compare=None, verbose=True):
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
			if scenario_number is not a positive integer;
			if dims_to_compare is not a subset of dims_to_simulate


	Example:

		>>> from slacgs.demo import *
		>>> set_report_service_conf(password, user_email)

		>>> params = (1, 1, 3, -0.2, 0, 0)
		>>> scenario_number = 5
		>>> dims_to_simulate = (1, 2, 3)
		>>> dims_to_compare = (2, 3)
		>>> add_simulation_to_custom_scenario_spreadsheet_test(params, scenario_number, dims_to_simulate, dims_to_compare, verbose=False)

	"""

	if not isinstance(params, (list, tuple)):
		raise TypeError("params must be a list or tuple")

	if not isinstance(scenario_number, int):
		raise TypeError("scenario_number must be an int")

	if dims_to_simulate and not isinstance(dims_to_simulate, (list, tuple)):
		raise TypeError("dims_to_simulate must be a list or tuple")

	if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if scenario_number < 1:
		raise ValueError("scenario_number must be a positive integer")


	if dims_to_compare and not set(dims_to_compare).issubset(set(dims_to_simulate)):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")


	## update scenario gif
	save_scenario_figures_as_gif([params], scenario_number, verbose=verbose)

	start_google_drive_service()

	## create Model object to test parameter set before continuing
	model = Model(params)

	## create Simulator object to test parameters before continuing
	slacgs = Simulator(model, dims=dims_to_simulate, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4,
	                   augmentation_until_n=1024, verbose=verbose)

	if not set(dims_to_compare).issubset(set(dims_to_simulate)):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
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
	gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def run_custom_simulation_test(params, dims_to_simulate=None, dims_to_compare=None, verbose=True):
	""" run a custom simulation for any dimensionality and cardinality

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
			if dims_to_compare is not a subset of dims_to_simulate


	:Example:
		>>> from slacgs.demo import *
		>>> set_report_service_conf(password, user_email)

		>>> ## 2 features
		>>> params = [1, 2, 0.4]
		>>> dims_to_compare = (1, 2)
		>>> run_custom_simulation_test(params, dims_to_compare, verbose=False)

		>>> ## 3 features
		>>> params = [1, 1, 4, -0.2, 0.1, 0.1]
		>>> dims_to_compare = (2, 3)
		>>> run_custom_simulation_test(params, dims_to_compare, verbose=False)

		>>> ## 4 features
		>>> params = [1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
		>>> dims_to_compare = (3, 4)
		>>> run_custom_simulation_test(params, dims_to_compare, verbose=False)

		>>> ## 5 features
		>>> params = [1, 1, 2, 2, 2, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.2, 0, 0, 0]
		>>> dims_to_compare = (2, 5)
		>>> run_custom_simulation_test(params, dims_to_compare, verbose=False)


	"""

	if not isinstance(params, (list, tuple)):
		raise TypeError("params must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in params):
		raise TypeError("params must be a list or tuple of int or float")

	if dims_to_compare and not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if dims_to_compare and not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	if dims_to_compare and not set(dims_to_compare).issubset(set(dims_to_simulate)):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")

	## initialize gdrive client if it hasn't been initialized yet
	start_google_drive_service()

	## create model object
	model = Model(params)

	## create simulator object
	slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4,
	                   augmentation_until_n=1024, verbose=verbose)


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + GDC.gdrive_account_email

	## create folder if it doesn't exist
	if not GDC.check_folder_existence(REPORT_FOLDER_NAME):
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
	gsc = GspreadClient(report_service_conf['pygsheets_service'], SPREADSHEET_TITLE)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, dims_to_compare=dims_to_compare, verbose=verbose)

	return True


def run_experiment_test(start_scenario=1, verbose=True):
	""" run all simulations in all experiment scenarios

	Reports with results will be stored in a Google Spreadsheet for each:  Experiment Scenario, Custom Experiment Scenario
	and another one for the Custom Simulations.
	The Spreadsheets are stored in a Google Drive folder named 'slacgs.demo.<user_email>'	owned by slacgs' google service
	account and shared with the user's Google Drive account.
	Also, images with data visualization will be exported to a local folder inside project's root folder ( slacgs/images/ )

	Reports Exported:
		Loss Report: Contains mainly results focused on Loss Functions evaluations for each dimensionality of the model.
		Compare Resport: Contains mainly results focused on comparing the performance of the Model using 2 features and 3 features.
		Home Report: Contains results from all simulations in a Scenario and links to the other reports. (available only for comparison between 2D and 3D)

	Images Exported:
		Scenario Data plots .gif: Contains a gif with all plots with the data points (n = 1024, dims=[2,3] ) generated for all Models in an Experiment Scenario.
		Simulation Data plot .png: Contains a plot with the data points (n = 1024, dims=[2,3] ) generated for a Model in a Simulation.
		Simulation Loss plot .png: Contains a plot with the loss values (Theoretical, Empirical with Train Data, Empirical with Test data) generated for a Model in a Simulation.

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
		N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]  (+ [2048, 4096, 8192] if L_3 > L_2)

	Parameters:
		start_scenario: scenario to start the experiment test
		verbose: if True, prints messages to console output

	Returns:
	 		0 if all parameters have been simulated


	Raises:
		ValueError: if start_scenario is not between 1 and 4
		TypeError: if start_scenario is not an int


	:param start_scenario: scenario to start the experiment test
	:type start_scenario: int

	:returns: 0 if all parameters have been simulated
	:rtype: int

	:raises ValueError: if start_scenario is not between 1 and 4
	:raises TypeError: if start_scenario is not an int

	:Example:
		>>> from slacgs.demo import *
		>>> set_report_service_conf(password, user_email)
		>>> run_experiment_test()

	"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if start_scenario < 1 or start_scenario > 4:
		raise ValueError("start_scenario must be between 1 and 4")

	for index in range(len(SCENARIOS)):
		save_scenario_figures_as_gif(SCENARIOS[index], index + 1, verbose=verbose)

	while run_experiment_simulation_test(start_scenario):
		continue

	if verbose:
		print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
	return 0


def save_scenario_figures_as_gif(scenario, scenario_number, export_path=None, duration=200, loop=0, verbose=True):
	"""
	Save a list of matplotlib Figure objects as an animated GIF.

	Parameters:
		scenario (list[tuple[float|int]]): A list of parameter sets to simulate.
		scenario_number (int): The scenario number.
		export_path (str): The file path where the animated GIF will be saved.
		duration (int, optional): The duration (in milliseconds) between frames. Default is 200ms.
		loop (int, optional): The number of loops for the animation. Use 0 for infinite looping (default).

	Returns:
			None
	"""
	# Ensure the export path has the ".gif" extension
	if export_path is None:
		export_path = get_grandparent_folder_path()
		export_path += '\\images\\' if os.name == 'nt' else '/images/'
		export_path += 'scenario' + str(scenario_number) + '.gif'
	elif not export_path.endswith(".gif"):
		export_path += ".gif"

	# Get the list of figure objects
	param_figures_list = [(model.params, model.data_points_plot) for model in [Model(params) for params in scenario]]

	# Create a temporary directory to store the individual frame images
	scenario_figs_dir = get_grandparent_folder_path()
	scenario_figs_dir += '\\images\\' if os.name == 'nt' else '/images/'
	scenario_figs_dir += 'scenario' + str(scenario_number) + '_figures'
	try:
		os.makedirs(scenario_figs_dir)
	except OSError:
		pass


	try:
		# Save each figure as an individual frame image
		for i in range(len(param_figures_list)):
			params = param_figures_list[i][0]
			fig = param_figures_list[i][1]
			frame_path = os.path.join(scenario_figs_dir, str(params) + '.png')
			if not os.path.exists(frame_path):
				fig.savefig(frame_path, format="png", dpi=300)
			plt.close(fig)

		# Create the animated GIF from the frame images
		frames = [Image.open(os.path.join(scenario_figs_dir, f)) for f in os.listdir(scenario_figs_dir) if f.endswith(".png")]
		frames[0].save(export_path, format="GIF", append_images=frames[1:], save_all=True, duration=duration, loop=loop)

		if verbose:
			print(f"Animated GIF saved as: {export_path}")
	except Exception as e:
		print(f"Failed to save the animated GIF: {e}")



def print_experiment_scenarios():

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
    table = tabulate(indexed_data , tablefmt='grid', headers=['Scenario 1','Scenario 2','Scenario 3','Scenario 4'])
    print(table)