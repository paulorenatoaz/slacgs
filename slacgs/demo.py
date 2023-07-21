import contextlib
import io
from math import sqrt
from .model import Model
from .simulator import Simulator
from .gspread_client import GspreadClient
from .gdrive_client import GdriveClient
from .utils import is_valid_email
import os
import re

## define list of parameters for scenario 1
SCENARIO1 = [[1, 1, round(1 + 0.1 * sigma3, 2), 0, 0, 0] for sigma3 in range(3, 10)]
SCENARIO1 += [[1, 1, sigma3 / 2, 0, 0, 0] for sigma3 in range(4, 11, 1)]
SCENARIO1 += [[1, 1, sigma3, 0, 0, 0] for sigma3 in range(6, 14, 1)]

## define list of parameters for scenario 2
SCENARIO2 = [[1, 1, 2, round(rho12 * 0.1, 1), 0, 0] for rho12 in range(-8, 9)]

## define list of parameters for scenario 3
rho12=0
SCENARIO3 = []
for r in range(-8,8):
  if  abs(round(0.1*r,1)) < sqrt((1+rho12)/2) :
    SCENARIO3 += [[1, 1, 2, rho12, round(0.1 * r, 1), round(0.1 * r, 1)]]

## define list of parameters for scenario 4
rho12=-0.1
SCENARIO4 = []
for r in range(-8,8):
  if  abs(round(0.1*r,1)) < sqrt((1+rho12)/2) :
    SCENARIO4 += [[1, 1, 2, rho12, round(0.1 * r, 1), round(0.1 * r, 1)]]

## create list of scenarios
SCENARIOS = [SCENARIO1, SCENARIO2, SCENARIO3, SCENARIO4]

if os.name == 'nt':  # check if running on Windows
	KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\key.py'
else:  # running on Linux or Mac
	KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '/key.py'

## create GdriveClient object
gdc = GdriveClient(KEY_PATH)


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

	## define path to Key file for accessing Google Sheets API via Service Account Credentials
	if not gdc.gdrive_account_mail:

		while True:
			user_email = input(
				"Please enter your google account email address so I can share results with you:\n(if you don't have a google account, create one at https://accounts.google.com/signup)\n")
			if is_valid_email(user_email):
				print("Valid email address!")
				break
			else:
				print("Invalid email address. Please try again.")

		# Store the user's input email as a string in the GdriveClient object
		gdc.gdrive_account_mail = user_email

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + gdc.gdrive_account_mail

	## create folder if it doesn't exist
	if not gdc.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME)  # create folder
		gdc.share_folder_with_gdrive_account(folder_id)  # share folder with user's google drive account

	SPREADSHEET_TITLE = 'sscenario1'
	## create spreadsheet for the first simulation if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id)
		gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)
		PARAM = SCENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(start_scenario - 1, len(SCENARIOS)):
			SPREADSHEET_TITLE = 'scenario' + str(i + 1)

			## create spreadsheet if it doesn't exist
			if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
				folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
				gdc.move_file_to_folder(spreadsheet_id, folder_id)

			gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

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

	:param dims_to_compare: dimensions to compare
	:type dims_to_compare: list[int] or tuple[int]

	:returns: 0 if simulation was successful
	:rtype: int

	:raises TypeError:
		if param is not a list[int|float] or tuple[int|float];
		if dims_to_compare is not a list[int] or tuple[int];

	:raises ValueError:
		if param is not a valid parameter list;
		if dims_to_compare is not a valid dimensions list;

	"""

	if not isinstance(param, (list, tuple)):
		raise TypeError("param must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in param):
		raise TypeError("param must be a list or tuple of int or float")

	if not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")


	## create model object
	model = Model(param)

	## create simulator object
	slacgs = Simulator(model, dims=dims_to_compare)

	## share results with user if not already shared
	if not gdc.gdrive_account_mail:
		# Ask user for google account email address
		while True:
			user_email = input(
				"Please enter your google account email address so I can share results with you:\n(if you don't have a google account, create one at https://accounts.google.com/signup)\n")
			if is_valid_email(user_email):
				print("Valid email address!")
				break
			else:
				print("Invalid email address. Please try again.")

		# Store the user's input email as a string in the GdriveClient object
		gdc.gdrive_account_mail = user_email

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + gdc.gdrive_account_mail

	## create folder if it doesn't exist
	if not gdc.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME)  # create folder
		gdc.share_folder_with_gdrive_account(folder_id)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_simulations'

	## create spreadsheet if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id)

	## create gspread client object
	gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc)


def run_custom_scenario(scenario_list, scenario_number, dims_to_simulate, dims_to_compare):
	""" run a custom scenario

	:param scenario_list: scenario list
	:type scenario_list: list[list[float|int]] or tuple[list[float|int]]

	:param dims_to_simulate: dimensions to simulate
	:type dims_to_simulate: list[int] or tuple[int]

	:param dims_to_compare: dimensions to compare
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

	if not all(dim in dims_to_simulate for dim in dims_to_compare):
		raise ValueError("dims_to_compare must be a subset of dims_to_simulate")

	## share results with user if not already shared
	if not gdc.gdrive_account_mail:
		# Ask user for google account email address
		while True:
			user_email = input(
				"Please enter your google account email address so I can share results with you:\n(if you don't have a google account, create one at https://accounts.google.com/signup)\n")
			if is_valid_email(user_email):
				print("Valid email address!")
				break
			else:
				print("Invalid email address. Please try again.")

		# Store the user's input email as a string in the GdriveClient object
		gdc.gdrive_account_mail = user_email

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + gdc.gdrive_account_mail

	## create folder if it doesn't exist
	if not gdc.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME)  # create folder
		gdc.share_folder_with_gdrive_account(folder_id)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_scenario' + str(scenario_number)

	## create spreadsheet if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id)

	## create gspread client object
	gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

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


def add_simulation_to_experiment_scenario_spreadsheet(scenario, param):
	""" add simulation results to the scenario spreadsheet on google drive

	:param scenario: scenario number
	:type scenario: int

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

	if not isinstance(scenario, int):
		raise TypeError("scenario must be an int")

	if not isinstance(param, (list, tuple)):
		raise TypeError("param must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in param):
		raise TypeError("param must be a list or tuple of int or float")

	if scenario < 1 or scenario > 4:
		raise ValueError("scenario must be between 1 and 4")

	if len(param) != 6:
		raise ValueError("param must be a list or tuple of 6 elements for this experiment")

	if scenario == 1:
		if param[0] != 1 or param[1] != 1 or param[3] != 0 or param[4] != 0 or param[5] != 0:
			raise ValueError("for scenario 1, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[3] = 0, param[4] = 0 and param[5] = 0")

	elif scenario == 2:
		if param[0] != 1 or param[1] != 1 or param[2] != 2 or param[4] != 0 or param[5] != 0:
			raise ValueError("for scenario 2, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2, param[4] = 0 and param[5] = 0")
		if param[3] < -0.8 or param[3] > 0.8:
			raise ValueError("for scenario 2, param[3] must be between -0.8 and 0.8")

	elif scenario == 3:
		if param[0] != 1 or param[1] != 1 or param[2] != 2 or param[3] != 0:
			raise ValueError("for scenario 3, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2 and param[3] = 0")
		if param[4] < -0.7 or param[4] > 0.7 or param[4] != param[5]:
			raise ValueError("for scenario 3, param[4] must be between -0.7 and 0.7 and param[4] must be equal to param[5]")

	elif scenario == 4:
		if param[0] != 1 or param[1] != 1 or param[2] != 2 or param[3] != -0.1:
			raise ValueError("for scenario 4, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2 and param[3] = -0.1")
		if param[4] < -0.6 or param[4] > 0.6 or param[4] != param[5]:
			raise ValueError("for scenario 4, param[4] must be between -0.7 and 0.7 and param[4] must be equal to param[5]")

	## share results with user if not already shared
	if not gdc.gdrive_account_mail:
		# Ask user for google account email address
		while True:
			user_email = input(
				"Please enter your google account email address so I can share results with you:\n(if you don't have a google account, create one at https://accounts.google.com/signup)\n")
			if is_valid_email(user_email):
				print("Valid email address!")
				break
			else:
				print("Invalid email address. Please try again.")

		# Store the user's input email as a string in the GdriveClient object
		gdc.gdrive_account_mail = user_email

	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.demo.' + gdc.gdrive_account_mail

	## create folder if it doesn't exist
	if not gdc.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME)  # create folder
		gdc.share_folder_with_gdrive_account(folder_id)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'scenario' + str(scenario)

	## create spreadsheet if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id)

	## create gspread client object
	gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

	## create model object
	model = Model(param)

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

	while run_experiment_simulation(start_scenario):
		continue

	print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
	return 0
