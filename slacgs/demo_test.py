from .demo import *



def doctest_next_parameter():
	""" return the next parameter to be simulated on doctests, and also the adequate spreadsheet title

	:returns: PARAM, SPREADSHEET_TITLE
	:rtype: tuple

	:Example:
		>>> from slacgs.demo_test import *
		>>> params, spreadsheet_title = doctest_next_parameter()


	"""

	REPORT_FOLDER_NAME = 'slacgs.doctest'
	SPREADSHEET_TITLE = 'scenario1.doctest'
	## create spreadsheet for the first simulation if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		with contextlib.redirect_stdout(io.StringIO()):
			## do operations without printing to stdout
			spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
			folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
			gdc.move_file_to_folder(spreadsheet_id, folder_id)
		PARAM = SCENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(len(SCENARIOS)):
			SPREADSHEET_TITLE = 'scenario' + str(i + 1) + '.doctest'

			## create spreadsheet if it doesn't exist
			if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
				folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
				gdc.move_file_to_folder(spreadsheet_id, folder_id)

			gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for param in SCENARIOS[i]:
				if gsc.param_not_in_home(param):
					PARAM = param
					break

			## if all parameters are in the spreadsheet report home, then go to the next spreadsheet
			if PARAM:
				break

	return PARAM, SPREADSHEET_TITLE


def run_experiment_test_simulation(start_scenario=1, verbose=True):
	""" run the simulation test for the simulator and return True if there are still parameters to be simulated and False otherwise

	:param start_scenario: scenario to start the simulation test
	:type start_scenario: int
	:returns: True if there are still parameters to be simulated and False otherwise
	:rtype: bool

	:raises ValueError: if start_scenario is not between 1 and 4
	:raises TypeError: if start_scenario is not an int

	:Example:
		>>> from slacgs.demo_test import *
		>>> gdc.gdrive_account_mail = user_email
		>>> run_experiment_test_simulation(verbose=False)
		>>> run_experiment_test_simulation(start_scenario=2, verbose=False)
		>>> run_experiment_test_simulation(start_scenario=3, verbose=False)
		>>> run_experiment_test_simulation(start_scenario=4, verbose=False)

	"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if not 1 <= start_scenario <= 4:
		raise ValueError("start_scenario must be between 1 and 4")

	## define path to Key file for accessing Google Sheets API via Service Account Credentials
	if not gdc.gdrive_account_mail:

		while True:
			user_email = input(
				"Please enter your google account email address so I can share results with you:\n(if you don't have a google account, create one at https://accounts.google.com/signup) ")
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
		gdc.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	SPREADSHEET_TITLE = 'scenario1.test'
	## create spreadsheet for the first simulation if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)
		gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)
		PARAM = SCENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(start_scenario - 1, len(SCENARIOS)):
			SPREADSHEET_TITLE = 'scenario' + str(i + 1) + '.test'

			## create spreadsheet if it doesn't exist
			if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
				folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
				gdc.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

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
	slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n=1024,
	                   verbose=verbose)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)

	return True


def run_custom_test_simulation(params, dims_to_compare, verbose=True):
	""" run a custom simulation

	:param params: parameters list
	:type params: list[float|int] or tuple[float|int]

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

	:Example:
		>>> from slacgs.demo_test import *

		>>> ## 2 features
		>>> param = [1, 2, 0.4]
		>>> dims_to_compare = (1, 2)
		>>> gdc.gdrive_account_mail = user_email
		>>> run_custom_test_simulation(params, dims_to_compare, verbose=False)

		>>> ## 3 features
		>>> param = [1, 1, 4, -0.2, 0.1, 0.1]
		>>> dims_to_compare = (2, 3)
		>>> gdc.gdrive_account_mail = user_email
		>>> run_custom_test_simulation(params, dims_to_compare, verbose=False)

		>>> ## 4 features
		>>> param = [1, 1, 1, 2, 0, 0, 0, 0, 0, 0]
		>>> dims_to_compare = (3, 4)
		>>> gdc.gdrive_account_mail = user_email
		>>> run_custom_test_simulation(params, dims_to_compare, verbose=False)

		>>> ## 5 features
		>>> param = [1, 1, 2, 2, 2, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.2, 0, 0, 0]
		>>> dims_to_compare = (2, 5)
		>>> gdc.gdrive_account_mail = user_email
		>>> run_custom_test_simulation(params, dims_to_compare, verbose=False)


	"""

	if not isinstance(params, (list, tuple)):
		raise TypeError("param must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in params):
		raise TypeError("param must be a list or tuple of int or float")

	if not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	## create model object
	model = Model(params)

	## create simulator object
	slacgs = Simulator(model, dims=dims_to_compare, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4,
	                   augmentation_until_n=1024, verbose=verbose)

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
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		gdc.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_simulations.test'

	## create spreadsheet if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)


def add_test_simulation_to_test_experiment_scenario_spreadsheet(params, scenario_number, verbose=True):
	""" add simulation results to the scenario spreadsheet on google drive

	:param params: parameters list
	:type params: list[float|int] or tuple[float|int]

	:param scenario_number: scenario number
	:type scenario_number: int

	:returns: 0 if simulation was successful
	:rtype: int

	:raises TypeError:
		if scenario is not an int;
		if param is not a list[int|float] or tuple[int|float]

	:raises ValueError:
		if param is not a valid parameter list;
		if scenario is not between 1 and 4

	:Example:
		>>> from slacgs.demo_test import *
		>>> scenario_number = 1
		>>> param = [1, 1, 2.1, 0, 0, 0]
		>>> gdc.gdrive_account_mail = user_email
		>>> add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number, verbose=False)

		>>> scenario_number = 2
		>>> param = [1, 1, 2, 0, 0.15, 0.15]
		>>> gdc.gdrive_account_mail = user_email
		>>> add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number, verbose=False)

		>>> scenario_number = 4
		>>> param = [1, 1, 2, -0.1, 0.15, 0.15]
		>>> gdc.gdrive_account_mail = user_email
		>>> add_simulation_to_experiment_scenario_spreadsheet(params, scenario_number, verbose=False)


	"""

	if not isinstance(scenario_number, int):
		raise TypeError("scenario must be an int")

	if not isinstance(params, (list, tuple)):
		raise TypeError("param must be a list or tuple")

	if not all(isinstance(x, (int, float)) for x in params):
		raise TypeError("param must be a list or tuple of int or float")

	if scenario_number < 1 or scenario_number > 4:
		raise ValueError("scenario must be between 1 and 4")

	if len(params) != 6:
		raise ValueError("param must be a list or tuple of 6 elements for this experiment")

	if scenario_number == 1:
		if params[0] != 1 or params[1] != 1 or params[3] != 0 or params[4] != 0 or params[5] != 0:
			raise ValueError(
				"for scenario 1, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[3] = 0, param[4] = 0 and param[5] = 0")

	elif scenario_number == 2:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[4] != 0 or params[5] != 0:
			raise ValueError(
				"for scenario 2, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2, param[4] = 0 and param[5] = 0")
		if params[3] < -0.8 or params[3] > 0.8:
			raise ValueError("for scenario 2, param[3] must be between -0.8 and 0.8")

	elif scenario_number == 3:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[3] != 0:
			raise ValueError(
				"for scenario 3, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2 and param[3] = 0")
		if params[4] < -0.7 or params[4] > 0.7 or params[4] != params[5]:
			raise ValueError("for scenario 3, param[4] must be between -0.7 and 0.7 and param[4] must be equal to param[5]")

	elif scenario_number == 4:
		if params[0] != 1 or params[1] != 1 or params[2] != 2 or params[3] != -0.1:
			raise ValueError(
				"for scenario 4, param must be a list or tuple of 6 elements where param[0] = 1, param[1] = 1, param[2] = 2 and param[3] = -0.1")
		if params[4] < -0.6 or params[4] > 0.6 or params[4] != params[5]:
			raise ValueError("for scenario 4, param[4] must be between -0.6 and 0.6 and param[4] must be equal to param[5]")

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
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		gdc.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'scenario' + str(scenario_number) + '.test'

	## create spreadsheet if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

	## create model object
	model = Model(params)

	## create simulator object
	slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n=1024,
	                   verbose=verbose)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)


def run_custom_test_scenario(scenario_list, scenario_number, dims_to_simulate, dims_to_compare, verbose=True):
	""" run a custom test scenario and write the results to a Google Spreadsheet shared with the user

	:param scenario_list: scenario list
	:type scenario_list: list[list[float|int]] or tuple[list[float|int]]

	:param scenario_number: scenario number
	:type scenario_number: int

	:param dims_to_simulate: dimensions to simulate
	:type dims_to_simulate: list[int] or tuple[int]

	:param dims_to_compare: dimensions to compare
	:type dims_to_compare: list[int] or tuple[int]

	:returns: None
	:rtype: None

	:raises TypeError:
		if scenario_list is not a list[list[float|int]] or tuple[list[float|int]];
		if scenario_number is not an int;
		if dims_to_simulate is not a list[int] or tuple[int];
		if dims_to_compare is not a list[int] or tuple[int];

	:raises ValueError:
		if scenario_list is not a valid scenario list;
		if scenario_number is not a positive integer;
		if dims_to_compare is not a subset of dims_to_simulate;

	:Example:
		>>> from slacgs.demo_test import *
		>>> scenario_list = [[1,1,3,round(0.1*rho,1),0,0] for rho in range(-1,2)]
		>>> scenario_number = 5
		>>> dims_to_simulate = (1,2,3)
		>>> dims_to_compare = (2,3)
		>>> gdc.gdrive_account_mail = user_email
		>>> run_custom_test_scenario(scenario_list, scenario_number, dims_to_simulate, dims_to_compare, verbose=False)
	"""

	if not isinstance(scenario_list, (list, tuple)):
		raise TypeError("scenario must be a list or tuple")

	if not all(isinstance(x, (list, tuple)) for x in scenario_list):
		raise TypeError("scenario must be a list or tuple of list or tuple")

	if not all(isinstance(x, (int, float)) for y in scenario_list for x in y):
		raise TypeError("scenario must be a list or tuple of list or tuple of int or float")

	if not isinstance(scenario_number, int):
		raise TypeError("scenario_number must be an int")

	if not isinstance(dims_to_simulate, (list, tuple)):
		raise TypeError("dims_to_simulate must be a list or tuple")

	if not all(isinstance(x, int) for x in dims_to_simulate):
		raise TypeError("dims_to_simulate must be a list or tuple of int")

	if not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if not all(isinstance(x, int) for x in dims_to_compare):
		raise TypeError("dims_to_compare must be a list or tuple of int")

	if scenario_number < 1:
		raise ValueError("scenario_number must be a positive integer")

	## create Model objects to test each parameter set before continuing
	models = []
	for param in scenario_list:
		models.append(Model(param))

	## create Simulator objects to test each parameter set before continuing
	simulators = []
	for model in models:
		simulators.append(
			Simulator(model, dims=dims_to_simulate, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4,
			          augmentation_until_n=1024, verbose=verbose))

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
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		gdc.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_scenario' + str(scenario_number) + '.test'

	## create spreadsheet if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE, verbose=verbose)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

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


def add_test_simulation_to_custom_test_scenario_spreadsheet(params, scenario_number, dims_to_simulate, dims_to_compare, verbose=True):
	""" add a test simulation to a custom test scenario

	:param params: param
	:type params: list[float|int] or tuple[float|int]

	:param scenario_number: scenario number
	:type scenario_number: int

	:param dims_to_simulate: dimensions to simulate
	:type dims_to_simulate: list[int] or tuple[int]

	:param dims_to_compare: dimensions to compare
	:type dims_to_compare: list[int] or tuple[int]

	:returns: None
	:rtype: None

	:raises TypeError:
		if param is not a list[float|int] or tuple[float|int];
		if scenario_number is not an int;
		if dims_to_simulate is not a list[int] or tuple[int];
		if dims_to_compare is not a list[int] or tuple[int]


	:raises ValueError:
		if scenario_number is not a positive integer;
		if dims_to_compare is not a subset of dims_to_simulate;

	:Example:
		>>> from slacgs.demo_test import *
		>>> param = (1, 1, 2, -0.2, 0, 0)
		>>> scenario_number = 2
		>>> dims_to_simulate = (1, 2, 3)
		>>> dims_to_compare = (2, 3)
		>>> gdc.gdrive_account_mail = user_email
		>>> add_test_simulation_to_custom_test_scenario_spreadsheet(params, scenario_number, dims_to_simulate, dims_to_compare, verbose=False)
	"""

	if not isinstance(params, (list, tuple)):
		raise TypeError("param must be a list or tuple")

	if not isinstance(scenario_number, int):
		raise TypeError("scenario_number must be an int")

	if not isinstance(dims_to_simulate, (list, tuple)):
		raise TypeError("dims_to_simulate must be a list or tuple")

	if not isinstance(dims_to_compare, (list, tuple)):
		raise TypeError("dims_to_compare must be a list or tuple")

	if scenario_number < 1:
		raise ValueError("scenario_number must be a positive integer")

	## create Model object to test parameter set before continuing
	model = Model(params)

	## create Simulator object to test parameters before continuing
	slacgs = Simulator(model, dims=dims_to_simulate, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4,
	                   augmentation_until_n=1024, verbose=verbose)

	if not set(dims_to_compare).issubset(set(dims_to_simulate)):
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
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME, verbose=verbose)  # create folder
		gdc.share_folder_with_gdrive_account(folder_id, verbose=verbose)  # share folder with user's google drive account

	## define spreadsheet title
	SPREADSHEET_TITLE = 'custom_scenario' + str(scenario_number) + '.test'

	## create spreadsheet if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id, verbose=verbose)

	## create gspread client object
	gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc, verbose=verbose)


def run_test_experiment(start_scenario=1):
	""" run the experiment test for the simulator and return 0 if all parameters have been simulated

	:param start_scenario: scenario to start the experiment test
	:type start_scenario: int

	:returns: 0 if all parameters have been simulated
	:rtype: int

	:raises ValueError: if start_scenario is not between 1 and 4
	:raises TypeError: if start_scenario is not an int

	:Example:
		>>> from slacgs.demo_test import *
		>>> gdc.gdrive_account_mail = user_email
		>>> run_test_experiment()

	"""

	if not isinstance(start_scenario, int):
		raise TypeError("start_scenario must be an int")

	if start_scenario < 1 or start_scenario > 4:
		raise ValueError("start_scenario must be between 1 and 4")

	while run_experiment_test_simulation(start_scenario):
		continue

	print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
	return 0

