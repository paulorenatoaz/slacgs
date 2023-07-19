import contextlib
import io
from math import sqrt
from .model import Model
from .simulator import Simulator
from .gspread_client import GspreadClient
from .gdrive_client import GdriveClient
import os
import re

## define list of parameters for cenario 1
CENARIO1 = [[1, 1, round(1 + 0.1 * sigma3, 2), 0, 0, 0] for sigma3 in range(3, 10)]
CENARIO1 += [[1, 1, sigma3 / 2, 0, 0, 0] for sigma3 in range(4, 11, 1)]
CENARIO1 += [[1, 1, sigma3, 0, 0, 0] for sigma3 in range(6, 14, 1)]

## define list of parameters for cenario 2
CENARIO2 = [[1, 1, 2, round(rho12 * 0.1, 1), 0, 0] for rho12 in range(-8, 9)]

## define list of parameters for cenario 3
rho12=0
CENARIO3 = []
for r in range(-8,8):
  if  abs(round(0.1*r,1)) < sqrt((1+rho12)/2) :
    CENARIO3 += [[1, 1, 2, rho12, round(0.1 * r, 1), round(0.1 * r, 1)]]

## define list of parameters for cenario 4
rho12=-0.1
CENARIO4 = []
for r in range(-8,8):
  if  abs(round(0.1*r,1)) < sqrt((1+rho12)/2) :
    CENARIO4 += [[1, 1, 2, rho12, round(0.1 * r, 1), round(0.1 * r, 1)]]

## create list of cenarios
CENARIOS = [CENARIO1, CENARIO2, CENARIO3, CENARIO4]


if os.name == 'nt':  # check if running on Windows
	KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\key.py'
else:  # running on Linux or Mac
	KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '/key.py'

## create GdriveClient object
gdc = GdriveClient(KEY_PATH)

def experiment_test(start_cenario=1):
	""" run the experiment test for the simulator and return 0 if all parameters have been simulated

	:param start_cenario: cenario to start the experiment test
	:type start_cenario: int

	:returns: 0 if all parameters have been simulated
	:rtype: int

	:raises ValueError: if start_cenario is not between 1 and 4
	:raises TypeError: if start_cenario is not an int

	"""

	if not isinstance(start_cenario, int):
		raise TypeError("start_cenario must be an int")

	if start_cenario < 1 or start_cenario > 4:
		raise ValueError("start_cenario must be between 1 and 4")

	while simulation_test(start_cenario):
		continue

	print("All parameters have been simulated. Please check your google drive section: 'Shared with me' for results.")
	return 0

def doctest_next_parameter():
	""" return the next parameter to be simulated on doctests, and also the adequate spreadsheet title

	:returns: PARAM, SPREADSHEET_TITLE
	:rtype: tuple

	"""

	REPORT_FOLDER_NAME = 'slacgs.doctest'
	SPREADSHEET_TITLE = 'cenario1.doctest'
	## create spreadsheet for the first simulation if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		with contextlib.redirect_stdout(io.StringIO()):
			## do operations without printing to stdout
			spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
			folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
			gdc.move_file_to_folder(spreadsheet_id, folder_id)
		PARAM = CENARIOS[0][0]
	else:  # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(len(CENARIOS)):
			SPREADSHEET_TITLE = 'cenario' + str(i + 1) + '.doctest'

			## create spreadsheet if it doesn't exist
			if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
				folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
				gdc.move_file_to_folder(spreadsheet_id, folder_id)

			gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for param in CENARIOS[i]:
				if gsc.param_not_in_home(param):
					PARAM = param
					break

			## if all parameters are in the spreadsheet report home, then go to the next spreadsheet
			if PARAM:
				break

	return PARAM, SPREADSHEET_TITLE



def simulation_test(start_cenario=1):
	""" run the simulation test for the simulator and return True if there are still parameters to be simulated and False otherwise

	:param start_cenario: cenario to start the simulation test
	:type start_cenario: int
	:returns: True if there are still parameters to be simulated and False otherwise
	:rtype: bool

	:raises ValueError: if start_cenario is not between 1 and 4
	:raises TypeError: if start_cenario is not an int

	"""

	if not isinstance(start_cenario, int):
		raise TypeError("start_cenario must be an int")

	if not 1 <= start_cenario <= 4:
		raise ValueError("start_cenario must be between 1 and 4")


	## define path to Key file for accessing Google Sheets API via Service Account Credentials
	if not gdc.gdrive_account_mail:
		def is_valid_email(email):
			# Regular expression pattern for email validation
			pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
			return re.match(pattern, email)

		while True:
			user_email = input("Please enter your google account email address so I can share results with you:\n(if you don't have a google account, create one at https://accounts.google.com/signup) ")
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
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME) # create folder
		gdc.share_folder_with_gdrive_account(folder_id) # share folder with user's google drive account


	SPREADSHEET_TITLE = 'cenario1.test'
	## create spreadsheet for the first simulation if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id)
		gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)
		PARAM = CENARIOS[0][0]
	else: # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(start_cenario-1,len(CENARIOS)):
			SPREADSHEET_TITLE = 'cenario' + str(i+1) + '.test'

			## create spreadsheet if it doesn't exist
			if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
				folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
				gdc.move_file_to_folder(spreadsheet_id, folder_id)

			gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for j in range(CENARIOS[i]):
				param = CENARIOS[i][j]
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
	model = Model(PARAM, N=[2 ** i for i in range(1, 11)], max_n=1024)

	## create simulator object
	slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n=1024)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc)

	return True

def simulation():
	## define path to Key file for accessing Google Sheets API via Service Account Credentials
	if not gdc.gdrive_account_mail:
		def is_valid_email(email):
			# Regular expression pattern for email validation
			pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
			return re.match(pattern, email)

		while True:
			user_email = input("Please enter your google account email address so I can share results with you:\n(if you don't have a google account, create one at https://accounts.google.com/signup) ")
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
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME) # create folder
		gdc.share_folder_with_gdrive_account(folder_id) # share folder with user's google drive account


	SPREADSHEET_TITLE = 'cenario1'
	## create spreadsheet for the first simulation if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id)
		gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)
		PARAM = CENARIOS[0][0]
	else: # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(len(CENARIOS)):
			SPREADSHEET_TITLE = 'cenario' + str(i+1) + '.test'

			## create spreadsheet if it doesn't exist
			if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
				folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
				gdc.move_file_to_folder(spreadsheet_id, folder_id)

			gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for param in CENARIOS[i]:
				print(gsc.param_not_in_home(param))
				if gsc.param_not_in_home(param):
					PARAM = param
					break

			## if all parameters are in the spreadsheet report home, then go to the next spreadsheet
			if PARAM:
				break


	## create model object
	model = Model(PARAM, N=[2 ** i for i in range(1, 11)], max_n=1024)

	## create simulator object
	slacgs = Simulator(model)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc)

def simulation_doctest():
	## define path to Key file for accessing Google Sheets API via Service Account Credentials
	if not gdc.gdrive_account_mail:
		def is_valid_email(email):
			# Regular expression pattern for email validation
			pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
			return re.match(pattern, email)


	## define folder name for storing reports
	REPORT_FOLDER_NAME = 'slacgs.doctest.' + gdc.gdrive_account_mail

	## create folder if it doesn't exist
	if not gdc.check_folder_existence(REPORT_FOLDER_NAME):
		folder_id = gdc.create_folder(REPORT_FOLDER_NAME) # create folder
		gdc.share_folder_with_gdrive_account(folder_id) # share folder with user's google drive account


	SPREADSHEET_TITLE = 'cenario1.doctest'
	## create spreadsheet for the first simulation if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id)
		gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)
		PARAM = CENARIOS[0][0]
	else: # if spreadsheet already exists, then find the first parameter that is not in the spreadsheet report home
		for i in range(len(CENARIOS)):
			SPREADSHEET_TITLE = 'cenario' + str(i+1) + '.doctest'

			## create spreadsheet if it doesn't exist
			if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
				spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
				folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
				gdc.move_file_to_folder(spreadsheet_id, folder_id)

			gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

			## retrieve the first parameter that is not in the spreadsheet report home
			PARAM = None
			for param in CENARIOS[i]:
				if gsc.param_not_in_home(param):
					PARAM = param
					break

			## if all parameters are in the spreadsheet report home, then go to the next spreadsheet
			if PARAM:
				break


	## create model object
	model = Model(PARAM, N=[2 ** i for i in range(1, 11)], max_n=1024)

	## create simulator object
	slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n=1024)

	## run simulation
	slacgs.run()

	## write results to spreadsheet
	slacgs.report.write_to_spreadsheet(gsc)
