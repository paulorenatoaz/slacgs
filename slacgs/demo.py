from math import sqrt
from .model import Model
from .simulator import Simulator
from .gspread_client import GspreadClient
from .gdrive_client import GdriveClient
import os

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

def simulation_test():
	## define path to Key file for accessing Google Sheets API via Service Account Credentials

	try:  # check if running on Google Colab
		import google.colab
		IN_COLAB = True
	except:
		IN_COLAB = False

	if IN_COLAB:
		KEY_PATH = '/content/key.json'
	else:
		if os.name == 'nt':  # check if running on Windows
			KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\key.json'
		else:  # running on Linux or Mac
			KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '/key.json'

	## create GdriveClient object
	gdc = GdriveClient(KEY_PATH)

	## define folder name
	REPORT_FOLDER_NAME = 'slacgs.demo'  # Replace with your desired folder name

	## create folder if it doesn't exist
	if not gdc.check_folder_existence(REPORT_FOLDER_NAME):
		gdc.create_folder(REPORT_FOLDER_NAME) # create folder


	SPREADSHEET_TITLE = 'cenario1.test'
	## create spreadsheet for the first simulation if it doesn't exist
	if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
		spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
		folder_id = gdc.get_folder_id_by_name(REPORT_FOLDER_NAME)
		gdc.move_file_to_folder(spreadsheet_id, folder_id)
		gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)
		PARAM = CENARIOS[0][0]
	else:
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
