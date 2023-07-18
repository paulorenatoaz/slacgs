from slacgs import Model
from slacgs import Simulator
from slacgs import GspreadClient
from slacgs import GdriveClient
import os

## define path to Key file for accessing Google Sheets API via Service Account Credentials

try:  # check if running on Google Colab
	import google.colab
	IN_COLAB = True
except:
	IN_COLAB = False

if IN_COLAB:
	KEY_PATH = '/content/key.py'
else:
	if os.name == 'nt':  # check if running on Windows
		KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\key.py'
	else:  # running on Linux or Mac
		KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '/key.py'
		
## define spreadsheet title and folder name
REPORT_FOLDER_NAME = 'slacgs.test'  # Replace with your desired folder name
SPREADSHEET_TITLE = 'sim_test'  # Replace with your desired spreadsheet name

## create GdriveClient object
gdc = GdriveClient(KEY_PATH)

## create folder if it doesn't exist
if not gdc.check_spreadsheet_existence(SPREADSHEET_TITLE):
	spreadsheet_id = gdc.create_spreadsheet(SPREADSHEET_TITLE)
	folder_id = gdc.get_folder_id_by_name('slacgs.test')
	gdc.move_file_to_folder(spreadsheet_id, folder_id)

## create GspreadClient object
gsc = GspreadClient(KEY_PATH, SPREADSHEET_TITLE)

## create model object
param = [1, 1, 3, 0, 0, 0]
model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)

## create simulator object
slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n=1024)

## run simulation
slacgs.run()

## write results to spreadsheet
slacgs.report.write_to_spreadsheet(gsc)
