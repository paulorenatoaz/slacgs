from slacgs.model import Model
from slacgs import Simulator
from slacgs import GspreadClient
from slacgs import GdriveClient
import os


try:
	import google.colab
	IN_COLAB = True
except:
	IN_COLAB = False

if IN_COLAB:
	KEY_PATH = '/content/key.json'
else:
	if os.name == 'nt':
		KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '\\key.json'
	else:
		KEY_PATH = os.path.dirname(os.path.abspath(__file__)) + '/key.json'

spreadsheet_name = 'sim_test.py'  # Replace with your desired spreadsheet name

gdc = GdriveClient(KEY_PATH)

spreadsheet_id = gdc.create_spreadsheet(spreadsheet_name)
folder_id = gdc.get_folder_id_by_name('slacgs.test')

gdc.move_file_to_folder(spreadsheet_id, folder_id)

gsc = GspreadClient(KEY_PATH, 'sim_test.py')

param = [1, 1, 2, 0, 0, 0]

model = Model(param, N=[2**i for i in range(1,11)], max_n=1024)

slacgs = Simulator(model, iters_per_step=1, max_steps=10, first_step=5, precision=1e-4, augmentation_until_n=1024)

slacgs.run()

slacgs.report.write_to_spreadsheet(gsc)
