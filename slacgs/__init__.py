from .model import Model
from .simulator import Simulator
from .report import Report
from .enumtypes import DictionaryType, LossType
from .gspread_client import GspreadClient
from .gdrive_client import GdriveClient
from .utils import *
from .demo import *

## Report Service Folder Structure
images_path = report_service_conf['images_path']
reports_path = report_service_conf['reports_path']

try:
	os.makedirs(images_path)
	print(f"Folder created at '{images_path}'.")
except OSError as e:
	print(f"Images Folder at '{images_path}'.")

try:
	os.makedirs(reports_path)
	print(f"Folder created at '{reports_path}'.")
except OSError as e:
	print(f"Reports Folder at '{reports_path}'.\n")
