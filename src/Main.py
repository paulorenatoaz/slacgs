import pygsheets
from GspreadClient import GspreadClient
from Model import Model
from Simulator import Simulator


key_path = GSPREAD_KEY
spredsheet_title = SPREADSHEET_TITLE

gc = pygsheets.authorize(service_file=key_path)
sh = gc.open(spredsheet_title)

ws_home = sh.worksheet(value=0)

gc_client = GspreadClient(key_path, spredsheet_title)

already_done = ws_home.get_values((2, 1), (ws_home.rows, 6), value_render='FORMULA')

model = Model(param, N=[2 ** i for i in range(1, 10)])

#start simulation with default values  for model
sim = Simulator(model)

sim.start()

report = sim.report


dims_to_compare = DIMS_PAIR_TO_COMPARE
gc_client.write_loss_report_to_spreadsheet(report)
gc_client.write_compare_report_to_spreadsheet(report, dims_to_compare)
gc_client.update_N_report_on_spreadsheet(report, dims_to_compare)
