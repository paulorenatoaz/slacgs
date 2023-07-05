import numpy as np
import pygsheets
from GspreadClient import GspreadClient
from Model import Model
from Simulator import Simulator
from docs.cenarios_0 import cenario1_0, cenario2_0, cenario3_0, cenario4_0
from docs.cenarios_1 import cenario1, cenario2, cenario3, cenario4

cenarios = [cenario1, cenario2, cenario3, cenario4]
cenarios_0 = [cenario1_0, cenario2_0, cenario3_0, cenario4_0]
selected_cenario = 0

key_path = GSPREAD_KEY
spredsheet_title = 'cenario' + str(selected_cenario)
gc = pygsheets.authorize(service_file=key_path)
sh = gc.open(spredsheet_title)

ws_home = sh.worksheet(value=0)

gc_client = GspreadClient(key_path, spredsheet_title)

already_done = ws_home.get_values((2, 1), (ws_home.rows, 6), value_render='FORMULA')

for param in cenarios[selected_cenario - 1]:
    if param not in cenarios_0[selected_cenario - 1] and param not in already_done:

        model = Model(param, N=[2 ** i for i in range(1, 10)])

        sim = Simulator(model)

        try:
            sim.start()
        except np.linalg.LinAlgError:
            continue
        except ValueError:
            continue

        report = sim.report

        dims_to_compare = [2, 3]
        gc_client.write_loss_report_to_spreadsheet(report)
        gc_client.write_compare_report_to_spreadsheet(report, dims_to_compare)
        gc_client.update_N_report_on_spreadsheet(report, dims_to_compare)
