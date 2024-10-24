import os
import sys

# Add the 'src' directory to the sys.path
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(src_path)
print("Source directory added to sys.path:", src_path)

from slacgs import *

if __name__ == "__main__":

	for scenario in SCENARIOS:

		for param in scenario:

			# check if param is allready in simulatio_reports.json

			if is_param_in_simulation_reports(param):
				print(f'Param {param} is allready in simulation_reports.json. Skipping...')
				continue

			# create model object
			model = Model(param)

			# create simulator object
			slacgs = Simulator(model)

			# run simulation
			slacgs.run()

			# export and save simulation graphs
			slacgs.report.save_graphs_png_images_files()

			# export and save simulation tables
			slacgs.report.create_report_tables()

			# write results to json data file
			slacgs.report.write_to_json()

			# create simulation html report
			slacgs.report.create_html_report()