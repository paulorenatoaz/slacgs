# import os
# import sys
#
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(src_path)
# print("Source directory added to sys.path:", src_path)
from slacgs import *

if __name__ == "__main__":

	PARAM = [1, 4, 0.6]

	# create model object
	model = Model(PARAM)

	# create simulator object
	slacgs = Simulator(model, test_mode=False)

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

