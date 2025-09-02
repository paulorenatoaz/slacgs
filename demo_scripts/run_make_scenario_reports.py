# import os
# import sys
#
# # Add the 'src' directory to the sys.path
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(src_path)
# print("Source directory added to sys.path:", src_path)
# TODO(TASK-011): Move report_utils into reporting subpackage; update import after src/ move
from slacgs.report_utils import create_scenario_report

if __name__ == "__main__":

	scenario_params = [[1, 1, 0], [1, 2, 0], [1, 3, 0], [1, 4, 0], [1, 5, 0], [1, 6, 0], [1, 7, 0]]

	# create_scenario_report(scenario_params, 5)

	# scenario_params = [[1, 1, -0.8], [1, 1, -0.4], [1, 1, 0], [1, 1, 0.4], [1, 1, 0.8]]
	scenario_params = [[1, 1, 0.6]]

	create_scenario_report(scenario_params, 6)


	# Define the scenario with manually typed parameters
	scenario_params = [
		[1, 1, 1.7, 0, 0, 0],
		[1, 1, 2, 0, 0, 0],
		[1, 1, 2.5, 0, 0, 0],
		[1, 1, 3.5, 0, 0, 0],
		[1, 1, 7, 0, 0, 0],
		[1, 1, 10, 0, 0, 0]
	]
	create_scenario_report(scenario_params,1)

	scenario_params = [[1, 1, 2, -0.8, 0, 0],
						# [1, 1, 2, -0.7, 0, 0],
						# [1, 1, 2, -0.6, 0, 0],
						# [1, 1, 2, -0.5, 0, 0],
						[1, 1, 2, -0.4, 0, 0],
						# [1, 1, 2, -0.3, 0, 0],
						# [1, 1, 2, -0.2, 0, 0],
						# [1, 1, 2, -0.1, 0, 0],
						[1, 1, 2, 0.0, 0, 0],
						# [1, 1, 2, 0.1, 0, 0],
						# [1, 1, 2, 0.2, 0, 0],
						# [1, 1, 2, 0.3, 0, 0],
						[1, 1, 2, 0.4, 0, 0],
						# [1, 1, 2, 0.5, 0, 0],
						# [1, 1, 2, 0.6, 0, 0],
						# [1, 1, 2, 0.7, 0, 0],
						[1, 1, 2, 0.8, 0, 0]]

	# create_scenario_report(scenario_params, 2)

	scenario_params = [[1, 1, 2, 0, -0.7, -0.7],
						# [1, 1, 2, 0, -0.6, -0.6],
						# [1, 1, 2, 0, -0.5, -0.5],
						[1, 1, 2, 0, -0.4, -0.4],
						# [1, 1, 2, 0, -0.3, -0.3],
						[1, 1, 2, 0, -0.2, -0.2],
						# [1, 1, 2, 0, -0.1, -0.1],
						[1, 1, 2, 0, 0.0, 0.0],
						# [1, 1, 2, 0, 0.1, 0.1],
						[1, 1, 2, 0, 0.2, 0.2],
						# [1, 1, 2, 0, 0.3, 0.3],
						[1, 1, 2, 0, 0.4, 0.4],
						# [1, 1, 2, 0, 0.5, 0.5],
						# [1, 1, 2, 0, 0.6, 0.6],
						[1, 1, 2, 0, 0.7, 0.7]]

	# create_scenario_report(scenario_params, 3)

	scenario_params = [
						# [1, 1, 1, -0.1, -0.6, -0.6],
						# [1, 1, 1, -0.1, -0.5, -0.5],
						# [1, 1, 1, -0.1, -0.4, -0.4],
						# [1, 1, 1, -0.1, -0.3, -0.3],
						# [1, 1, 1, -0.1, -0.2, -0.2],
						# [1, 1, 1, -0.1, -0.1, -0.1],
						[1, 1, 1, -0.1, 0.0, 0.0],
						# [1, 1, 1, -0.1, 0.1, 0.1],
						# [1, 1, 1, -0.1, 0.2, 0.2],
						[1, 1, 1, -0.1, 0.3, 0.3],
						[1, 1, 1, -0.1, 0.4, 0.4],
						# [1, 1, 1, -0.1, 0.5, 0.5],
						# [1, 1, 1, -0.1, 0.6, 0.6]
	                   ]

	# create_scenario_report(scenario_params, 4)

	scenario_params = [
		[1, 4, -0.8],
		[1, 4, -0.4],
		[1, 4, 0],
		[1, 4, 0.4],
		[1, 4, 0.5],
		[1, 4, 0.6],
		[1, 4, 0.7],

	]

# create_scenario_report(scenario_params, 7)
