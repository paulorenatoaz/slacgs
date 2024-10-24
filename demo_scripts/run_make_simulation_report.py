# import os
# import sys
#
# # Add the 'src' directory to the sys.path
# src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
# sys.path.append(src_path)
print("Source directory added to sys.path:", src_path)

if __name__ == "__main__":

	PARAMS = [1, 1, 1, -0.1, 0.3, 0.3]


	report = Report(params=PARAMS)

	# create simulation html report
	# report.create_html_report()

	# recreate simulation graphs
	report.save_graphs_png_images_files()
