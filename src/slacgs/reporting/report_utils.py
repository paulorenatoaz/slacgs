import math
import os
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from slacgs.utils import report_service_conf


def load_simulation_data(file_path):
	with open(file_path, 'r') as f:
		return json.load(f)


def filter_simulations_by_scenario(simulations, scenario_params):
	scenario = []
	for sim in simulations:
		if sim["model_tag"]["params"] in scenario_params:
			scenario.append(sim)
	return sorted(scenario, key=lambda x: x["model_tag"]["params"])


def detect_alpha_index(scenario_params, scenario_dim):
	# Assuming the parameters that vary are the same in all sets
	alpha_index = None
	for i in range(len(scenario_params[0])):
		if len(set(param[i] for param in scenario_params)) > 1:
			alpha_index = i
			break

	# If no varying parameter found (single simulation or all same params), use first sigma
	if alpha_index is None:
		alpha_index = 0
		alpha_index_text = r'$\sigma_1$'
	elif alpha_index + 1 <= scenario_dim:
		alpha_index_text = f'$\\sigma_{alpha_index + 1}$'
	else:
		if alpha_index + 1 == scenario_dim + 1:
			alpha_index_text = r'$\rho_{12}$'
		else:
			alpha_index_text = r'$\rho_{13} = \rho_{23}$'
	return alpha_index, alpha_index_text


def create_scenario_gif(scenario, scenario_id):
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))

	# Create the output directory if it does not exist
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	images = []
	for sim in scenario:
		png_path = os.path.join(report_service_conf['reports_path'], sim["export_path_visualizations_dir"],
		                        'datapoints' + str(sim["model_tag"]["params"]) + '.png')
		if os.path.exists(png_path):
			images.append(Image.open(png_path))

	scenario_gif_path = os.path.join(output_dir, 'scenario_' + str(scenario_id) + '_visualization.gif')
	images[0].save(
		scenario_gif_path,
		save_all=True,
		append_images=images[1:],
		duration=1000,  # Duration in milliseconds
		loop=0
	)
	return scenario_gif_path


def make_gif_html(scenario_gif_path):
	scenario_gif_rel_path = os.path.relpath(scenario_gif_path, report_service_conf['reports_path'])
	return f'<img src="{scenario_gif_rel_path}" alt="Scenario Visualization GIF"><br>'


def create_loss_vs_n_graphs(scenario, scenario_id, alpha_index, alpha_index_text):
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Detect loss types from the first simulation
	loss_types = scenario[0]["sim_tag"]["loss_types"]
	model_dim = scenario[0]["model_tag"]["dim"]

	# Create a graph for each loss type
	graph_paths = []

	for loss_type in loss_types:
		plt.figure()



		# plot max dimension curves
		for sim in scenario:

			alpha_value = sim["model_tag"]["params"][alpha_index]
			loss_values = sim["loss_N"][str(model_dim)][loss_type]
			# Use only N values that have corresponding loss data (may be fewer in test mode)
			num_points = len(loss_values)
			n_values = np.log2(sim['model_tag']['N'][:num_points])
			plt.plot(n_values, loss_values, label=f'{model_dim} feat; {alpha_index_text} = {alpha_value}')

		# plot last but one dimension(s) curves
		if alpha_index != 3:
			sim = scenario[0]

			loss_values = sim["loss_N"][str(model_dim - 1)][loss_type]
			# Use only N values that have corresponding loss data
			num_points = len(loss_values)
			n_values = np.log2(sim['model_tag']['N'][:num_points])
			plt.plot(n_values, loss_values, '--', label=f'{model_dim - 1} feat')
		else:
			for sim in scenario:
				alpha_value = sim["model_tag"]["params"][alpha_index]
				loss_values = sim["loss_N"][str(model_dim - 1)][loss_type]
				# Use only N values that have corresponding loss data
				num_points = len(loss_values)
				n_values = np.log2(sim['model_tag']['N'][:num_points])
				plt.plot(n_values, loss_values, '--', label=f'{model_dim - 1} feat; {alpha_index_text} = {alpha_value}')

		single_graph_title = loss_type + ' loss'

		plt.xlabel('$\log_2(n)$')
		plt.ylabel('$P(error)$')
		plt.title(single_graph_title)
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		graph_path = os.path.join(output_dir, f'{loss_type}_loss_vs_n.png')
		
		plt.grid()
		plt.savefig(graph_path, bbox_inches='tight')
		plt.close()
		graph_paths.append(graph_path)

	return graph_paths

def create_loss_vs_alpha_graphs(scenario, scenario_id, alpha_index, alpha_index_text):
    output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Detect loss types from the first simulation
    loss_types = scenario[0]["sim_tag"]["loss_types"]
    model_dim = scenario[0]["model_tag"]["dim"]

    # Create a graph for each loss type
    graphs = []

    # Define color map
    color_map = plt.get_cmap('tab10')  # You can choose a different color map if you prefer

    for loss_type in loss_types:
        plt.figure()
        alpha_values = [sim["model_tag"]["params"][alpha_index] for sim in scenario]

        n_values = scenario[0]['model_tag']['N']
        n_values_to_plot = [2**3, 2**6]
        # Only plot N values that exist in the simulated data
        n_indexes_to_plot = []
        for n in n_values_to_plot:
            if n in n_values:
                idx = n_values.index(n)
                # Check if this index exists in the loss data
                first_sim_losses = scenario[0]["loss_N"][str(model_dim)][loss_type]
                if idx < len(first_sim_losses):
                    n_indexes_to_plot.append(idx)

        # plot curves for n in n_values_to_plot for 3D (continuous lines)
        for i, n_index in enumerate(n_indexes_to_plot):
            loss_values = [sim["loss_N"][str(model_dim)][loss_type][n_index] for sim in scenario]
            plt.plot(alpha_values, loss_values, label=f'$n = {n_values[n_index]}$; {model_dim} feat', color=color_map(i))

        # plot curve for n = infinty (loss_bayes) for 3D
        loss_values = [sim["loss_bayes"][str(model_dim)] for sim in scenario]
        plt.plot(alpha_values, loss_values, label=f'$n = \infty$; {model_dim} feat', color='black')

        # plot constant lines for n in n_values_to_plot for 2D (dashed lines)
        for i, n_index in enumerate(n_indexes_to_plot):
            loss_values = [sim["loss_N"][str(model_dim - 1)][loss_type][n_index] for sim in scenario]
            plt.plot(alpha_values, loss_values, '--', label=f'$n = {n_values[n_index]}$; {model_dim - 1} feat', color=color_map(i))

        # plot constant line for n = infinty (loss_bayes) for 2D
        loss_values = [sim["loss_bayes"][str(model_dim - 1)] for sim in scenario]
        plt.plot(alpha_values, loss_values, '--', label=f'$n = \infty$; {model_dim - 1} feat', color='gray')

        single_graph_title = f'{loss_type} loss'

        plt.xlabel(f'{alpha_index_text}')
        plt.ylabel('$P(error)$')
        plt.title(single_graph_title)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        alpha_index_path_text = alpha_index_text.replace('\\', '').replace('$', '')
        graph_path = os.path.join(output_dir, f'{loss_type}_loss_vs_{alpha_index_path_text}.png')

        plt.grid()
        plt.savefig(graph_path, bbox_inches='tight')
        plt.close()
        graphs.append(graph_path)

    return graphs


def create_n_star_vs_alpha_graphs(scenario, scenario_id, alpha_index, alpha_index_text):
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Detect loss types from the first simulation
	loss_types = scenario[0]["sim_tag"]["loss_types"][1:]
	model_dim = scenario[0]["model_tag"]["dim"]

	# Create a graph for each loss type
	graphs = []

	for loss_type in loss_types:
		plt.figure(figsize=(6.4,6.2))
		alpha_values = [sim["model_tag"]["params"][alpha_index] for sim in scenario]

		n_star_log2_values = [np.log2(math.ceil(sim[f'N_star_matrix_{loss_type.lower()}'][model_dim-1][model_dim-2]))
	                      if sim.get(f'N_star_matrix_{loss_type.lower()}') is not None 
	                      and not isinstance(sim[f'N_star_matrix_{loss_type.lower()}'][model_dim-1][model_dim-2], str)
		                    else 0
		                for sim in scenario]
		plt.scatter(alpha_values, n_star_log2_values)

		for i in range(len(alpha_values)):
			plt.text(alpha_values[i], n_star_log2_values[i], f'({alpha_values[i]}, {round(n_star_log2_values[i],1)})', fontsize=16, ha='center', va='bottom')

		single_graph_title = f'{loss_type} loss'
		plt.xlabel(f'{alpha_index_text}')
		plt.ylabel('$\log_2(n^*)$')
		plt.title(single_graph_title)
		alpha_index_path_text = alpha_index_text.replace('\\', '').replace('$', '')
		graph_path = os.path.join(output_dir, f'{loss_type}_n_star_vs_{alpha_index_path_text}.png')
		
		plt.grid()
		plt.savefig(graph_path)
		plt.close()
		graphs.append(graph_path)

	return graphs


def make_graphs_html(graph_paths):
	graph_html = ""
	for graph_path in graph_paths:
		graph_rel_path = os.path.relpath(graph_path, report_service_conf['reports_path'])
		graph_html += f'<img src="{graph_rel_path}" alt="Graph"><br>'
	return graph_html


def create_n_vs_alpha_csv_tables(scenario, scenario_id, alpha_index, alpha_index_text):
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Detect loss types from the first simulation
	loss_types = scenario[0]["sim_tag"]["loss_types"][1:]
	model_dim = scenario[0]["model_tag"]["dim"]
	n_values = scenario[0]['model_tag']['N']

	# Create a table for each loss type
	tables = []
	tables_titles = []

	for loss_type in loss_types:
		alpha_index_path_text = alpha_index_text.replace('\\', '').replace('$', '')
		table_path = os.path.join(output_dir, f'{loss_type}_n_vs_{alpha_index_path_text}.csv')
		title = f'{loss_type}; loss(n) vs. {alpha_index_path_text}'
		if alpha_index != 3:
			header = ['n'] + [f'{model_dim-1} feat'] + [f'{alpha_index_path_text} = {sim["model_tag"]["params"][alpha_index]}' for sim in scenario]
			rows = []
			for n in n_values:
				row = [n] + [scenario[0]['loss_N'][str(model_dim - 1)][loss_type][n_values.index(n)]] + [sim["loss_N"][str(model_dim)][loss_type][n_values.index(n)] for sim in scenario]
				rows.append(row)
		else:
			last_but_one_dim_header = [f'{model_dim-1} feat; {alpha_index_path_text} = {sim["model_tag"]["params"][alpha_index]}' for sim in scenario]
			last_dim_header = [f'{model_dim } feat; {alpha_index_path_text} = {sim["model_tag"]["params"][alpha_index]}' for sim in scenario]
			header = ['n'] + last_but_one_dim_header + last_dim_header
			rows = []
			for n in n_values:
				last_but_one_dim_row = [sim["loss_N"][str(model_dim-1)][loss_type][n_values.index(n)] for sim in scenario]
				last_dim_row = [sim["loss_N"][str(model_dim)][loss_type][n_values.index(n)] for sim in scenario]
				row = [n] + last_but_one_dim_row + last_dim_row
				rows.append(row)

		df = pd.DataFrame(rows, columns=header)
		df.to_csv(table_path, index=False)
		tables.append(df)
		tables_titles.append(title)

	return tables, tables_titles


def create_n_star_vs_alpha_csv_tables(scenario, scenario_id, alpha_index, alpha_index_text):
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# Detect loss types from the first simulation
	loss_types = scenario[0]["sim_tag"]["loss_types"][1:]
	model_dim = scenario[0]["model_tag"]["dim"]

	# Create a table for each loss type
	tables = []
	tables_titles = []

	for loss_type in loss_types:
		alpha_index_path_text = alpha_index_text.replace('\\', '').replace('$', '')
		table_path = os.path.join(output_dir, f'{loss_type}_n_star_vs_{alpha_index_path_text}.csv')
		title = f'{loss_type}; n_star vs. {alpha_index_path_text}'
		header = [f'{alpha_index_path_text}'] + [str(sim['model_tag']['params'][alpha_index]) for sim in scenario]
		row = ['n_star'] + [math.ceil(sim[f'N_star_matrix_{loss_type.lower()}'][model_dim-1][model_dim-2])
		                    if not isinstance(sim[f'N_star_matrix_{loss_type.lower()}'][model_dim-1][model_dim-2], str)
		                    else 0
		                    for sim in scenario]

		df = pd.DataFrame([row], columns=header)
		df.to_csv(table_path, index=False)
		tables.append(df)
		tables_titles.append(title)

	return tables, tables_titles


def create_report_links_df_table(scenario, scenario_id, alpha_index, alpha_index_text):
	''' create a table with links to the simulation reports html files

		Args:
			scenario (list): list of simulations
			scenario_id (int): scenario id
			alpha_index (int): index of the alpha parameter
			alpha_index_text (str): text of the alpha parameter

		Returns:
			pandas.DataFrame: table with links to the simulation reports html files

	'''

	header = ['reports links']
	rows = []
	for sim in scenario:
		sim_params = sim["model_tag"]["params"]
		html_link = f'<a href="{sim["export_path_html_report"]}">{str(sim_params)}</a>'
		rows.append([html_link])

	df = pd.DataFrame(rows, columns=header)

	return df


def make_table_html(table_df, table_title):
	return f"<h3>{table_title}</h3><div class='table-container'>{table_df.to_html(index=False, escape=False)}</div>"


def create_html_scenario_report(scenario, scenario_id, test_mode=False):
	test_suffix = '[test]' if test_mode else ''
	output_path = os.path.join(report_service_conf['reports_path'], f'scenario_{scenario_id}_report{test_suffix}.html')

	plt.rcParams.update({'font.size': 20})

	# Create the scenario visualization GIF
	scenario_gif_path = create_scenario_gif(scenario, scenario_id)
	gifs_html = make_gif_html(scenario_gif_path)

	# Create the graphs
	scenario_params = [sim["model_tag"]["params"] for sim in scenario]
	scenario_dim = scenario[0]["model_tag"]["dim"]

	alpha_index, alpha_index_text = detect_alpha_index(scenario_params, scenario_dim)

	loss_vs_n_graph_paths = create_loss_vs_n_graphs(scenario, scenario_id, alpha_index, alpha_index_text)
	loss_vs_n_graphs_html = make_graphs_html(loss_vs_n_graph_paths)

	loss_vs_alpha_graph_paths = create_loss_vs_alpha_graphs(scenario, scenario_id, alpha_index, alpha_index_text)
	loss_vs_alpha_graphs_html = make_graphs_html(loss_vs_alpha_graph_paths)

	n_star_vs_alpha_graph_paths = create_n_star_vs_alpha_graphs(scenario, scenario_id, alpha_index, alpha_index_text)
	n_star_vs_alpha_graphs_html = make_graphs_html(n_star_vs_alpha_graph_paths)

	# Create the tables
	tables, tables_titles = create_n_vs_alpha_csv_tables(scenario, scenario_id, alpha_index, alpha_index_text)
	tables_html = [make_table_html(table_path, title) for table_path, title in zip(tables, tables_titles)]

	tables, tables_titles = create_n_star_vs_alpha_csv_tables(scenario, scenario_id, alpha_index, alpha_index_text)
	tables_html += [make_table_html(table_path, title) for table_path, title in zip(tables, tables_titles)]

	links_table = create_report_links_df_table(scenario, scenario_id, alpha_index, alpha_index_text)
	links_table_html = make_table_html(links_table, 'Links to simulation reports')

	alpha_index_path_text = alpha_index_text.replace('\\', '').replace('$', '')
	html_content = f"""
    <html>
    <head>
        <title>Scenario Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; }}
            .section {{ margin-bottom: 40px; }}
            .section-title {{ font-size: 24px; margin-bottom: 10px; }}
            .sub-section-title {{ font-size: 20px; margin-bottom: 10px; }}
            .gif-container, .graphs-container, .tables-container {{ margin: 20px 0; }}
            .gif-container img, .graphs-container img {{ border: 2px solid black; width: 100%; max-width: 1000px; }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 10px;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 8px;
                text-align: left;
            }}
            th {{
                background-color: #f2f2f2;
            }}
        </style>
    </head>
    <body>
        <div class="section">
            <div class="section-title">1. Scenario {scenario_id} Visualization and simulation reports links</div>
            <div class="gif-container">
                {gifs_html}
                {links_table_html}
            </div>
        </div>
        <div class="section">
            <div class="section-title">2. Graphs</div>
            <div class="graphs-container">
                <div class="sub-section-title">2.1 Loss vs. N</div>
                {loss_vs_n_graphs_html}
                <div class="sub-section-title">2.2 Loss vs. {alpha_index_path_text}</div>
                {loss_vs_alpha_graphs_html}
                <div class="sub-section-title">2.3 log_2(n*) vs. {alpha_index_path_text}</div>
                {n_star_vs_alpha_graphs_html}
            </div>
        </div>
        <div class="section">
            <div class="section-title">3. Tables</div>
            <div class="tables-container">
                {tables_html[0]}
                {tables_html[2]}
                {tables_html[1]}
                {tables_html[3]}
            </div>
        </div>
    </body>
    </html>
    """

	# Save the HTML content to a file, report success and failure with try and except
	try:
		with open(output_path, 'w') as f:
			f.write(html_content)
		print(f"Scenario report saved to: {output_path}")
	except Exception as e:
		print(f"Failed to save scenario report to {output_path}: {e}")


def create_scenario_report(scenario_params, scenario_id, test_mode=False):
	# Path to the JSON file containing the simulation data
	json_filename = 'simulation_reports_test.json' if test_mode else 'simulation_reports.json'
	file_path = os.path.join(report_service_conf['data_path'], json_filename)

	# Load simulation data
	simulations_data = load_simulation_data(file_path)

	# Filter simulations based on the defined scenario parameters
	scenario_data = filter_simulations_by_scenario(simulations_data, scenario_params)

	# Create the HTML report
	create_html_scenario_report(scenario_data, scenario_id, test_mode=test_mode)



