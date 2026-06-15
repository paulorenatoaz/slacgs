import math
import os
import json

import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from cosensim.utils import report_service_conf
from cosensim.reporting.html_render import format_param_label


def load_simulation_data(file_path):
	with open(file_path, 'r') as f:
		return json.load(f)


def filter_simulations_by_scenario(simulations, scenario_params):
	"""Select one simulation per scenario parameter set, deduplicating by id.

	The ``simulation_reports*.json`` files grow over time, so the same
	``params`` vector may appear in multiple historical entries. For
	scenario reporting we want exactly one simulation per requested
	parameter set and we prefer the most recent run (highest ``id``).

	Args:
		simulations: List of simulation report dicts loaded from JSON.
		scenario_params: Ordered list of parameter vectors that define
			the scenario.

	Returns:
		List of simulation dicts — at most one per ``scenario_params``
		entry — in the same order as ``scenario_params``. Missing
		parameter sets are silently skipped.
	"""
	by_params = {}
	for sim in simulations:
		params = sim["model_tag"]["params"]
		if params not in scenario_params:
			continue
		key = tuple(params)
		prev = by_params.get(key)
		if prev is None or sim.get("id", -1) > prev.get("id", -1):
			by_params[key] = sim

	ordered = []
	seen_ids = set()
	for params in scenario_params:
		sim = by_params.get(tuple(params))
		if sim is None:
			continue
		sim_id = sim.get("id")
		if sim_id in seen_ids:
			continue
		seen_ids.add(sim_id)
		ordered.append(sim)
	return ordered


def detect_alpha_index(scenario_params, scenario_dim):
	"""Detect which parameter index varies across a scenario.

	Args:
		scenario_params: List of parameter vectors.
		scenario_dim: Number of features (``model_tag['dim']``).

	Returns:
		Tuple ``(alpha_index, labels)`` where ``labels`` is a dict with
		``"html"`` (Unicode/HTML, e.g. ``"ρ₁₂"``), ``"mathtext"``
		(matplotlib mathtext for axis text), and ``"plain"`` (filename-safe).
	"""
	# Assuming the parameters that vary are the same in all sets
	alpha_index = None
	for i in range(len(scenario_params[0])):
		if len(set(param[i] for param in scenario_params)) > 1:
			alpha_index = i
			break

	# If no varying parameter found (single simulation or all same params), use first sigma
	if alpha_index is None:
		alpha_index = 0
		labels = {
			"html": "σ<sub>1</sub>",
			"unicode": "σ₁",
			"mathtext": r"$\sigma_1$",
			"plain": "sigma1",
		}
		return alpha_index, labels

	# Special case in 3D: the CoSenSim scenario 4 uses a single slider for
	# rho_{13} = rho_{23} stored in the same position. Preserve the original
	# combined label when index points to the rho_{13} slot of a 3D model.
	if scenario_dim == 3 and alpha_index == scenario_dim + 1:
		labels = {
			"html": "ρ<sub>cond,13</sub> = ρ<sub>cond,23</sub>",
			"unicode": "ρ_cond,13 = ρ_cond,23",
			"mathtext": r"$\rho_{\mathrm{cond},13} = \rho_{\mathrm{cond},23}$",
			"plain": "rho_cond_13_eq_rho_cond_23",
		}
		return alpha_index, labels

	labels = format_param_label(alpha_index, scenario_dim)
	return alpha_index, labels


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


def create_loss_vs_n_graphs(scenario, scenario_id, alpha_index, alpha_labels):
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	alpha_unicode = alpha_labels['unicode']

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
			num_points = len(loss_values)
			n_values = np.log2(sim['model_tag']['N'][:num_points])
			plt.plot(n_values, loss_values, label=f'{model_dim} feat; {alpha_unicode} = {alpha_value}')

		# plot last but one dimension(s) curves
		if alpha_index != 3:
			sim = scenario[0]
			loss_values = sim["loss_N"][str(model_dim - 1)][loss_type]
			num_points = len(loss_values)
			n_values = np.log2(sim['model_tag']['N'][:num_points])
			plt.plot(n_values, loss_values, '--', label=f'{model_dim - 1} feat')
		else:
			for sim in scenario:
				alpha_value = sim["model_tag"]["params"][alpha_index]
				loss_values = sim["loss_N"][str(model_dim - 1)][loss_type]
				num_points = len(loss_values)
				n_values = np.log2(sim['model_tag']['N'][:num_points])
				plt.plot(n_values, loss_values, '--', label=f'{model_dim - 1} feat; {alpha_unicode} = {alpha_value}')

		plt.xlabel('log₂(n)')
		plt.ylabel('P(error)')
		plt.title(f'{loss_type} loss')
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		graph_path = os.path.join(output_dir, f'{loss_type}_loss_vs_n.png')
		plt.grid()
		plt.savefig(graph_path, bbox_inches='tight')
		plt.close()
		graph_paths.append(graph_path)

	return graph_paths

def create_loss_vs_alpha_graphs(scenario, scenario_id, alpha_index, alpha_labels):
    output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    alpha_unicode = alpha_labels['unicode']
    alpha_plain = alpha_labels['plain']

    loss_types = scenario[0]["sim_tag"]["loss_types"]
    model_dim = scenario[0]["model_tag"]["dim"]

    graphs = []
    color_map = plt.get_cmap('tab10')

    for loss_type in loss_types:
        plt.figure()
        alpha_values = [sim["model_tag"]["params"][alpha_index] for sim in scenario]

        n_values = scenario[0]['model_tag']['N']
        n_values_to_plot = [2**3, 2**6]
        n_indexes_to_plot = []
        for n in n_values_to_plot:
            if n in n_values:
                idx = n_values.index(n)
                first_sim_losses = scenario[0]["loss_N"][str(model_dim)][loss_type]
                if idx < len(first_sim_losses):
                    n_indexes_to_plot.append(idx)

        for i, n_index in enumerate(n_indexes_to_plot):
            loss_values = [sim["loss_N"][str(model_dim)][loss_type][n_index] for sim in scenario]
            plt.plot(alpha_values, loss_values, label=f'n = {n_values[n_index]}; {model_dim} feat', color=color_map(i))

        loss_values = [sim["loss_bayes"][str(model_dim)] for sim in scenario]
        plt.plot(alpha_values, loss_values, label=f'n = ∞; {model_dim} feat', color='black')

        for i, n_index in enumerate(n_indexes_to_plot):
            loss_values = [sim["loss_N"][str(model_dim - 1)][loss_type][n_index] for sim in scenario]
            plt.plot(alpha_values, loss_values, '--', label=f'n = {n_values[n_index]}; {model_dim - 1} feat', color=color_map(i))

        loss_values = [sim["loss_bayes"][str(model_dim - 1)] for sim in scenario]
        plt.plot(alpha_values, loss_values, '--', label=f'n = ∞; {model_dim - 1} feat', color='gray')

        plt.xlabel(alpha_unicode)
        plt.ylabel('P(error)')
        plt.title(f'{loss_type} loss')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        graph_path = os.path.join(output_dir, f'{loss_type}_loss_vs_{alpha_plain}.png')
        plt.grid()
        plt.savefig(graph_path, bbox_inches='tight')
        plt.close()
        graphs.append(graph_path)

    return graphs


def _n_star_log2(sim, loss_type, model_dim):
	"""Return log2(ceil(N*)) for the (top, top-1) dim pair, or None if missing.

	Args:
		sim: A simulation report dict.
		loss_type: Loss type name (e.g. ``"EMPIRICAL_TEST"``).
		model_dim: Top dimension of the model.

	Returns:
		Float log2 of the integer ceil of N* in the (model_dim, model_dim-1)
		pair, or ``None`` if the matrix is missing or the entry is a placeholder.
	"""
	mat = sim.get(f'N_star_matrix_{loss_type.lower()}')
	if mat is None:
		return None
	try:
		val = mat[model_dim - 1][model_dim - 2]
	except (IndexError, TypeError):
		return None
	if isinstance(val, str) or val is None or val <= 0:
		return None
	return float(np.log2(math.ceil(val)))


def create_n_star_vs_alpha_graphs(scenario, scenario_id, alpha_index, alpha_labels):
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	alpha_mathtext = alpha_labels['mathtext']
	alpha_plain = alpha_labels['plain']

	loss_types = scenario[0]["sim_tag"]["loss_types"][1:]
	model_dim = scenario[0]["model_tag"]["dim"]

	graphs = []

	for loss_type in loss_types:
		plt.figure(figsize=(6.4, 6.2))
		alpha_values = [sim["model_tag"]["params"][alpha_index] for sim in scenario]
		n_star_log2_values = [_n_star_log2(sim, loss_type, model_dim) or 0 for sim in scenario]

		plt.scatter(alpha_values, n_star_log2_values)
		for i in range(len(alpha_values)):
			plt.text(alpha_values[i], n_star_log2_values[i],
			         f'({alpha_values[i]}, {round(n_star_log2_values[i], 1)})',
			         fontsize=12, ha='center', va='bottom')

		plt.xlabel(alpha_mathtext)
		plt.ylabel(r'$\log_2(N^*)$')
		plt.title(f'{loss_type} loss')
		graph_path = os.path.join(output_dir, f'{loss_type}_n_star_vs_{alpha_plain}.png')
		plt.grid()
		plt.savefig(graph_path, bbox_inches='tight')
		plt.close()
		graphs.append(graph_path)

	return graphs


def make_graphs_html(graph_paths):
	graph_html = ""
	for graph_path in graph_paths:
		graph_rel_path = os.path.relpath(graph_path, report_service_conf['reports_path'])
		graph_html += f'<img src="{graph_rel_path}" alt="Graph"><br>'
	return graph_html


def create_n_vs_alpha_csv_tables(scenario, scenario_id, alpha_index, alpha_labels):
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	alpha_index_path_text = alpha_labels['plain']
	alpha_html = alpha_labels['html']
	alpha_unicode = alpha_labels['unicode']

	# Detect loss types from the first simulation
	loss_types = scenario[0]["sim_tag"]["loss_types"][1:]
	model_dim = scenario[0]["model_tag"]["dim"]
	n_values = scenario[0]['model_tag']['N']

	# Create a table for each loss type
	tables = []
	tables_titles = []

	for loss_type in loss_types:
		table_path = os.path.join(output_dir, f'{loss_type}_n_vs_{alpha_index_path_text}.csv')
		title = f'{loss_type}; loss(n) vs. {alpha_html}'
		if alpha_index != 3:
			header = ['n'] + [f'{model_dim-1} feat'] + [f'{alpha_unicode} = {sim["model_tag"]["params"][alpha_index]}' for sim in scenario]
			rows = []
			for n in n_values:
				row = [n] + [scenario[0]['loss_N'][str(model_dim - 1)][loss_type][n_values.index(n)]] + [sim["loss_N"][str(model_dim)][loss_type][n_values.index(n)] for sim in scenario]
				rows.append(row)
		else:
			last_but_one_dim_header = [f'{model_dim-1} feat; {alpha_unicode} = {sim["model_tag"]["params"][alpha_index]}' for sim in scenario]
			last_dim_header = [f'{model_dim } feat; {alpha_unicode} = {sim["model_tag"]["params"][alpha_index]}' for sim in scenario]
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


def create_n_star_vs_alpha_csv_tables(scenario, scenario_id, alpha_index, alpha_labels):
	"""Persist N* values as a CSV (one row per loss type) for the record.

	Args:
		scenario: List of simulation report dicts.
		scenario_id: Scenario identifier.
		alpha_index: Index of the varying parameter.
		alpha_labels: Dict returned by :func:`detect_alpha_index`.

	Returns:
		Tuple ``(dataframes, titles)`` where titles are plain HTML strings.
	"""
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	alpha_plain = alpha_labels['plain']
	alpha_html = alpha_labels['html']

	loss_types = scenario[0]["sim_tag"]["loss_types"][1:]
	model_dim = scenario[0]["model_tag"]["dim"]

	tables = []
	tables_titles = []

	for loss_type in loss_types:
		table_path = os.path.join(output_dir, f'{loss_type}_n_star_vs_{alpha_plain}.csv')
		title = f'{loss_type}; N* vs. {alpha_html}'
		header = [alpha_plain] + [str(sim['model_tag']['params'][alpha_index]) for sim in scenario]
		row = ['n_star']
		for sim in scenario:
			mat = sim.get(f'N_star_matrix_{loss_type.lower()}')
			try:
				val = mat[model_dim - 1][model_dim - 2]
				row.append(math.ceil(val) if not isinstance(val, str) else 0)
			except (TypeError, IndexError):
				row.append(0)

		df = pd.DataFrame([row], columns=header)
		df.to_csv(table_path, index=False)
		tables.append(df)
		tables_titles.append(title)

	return tables, tables_titles


def _offdiag_value(matrix, i=0, j=1):
	"""Safely fetch off-diagonal entry ``(i, j)`` from a matrix or return None."""
	if matrix is None:
		return None
	try:
		return float(np.asarray(matrix, dtype=float)[i, j])
	except (ValueError, IndexError, TypeError):
		return None


def create_n_star_vs_corr_graphs(scenario, scenario_id):
	"""Create N* vs. three correlation references (separate plots, no dual axes).

	For each loss type (except the first), produces three scatter plots of
	log2(N*) against:

	1. Theoretical conditional correlation (off-diag ρ from within-class Σ).
	2. Expected global correlation (off-diag from law of total covariance).
	3. Empirical global correlation (off-diag from Monte-Carlo measurement).

	Args:
		scenario: List of simulation report dicts.
		scenario_id: Scenario identifier.

	Returns:
		Dict mapping ``loss_type -> {"cond": path, "expected": path,
		"empirical": path}`` (any value may be ``None`` if data is missing).
	"""
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	loss_types = scenario[0]["sim_tag"]["loss_types"][1:]
	model_dim = scenario[0]["model_tag"]["dim"]

	# Conditional correlation off-diagonal: derive from theoretical_conditional_cov
	# (within-class covariance Sigma) using corrcoef-equivalent normalization.
	def _cond_corr_offdiag(sim):
		cov = sim.get('theoretical_conditional_cov') or sim['model_tag'].get('cov')
		if cov is None:
			return None
		arr = np.asarray(cov, dtype=float)
		std = np.sqrt(np.diag(arr))
		if std[0] == 0 or std[1] == 0:
			return None
		return float(arr[0, 1] / (std[0] * std[1]))

	results = {}
	for loss_type in loss_types:
		paths = {"cond": None, "expected": None, "empirical": None}
		nstar_log2 = [_n_star_log2(sim, loss_type, model_dim) for sim in scenario]

		series = {
			"cond": [_cond_corr_offdiag(sim) for sim in scenario],
			"expected": [_offdiag_value(sim.get('expected_global_corr')) for sim in scenario],
			"empirical": [_offdiag_value(sim.get('empirical_global_corr')) for sim in scenario],
		}
		titles = {
			"cond": "log₂(N*) vs Conditional Correlation ρ_cond,12",
			"expected": "log₂(N*) vs Expected Global Correlation ρ_global,expected,12",
			"empirical": "log₂(N*) vs Empirical Global Correlation ρ_global,empirical,12",
		}
		mathlabels = {
			"cond": "ρ_cond,12",
			"expected": "ρ_global,expected,12",
			"empirical": "ρ_global,empirical,12",
		}

		for kind, xs in series.items():
			pairs = [(x, y) for x, y in zip(xs, nstar_log2) if x is not None and y is not None]
			if not pairs:
				continue
			xs_p, ys_p = zip(*pairs)
			plt.figure(figsize=(6.4, 5.2))
			plt.scatter(xs_p, ys_p)
			for xv, yv in zip(xs_p, ys_p):
				plt.text(xv, yv, f'({xv:.2f}, {yv:.1f})',
				         fontsize=10, ha='center', va='bottom')
			plt.xlabel(mathlabels[kind])
			plt.ylabel('log₂(N*)')
			plt.title(f'{loss_type} — {titles[kind]}')
			plt.grid(True)
			p = os.path.join(output_dir, f'{loss_type}_n_star_vs_{kind}_corr.png')
			plt.savefig(p, bbox_inches='tight')
			plt.close()
			paths[kind] = p
		results[loss_type] = paths
	return results


def build_n_star_table_html(scenario, alpha_index, alpha_labels):
	"""Render the enriched N* HTML table referencing three correlation values.

	Each loss type produces one table. The first column lists the loss type
	and N* row label; the remaining columns are simulations. Header is split
	into 4 rows: simulation parameters, ρ_cond,12, ρ_global,expected,12 and
	ρ_global,empirical,12. The body row holds the ceil(N*) values. N* values
	are not duplicated across multiple rows.

	Args:
		scenario: List of simulation report dicts.
		alpha_index: Index of the varying parameter (unused for header but
			kept for API consistency).
		alpha_labels: Dict returned by :func:`detect_alpha_index`.

	Returns:
		HTML fragment string.
	"""
	loss_types = scenario[0]["sim_tag"]["loss_types"][1:]
	model_dim = scenario[0]["model_tag"]["dim"]

	def _fmt(x):
		return "—" if x is None else f"{x:.3f}"

	# Per-sim correlation refs (off-diag (1,2))
	cond_vals, exp_vals, emp_vals = [], [], []
	for sim in scenario:
		cov = sim.get('theoretical_conditional_cov') or sim['model_tag'].get('cov')
		if cov is not None:
			arr = np.asarray(cov, dtype=float)
			std = np.sqrt(np.diag(arr))
			if std.size >= 2 and std[0] > 0 and std[1] > 0:
				cond_vals.append(float(arr[0, 1] / (std[0] * std[1])))
			else:
				cond_vals.append(None)
		else:
			cond_vals.append(None)
		exp_vals.append(_offdiag_value(sim.get('expected_global_corr')))
		emp_vals.append(_offdiag_value(sim.get('empirical_global_corr')))

	param_cells = "".join(
		f'<th>{str(sim["model_tag"]["params"])}</th>' for sim in scenario
	)
	cond_cells = "".join(f'<td>{_fmt(v)}</td>' for v in cond_vals)
	exp_cells = "".join(f'<td>{_fmt(v)}</td>' for v in exp_vals)
	emp_cells = "".join(f'<td>{_fmt(v)}</td>' for v in emp_vals)

	tables_html = ""
	for loss_type in loss_types:
		n_star_cells = ""
		for sim in scenario:
			mat = sim.get(f'N_star_matrix_{loss_type.lower()}')
			val = None
			try:
				v = mat[model_dim - 1][model_dim - 2]
				if not isinstance(v, str) and v is not None:
					val = int(math.ceil(v))
			except (TypeError, IndexError):
				val = None
			n_star_cells += f'<td>{val if val is not None else "—"}</td>'

		tables_html += (
			f'<div class="table-wrap"><div class="table-title">'
			f'N* table — {loss_type} loss (varying {alpha_labels["html"]})'
			f'</div><table class="matrix-table">'
			f'<thead>'
			f'<tr><th>params</th>{param_cells}</tr>'
			f'<tr><th>ρ<sub>cond,12</sub></th>{cond_cells}</tr>'
			f'<tr><th>ρ<sub>global,expected,12</sub></th>{exp_cells}</tr>'
			f'<tr><th>ρ<sub>global,empirical,12</sub></th>{emp_cells}</tr>'
			f'</thead>'
			f'<tbody><tr><th>ceil(N*)</th>{n_star_cells}</tr></tbody>'
			f'</table></div>'
		)
	return tables_html


def create_report_links_df_table(scenario, scenario_id, alpha_index=None, alpha_labels=None):
	"""Create a DataFrame with links to per-simulation HTML reports.

	Args:
		scenario: List of simulation report dicts.
		scenario_id: Scenario identifier (kept for API parity).
		alpha_index: Unused; kept for backward compatibility.
		alpha_labels: Unused; kept for backward compatibility.

	Returns:
		A single-column ``pandas.DataFrame`` whose rows are anchor tags
		pointing to each simulation's HTML report.
	"""
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


def _extract_off_diagonal_series(scenario, matrix_key):
	"""Extract off-diagonal series from a per-simulation matrix field.

	Args:
		scenario: List of simulation report dicts.
		matrix_key: JSON field name holding a 2D matrix (e.g.
			``"empirical_global_corr"``).

	Returns:
		Tuple ``(labels, series)``:
		- ``labels`` is a list of strings like ``"corr(x1,x2)"`` enumerating
		  the upper-triangular off-diagonal entries.
		- ``series`` is a list, one per label, of values across simulations
		  (``None`` placed where the matrix is missing for a given sim).
	"""
	labels = None
	series = None
	for sim in scenario:
		mat = sim.get(matrix_key)
		if mat is None:
			continue
		arr = np.asarray(mat, dtype=float)
		if labels is None:
			d = arr.shape[0]
			labels = []
			for i in range(d):
				for j in range(i + 1, d):
					labels.append(f'(x{i + 1},x{j + 1})')
			series = [[] for _ in labels]
		k = 0
		for i in range(arr.shape[0]):
			for j in range(i + 1, arr.shape[1]):
				series[k].append(float(arr[i, j]))
				k += 1
	if labels is None:
		return [], []
	return labels, series


def _save_corr_graph(alpha_values, labels, series, title, ylabel,
                     alpha_axis_label, output_path):
	"""Save a single off-diagonal correlation/covariance graph as PNG.

	Args:
		alpha_values: X-axis values (varying scenario parameter).
		labels: One legend label per off-diagonal pair (matplotlib mathtext).
		series: One list of y-values per pair (aligned with ``alpha_values``).
		title: Plot title.
		ylabel: Y-axis label.
		alpha_axis_label: Matplotlib mathtext label for the X-axis.
		output_path: PNG destination path.
	"""
	plt.figure()
	for label, ys in zip(labels, series):
		if len(ys) == len(alpha_values):
			plt.plot(alpha_values, ys, marker='o', label=label)
	plt.xlabel(alpha_axis_label)
	plt.ylabel(ylabel)
	plt.title(title)
	plt.grid(True)
	if labels:
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
	plt.savefig(output_path, bbox_inches='tight')
	plt.close()


def create_global_corr_graphs(scenario, scenario_id, alpha_index, alpha_labels):
	"""Create scenario-level Expected vs Empirical Global Correlation graphs.

	Args:
		scenario: List of simulation report dicts.
		scenario_id: Scenario identifier (used in the output directory).
		alpha_index: Index of the varying parameter in each ``params`` list.
		alpha_labels: Dict returned by :func:`detect_alpha_index`.

	Returns:
		Dict mapping ``"expected"``, ``"empirical"``, ``"comparison"`` to
		absolute PNG paths (any value may be ``None`` if the source data is
		missing).
	"""
	output_dir = os.path.join(report_service_conf['reports_path'], 'scenario_' + str(scenario_id))
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	alpha_values = [sim['model_tag']['params'][alpha_index] for sim in scenario]
	alpha_axis_label = alpha_labels['mathtext']

	exp_labels, exp_series = _extract_off_diagonal_series(scenario, 'expected_global_corr')
	emp_labels, emp_series = _extract_off_diagonal_series(scenario, 'empirical_global_corr')

	paths = {'expected': None, 'empirical': None, 'comparison': None}

	if exp_series:
		p = os.path.join(output_dir, 'expected_global_corr.png')
		_save_corr_graph(alpha_values, exp_labels, exp_series,
		                 'Expected Global Correlation (analytical)',
		                 'correlation', alpha_axis_label, p)
		paths['expected'] = p

	if emp_series:
		p = os.path.join(output_dir, 'empirical_global_corr.png')
		_save_corr_graph(alpha_values, emp_labels, emp_series,
		                 'Empirical Global Correlation (measured)',
		                 'correlation', alpha_axis_label, p)
		paths['empirical'] = p

	# Comparison overlay: same off-diagonal pairs, expected dashed vs empirical solid
	if exp_series and emp_series and exp_labels == emp_labels:
		p = os.path.join(output_dir, 'global_corr_comparison.png')
		plt.figure()
		color_map = plt.get_cmap('tab10')
		for k, label in enumerate(exp_labels):
			c = color_map(k % 10)
			if len(exp_series[k]) == len(alpha_values):
				plt.plot(alpha_values, exp_series[k], '--', color=c, label=f'expected {label}')
			if len(emp_series[k]) == len(alpha_values):
				plt.plot(alpha_values, emp_series[k], '-o', color=c, label=f'empirical {label}')
		plt.xlabel(alpha_axis_label)
		plt.ylabel('correlation')
		plt.title('Expected vs Empirical Global Correlation')
		plt.grid(True)
		plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
		plt.savefig(p, bbox_inches='tight')
		plt.close()
		paths['comparison'] = p

	return paths


def create_html_scenario_report(scenario, scenario_id, test_mode=False):
	"""Render the scenario HTML report (normal + portable embedded variants).

	Args:
		scenario: List of simulation report dicts belonging to the scenario.
		scenario_id: Scenario identifier (used in filenames and headers).
		test_mode: If True, append ``[test]`` to the output filename.

	Side Effects:
		Writes two files into ``report_service_conf['reports_path']``:
		``scenario_{id}_report{[test]}.html`` (local image references) and
		``scenario_{id}_report{[test]}_embedded.html`` (base64-inlined).
	"""
	from cosensim.reporting.html_render import (
		render_section,
		render_image_card,
		render_dataframe_table,
		render_page,
	)

	test_suffix = '[test]' if test_mode else ''
	reports_root = report_service_conf['reports_path']
	output_path = os.path.join(reports_root, f'scenario_{scenario_id}_report{test_suffix}.html')
	embedded_path = os.path.join(reports_root, f'scenario_{scenario_id}_report{test_suffix}_embedded.html')

	plt.rcParams.update({'font.size': 14})

	scenario_params = [sim['model_tag']['params'] for sim in scenario]
	scenario_dim = scenario[0]['model_tag']['dim']
	alpha_index, alpha_labels = detect_alpha_index(scenario_params, scenario_dim)
	alpha_html = alpha_labels['html']
	alpha_plain = alpha_labels['plain']

	# ---- Generate all artifacts up front (PNG, CSV, DataFrames) ----
	scenario_gif_path = create_scenario_gif(scenario, scenario_id)
	loss_vs_n_paths = create_loss_vs_n_graphs(scenario, scenario_id, alpha_index, alpha_labels)
	loss_vs_alpha_paths = create_loss_vs_alpha_graphs(scenario, scenario_id, alpha_index, alpha_labels)
	corr_paths = create_global_corr_graphs(scenario, scenario_id, alpha_index, alpha_labels)
	n_star_corr_paths = create_n_star_vs_corr_graphs(scenario, scenario_id)
	n_star_html_table = build_n_star_table_html(scenario, alpha_index, alpha_labels)

	n_vs_alpha_tables, n_vs_alpha_titles = create_n_vs_alpha_csv_tables(
		scenario, scenario_id, alpha_index, alpha_labels)
	n_star_tables, n_star_titles = create_n_star_vs_alpha_csv_tables(
		scenario, scenario_id, alpha_index, alpha_labels)
	links_table = create_report_links_df_table(scenario, scenario_id)

	def _build_html(embed: bool) -> str:
		"""Render the full scenario HTML string for either embed mode.

		Args:
			embed: If True, inline images as base64 data URIs.

		Returns:
			Complete HTML document string.
		"""
		# 1. Overview
		overview = (
			f"<p>This scenario aggregates <strong>{len(scenario)}</strong> "
			f"CoSenSim simulations sharing a common Gaussian model template; "
			f"the parameter <strong>{alpha_html}</strong> varies across "
			f"simulations while the others are held fixed."
			f"{' Run in <strong>test mode</strong>.' if test_mode else ''}"
			f"</p>"
		)

		# 2. Scenario Definition
		definition_df = pd.DataFrame(
			[[i + 1, str(p), p[alpha_index]] for i, p in enumerate(scenario_params)],
			columns=['#', 'params', alpha_plain],
		)
		definition_html = render_dataframe_table(
			'Per-simulation parameter sets',
			definition_df,
			caption=f'Varying parameter: {alpha_html} (column index {alpha_index}).',
		)

		# 3. Simulations Included — visualization GIF only (links moved to section 8)
		gif_card = render_image_card(
			'Scenario visualization (animated)',
			scenario_gif_path,
			caption='Per-simulation data point clouds, in scenario order.',
			embed=embed, base_dir=reports_root,
		) if scenario_gif_path else ''
		sims_included_html = gif_card or '<p><em>No visualization GIF available.</em></p>'

		# 4. Loss / Performance Summary
		perf_html = "".join(
			render_image_card(
				f'Loss vs log<sub>2</sub>(n) — graph {i + 1}',
				p, embed=embed, base_dir=reports_root,
			) for i, p in enumerate(loss_vs_n_paths)
		)
		perf_html += "".join(
			render_image_card(
				f'Loss vs {alpha_html} — graph {i + 1}',
				p, embed=embed, base_dir=reports_root,
			) for i, p in enumerate(loss_vs_alpha_paths)
		)

		# 5. N* / Crossover Summary — three standardized correlation plots only.
		# The old generic "log2(N*) vs alpha" plot is intentionally omitted to
		# avoid a duplicate that does not distinguish conditional from global.
		n_star_html = (
			"<p>N* is the smallest sample size at which adding one more "
			"feature stops hurting performance. The plots below show "
			"log<sub>2</sub>(N*) against three correlation references: "
			"conditional (within-class), expected global (analytical) and "
			"empirical global (measured). The enriched table below lists a "
			"single ceil(N*) row per loss type, with header rows that expose "
			"all three correlations side by side.</p>"
		)
		for lt, kinds in n_star_corr_paths.items():
			for kind, label in (
				("cond", "ρ<sub>cond,12</sub>"),
				("expected", "ρ<sub>global,expected,12</sub>"),
				("empirical", "ρ<sub>global,empirical,12</sub>"),
			):
				n_star_html += render_image_card(
					f'log<sub>2</sub>(N*) vs {label} — {lt}',
					kinds.get(kind),
					embed=embed, base_dir=reports_root,
				)
		n_star_html += n_star_html_table

		# 6. Correlation Analysis
		corr_intro = (
			"<p>Real-world correlation-based experiments only have access to "
			"<em>global</em> correlations — i.e. correlations measured on the "
			"unlabeled mixed-class data. The graphs below show how the "
			"<em>off-diagonal</em> global correlation entries evolve across "
			"this scenario, both analytically (expected, from the law of "
			"total covariance) and empirically (measured across Monte-Carlo "
			"iterations). They are the most directly comparable diagnostic "
			"with sensor-data experiments.</p>"
		)
		corr_html = corr_intro
		corr_html += render_image_card(
			f'Expected Global Correlation vs {alpha_html}',
			corr_paths['expected'],
			caption='Analytical off-diagonal correlations across the scenario.',
			embed=embed, base_dir=reports_root,
		)
		corr_html += render_image_card(
			f'Empirical Global Correlation vs {alpha_html}',
			corr_paths['empirical'],
			caption='Monte-Carlo measured off-diagonal correlations across the scenario.',
			embed=embed, base_dir=reports_root,
		)
		if corr_paths.get('comparison'):
			corr_html += render_image_card(
				'Expected vs Empirical (overlay)',
				corr_paths['comparison'],
				caption='Dashed: expected. Solid markers: empirical.',
				embed=embed, base_dir=reports_root,
			)

		# 7. Scenario Tables — pass escape=False because some headers already
		# contain controlled HTML such as ρ<sub>12</sub>=value built from
		# alpha_labels['html'].
		tables_html = "".join(
			render_dataframe_table(t, df, escape=False)
			for df, t in zip(n_vs_alpha_tables, n_vs_alpha_titles)
		)
		tables_html += "".join(
			render_dataframe_table(t, df, escape=False)
			for df, t in zip(n_star_tables, n_star_titles)
		)

		# 8. Links to Individual Simulation Reports
		links_html = render_dataframe_table(
			'Links to individual simulation reports', links_table, escape=False,
		)

		sections = (
			render_section('1. Overview', overview, 'overview')
			+ render_section('2. Scenario Definition', definition_html, 'definition')
			+ render_section('3. Simulations Included', sims_included_html, 'sims-included')
			+ render_section('4. Loss / Performance Summary', perf_html, 'performance')
			+ render_section('5. N* / Crossover Summary', n_star_html, 'n-star')
			+ render_section('6. Correlation Analysis', corr_html, 'correlation')
			+ render_section('7. Scenario Tables', tables_html, 'tables')
			+ render_section('8. Links to Individual Simulation Reports', links_html, 'links')
		)
		toc = [
			('overview', 'Overview'),
			('definition', 'Scenario Definition'),
			('sims-included', 'Simulations Included'),
			('performance', 'Loss / Performance Summary'),
			('n-star', 'N* / Crossover Summary'),
			('correlation', 'Correlation Analysis'),
			('tables', 'Scenario Tables'),
			('links', 'Links to Individual Simulation Reports'),
		]
		subtitle = (
			f"Scenario {scenario_id} &middot; {len(scenario)} simulations "
			f"&middot; varying {alpha_html}"
			f"{' &middot; test mode' if test_mode else ''}"
		)
		return render_page(
			title=f"CoSenSim Scenario Report — Scenario {scenario_id}",
			subtitle=subtitle,
			toc_items=toc,
			sections_html=sections,
		)

	try:
		with open(output_path, 'w') as f:
			f.write(_build_html(embed=False))
		with open(embedded_path, 'w') as f:
			f.write(_build_html(embed=True))
		print(f"Scenario report saved to: {output_path}")
		print(f"Embedded scenario report saved to: {embedded_path}")
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



