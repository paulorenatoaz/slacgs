"""Quick validation: run one minimal scenario with test_mode and verify the
empirical global covariance is computed and rendered into the HTML report.
"""
import os
import sys
import json

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, src_path)

from slacgs.core.model import Model
from slacgs.core.simulator import Simulator
from slacgs.utils import report_service_conf

# Use a fast 2D parameter set from scenario_6_min
params = [1, 1, 0.4]
model = Model(params)
sim = Simulator(model, test_mode=True, verbose=False)
sim.run()

print("\n--- Validation ---")
print("Theoretical conditional cov:", model.cov)
print("Empirical global cov:", sim.report.empirical_global_cov)
print("Expected global corr:", sim.report.expected_global_corr)
print("Empirical global corr:", sim.report.empirical_global_corr)
assert sim.report.empirical_global_cov is not None
assert sim.report.empirical_global_corr is not None
assert sim.report.expected_global_corr is not None

# Sanity: for [1,1,rho], expected global corr off-diagonal ~ (1+rho)/2
import numpy as _np
rho = params[2]
expected_off = (1.0 + rho) / 2.0
got_off = sim.report.expected_global_corr[0][1]
print(f"Sanity: expected off-diag = {expected_off:.4f}  got = {got_off:.4f}")
assert abs(got_off - expected_off) < 1e-9, "expected global corr formula mismatch"

# Persist + render
sim.report.print_N_star_matrix_between_all_dims()
sim.report.save_graphs_png_images_files()
sim.report.create_report_tables()
sim.report.write_to_json()
sim.report.create_html_report()

# Check JSON content
data_path = os.path.join(report_service_conf['data_path'], 'simulation_reports_test.json')
with open(data_path) as f:
    entries = json.load(f)
last = entries[-1]
for key in ("empirical_global_cov", "empirical_global_corr",
            "theoretical_conditional_cov", "expected_global_corr"):
    assert key in last and last[key] is not None, f"{key} missing in JSON entry"
print("JSON entry id:", last["id"], "all four diagnostics persisted.")

# Check HTML content
html_path = sim.report.export_path_html_report
with open(html_path) as f:
    html = f.read()
for label in ("Correlation and Covariance Diagnostics",
              "Theoretical Conditional Covariance",
              "Empirical Global Covariance",
              "Expected Global Correlation",
              "Empirical Global Correlation"):
    assert label in html, f"missing in HTML: {label}"
print("HTML report:", html_path)
print("OK: all four diagnostic tables and section heading present in HTML.")
