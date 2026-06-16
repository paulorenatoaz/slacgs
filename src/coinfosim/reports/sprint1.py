"""
Sprint 1 HTML report generator for CoInfoSim.

Generates a self-contained HTML report for a Synthetic Scenario 1 simulation
run. All figures are embedded as base64 PNGs so the report is a single file.

Sprint 1 reports empirical test loss only. Empirical train loss, theoretical
loss, and Bayes error are intentionally excluded.
"""

from __future__ import annotations

import base64
import html
import io
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend for report figures
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from coinfosim.classifiers.registry import classifier_label
from coinfosim.results.analysis import (
    best_subset_rankings,
    standard_threshold_comparisons,
)
from coinfosim.results.summary import summary_dataframe
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.scenarios.synthetic import SyntheticScenario
from coinfosim.simulation.monte_carlo import SimulationResult
from coinfosim.simulation.subsets import subset_label

_EXCLUSION_NOTICE = (
    "Sprint 1 reports empirical test loss only. "
    "Empirical train loss, theoretical loss, and Bayes error are "
    "intentionally excluded."
)

# Distinct colors for the (up to seven) subsets.
_SUBSET_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
]


def _fig_to_base64(fig) -> str:
    """Render a Matplotlib figure to a base64-encoded PNG ``data:`` URI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _loss_curve_image(
    result: SimulationResult, classifier_name: str
) -> str:
    """Plot empirical test-loss curves vs n_per_class for one classifier."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    sample_sizes = result.sample_sizes
    for color, subset in zip(_SUBSET_COLORS, result.subsets):
        means = [
            result.accumulator.mean_loss(n, subset, classifier_name)
            for n in sample_sizes
        ]
        errs = [
            result.accumulator.standard_error(n, subset, classifier_name)
            for n in sample_sizes
        ]
        ax.errorbar(
            sample_sizes,
            means,
            yerr=errs,
            marker="o",
            capsize=3,
            color=color,
            label=subset_label(subset),
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("n_per_class (training samples per class)")
    ax.set_ylabel("Empirical test loss (misclassification rate)")
    ax.set_title(f"Empirical test loss — {classifier_label(classifier_name)}")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title="Channel subset", fontsize=8, ncol=2)
    return _fig_to_base64(fig)


def _scatter_image(
    sampler: GaussianClassConditionalSampler,
    channels: Sequence[int],
    n_show: int = 300,
) -> str:
    """2D scatter plot of the two given channels from the fixed test set."""
    test = sampler.sample_test()
    i, j = channels
    fig, ax = plt.subplots(figsize=(4.5, 4))
    for label, color in zip((0, 1), ("#1f77b4", "#d62728")):
        mask = test.y == label
        x = test.X[mask, i][:n_show]
        y = test.X[mask, j][:n_show]
        ax.scatter(x, y, s=10, alpha=0.5, color=color, label=f"class {label}")
    ax.set_xlabel(f"X{i + 1}")
    ax.set_ylabel(f"X{j + 1}")
    ax.set_title(f"(X{i + 1}, X{j + 1})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    return _fig_to_base64(fig)


def _scatter_3d_image(
    sampler: GaussianClassConditionalSampler, n_show: int = 300
) -> Optional[str]:
    """Optional 3D scatter of all three channels."""
    test = sampler.sample_test()
    if test.d < 3:
        return None
    fig = plt.figure(figsize=(5, 4.5))
    ax = fig.add_subplot(111, projection="3d")
    for label, color in zip((0, 1), ("#1f77b4", "#d62728")):
        mask = test.y == label
        pts = test.X[mask][:n_show]
        ax.scatter(
            pts[:, 0], pts[:, 1], pts[:, 2],
            s=8, alpha=0.5, color=color, label=f"class {label}",
        )
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("X3")
    ax.set_title("(X1, X2, X3)")
    ax.legend(fontsize=8)
    return _fig_to_base64(fig)


def _matrix_html(matrix: np.ndarray) -> str:
    """Render a small numeric matrix as an HTML table."""
    rows = []
    for row in matrix:
        cells = "".join(f"<td>{v:.2f}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return f"<table class='matrix'>{''.join(rows)}</table>"


def _vector_html(vector: np.ndarray) -> str:
    cells = ", ".join(f"{v:.2f}" for v in vector)
    return f"[{cells}]"


def _dataframe_html(df, float_cols: Optional[Dict[str, str]] = None) -> str:
    """Render a pandas DataFrame to an HTML table with escaped values."""
    float_cols = float_cols or {}
    headers = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body_rows = []
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            val = row[col]
            if col in float_cols and val is not None and not _is_nan(val):
                cells.append(f"<td>{float(val):{float_cols[col]}}</td>")
            elif val is None or _is_nan(val):
                cells.append("<td>—</td>")
            else:
                cells.append(f"<td>{html.escape(str(val))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    return (
        f"<table class='data'><thead><tr>{headers}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _is_nan(value) -> bool:
    try:
        return bool(np.isnan(value))
    except (TypeError, ValueError):
        return False


def _stopping_table_html(result: SimulationResult) -> str:
    rows = []
    for n in result.sample_sizes:
        info = result.stopping_info[n]
        rows.append(
            f"<tr><td>{n}</td><td>{info.replications}</td>"
            f"<td>{html.escape(info.reason)}</td>"
            f"<td>{info.max_ci_half_width:.4f}</td></tr>"
        )
    return (
        "<table class='data'><thead><tr>"
        "<th>n_per_class</th><th>replications</th>"
        "<th>stopping reason</th><th>max CI half-width</th>"
        "</tr></thead><tbody>" + "".join(rows) + "</tbody></table>"
    )


def generate_sprint1_report(
    scenario: SyntheticScenario,
    result: SimulationResult,
    output_dir,
    sampler: Optional[GaussianClassConditionalSampler] = None,
    filename: str = "synthetic_scenario_1_report.html",
) -> Path:
    """Generate the Sprint 1 HTML report and return its path.

    Parameters
    ----------
    scenario:
        The scenario that was simulated.
    result:
        The :class:`SimulationResult` from the cooperative simulator.
    output_dir:
        Directory in which to write the report (created if missing).
    sampler:
        Sampler used to draw scatter plots from the fixed test set. If not
        provided, a fresh sampler is built from the run configuration.
    filename:
        Output filename.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    model = scenario.model
    config = result.config

    if sampler is None:
        sampler = GaussianClassConditionalSampler(
            model,
            base_seed=config.base_seed,
            test_samples_per_class=config.test_samples_per_class,
        )

    # --- Figures --------------------------------------------------------------
    loss_curves = {
        clf: _loss_curve_image(result, clf) for clf in result.classifier_names
    }
    scatter_2d = {
        "X1,X2": _scatter_image(sampler, (0, 1)),
        "X1,X3": _scatter_image(sampler, (0, 2)),
        "X2,X3": _scatter_image(sampler, (1, 2)),
    }
    scatter_3d = _scatter_3d_image(sampler)

    # --- Tables ---------------------------------------------------------------
    summary_df = summary_dataframe(
        result.accumulator,
        result.sample_sizes,
        result.subsets,
        result.classifier_names,
    )
    rankings_df = best_subset_rankings(
        result.accumulator,
        result.classifier_names,
        result.sample_sizes,
        result.subsets,
    )
    thresholds_df = standard_threshold_comparisons(
        result.accumulator,
        result.classifier_names,
        result.sample_sizes,
        result.subsets,
    )

    # Trim display columns for readability.
    rankings_display = rankings_df[
        ["classifier_label", "n_per_class", "best_subset_label", "mean_loss"]
    ].rename(
        columns={
            "classifier_label": "Classifier",
            "n_per_class": "n_per_class",
            "best_subset_label": "Best subset",
            "mean_loss": "Mean test loss",
        }
    )
    thresholds_display = thresholds_df[
        ["classifier_label", "comparison", "subset_a_label", "subset_b_label", "n_star"]
    ].rename(
        columns={
            "classifier_label": "Classifier",
            "comparison": "Comparison",
            "subset_a_label": "A",
            "subset_b_label": "B (cooperative)",
            "n_star": "N*",
        }
    )
    summary_display = summary_df[
        ["n_per_class", "subset_label", "classifier_label",
         "mean_loss", "standard_error", "replications"]
    ].rename(
        columns={
            "n_per_class": "n_per_class",
            "subset_label": "Subset",
            "classifier_label": "Classifier",
            "mean_loss": "Mean test loss",
            "standard_error": "Std. error",
            "replications": "Reps",
        }
    )

    # --- HTML assembly --------------------------------------------------------
    classifiers_list = ", ".join(
        classifier_label(c) for c in result.classifier_names
    )
    subsets_list = ", ".join(subset_label(s) for s in result.subsets)

    loss_curve_html = "".join(
        f"<div class='figure'><img src='{src}' alt='loss curve "
        f"{html.escape(classifier_label(clf))}'/></div>"
        for clf, src in loss_curves.items()
    )
    scatter_html = "".join(
        f"<div class='figure inline'><img src='{src}' alt='scatter {html.escape(name)}'/></div>"
        for name, src in scatter_2d.items()
    )
    scatter_3d_html = (
        f"<div class='figure'><img src='{scatter_3d}' alt='3D scatter'/></div>"
        if scatter_3d
        else ""
    )

    doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>CoInfoSim — Synthetic Scenario 1 Report</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
         margin: 0 auto; max-width: 1000px; padding: 2rem; color: #222; line-height: 1.5; }}
  h1 {{ font-size: 1.8rem; border-bottom: 3px solid #1f77b4; padding-bottom: .4rem; }}
  h2 {{ font-size: 1.3rem; margin-top: 2rem; color: #1f3b66; border-bottom: 1px solid #ddd; padding-bottom: .2rem; }}
  .notice {{ background: #fff8e1; border-left: 4px solid #f0ad4e; padding: .8rem 1rem; margin: 1rem 0; }}
  .question {{ font-style: italic; background: #eef5fb; padding: .8rem 1rem; border-left: 4px solid #1f77b4; }}
  table.data {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }}
  table.data th, table.data td {{ border: 1px solid #ccc; padding: .35rem .6rem; text-align: center; }}
  table.data th {{ background: #f0f4f8; }}
  table.matrix {{ border-collapse: collapse; display: inline-block; margin: .3rem 0; }}
  table.matrix td {{ border: 1px solid #bbb; padding: .25rem .6rem; text-align: right; font-family: monospace; }}
  .figure {{ text-align: center; margin: 1.2rem 0; }}
  .figure img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  .figure.inline {{ display: inline-block; width: 32%; margin: .3rem .3%; vertical-align: top; }}
  dl.meta {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; }}
  dl.meta dt {{ font-weight: 600; color: #444; }}
  code {{ background: #f5f5f5; padding: .1rem .3rem; border-radius: 3px; }}
</style>
</head>
<body>

<h1>CoInfoSim — Synthetic Scenario 1 Report</h1>
<p><strong>Scenario:</strong> {html.escape(scenario.name)}</p>

<div class="notice"><strong>Metric notice.</strong> {html.escape(_EXCLUSION_NOTICE)}</div>

<h2>Scientific question</h2>
<p class="question">{html.escape(scenario.question)}</p>

<h2>Model parameters</h2>
<dl class="meta">
  <dt>Channels (d)</dt><dd>{model.d}</dd>
  <dt>Classes (K)</dt><dd>{model.K} (labels {list(model.class_labels)})</dd>
  <dt>μ₀</dt><dd>{_vector_html(model.mean(0))}</dd>
  <dt>μ₁</dt><dd>{_vector_html(model.mean(1))}</dd>
  <dt>Σ₀</dt><dd>{_matrix_html(model.covariance(0))}</dd>
  <dt>Σ₁</dt><dd>{_matrix_html(model.covariance(1))}</dd>
</dl>

<h2>Run configuration</h2>
<dl class="meta">
  <dt>Execution mode</dt><dd><code>{html.escape(config.mode)}</code></dd>
  <dt>Sample sizes (n_per_class)</dt><dd>{list(config.sample_sizes)}</dd>
  <dt>Fixed test-set size per class</dt><dd>{config.test_samples_per_class}</dd>
  <dt>Monte Carlo stopping rule</dt><dd>Standard-error rule (stop when max 1.96·SE ≤ target)</dd>
  <dt>Target CI half-width</dt><dd>{config.ci_half_width_target}</dd>
  <dt>Min / max replications</dt><dd>{config.min_replications} / {config.max_replications}</dd>
  <dt>Replication batch size</dt><dd>{config.replication_batch_size}</dd>
  <dt>Base seed</dt><dd>{config.base_seed}</dd>
  <dt>Classifiers</dt><dd>{html.escape(classifiers_list)}</dd>
  <dt>Channel subsets</dt><dd>{html.escape(subsets_list)}</dd>
  <dt>Runtime</dt><dd>{result.runtime_seconds:.2f} s</dd>
</dl>

<h2>Monte Carlo stopping &amp; replications</h2>
{_stopping_table_html(result)}

<h2>Empirical test-loss curves</h2>
<p>One panel per classifier; error bars show the standard error of the mean across replications.</p>
{loss_curve_html}

<h2>Best-subset rankings</h2>
<p>For each classifier and sample size, the channel subset with the lowest mean empirical test loss
(<code>A*_f(n) = argmin_A Lbar_(A,f)(n)</code>).</p>
{_dataframe_html(rankings_display, float_cols={"Mean test loss": ".4f"})}

<h2>Cooperative advantage thresholds N*</h2>
<p><code>N*(A, B; f) = min {{ n : Lbar_(B,f)(n) &lt; Lbar_(A,f)(n) }}</code> — the smallest
training size per class at which the cooperative subset B beats subset A. "—" means no
threshold was observed within the evaluated sample sizes.</p>
{_dataframe_html(thresholds_display)}

<h2>Monte Carlo uncertainty summary</h2>
<p>Per-cell mean empirical test loss, standard error, and replication count.</p>
{_dataframe_html(summary_display, float_cols={"Mean test loss": ".4f", "Std. error": ".4f"})}

<h2>Data diagnostics — 2D scatter plots</h2>
<p>Drawn from the fixed test set.</p>
{scatter_html}

<h2>Data diagnostics — 3D scatter plot</h2>
{scatter_3d_html if scatter_3d_html else "<p>(not available)</p>"}

<hr/>
<p class="notice">{html.escape(_EXCLUSION_NOTICE)}</p>

</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path
