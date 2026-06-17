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
import pandas as pd  # noqa: E402

from coinfosim.classifiers.registry import classifier_label
from coinfosim.results.analysis import (
    best_subset_by_sample_size,
    final_ranking_at_largest_n,
    standard_threshold_comparisons,
)
from coinfosim.results.summary import hardest_cell_summary, summary_dataframe
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.scenarios.synthetic import SyntheticScenario
from coinfosim.simulation.monte_carlo import SimulationResult
from coinfosim.simulation.subsets import subset_label
from coinfosim.reports.visualizations import (
    plot_1d_grid,
    plot_2d_grid,
    plot_3d_grid,
    generate_growth_gif,
)

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
    """Plot empirical test-loss curves vs n for one classifier.

    The y-axis uses the math symbol ``$\\hat{L}$`` (empirical misclassification
    rate on the fixed test set); the classifier name alone is the title.
    """
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
            markersize=4,
            linewidth=1.6,
            capsize=3,
            color=color,
            label=subset_label(subset),
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(sample_sizes)
    ax.set_xticklabels([str(n) for n in sample_sizes])
    ax.set_xlabel(r"$n_{\mathrm{per\ class}}$")
    ax.set_ylabel(r"$\hat{L}$")
    ax.set_title(classifier_label(classifier_name), fontsize=12, fontweight="bold")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(title=r"Subset $A$", fontsize=8, ncol=2, framealpha=0.9)
    return _fig_to_base64(fig)


def _matrix_html(matrix: np.ndarray) -> str:
    """Render a small numeric matrix as a bracketed HTML matrix."""
    rows = []
    for row in matrix:
        cells = "".join(f"<td>{v:.2f}</td>" for v in row)
        rows.append(f"<tr>{cells}</tr>")
    return (
        "<span class='matrix-wrap'><span class='bracket'>[</span>"
        f"<table class='matrix'>{''.join(rows)}</table>"
        "<span class='bracket'>]</span></span>"
    )


def _vector_html(vector: np.ndarray) -> str:
    """Render a vector as a bracketed inline row."""
    cells = "".join(f"<td>{v:.2f}</td>" for v in vector)
    return (
        "<span class='matrix-wrap'><span class='bracket'>[</span>"
        f"<table class='matrix'><tr>{cells}</tr></table>"
        "<span class='bracket'>]</span></span>"
    )


def _dataframe_html(
    df,
    float_cols: Optional[Dict[str, str]] = None,
    show_index: bool = False,
    index_header: str = "",
) -> str:
    """Render a pandas DataFrame to an HTML table with escaped values.

    When ``show_index`` is true the DataFrame index is rendered as the first
    column with header ``index_header``.
    """
    float_cols = float_cols or {}
    header_cells = ""
    if show_index:
        header_cells += f"<th>{html.escape(index_header)}</th>"
    header_cells += "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    body_rows = []
    for idx, row in df.iterrows():
        cells = []
        if show_index:
            cells.append(f"<td><strong>{html.escape(str(idx))}</strong></td>")
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
        f"<table class='data'><thead><tr>{header_cells}</tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )


def _final_ranking_html(final_ranking_df, classifier_names: Sequence[str]) -> str:
    """Render the final-ranking-at-largest-n as one compact table per classifier."""
    blocks = []
    for clf in classifier_names:
        sub = final_ranking_df[final_ranking_df["classifier"] == clf]
        sub = sub.sort_values("rank")
        label = classifier_label(clf)
        display = sub[["rank", "subset_label", "mean_loss", "standard_error"]].rename(
            columns={
                "rank": "Rank",
                "subset_label": "Subset",
                "mean_loss": "Mean test loss",
                "standard_error": "Std. error",
            }
        )
        table = _dataframe_html(
            display,
            float_cols={"Mean test loss": ".4f", "Std. error": ".4f"},
        )
        blocks.append(f"<h3 class='rank-h3'>{html.escape(label)}</h3>{table}")
    return "".join(blocks)


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
    make_gif: bool = True,
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
    make_gif:
        Whether to render the sample-growth animation GIF and reference it in
        the report. Disable to speed up report generation (e.g. in tests).
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
    # SLACGS-inspired geometric diagnostics. The training replication used for
    # the Linear SVM separators is fixed (replication_id=0) and the sample size
    # is the largest evaluated one, so the figures are deterministic.
    geometry_n = max(result.sample_sizes)
    grid_1d = _fig_to_base64(
        plot_1d_grid(model, sampler, n_per_class=geometry_n, replication_id=0)
    )
    grid_2d = _fig_to_base64(
        plot_2d_grid(model, sampler, n_per_class=geometry_n, replication_id=0)
    )
    grid_3d_fig = plot_3d_grid(model, sampler, n_per_class=geometry_n, replication_id=0)
    grid_3d = _fig_to_base64(grid_3d_fig) if grid_3d_fig is not None else None

    # Sample-growth animation (geometry + per-classifier loss curves with N*
    # markers). Embedded inline when small enough, otherwise linked as a
    # sibling artifact next to the report.
    gif_src: Optional[str] = None
    if make_gif:
        gif_path = output_dir / "synthetic_scenario_1_growth.gif"
        generate_growth_gif(model, sampler, result, gif_path)
        gif_bytes = gif_path.read_bytes()
        if len(gif_bytes) <= 4_000_000:
            encoded = base64.b64encode(gif_bytes).decode("ascii")
            gif_src = f"data:image/gif;base64,{encoded}"
        else:
            gif_src = gif_path.name  # link to sibling file

    # --- Tables ---------------------------------------------------------------
    summary_df = summary_dataframe(
        result.accumulator,
        result.sample_sizes,
        result.subsets,
        result.classifier_names,
    )
    best_by_n_df = best_subset_by_sample_size(
        result.accumulator,
        result.classifier_names,
        result.sample_sizes,
        result.subsets,
    )
    final_ranking_df = final_ranking_at_largest_n(
        result.accumulator,
        result.classifier_names,
        result.sample_sizes,
        result.subsets,
    )
    hardest_df = hardest_cell_summary(
        result.accumulator,
        result.sample_sizes,
        result.subsets,
        result.classifier_names,
    )
    thresholds_df = standard_threshold_comparisons(
        result.accumulator,
        result.classifier_names,
        result.sample_sizes,
        result.subsets,
    )

    thresholds_display = thresholds_df[
        [
            "classifier_label",
            "comparison",
            "subset_a_label",
            "subset_b_label",
            "n_star_grid",
            "n_star_interp",
        ]
    ].copy()
    thresholds_display["n_star_interp"] = thresholds_display["n_star_interp"].map(
        lambda v: "—" if v is None or pd.isna(v) else f"{float(v):.1f}"
    )
    thresholds_display["n_star_grid"] = thresholds_display["n_star_grid"].map(
        lambda v: "—" if v is None or pd.isna(v) else f"{int(v)}"
    )
    thresholds_display = thresholds_display.rename(
        columns={
            "classifier_label": "Classifier",
            "comparison": "Comparison",
            "subset_a_label": "A",
            "subset_b_label": "B (cooperative)",
            "n_star_grid": "N* (grid)",
            "n_star_interp": "N* (interp)",
        }
    )
    hardest_display = hardest_df[
        ["n_per_class", "subset_label", "classifier_label",
         "ci_half_width", "mean_loss"]
    ].rename(
        columns={
            "n_per_class": "n_per_class",
            "subset_label": "Hardest subset",
            "classifier_label": "Classifier",
            "ci_half_width": "Max CI half-width",
            "mean_loss": "Mean test loss",
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

    # Final ranking rendered per classifier as compact grouped tables.
    final_ranking_html = _final_ranking_html(final_ranking_df, result.classifier_names)

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
    grid_3d_html = (
        f"<div class='figure'><img src='{grid_3d}' alt='3D model geometry grid'/></div>"
        if grid_3d
        else "<p>(3-D geometry requires at least three channels.)</p>"
    )
    if gif_src is not None:
        gif_html = (
            f"<div class='figure'><img src='{gif_src}' "
            f"alt='sample-growth animation'/></div>"
        )
    else:
        gif_html = "<p>(Animation not generated for this report.)</p>"

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
  .define {{ background: #eef9f0; border-left: 4px solid #2ca02c; padding: .8rem 1rem; margin: 1rem 0; }}
  .question {{ font-style: italic; background: #eef5fb; padding: .8rem 1rem; border-left: 4px solid #1f77b4; }}
  table.data {{ border-collapse: collapse; width: 100%; margin: 1rem 0; font-size: .9rem; }}
  table.data th, table.data td {{ border: 1px solid #ccc; padding: .35rem .6rem; text-align: center; }}
  table.data th {{ background: #f0f4f8; }}
  /* Cards for model parameters */
  .cards {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; margin: 1rem 0; }}
  .card {{ border: 1px solid #dde3ea; border-radius: 8px; padding: .8rem 1rem;
           background: #fafcff; box-shadow: 0 1px 2px rgba(0,0,0,.04); }}
  .card .card-title {{ font-weight: 700; color: #1f3b66; margin-bottom: .4rem; font-size: 1rem; }}
  .card .card-sub {{ color: #667; font-size: .82rem; margin-bottom: .5rem; }}
  .card.wide {{ grid-column: 1 / -1; }}
  .matrix-wrap {{ display: inline-flex; align-items: stretch; vertical-align: middle; }}
  .matrix-wrap .bracket {{ font-size: 1.9em; line-height: 1; display: flex; align-items: center; color: #355; }}
  table.matrix {{ border-collapse: collapse; display: inline-block; margin: 0 .2rem; }}
  table.matrix td {{ padding: .15rem .55rem; text-align: right; font-family: 'Courier New', monospace; }}
  .figure {{ text-align: center; margin: 1.2rem 0; }}
  .figure img {{ max-width: 100%; border: 1px solid #eee; border-radius: 4px; }}
  .figure.inline {{ display: inline-block; width: 32%; margin: .3rem .3%; vertical-align: top; }}
  dl.meta {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; }}
  dl.meta dt {{ font-weight: 600; color: #444; }}
  code {{ background: #f5f5f5; padding: .1rem .3rem; border-radius: 3px; }}
  h3.rank-h3 {{ font-size: 1rem; color: #1f3b66; margin: 1rem 0 .2rem; }}
  details {{ margin: 1rem 0; }}
  details > summary {{ cursor: pointer; font-weight: 600; color: #1f3b66;
                       padding: .4rem 0; }}
</style>
</head>
<body>

<h1>CoInfoSim — Synthetic Scenario 1 Report</h1>
<p><strong>Scenario:</strong> {html.escape(scenario.name)}</p>

<div class="notice"><strong>Metric notice.</strong> {html.escape(_EXCLUSION_NOTICE)}</div>

<h2>Scientific question</h2>
<p class="question">{html.escape(scenario.question)}</p>

<h2>Model parameters</h2>
<p>Two Gaussian classes over <em>d</em> = {model.d} channels
(<code>X1</code>, <code>X2</code>, <code>X3</code>), with class labels {list(model.class_labels)}.
Each class <em>k</em> is distributed as <code>N(μ_k, Σ_k)</code>.</p>
<div class="cards">
  <div class="card">
    <div class="card-title">Class 0</div>
    <div class="card-sub">mean vector μ₀</div>
    {_vector_html(model.mean(0))}
  </div>
  <div class="card">
    <div class="card-title">Class 1</div>
    <div class="card-sub">mean vector μ₁</div>
    {_vector_html(model.mean(1))}
  </div>
  <div class="card">
    <div class="card-title">Covariance Σ₀</div>
    <div class="card-sub">class 0 (channels X1, X2, X3)</div>
    {_matrix_html(model.covariance(0))}
  </div>
  <div class="card">
    <div class="card-title">Covariance Σ₁</div>
    <div class="card-sub">class 1 (channels X1, X2, X3)</div>
    {_matrix_html(model.covariance(1))}
  </div>
</div>

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
<div class="define"><strong>Notation.</strong> <em>L&#770;</em> (the y-axis of every panel)
denotes the <strong>empirical misclassification rate on the fixed test set</strong>,
i.e. the fraction of test points misclassified. The x-axis
<em>n</em><sub>per&nbsp;class</sub> is the number of training samples per class
(log₂ scale). One panel is shown per classifier; error bars are the standard error of the
mean across replications, and each curve corresponds to a channel subset <em>A</em>.</div>
{loss_curve_html}

<h2>Best subset by sample size</h2>
<p>For each classifier (columns) and training size <em>n</em><sub>per&nbsp;class</sub> (rows),
the channel subset <em>A*</em> with the lowest mean test loss,
<em>A*<sub>f</sub>(n) = argmin<sub>A</sub> L&#770;<sub>A,f</sub>(n)</em>.</p>
{_dataframe_html(best_by_n_df, show_index=True, index_header="n_per_class")}

<h2>Final ranking at largest n</h2>
<p>All channel subsets ranked by mean test loss at the largest evaluated
training size (<em>n</em><sub>per&nbsp;class</sub> = {max(result.sample_sizes)}),
one table per classifier. Rank&nbsp;1 is the best (lowest <em>L&#770;</em>).</p>
{final_ranking_html}

<h2>Cooperative advantage thresholds N*</h2>
<p><em>N*(A, B; f) = min {{ n : L&#770;<sub>B,f</sub>(n) &lt; L&#770;<sub>A,f</sub>(n) }}</em> — the
smallest training size per class at which the cooperative subset <em>B</em> beats subset
<em>A</em>. <strong>N* (grid)</strong> is the smallest evaluated sample size where the
crossing occurs; <strong>N* (interp)</strong> linearly interpolates the crossing between
the two bracketing sample sizes for a finer estimate. "—" means no threshold was
observed within the evaluated sample sizes.</p>
{_dataframe_html(thresholds_display)}

<h2>Monte Carlo uncertainty</h2>
<p>The standard-error stopping rule halts each <em>n</em><sub>per&nbsp;class</sub> when the
largest 95% CI half-width (<em>1.96·SE</em>) across all subset/classifier cells meets the
target. The hardest cell below is the slowest-converging one driving that decision.</p>
{_dataframe_html(hardest_display, float_cols={"Max CI half-width": ".4f", "Mean test loss": ".4f"})}

<details>
<summary>Full per-cell uncertainty summary (all subsets × classifiers × n)</summary>
{_dataframe_html(summary_display, float_cols={"Mean test loss": ".4f", "Std. error": ".4f"})}
</details>

<h2>Synthetic data geometry</h2>
<div class="define">
<p>These diagnostics illustrate the <strong>geometry of the Gaussian model</strong> and how a
linear boundary separates the classes channel-by-channel. The ellipses (2-D) and ellipsoids
(3-D) are computed from the <em>model</em> means and covariances, not from sample estimates.</p>
<p>The straight separators are drawn with a <strong>Linear SVM only</strong>, purely for
illustration. They are independent of the three classifiers evaluated by the simulation
(Linear SVM, Logistic Regression, Gaussian Naive Bayes) — the loss curves above still use all
three. Separators are fit on a single training replication
(<em>n</em><sub>per&nbsp;class</sub> = {geometry_n}, replication&nbsp;0).</p>
</div>

<h3>1-D single-channel views</h3>
<p>Per-channel class samples, model density curves, class means with one-sigma markers, and the
Linear SVM threshold.</p>
<div class='figure'><img src='{grid_1d}' alt='1D model geometry grid'/></div>

<h3>2-D pairwise views</h3>
<p>Pairwise scatter with class centers, model Gaussian ellipses (1σ solid, 2σ dashed), and the
Linear SVM separating line. Axis limits adapt to the data and ellipses.</p>
<div class='figure'><img src='{grid_2d}' alt='2D model geometry grid'/></div>

<h3>3-D triple-wise views</h3>
<p>Triple-wise scatter with class centers, model Gaussian ellipsoids (1σ), and the Linear SVM
separating hyperplane.</p>
{grid_3d_html}

<h2>Sample-growth animation</h2>
<p>The animation below shows the simulation building up over
<em>n</em><sub>per&nbsp;class</sub>. The left panel is a 2-D geometric view of a single training
replication (replication&nbsp;0) growing with <em>n</em>, with the model Gaussian ellipses and
the Linear SVM separating line. The three right panels show the empirical test-loss curves for
the three evaluated classifiers, drawn progressively. A <span style="color:red;font-weight:700">
red marker</span> appears in each classifier panel once its cooperative threshold <em>N*</em>
(full subset vs best single channel) has been reached.</p>
{gif_html}

<hr/>
<p class="notice">{html.escape(_EXCLUSION_NOTICE)}</p>

</body>
</html>"""

    out_path.write_text(doc, encoding="utf-8")
    return out_path
