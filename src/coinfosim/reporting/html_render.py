"""
Shared HTML rendering helpers for CoInfoSim reports.

This module centralizes the CSS and rendering primitives used by both the
individual simulation report and the scenario report, so both share a
consistent academic visual language and support a "portable embedded"
mode that inlines images as base64 data URIs (no external dependencies).

The helpers are intentionally small and composable. They do not change any
scientific logic; they only format already-computed artifacts (matrices,
DataFrames, PNG files) as HTML.

Example:
    >>> from coinfosim.reporting.html_render import render_section, render_matrix_table
    >>> html = render_section("Diagnostics", render_matrix_table("Sigma", [[1, 0], [0, 1]]))
"""

import base64
import mimetypes
import os
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# Shared CSS used by every CoInfoSim report. Inspired by the Sensor Crossover
# layout: centered max-width column, sober academic colors, card-like
# sections, responsive figures and tables.
REPORT_CSS = """
:root {
    --bg: #ffffff;
    --fg: #1a1a1a;
    --muted: #5f6b7a;
    --accent: #2d4f8a;
    --border: #e2e6ee;
    --card-bg: #fafbfd;
    --table-stripe: #f4f6fa;
}

* { box-sizing: border-box; }

html, body {
    margin: 0;
    padding: 0;
    background: var(--bg);
    color: var(--fg);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 "Helvetica Neue", Arial, "Noto Sans", sans-serif;
    font-size: 16px;
    line-height: 1.55;
}

.container {
    max-width: 1080px;
    margin: 0 auto;
    padding: 32px 24px 64px 24px;
}

header.report-header {
    border-bottom: 1px solid var(--border);
    padding-bottom: 18px;
    margin-bottom: 28px;
}

header.report-header h1 {
    margin: 0 0 6px 0;
    font-size: 28px;
    color: var(--accent);
    font-weight: 600;
}

header.report-header .subtitle {
    color: var(--muted);
    font-size: 15px;
}

nav.toc {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 18px;
    margin-bottom: 32px;
    font-size: 14px;
}

nav.toc strong {
    display: block;
    margin-bottom: 6px;
    color: var(--accent);
}

nav.toc ol {
    margin: 0;
    padding-left: 22px;
}

nav.toc a {
    color: var(--fg);
    text-decoration: none;
}

nav.toc a:hover { color: var(--accent); text-decoration: underline; }

section.card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 22px 24px;
    margin-bottom: 26px;
}

section.card h2 {
    margin: 0 0 12px 0;
    font-size: 20px;
    color: var(--accent);
    font-weight: 600;
    border-bottom: 1px solid var(--border);
    padding-bottom: 6px;
}

section.card h3 {
    font-size: 16px;
    color: var(--fg);
    margin: 18px 0 8px 0;
    font-weight: 600;
}

section.card p {
    color: var(--fg);
    margin: 0 0 10px 0;
}

section.card p.note {
    color: var(--muted);
    font-size: 14px;
    font-style: italic;
}

/* Figures */
.figure {
    margin: 14px 0 22px 0;
    padding: 12px;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
}

.figure .figure-title {
    font-weight: 600;
    color: var(--fg);
    margin-bottom: 6px;
}

.figure img {
    display: block;
    max-width: 100%;
    height: auto;
    margin: 0 auto;
    border: none;
    border-radius: 4px;
}

.figure .figure-caption {
    color: var(--muted);
    font-size: 13px;
    margin-top: 8px;
    text-align: center;
}

.figure.missing {
    color: var(--muted);
    font-style: italic;
    background: var(--card-bg);
}

/* Tables */
.table-wrap {
    margin: 10px 0 18px 0;
    overflow-x: auto;
}

.table-wrap .table-title {
    font-weight: 600;
    margin-bottom: 6px;
}

.table-wrap table {
    border-collapse: collapse;
    width: 100%;
    font-size: 14px;
    background: var(--bg);
}

.table-wrap th, .table-wrap td {
    border: 1px solid var(--border);
    padding: 6px 10px;
    text-align: right;
}

.table-wrap th {
    background: var(--accent);
    color: #ffffff;
    font-weight: 600;
    text-align: center;
}

.table-wrap tr:nth-child(even) td {
    background: var(--table-stripe);
}

.table-wrap td:first-child,
.table-wrap th:first-child {
    text-align: left;
}

.matrix-table td { font-family: ui-monospace, "SFMono-Regular", Menlo, Consolas, monospace; }

.unavailable {
    color: var(--muted);
    font-style: italic;
    padding: 8px 0;
}

/* Two-column figure grid for compact pairs */
.figure-grid {
    display: grid;
    grid-template-columns: 1fr;
    gap: 16px;
}

@media (min-width: 820px) {
    .figure-grid.two { grid-template-columns: 1fr 1fr; }
}

footer.report-footer {
    margin-top: 36px;
    padding-top: 14px;
    border-top: 1px solid var(--border);
    color: var(--muted);
    font-size: 13px;
    text-align: center;
}
"""


_SUB_DIGITS = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")


def _subscript_digits(s: str) -> str:
    """Convert ASCII digits in ``s`` to their Unicode subscript variants.

    Args:
        s: String potentially containing ASCII digits.

    Returns:
        String with digits replaced by ``₀-₉``.
    """
    return s.translate(_SUB_DIGITS)


def format_param_label(param_index: int, dim: int) -> dict:
    """Return human-readable labels for a model parameter index.

    CoInfoSim parameter vectors layout (for ``dim`` features) is
    ``[σ₁, ..., σ_dim, ρ₁₂, ρ₁₃, ρ₂₃, ...]``. This helper returns the
    label associated with ``param_index`` in four flavors so the same
    information can be used in HTML text, matplotlib axes, plain text
    contexts (CSV, console, table headers) and filenames.

    Args:
        param_index: Zero-based index into the parameter vector.
        dim: Number of features in the model.

    Returns:
        Dict with keys ``"html"`` (Unicode + ``<sub>``), ``"unicode"``
        (plain Unicode using subscript characters and ``ρ`` / ``σ``,
        safe for matplotlib labels and CSV), ``"mathtext"`` (matplotlib
        mathtext, kept for backwards compatibility), and ``"plain"``
        (ASCII-safe token usable in filenames).
    """
    if param_index < dim:
        idx = param_index + 1
        return {
            "html": f"σ<sub>{idx}</sub>",
            "unicode": f"σ{_subscript_digits(str(idx))}",
            "mathtext": rf"$\sigma_{{{idx}}}$",
            "plain": f"sigma{idx}",
        }
    rho_index = param_index - dim
    pairs = [(i + 1, j + 1) for i in range(dim) for j in range(i + 1, dim)]
    if rho_index >= len(pairs):
        return {
            "html": f"param[{param_index}]",
            "unicode": f"param[{param_index}]",
            "mathtext": f"param[{param_index}]",
            "plain": f"param{param_index}",
        }
    i, j = pairs[rho_index]
    # When the varying parameter is a model rho, it is the within-class
    # (conditional) correlation by definition. Use the standardized
    # ``ρ_cond,ij`` notation to distinguish from ρ_global,* references.
    return {
        "html": f"ρ<sub>cond,{i}{j}</sub>",
        "unicode": f"ρ_cond,{i}{j}",
        "mathtext": rf"$\rho_{{\mathrm{{cond}},{i}{j}}}$",
        "plain": f"rho_cond_{i}{j}",
    }


def extract_offdiag_pairs(matrix) -> List[Tuple[str, str, float]]:
    """Enumerate upper-triangular off-diagonal entries of a square matrix.

    Args:
        matrix: 2D array-like (or None).

    Returns:
        List of ``(html_label, mathtext_label, value)`` tuples where labels
        identify pairs like ``"ρ₁₂"`` / ``r"$\\rho_{12}$"``.
    """
    if matrix is None:
        return []
    arr = np.asarray(matrix, dtype=float)
    out: List[Tuple[str, str, float]] = []
    n = arr.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            out.append((
                f"ρ<sub>{i + 1}{j + 1}</sub>",
                rf"$\rho_{{{i + 1}{j + 1}}}$",
                float(arr[i, j]),
            ))
    return out


def image_to_data_uri(path: str) -> Optional[str]:
    """Encode an image file as a base64 data URI.

    Args:
        path: Filesystem path to the image.

    Returns:
        ``data:<mime>;base64,...`` string, or ``None`` if the file is missing
        or unreadable.
    """
    if not path or not os.path.exists(path):
        return None
    mime, _ = mimetypes.guess_type(path)
    if mime is None:
        mime = "application/octet-stream"
    try:
        with open(path, "rb") as fh:
            data = fh.read()
    except OSError:
        return None
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _rel_src(image_path: str, base_dir: str) -> str:
    """Compute a relative path from base_dir to image_path using forward slashes."""
    return os.path.relpath(image_path, base_dir).replace(os.path.sep, "/")


def render_image_card(
    title: str,
    image_path: Optional[str],
    caption: Optional[str] = None,
    embed: bool = False,
    base_dir: Optional[str] = None,
    alt: Optional[str] = None,
) -> str:
    """Render an image inside a styled figure card.

    Args:
        title: Short figure title shown above the image.
        image_path: Absolute path to the PNG/JPG file.
        caption: Optional explanatory caption shown below the image.
        embed: If True, encode the image as a base64 data URI (portable);
            if False, emit a relative ``src`` from ``base_dir``.
        base_dir: Directory the report is written to (used to compute
            relative paths when ``embed`` is False).
        alt: Optional alt text. Defaults to ``title``.

    Returns:
        HTML fragment for the figure (always returns a card, even when the
        image is missing, in which case a "Not available" placeholder is
        rendered).
    """
    alt_text = alt or title
    if not image_path or not os.path.exists(image_path):
        return (
            f'<div class="figure missing">'
            f'<div class="figure-title">{title}</div>'
            f'<div class="unavailable">Image not available.</div>'
            f'</div>'
        )

    if embed:
        src = image_to_data_uri(image_path)
        if src is None:
            return (
                f'<div class="figure missing">'
                f'<div class="figure-title">{title}</div>'
                f'<div class="unavailable">Image not available.</div>'
                f'</div>'
            )
    else:
        if base_dir is None:
            src = image_path
        else:
            src = _rel_src(image_path, base_dir)

    caption_html = (
        f'<div class="figure-caption">{caption}</div>' if caption else ""
    )
    return (
        f'<div class="figure">'
        f'<div class="figure-title">{title}</div>'
        f'<img src="{src}" alt="{alt_text}">'
        f'{caption_html}'
        f'</div>'
    )


def render_matrix_table(
    title: str,
    matrix: Optional[Union[Sequence[Sequence[float]], np.ndarray]],
    caption: Optional[str] = None,
    fmt: str = "{:.4f}",
    row_labels: Optional[Sequence[str]] = None,
    col_labels: Optional[Sequence[str]] = None,
) -> str:
    """Render a numeric matrix as an HTML table inside a styled wrapper.

    Args:
        title: Title shown above the table.
        matrix: 2D array-like. ``None`` produces a "Not available" placeholder.
        caption: Optional caption shown below the table.
        fmt: Format string applied to each numeric cell.
        row_labels: Optional row labels prepended as the first column.
        col_labels: Optional column labels rendered as a header row.

    Returns:
        HTML fragment containing the matrix table.
    """
    if matrix is None:
        return (
            f'<div class="table-wrap">'
            f'<div class="table-title">{title}</div>'
            f'<div class="unavailable">Not available.</div>'
            f'</div>'
        )

    arr = np.asarray(matrix, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    if col_labels is None:
        col_labels = [f"x<sub>{i+1}</sub>" for i in range(arr.shape[1])]
    if row_labels is None:
        row_labels = [f"x<sub>{i+1}</sub>" for i in range(arr.shape[0])]

    header_cells = "".join(f"<th>{c}</th>" for c in col_labels)
    body_rows = []
    for i in range(arr.shape[0]):
        cells = "".join(f"<td>{fmt.format(arr[i, j])}</td>" for j in range(arr.shape[1]))
        body_rows.append(f"<tr><th>{row_labels[i]}</th>{cells}</tr>")

    caption_html = (
        f'<div class="figure-caption" style="text-align:left">{caption}</div>'
        if caption else ""
    )
    return (
        f'<div class="table-wrap">'
        f'<div class="table-title">{title}</div>'
        f'<table class="matrix-table"><thead><tr><th></th>{header_cells}</tr></thead>'
        f'<tbody>{"".join(body_rows)}</tbody></table>'
        f'{caption_html}'
        f'</div>'
    )


def render_dataframe_table(
    title: str,
    df: Optional[pd.DataFrame],
    caption: Optional[str] = None,
    escape: bool = True,
) -> str:
    """Render a pandas DataFrame as a styled HTML table.

    Args:
        title: Title shown above the table.
        df: DataFrame to render. ``None`` produces a "Not available" placeholder.
        caption: Optional caption shown below the table.
        escape: Whether to escape HTML inside cells. Set False for tables
            that contain anchor tags (e.g. links to other reports).

    Returns:
        HTML fragment containing the DataFrame as a table.
    """
    if df is None:
        return (
            f'<div class="table-wrap">'
            f'<div class="table-title">{title}</div>'
            f'<div class="unavailable">Not available.</div>'
            f'</div>'
        )

    table_html = df.to_html(index=False, escape=escape, border=0)
    caption_html = (
        f'<div class="figure-caption" style="text-align:left">{caption}</div>'
        if caption else ""
    )
    return (
        f'<div class="table-wrap">'
        f'<div class="table-title">{title}</div>'
        f'{table_html}'
        f'{caption_html}'
        f'</div>'
    )


def render_section(
    title: str,
    content: str,
    section_id: Optional[str] = None,
) -> str:
    """Wrap arbitrary HTML content in a styled report section card.

    Args:
        title: Section title (rendered as ``h2``).
        content: HTML fragment placed inside the section body.
        section_id: Optional DOM id so the section can be linked from the TOC.

    Returns:
        HTML fragment for the section.
    """
    id_attr = f' id="{section_id}"' if section_id else ""
    return (
        f'<section class="card"{id_attr}>'
        f'<h2>{title}</h2>'
        f'{content}'
        f'</section>'
    )


def render_toc(items: Iterable[Tuple[str, str]]) -> str:
    """Render a small inline table of contents.

    Args:
        items: Iterable of ``(section_id, title)`` tuples.

    Returns:
        HTML fragment with an ordered list of in-page links.
    """
    li = "".join(
        f'<li><a href="#{sid}">{title}</a></li>' for sid, title in items
    )
    return f'<nav class="toc"><strong>Contents</strong><ol>{li}</ol></nav>'


def render_page(
    title: str,
    subtitle: str,
    toc_items: Iterable[Tuple[str, str]],
    sections_html: str,
    footer: Optional[str] = None,
) -> str:
    """Wrap a set of sections in a complete HTML page with embedded CSS.

    Args:
        title: Page title and ``<h1>`` heading.
        subtitle: Subtitle line displayed under the title.
        toc_items: Iterable of ``(section_id, title)`` for the TOC.
        sections_html: Pre-rendered HTML for all sections.
        footer: Optional footer text. A default is used if omitted.

    Returns:
        Complete HTML document as a string.
    """
    footer_text = footer or (
        "Generated by CoInfoSim — Loss Analysis of Linear Classifiers on Gaussian Samples."
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{REPORT_CSS}</style>
</head>
<body>
<div class="container">
<header class="report-header">
<h1>{title}</h1>
<div class="subtitle">{subtitle}</div>
</header>
{render_toc(toc_items)}
{sections_html}
<footer class="report-footer">{footer_text}</footer>
</div>
</body>
</html>
"""
