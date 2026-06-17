"""
SLACGS-inspired geometric visualization helpers for CoInfoSim Sprint 1.

This module provides reusable, deterministic Matplotlib figures that
illustrate the *geometry* of a :class:`GaussianSimulationModel`:

* 1D single-channel views (samples, density curves, mean / one-sigma markers,
  Linear SVM threshold);
* 2D pairwise views (scatter, class centers, Gaussian ellipses, Linear SVM
  separating line);
* 3D triple-wise views (scatter, class centers, Gaussian ellipsoids, Linear
  SVM separating hyperplane).

Design notes
------------
* The geometric separator is **always a Linear SVM** and is purely
  illustrative. It is independent of the three classifiers evaluated by the
  Monte Carlo simulation (Linear SVM, Logistic Regression, Gaussian Naive
  Bayes).
* Ellipses / ellipsoids are drawn from the **model** means and covariances
  (the true geometry), not from empirical sample estimates.
* Axis limits adapt to the data and the drawn ellipses rather than using a
  fixed ``[-10, 10]`` window.
* All figures are deterministic: training samples used for the separator come
  from the sampler with a fixed ``replication_id``.
"""

from __future__ import annotations

import io
from itertools import combinations
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")  # headless backend for report figures
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from scipy.stats import norm  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.results.analysis import best_subset, cooperative_threshold
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.simulation.subsets import subset_label

# Class colors (class 0 / class 1) consistent with the report scatter plots.
_CLASS_COLORS = {0: "#1f77b4", 1: "#d62728"}
# Standard-deviation contours drawn for each class (1-sigma solid, 2-sigma dashed).
_SIGMA_LEVELS = (1.0, 2.0)
# Distinct colors for the (up to seven) channel subsets, matching the report.
_SUBSET_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#17becf",
]


def _channel_label(index: int) -> str:
    """One-based channel label, e.g. ``0 -> 'X1'``."""
    return f"X{index + 1}"


def gaussian_ellipse_points(
    mean: np.ndarray,
    cov: np.ndarray,
    n_std: float,
    num_points: int = 120,
) -> np.ndarray:
    """Return ``(num_points, 2)`` points tracing an ``n_std`` Gaussian ellipse.

    The ellipse is the level set of the 2-D Gaussian with the given ``mean``
    and ``cov`` at ``n_std`` standard deviations, obtained by rotating and
    scaling a unit circle with the eigen-decomposition of ``cov``.
    """
    mean = np.asarray(mean, dtype=float).reshape(2)
    cov = np.asarray(cov, dtype=float).reshape(2, 2)
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = n_std * np.sqrt(np.maximum(eigvals, 0.0))
    angles = np.linspace(0.0, 2.0 * np.pi, num_points)
    unit = np.column_stack([np.cos(angles), np.sin(angles)])
    ellipse = unit * radii  # scale along principal axes
    rotated = ellipse @ eigvecs.T  # rotate into data coordinates
    return rotated + mean


def _fit_linear_svm(X: np.ndarray, y: np.ndarray) -> Optional[SVC]:
    """Fit a Linear SVM, returning ``None`` if only one class is present."""
    if len(np.unique(y)) < 2:
        return None
    svm = SVC(kernel="linear")
    svm.fit(X, y)
    return svm


def _training_samples(
    sampler: GaussianClassConditionalSampler,
    channels: Sequence[int],
    n_per_class: int,
    replication_id: int,
):
    """Return ``(X, y)`` training arrays restricted to ``channels``."""
    train = sampler.sample_train(n_per_class, replication_id=replication_id)
    idx = list(channels)
    return train.X[:, idx], train.y


def plot_1d_grid(
    model: GaussianSimulationModel,
    sampler: GaussianClassConditionalSampler,
    n_per_class: int = 64,
    replication_id: int = 0,
    n_show: int = 200,
) -> plt.Figure:
    """Plot all single-channel geometric views.

    For each channel: a sample strip, model density curves, mean and one-sigma
    markers per class, and the Linear SVM decision threshold.
    """
    d = model.d
    fig, axes = plt.subplots(1, d, figsize=(5.0 * d, 3.8), squeeze=False)
    axes = axes[0]

    for ch in range(d):
        ax = axes[ch]
        X, y = _training_samples(sampler, [ch], n_per_class, replication_id)
        X = X[:, 0]

        x_values = []
        for label in model.class_labels:
            mu = float(model.mean(label)[ch])
            sigma = float(np.sqrt(model.covariance(label)[ch, ch]))
            x_values.extend([mu - 3.5 * sigma, mu + 3.5 * sigma])
        x_min, x_max = min(x_values), max(x_values)
        grid = np.linspace(x_min, x_max, 400)

        for label in model.class_labels:
            color = _CLASS_COLORS.get(label, None)
            mu = float(model.mean(label)[ch])
            sigma = float(np.sqrt(model.covariance(label)[ch, ch]))
            mask = y == label
            pts = X[mask][:n_show]
            ax.scatter(
                pts, np.zeros_like(pts), s=12, alpha=0.4, color=color,
                label=f"class {label}",
            )
            ax.plot(grid, norm.pdf(grid, loc=mu, scale=sigma), color=color, lw=1.6)
            ax.axvline(mu, color=color, ls="-", lw=1.0, alpha=0.7)
            ax.axvline(mu - sigma, color=color, ls="--", lw=0.8, alpha=0.6)
            ax.axvline(mu + sigma, color=color, ls="--", lw=0.8, alpha=0.6)

        svm = _fit_linear_svm(X.reshape(-1, 1), y)
        if svm is not None and abs(svm.coef_[0][0]) > 1e-12:
            threshold = -svm.intercept_[0] / svm.coef_[0][0]
            ax.axvline(threshold, color="k", lw=1.6, label="Linear SVM")

        ax.set_xlim(x_min, x_max)
        ax.set_yticks([])
        ax.set_xlabel(f"${_channel_label(ch)}$")
        ax.set_title(_channel_label(ch), fontsize=11, fontweight="bold")
        ax.grid(True, axis="x", alpha=0.3)
        if ch == 0:
            ax.legend(fontsize=8, framealpha=0.9)

    fig.suptitle(
        f"1-D channel geometry (Linear SVM separators, "
        f"$n_{{\\mathrm{{per\\ class}}}} = {n_per_class}$)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig


def plot_2d_grid(
    model: GaussianSimulationModel,
    sampler: GaussianClassConditionalSampler,
    n_per_class: int = 64,
    replication_id: int = 0,
    n_show: int = 250,
) -> plt.Figure:
    """Plot all pairwise channel views with ellipses and a Linear SVM line."""
    d = model.d
    pairs = list(combinations(range(d), 2))
    ncols = min(3, len(pairs))
    nrows = int(np.ceil(len(pairs) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 4.6 * nrows), squeeze=False
    )
    flat_axes = axes.ravel()

    for ax_idx, (i, j) in enumerate(pairs):
        ax = flat_axes[ax_idx]
        X, y = _training_samples(sampler, [i, j], n_per_class, replication_id)

        bounds_x = [X[:, 0].min(), X[:, 0].max()]
        bounds_y = [X[:, 1].min(), X[:, 1].max()]

        for label in model.class_labels:
            color = _CLASS_COLORS.get(label, None)
            mask = y == label
            pts = X[mask][:n_show]
            ax.scatter(
                pts[:, 0], pts[:, 1], s=14, alpha=0.45, color=color,
                label=f"class {label}",
            )
            mean = model.mean(label)[[i, j]]
            cov = model.covariance(label)[np.ix_([i, j], [i, j])]
            ax.scatter(
                mean[0], mean[1], s=70, marker="X", color=color,
                edgecolors="black", linewidths=0.8, zorder=5,
            )
            for level, style in zip(_SIGMA_LEVELS, ("-", "--")):
                ell = gaussian_ellipse_points(mean, cov, level)
                ax.plot(ell[:, 0], ell[:, 1], color=color, ls=style, lw=1.3)
                bounds_x += [ell[:, 0].min(), ell[:, 0].max()]
                bounds_y += [ell[:, 1].min(), ell[:, 1].max()]

        svm = _fit_linear_svm(X, y)
        if svm is not None and abs(svm.coef_[0][1]) > 1e-12:
            w = svm.coef_[0]
            b = svm.intercept_[0]
            xx = np.linspace(min(bounds_x), max(bounds_x), 100)
            yy = -(w[0] * xx + b) / w[1]
            ax.plot(xx, yy, "k-", lw=1.6, label="Linear SVM")

        margin_x = 0.08 * (max(bounds_x) - min(bounds_x) + 1e-9)
        margin_y = 0.08 * (max(bounds_y) - min(bounds_y) + 1e-9)
        ax.set_xlim(min(bounds_x) - margin_x, max(bounds_x) + margin_x)
        ax.set_ylim(min(bounds_y) - margin_y, max(bounds_y) + margin_y)
        ax.set_xlabel(f"${_channel_label(i)}$")
        ax.set_ylabel(f"${_channel_label(j)}$")
        ax.set_title(
            f"({_channel_label(i)}, {_channel_label(j)})",
            fontsize=11, fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.legend(fontsize=8, framealpha=0.9)

    for extra in range(len(pairs), len(flat_axes)):
        flat_axes[extra].set_visible(False)

    fig.suptitle(
        "2-D pairwise geometry (model ellipses + Linear SVM separators)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _ellipsoid_surface(mean: np.ndarray, cov: np.ndarray, n_std: float, res: int = 24):
    """Return ``(x, y, z)`` mesh arrays for an ``n_std`` Gaussian ellipsoid."""
    eigvals, eigvecs = np.linalg.eigh(cov)
    radii = n_std * np.sqrt(np.maximum(eigvals, 0.0))
    u = np.linspace(0.0, 2.0 * np.pi, res)
    v = np.linspace(0.0, np.pi, res)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    pts = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    rotated = pts @ eigvecs.T + mean
    shape = (res, res)
    return (
        rotated[:, 0].reshape(shape),
        rotated[:, 1].reshape(shape),
        rotated[:, 2].reshape(shape),
    )


def plot_3d_grid(
    model: GaussianSimulationModel,
    sampler: GaussianClassConditionalSampler,
    n_per_class: int = 64,
    replication_id: int = 0,
    n_show: int = 200,
) -> Optional[plt.Figure]:
    """Plot all triple channel views with ellipsoids and an SVM hyperplane.

    Returns ``None`` (graceful degradation) when the model has fewer than three
    channels.
    """
    d = model.d
    if d < 3:
        return None

    triples = list(combinations(range(d), 3))
    ncols = min(2, len(triples))
    nrows = int(np.ceil(len(triples) / ncols))
    fig = plt.figure(figsize=(6.5 * ncols, 5.5 * nrows))

    for idx, combo in enumerate(triples):
        ax = fig.add_subplot(nrows, ncols, idx + 1, projection="3d")
        X, y = _training_samples(sampler, list(combo), n_per_class, replication_id)

        for label in model.class_labels:
            color = _CLASS_COLORS.get(label, None)
            mask = y == label
            pts = X[mask][:n_show]
            ax.scatter(
                pts[:, 0], pts[:, 1], pts[:, 2], s=10, alpha=0.4, color=color,
                label=f"class {label}",
            )
            mean = model.mean(label)[list(combo)]
            cov = model.covariance(label)[np.ix_(combo, combo)]
            ax.scatter(
                mean[0], mean[1], mean[2], s=80, marker="X", color=color,
                edgecolors="black", linewidths=0.8,
            )
            ex, ey, ez = _ellipsoid_surface(mean, cov, 1.0)
            ax.plot_surface(
                ex, ey, ez, rstride=2, cstride=2, color=color,
                alpha=0.2, edgecolor="none",
            )

        svm = _fit_linear_svm(X, y)
        if svm is not None:
            w = svm.coef_[0]
            b = svm.intercept_[0]
            k = int(np.argmax(np.abs(w)))  # solve for the most informative axis
            if abs(w[k]) > 1e-12:
                others = [a for a in range(3) if a != k]
                lo = X.min(axis=0)
                hi = X.max(axis=0)
                g0 = np.linspace(lo[others[0]], hi[others[0]], 12)
                g1 = np.linspace(lo[others[1]], hi[others[1]], 12)
                m0, m1 = np.meshgrid(g0, g1)
                mk = -(b + w[others[0]] * m0 + w[others[1]] * m1) / w[k]
                surf = [None, None, None]
                surf[others[0]] = m0
                surf[others[1]] = m1
                surf[k] = mk
                ax.plot_surface(
                    surf[0], surf[1], surf[2], color="gray", alpha=0.2,
                )

        ax.set_xlabel(f"${_channel_label(combo[0])}$")
        ax.set_ylabel(f"${_channel_label(combo[1])}$")
        ax.set_zlabel(f"${_channel_label(combo[2])}$")
        ax.set_title(
            f"({_channel_label(combo[0])}, {_channel_label(combo[1])}, "
            f"{_channel_label(combo[2])})",
            fontsize=11, fontweight="bold",
        )
        ax.view_init(elev=22, azim=-50)
        if idx == 0:
            ax.legend(fontsize=8)

    fig.suptitle(
        "3-D triple-wise geometry (model ellipsoids + Linear SVM hyperplane)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def _fig_to_pil(fig, dpi: int = 90) -> Image.Image:
    """Render a Matplotlib figure to an RGB PIL image and close the figure."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _subset_n_star(accumulator, classifier_name, subsets, sample_sizes):
    """Return ``(n_star_grid, n_star_interp, baseline_subset, full_subset)``.

    Headline cooperative comparison: the full channel subset versus the best
    single channel for the given classifier (evaluated at the largest ``n``).
    """
    full = tuple(sorted(max(subsets, key=len)))
    singles = [tuple(s) for s in subsets if len(s) == 1]
    baseline = best_subset(
        accumulator, classifier_name, max(sample_sizes), singles
    ).subset
    thr = cooperative_threshold(
        accumulator, classifier_name, baseline, full, sample_sizes
    )
    return thr.n_star_grid, thr.n_star_interp, baseline, full


def generate_growth_gif(
    model: GaussianSimulationModel,
    sampler: GaussianClassConditionalSampler,
    result,
    output_path,
    replication_id: int = 0,
    geometry_pair: Optional[Sequence[int]] = None,
    frame_duration_ms: int = 1100,
    n_show: int = 220,
) -> "object":
    """Generate the sample-growth animation GIF and return its path.

    Each frame corresponds to one configured ``n_per_class``. The left panel
    shows a 2-D geometric view of a single training replication
    (``replication_id``) growing with ``n``, including the model Gaussian
    ellipses and the Linear SVM separating line fit on the data seen so far.
    The three right-hand panels show the empirical test-loss curves for the
    three evaluated classifiers, drawn progressively up to the current ``n``.
    A red vertical marker appears in each classifier panel once its cooperative
    threshold ``N*`` (full subset vs best single channel) has been reached.

    The animation is deterministic: training data is prefix-nested from a fixed
    replication, so each frame extends the previous one.
    """
    from pathlib import Path

    from coinfosim.classifiers.registry import classifier_label

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    accumulator = result.accumulator
    sample_sizes = list(result.sample_sizes)
    subsets = [tuple(s) for s in result.subsets]
    classifier_names = list(result.classifier_names)

    if geometry_pair is None:
        geometry_pair = (0, 1) if model.d >= 2 else (0, 0)
    gi, gj = int(geometry_pair[0]), int(geometry_pair[1])

    # Fixed geometry axis limits from the 2-sigma model ellipses (stable frames).
    geo_x, geo_y = [], []
    for label in model.class_labels:
        mean = model.mean(label)[[gi, gj]]
        cov = model.covariance(label)[np.ix_([gi, gj], [gi, gj])]
        ell = gaussian_ellipse_points(mean, cov, max(_SIGMA_LEVELS))
        geo_x += [ell[:, 0].min(), ell[:, 0].max()]
        geo_y += [ell[:, 1].min(), ell[:, 1].max()]
    gx_lo, gx_hi = min(geo_x), max(geo_x)
    gy_lo, gy_hi = min(geo_y), max(geo_y)
    gx_pad = 0.15 * (gx_hi - gx_lo + 1e-9)
    gy_pad = 0.15 * (gy_hi - gy_lo + 1e-9)
    geo_xlim = (gx_lo - gx_pad, gx_hi + gx_pad)
    geo_ylim = (gy_lo - gy_pad, gy_hi + gy_pad)

    # Fixed loss-curve y-limits across all cells (so curves don't jump).
    all_means = [
        accumulator.mean_loss(n, s, clf)
        for n in sample_sizes
        for s in subsets
        for clf in classifier_names
    ]
    y_lo = min(all_means)
    y_hi = max(all_means)
    y_pad = 0.08 * (y_hi - y_lo + 1e-9)
    loss_ylim = (max(0.0, y_lo - y_pad), y_hi + y_pad)

    # Cooperative thresholds per classifier (full subset vs best single channel).
    n_star_info = {
        clf: _subset_n_star(accumulator, clf, subsets, sample_sizes)
        for clf in classifier_names
    }

    frames = []
    for n_current in sample_sizes:
        fig = plt.figure(figsize=(13.5, 7.5))
        gs = fig.add_gridspec(
            len(classifier_names), 2, width_ratios=[1.15, 1.0]
        )
        ax_geo = fig.add_subplot(gs[:, 0])

        # --- Geometry panel ---------------------------------------------------
        X, y = _training_samples(
            sampler, [gi, gj], n_current, replication_id
        )
        for label in model.class_labels:
            color = _CLASS_COLORS.get(label)
            mask = y == label
            pts = X[mask][:n_show]
            ax_geo.scatter(
                pts[:, 0], pts[:, 1], s=16, alpha=0.5, color=color,
                label=f"class {label}",
            )
            mean = model.mean(label)[[gi, gj]]
            cov = model.covariance(label)[np.ix_([gi, gj], [gi, gj])]
            ax_geo.scatter(
                mean[0], mean[1], s=80, marker="X", color=color,
                edgecolors="black", linewidths=0.8, zorder=5,
            )
            for level, style in zip(_SIGMA_LEVELS, ("-", "--")):
                ell = gaussian_ellipse_points(mean, cov, level)
                ax_geo.plot(ell[:, 0], ell[:, 1], color=color, ls=style, lw=1.3)

        svm = _fit_linear_svm(X, y)
        if svm is not None and abs(svm.coef_[0][1]) > 1e-12:
            w = svm.coef_[0]
            b = svm.intercept_[0]
            xx = np.linspace(geo_xlim[0], geo_xlim[1], 100)
            yy = -(w[0] * xx + b) / w[1]
            ax_geo.plot(xx, yy, "k-", lw=1.6, label="Linear SVM")

        ax_geo.set_xlim(*geo_xlim)
        ax_geo.set_ylim(*geo_ylim)
        ax_geo.set_xlabel(f"${_channel_label(gi)}$")
        ax_geo.set_ylabel(f"${_channel_label(gj)}$")
        ax_geo.set_title(
            f"Training data ({_channel_label(gi)}, {_channel_label(gj)}), "
            f"$n_{{\\mathrm{{per\\ class}}}} = {n_current}$",
            fontsize=11, fontweight="bold",
        )
        ax_geo.grid(True, alpha=0.3)
        ax_geo.legend(fontsize=8, loc="upper left", framealpha=0.9)

        # --- Loss-curve panels (one per classifier) ---------------------------
        partial_sizes = [n for n in sample_sizes if n <= n_current]
        for row, clf in enumerate(classifier_names):
            ax = fig.add_subplot(gs[row, 1])
            for color, subset in zip(_SUBSET_COLORS, subsets):
                means = [
                    accumulator.mean_loss(n, subset, clf) for n in partial_sizes
                ]
                ax.plot(
                    partial_sizes, means, marker="o", markersize=3,
                    linewidth=1.4, color=color, label=subset_label(subset),
                )
            grid_star, interp_star, _, _ = n_star_info[clf]
            marker_n = interp_star if interp_star is not None else grid_star
            if grid_star is not None and n_current >= grid_star:
                ax.axvline(
                    marker_n, color="red", lw=1.6, ls="-", alpha=0.85,
                )
                ax.annotate(
                    f"N* = {marker_n:.1f}" if interp_star is not None
                    else f"N* = {grid_star}",
                    xy=(marker_n, loss_ylim[1]),
                    xytext=(-3, -10), textcoords="offset points",
                    ha="right", va="top",
                    color="red", fontsize=8, fontweight="bold",
                )
            ax.set_xscale("log", base=2)
            ax.set_xlim(min(sample_sizes), max(sample_sizes))
            ax.set_xticks(sample_sizes)
            ax.set_xticklabels([str(n) for n in sample_sizes], fontsize=7)
            ax.set_ylim(*loss_ylim)
            ax.set_ylabel(r"$\hat{L}$", fontsize=9)
            ax.set_title(classifier_label(clf), fontsize=10, fontweight="bold")
            ax.grid(True, which="both", alpha=0.3)
            if row == 0:
                ax.legend(fontsize=6, ncol=2, framealpha=0.9, loc="upper right")
            if row == len(classifier_names) - 1:
                ax.set_xlabel(r"$n_{\mathrm{per\ class}}$", fontsize=9)

        fig.suptitle(
            "Sample-growth diagnostics — geometry and empirical test loss "
            f"(replication {replication_id})",
            fontsize=13,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        frames.append(_fig_to_pil(fig))

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
        disposal=2,
    )
    return output_path

