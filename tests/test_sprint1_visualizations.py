"""Tests for Sprint 1 geometric visualization helpers (report upgrade CP4)."""

from itertools import combinations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.reports.visualizations import (
    gaussian_ellipse_points,
    plot_1d_grid,
    plot_2d_grid,
    plot_3d_grid,
)


def _model(d: int) -> GaussianSimulationModel:
    mu1 = np.full(d, 0.6)
    mu0 = -mu1
    cov = np.eye(d) + 0.1 * (np.ones((d, d)) - np.eye(d))
    return GaussianSimulationModel(
        means={0: mu0, 1: mu1},
        covariances={0: cov, 1: cov},
    )


def _sampler(model: GaussianSimulationModel) -> GaussianClassConditionalSampler:
    return GaussianClassConditionalSampler(
        model, base_seed=0, test_samples_per_class=50
    )


def test_gaussian_ellipse_points_unit_circle():
    # Identity covariance, 1-sigma ellipse is the unit circle centered at mean.
    mean = np.array([2.0, -1.0])
    cov = np.eye(2)
    pts = gaussian_ellipse_points(mean, cov, n_std=1.0, num_points=64)
    assert pts.shape == (64, 2)
    radii = np.linalg.norm(pts - mean, axis=1)
    assert np.allclose(radii, 1.0, atol=1e-6)


def test_gaussian_ellipse_points_scaling():
    mean = np.zeros(2)
    cov = np.diag([4.0, 1.0])  # sigma_x = 2, sigma_y = 1
    pts = gaussian_ellipse_points(mean, cov, n_std=1.0, num_points=200)
    # Extent along each axis equals the standard deviation (within the
    # discretization error of sampling the ellipse at 200 angles).
    assert pytest.approx(pts[:, 0].max(), abs=1e-3) == 2.0
    assert pytest.approx(pts[:, 1].max(), abs=1e-3) == 1.0


def test_plot_1d_grid_returns_figure():
    model = _model(3)
    fig = plot_1d_grid(model, _sampler(model), n_per_class=16)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3
    plt.close(fig)


def test_plot_2d_grid_returns_figure_d3():
    model = _model(3)
    fig = plot_2d_grid(model, _sampler(model), n_per_class=16)
    assert isinstance(fig, plt.Figure)
    # One visible axis per channel pair (3 choose 2 = 3).
    visible = [ax for ax in fig.axes if ax.get_visible()]
    assert len(visible) == len(list(combinations(range(3), 2)))
    plt.close(fig)


def test_plot_3d_grid_returns_figure_d3():
    model = _model(3)
    fig = plot_3d_grid(model, _sampler(model), n_per_class=16)
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == len(list(combinations(range(3), 3)))
    plt.close(fig)


def test_plot_3d_grid_degrades_for_d2():
    model = _model(2)
    fig = plot_3d_grid(model, _sampler(model), n_per_class=16)
    assert fig is None


def test_grids_work_for_d4():
    model = _model(4)
    sampler = _sampler(model)

    fig1 = plot_1d_grid(model, sampler, n_per_class=16)
    assert len(fig1.axes) == 4
    plt.close(fig1)

    fig2 = plot_2d_grid(model, sampler, n_per_class=16)
    visible2 = [ax for ax in fig2.axes if ax.get_visible()]
    assert len(visible2) == len(list(combinations(range(4), 2)))  # 6 pairs
    plt.close(fig2)

    fig3 = plot_3d_grid(model, sampler, n_per_class=16)
    assert len(fig3.axes) == len(list(combinations(range(4), 3)))  # 4 triples
    plt.close(fig3)
