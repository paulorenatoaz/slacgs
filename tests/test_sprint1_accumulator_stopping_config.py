"""Tests for Sprint 1 accumulator, summary, stopping rule, and config (Checkpoint 3)."""

import math

import numpy as np
import pandas as pd
import pytest

from coinfosim.results.accumulator import LossAccumulator
from coinfosim.results.summary import summary_dataframe
from coinfosim.simulation.config import (
    MonteCarloConfig,
    VALID_MODES,
    get_mode_config,
)
from coinfosim.simulation.stopping import StandardErrorStoppingRule


# --- Accumulator --------------------------------------------------------------

def test_accumulator_add_and_losses():
    acc = LossAccumulator()
    subset = (0,)
    for r, loss in enumerate([0.1, 0.2, 0.3]):
        acc.add(8, subset, "linear_svm", r, loss)
    losses = acc.losses(8, subset, "linear_svm")
    assert np.allclose(losses, [0.1, 0.2, 0.3])
    assert acc.count(8, subset, "linear_svm") == 3


def test_accumulator_mean_std_se():
    acc = LossAccumulator()
    subset = (0, 1)
    values = [0.2, 0.4, 0.6]
    for r, v in enumerate(values):
        acc.add(16, subset, "gaussian_nb", r, v)
    assert math.isclose(acc.mean_loss(16, subset, "gaussian_nb"), 0.4)
    expected_std = float(np.std(values, ddof=1))
    assert math.isclose(acc.std_loss(16, subset, "gaussian_nb"), expected_std)
    expected_se = expected_std / math.sqrt(3)
    assert math.isclose(acc.standard_error(16, subset, "gaussian_nb"), expected_se)


def test_accumulator_single_value_std_se_zero():
    acc = LossAccumulator()
    acc.add(2, (0,), "linear_svm", 0, 0.5)
    assert acc.std_loss(2, (0,), "linear_svm") == 0.0
    assert acc.standard_error(2, (0,), "linear_svm") == 0.0


def test_accumulator_empty_mean_is_nan():
    acc = LossAccumulator()
    assert math.isnan(acc.mean_loss(2, (0,), "linear_svm"))


def test_accumulator_replications_completed():
    acc = LossAccumulator()
    for r in range(5):
        acc.add(4, (0,), "linear_svm", r, 0.1)
        acc.add(4, (1,), "gaussian_nb", r, 0.2)
    assert acc.replications_completed(4) == 5
    assert acc.replications_completed(8) == 0
    assert acc.sample_sizes() == [4]


# --- Summary ------------------------------------------------------------------

def test_summary_dataframe():
    acc = LossAccumulator()
    subsets = [(0,), (0, 1)]
    for r in range(3):
        acc.add(8, (0,), "linear_svm", r, 0.3)
        acc.add(8, (0, 1), "linear_svm", r, 0.1)
    df = summary_dataframe(acc, [8], subsets, ["linear_svm"])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df["subset_label"]) == {"X1", "X1+X2"}
    row = df[df["subset_label"] == "X1+X2"].iloc[0]
    assert math.isclose(row["mean_loss"], 0.1)
    assert row["replications"] == 3
    assert row["classifier_label"] == "Linear SVM"


# --- Mode config --------------------------------------------------------------

@pytest.mark.parametrize("mode", ["smoke", "report_smoke", "fast", "full"])
def test_get_mode_config_valid(mode):
    cfg = get_mode_config(mode)
    assert isinstance(cfg, MonteCarloConfig)
    assert cfg.mode == mode
    assert len(cfg.sample_sizes) > 0
    assert cfg.min_replications >= 2
    assert cfg.max_replications >= cfg.min_replications
    assert cfg.replication_batch_size > 0
    assert cfg.test_samples_per_class > 0
    assert cfg.ci_half_width_target > 0


def test_valid_modes_constant():
    assert set(VALID_MODES) == {"smoke", "report_smoke", "fast", "full"}


def test_report_smoke_extends_to_32():
    cfg = get_mode_config("report_smoke")
    assert cfg.sample_sizes == (2, 4, 8, 16, 32)
    assert max(cfg.sample_sizes) == 32


def test_get_mode_config_invalid():
    with pytest.raises(ValueError):
        get_mode_config("turbo")


def test_monte_carlo_config_validation():
    with pytest.raises(ValueError):
        MonteCarloConfig(
            mode="x",
            sample_sizes=(),
            min_replications=5,
            max_replications=10,
            replication_batch_size=5,
            test_samples_per_class=10,
            ci_half_width_target=0.05,
        )
    with pytest.raises(ValueError):
        MonteCarloConfig(
            mode="x",
            sample_sizes=(2,),
            min_replications=10,
            max_replications=5,
            replication_batch_size=5,
            test_samples_per_class=10,
            ci_half_width_target=0.05,
        )


# --- Stopping rule ------------------------------------------------------------

def _fill(acc, n, cells, n_reps, loss_value):
    for subset, clf in cells:
        for r in range(n_reps):
            acc.add(n, subset, clf, r, loss_value)


def test_stopping_before_min_replications():
    rule = StandardErrorStoppingRule(
        min_replications=10, max_replications=100, ci_half_width_target=0.05
    )
    acc = LossAccumulator()
    cells = [((0,), "linear_svm")]
    _fill(acc, 8, cells, n_reps=5, loss_value=0.3)
    decision = rule.evaluate(acc, 8, cells)
    assert decision.should_stop is False
    assert decision.reason is None
    assert decision.replications == 5


def test_stopping_converged_after_min_replications():
    rule = StandardErrorStoppingRule(
        min_replications=5, max_replications=100, ci_half_width_target=0.05
    )
    acc = LossAccumulator()
    cells = [((0,), "linear_svm")]
    # Constant losses -> SE = 0 -> half width 0 <= target -> converge.
    _fill(acc, 8, cells, n_reps=10, loss_value=0.3)
    decision = rule.evaluate(acc, 8, cells)
    assert decision.should_stop is True
    assert decision.reason == "converged"
    assert decision.max_ci_half_width == pytest.approx(0.0, abs=1e-9)


def test_stopping_not_converged_high_variance():
    rule = StandardErrorStoppingRule(
        min_replications=4, max_replications=1000, ci_half_width_target=0.001
    )
    acc = LossAccumulator()
    cells = [((0,), "linear_svm")]
    for r, v in enumerate([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]):
        acc.add(8, (0,), "linear_svm", r, v)
    decision = rule.evaluate(acc, 8, cells)
    assert decision.should_stop is False
    assert decision.max_ci_half_width > 0.001


def test_stopping_max_budget():
    rule = StandardErrorStoppingRule(
        min_replications=4, max_replications=6, ci_half_width_target=0.001
    )
    acc = LossAccumulator()
    cells = [((0,), "linear_svm")]
    for r, v in enumerate([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]):
        acc.add(8, (0,), "linear_svm", r, v)
    decision = rule.evaluate(acc, 8, cells)
    assert decision.should_stop is True
    assert decision.reason == "max_budget"
    assert decision.replications == 6


def test_stopping_max_width_across_cells():
    rule = StandardErrorStoppingRule(
        min_replications=4, max_replications=100, ci_half_width_target=0.05
    )
    acc = LossAccumulator()
    # One low-variance cell, one high-variance cell. Max width should dominate.
    low = ((0,), "linear_svm")
    high = ((1,), "gaussian_nb")
    for r in range(8):
        acc.add(8, low[0], low[1], r, 0.3)
    for r, v in enumerate([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]):
        acc.add(8, high[0], high[1], r, v)
    decision = rule.evaluate(acc, 8, [low, high])
    assert decision.should_stop is False  # high-variance cell keeps it running


def test_stopping_rule_invalid_params():
    with pytest.raises(ValueError):
        StandardErrorStoppingRule(
            min_replications=1, max_replications=10, ci_half_width_target=0.05
        )
    with pytest.raises(ValueError):
        StandardErrorStoppingRule(
            min_replications=10, max_replications=5, ci_half_width_target=0.05
        )
