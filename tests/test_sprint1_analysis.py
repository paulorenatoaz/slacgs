"""Tests for Sprint 1 best-subset and cooperative-threshold analysis (Checkpoint 5)."""

import pandas as pd
import pytest

from coinfosim.results.accumulator import LossAccumulator
from coinfosim.results.analysis import (
    best_subset,
    best_subset_rankings,
    cooperative_threshold,
    standard_threshold_comparisons,
)
from coinfosim.simulation.subsets import all_nonempty_subsets


def _acc_from_table(table):
    """Build an accumulator from a {(n, subset, clf): mean_loss} table.

    Each cell gets two identical replications so the mean equals the value.
    """
    acc = LossAccumulator()
    for (n, subset, clf), value in table.items():
        acc.add(n, subset, clf, 0, value)
        acc.add(n, subset, clf, 1, value)
    return acc


def test_best_subset_known_table():
    table = {
        (8, (0,), "f"): 0.30,
        (8, (1,), "f"): 0.25,
        (8, (0, 1), "f"): 0.10,
    }
    acc = _acc_from_table(table)
    entry = best_subset(acc, "f", 8, [(0,), (1,), (0, 1)])
    assert entry.subset == (0, 1)
    assert entry.mean_loss == 0.10


def test_best_subset_rankings_dataframe():
    clf = "gaussian_nb"
    table = {
        (4, (0,), clf): 0.40,
        (4, (0, 1), clf): 0.20,
        (8, (0,), clf): 0.35,
        (8, (0, 1), clf): 0.15,
    }
    acc = _acc_from_table(table)
    df = best_subset_rankings(acc, [clf], [4, 8], [(0,), (0, 1)])
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    row8 = df[df["n_per_class"] == 8].iloc[0]
    assert row8["best_subset"] == (0, 1)
    assert row8["best_subset_label"] == "X1+X2"


def test_cooperative_threshold_found():
    # B = (0, 2) beats A = (0,) starting at n = 8.
    table = {
        (2, (0,), "f"): 0.30,
        (2, (0, 2), "f"): 0.35,  # B worse
        (4, (0,), "f"): 0.30,
        (4, (0, 2), "f"): 0.31,  # B still worse
        (8, (0,), "f"): 0.30,
        (8, (0, 2), "f"): 0.20,  # B now better -> N* = 8
        (16, (0,), "f"): 0.30,
        (16, (0, 2), "f"): 0.18,
    }
    acc = _acc_from_table(table)
    result = cooperative_threshold(acc, "f", (0,), (0, 2), [2, 4, 8, 16])
    assert result.n_star_grid == 8


def test_cooperative_threshold_interp_between_points():
    # Delta(n) = L_A - L_B. At n=4 Delta = -0.01, at n=8 Delta = +0.10.
    # Crossing at Delta=0: n = 4 + 0.01 * (8 - 4) / (0.10 - (-0.01))
    #                        = 4 + 0.01 * 4 / 0.11 = 4.3636...
    table = {
        (4, (0,), "f"): 0.30,
        (4, (0, 2), "f"): 0.31,  # Delta = -0.01
        (8, (0,), "f"): 0.30,
        (8, (0, 2), "f"): 0.20,  # Delta = +0.10
    }
    acc = _acc_from_table(table)
    result = cooperative_threshold(acc, "f", (0,), (0, 2), [4, 8])
    assert result.n_star_grid == 8
    expected = 4 + 0.01 * 4 / 0.11
    assert result.n_star_interp == pytest.approx(expected)


def test_cooperative_threshold_first_n():
    table = {
        (2, (0,), "f"): 0.50,
        (2, (0, 2), "f"): 0.40,  # B better immediately
        (4, (0,), "f"): 0.50,
        (4, (0, 2), "f"): 0.40,
    }
    acc = _acc_from_table(table)
    result = cooperative_threshold(acc, "f", (0,), (0, 2), [2, 4])
    # First evaluated point already favours B: grid == interp == first n.
    assert result.n_star_grid == 2
    assert result.n_star_interp == 2.0


def test_cooperative_threshold_not_found():
    # B never beats A.
    table = {
        (2, (0,), "f"): 0.20,
        (2, (0, 2), "f"): 0.30,
        (4, (0,), "f"): 0.20,
        (4, (0, 2), "f"): 0.25,
    }
    acc = _acc_from_table(table)
    result = cooperative_threshold(acc, "f", (0,), (0, 2), [2, 4])
    assert result.n_star_grid is None
    assert result.n_star_interp is None


def test_standard_threshold_comparisons():
    subsets = all_nonempty_subsets(3)
    sample_sizes = [2, 4, 8]
    # Construct a table where the full set eventually wins.
    table = {}
    for n in sample_sizes:
        for s in subsets:
            # Larger subsets get lower loss at larger n.
            base = 0.5 - 0.02 * len(s) - 0.001 * n * len(s)
            table[(n, s, "gaussian_nb")] = base
    acc = _acc_from_table(table)
    df = standard_threshold_comparisons(acc, ["gaussian_nb"], sample_sizes, subsets)
    assert isinstance(df, pd.DataFrame)
    # Four standard comparisons per classifier.
    assert len(df) == 4
    assert set(df["comparison"]) == {
        "best pair vs best single",
        "full subset vs best pair",
        "X1+X3 vs X1",
        "X1+X2+X3 vs X1+X2",
    }
    # Both grid and interpolated thresholds are present.
    assert "n_star_grid" in df.columns
    assert "n_star_interp" in df.columns
    for val in df["n_star_grid"]:
        assert val is None or isinstance(val, int)
    for val in df["n_star_interp"]:
        assert val is None or isinstance(val, float)
