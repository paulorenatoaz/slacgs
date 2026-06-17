"""Tests for Checkpoint 2 ranking and uncertainty redesign."""

import pandas as pd

from coinfosim.results.accumulator import LossAccumulator
from coinfosim.results.analysis import (
    best_subset_by_sample_size,
    final_ranking_at_largest_n,
)
from coinfosim.results.summary import hardest_cell_summary


def _acc_from_table(table):
    """Build an accumulator from a {(n, subset, clf): mean_loss} table.

    Two identical replications per cell so the mean equals the value and the
    standard error is (numerically) zero.
    """
    acc = LossAccumulator()
    for (n, subset, clf), value in table.items():
        acc.add(n, subset, clf, 0, value)
        acc.add(n, subset, clf, 1, value)
    return acc


def test_best_subset_by_sample_size_matrix():
    clf = "linear_svm"
    table = {
        (4, (0,), clf): 0.40,
        (4, (0, 1), clf): 0.20,
        (8, (0,), clf): 0.35,
        (8, (0, 1), clf): 0.15,
    }
    acc = _acc_from_table(table)
    df = best_subset_by_sample_size(acc, [clf], [4, 8], [(0,), (0, 1)])
    assert isinstance(df, pd.DataFrame)
    assert df.index.name == "n_per_class"
    assert list(df.index) == [4, 8]
    assert "Linear SVM" in df.columns
    assert df.loc[4, "Linear SVM"] == "X1+X2"
    assert df.loc[8, "Linear SVM"] == "X1+X2"


def test_best_subset_by_sample_size_multiple_classifiers():
    table = {}
    for n in (2, 4):
        table[(n, (0,), "linear_svm")] = 0.3
        table[(n, (1,), "linear_svm")] = 0.4
        table[(n, (0,), "gaussian_nb")] = 0.5
        table[(n, (1,), "gaussian_nb")] = 0.2
    acc = _acc_from_table(table)
    df = best_subset_by_sample_size(
        acc, ["linear_svm", "gaussian_nb"], [2, 4], [(0,), (1,)]
    )
    assert df.loc[2, "Linear SVM"] == "X1"
    assert df.loc[2, "Gaussian Naive Bayes"] == "X2"


def test_final_ranking_at_largest_n():
    clf = "logistic_regression"
    subsets = [(0,), (1,), (0, 1)]
    table = {
        (4, (0,), clf): 0.5,
        (4, (1,), clf): 0.4,
        (4, (0, 1), clf): 0.3,
        (8, (0,), clf): 0.30,
        (8, (1,), clf): 0.20,
        (8, (0, 1), clf): 0.10,
    }
    acc = _acc_from_table(table)
    df = final_ranking_at_largest_n(acc, [clf], [4, 8], subsets)
    # Only the largest n (8) is ranked.
    assert set(df["n_per_class"]) == {8}
    ranked = df.sort_values("rank")
    assert list(ranked["subset_label"]) == ["X1+X2", "X2", "X1"]
    assert list(ranked["rank"]) == [1, 2, 3]
    assert ranked.iloc[0]["mean_loss"] == 0.10


def test_hardest_cell_summary():
    subsets = [(0,), (0, 1)]
    clfs = ["linear_svm"]
    acc = LossAccumulator()
    # Cell (0,) low variance, cell (0,1) high variance at n=8.
    for r in range(6):
        acc.add(8, (0,), "linear_svm", r, 0.3)
    for r, v in enumerate([0.0, 1.0, 0.0, 1.0, 0.0, 1.0]):
        acc.add(8, (0, 1), "linear_svm", r, v)
    df = hardest_cell_summary(acc, [8], subsets, clfs)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["n_per_class"] == 8
    assert row["subset_label"] == "X1+X2"  # the high-variance cell
    assert row["ci_half_width"] > 0
