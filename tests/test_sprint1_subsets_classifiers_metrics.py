"""Tests for Sprint 1 subsets, classifier registry, and metrics (Checkpoint 2)."""

import numpy as np
import pytest
from sklearn.base import is_classifier

from coinfosim.simulation.subsets import (
    all_nonempty_subsets,
    subset_label,
    subset_labels,
)
from coinfosim.classifiers.registry import (
    CLASSIFIER_KEYS,
    available_classifiers,
    classifier_label,
    make_classifier,
)
from coinfosim.simulation.metrics import empirical_test_loss
from coinfosim.samplers.dataset import Dataset


# --- Subsets ------------------------------------------------------------------

def test_all_nonempty_subsets_d3():
    assert all_nonempty_subsets(3) == [
        (0,),
        (1,),
        (2,),
        (0, 1),
        (0, 2),
        (1, 2),
        (0, 1, 2),
    ]


def test_all_nonempty_subsets_count():
    assert len(all_nonempty_subsets(3)) == 7
    assert len(all_nonempty_subsets(4)) == 15


def test_all_nonempty_subsets_invalid():
    with pytest.raises(ValueError):
        all_nonempty_subsets(0)


def test_subset_labels_d3():
    subs = all_nonempty_subsets(3)
    assert subset_labels(subs) == [
        "X1",
        "X2",
        "X3",
        "X1+X2",
        "X1+X3",
        "X2+X3",
        "X1+X2+X3",
    ]


def test_subset_label_single():
    assert subset_label((2,)) == "X3"


def test_subset_label_empty_raises():
    with pytest.raises(ValueError):
        subset_label(())


# --- Classifier registry ------------------------------------------------------

def test_available_classifiers():
    assert available_classifiers() == [
        "linear_svm",
        "logistic_regression",
        "gaussian_nb",
    ]
    assert CLASSIFIER_KEYS == available_classifiers()


@pytest.mark.parametrize("key", ["linear_svm", "logistic_regression", "gaussian_nb"])
def test_make_classifier_fresh_unfitted(key):
    clf = make_classifier(key)
    assert hasattr(clf, "fit")
    assert hasattr(clf, "predict")
    assert is_classifier(clf)
    # Fresh and unfitted: a fitted attribute should not be present yet.
    assert not hasattr(clf, "classes_")


def test_make_classifier_returns_new_instances():
    a = make_classifier("gaussian_nb")
    b = make_classifier("gaussian_nb")
    assert a is not b


def test_make_classifier_unknown_raises():
    with pytest.raises(KeyError):
        make_classifier("rbf_svm")


def test_classifier_label():
    assert classifier_label("linear_svm") == "Linear SVM"
    assert classifier_label("logistic_regression") == "Logistic Regression"
    assert classifier_label("gaussian_nb") == "Gaussian Naive Bayes"


# --- Empirical test loss ------------------------------------------------------

class _DummyClassifier:
    """Predicts a constant label regardless of input."""

    def __init__(self, constant):
        self.constant = constant

    def predict(self, X):
        return np.full(X.shape[0], self.constant)


def test_empirical_test_loss_all_correct():
    X = np.zeros((10, 2))
    y = np.zeros(10)
    ds = Dataset(X, y)
    loss = empirical_test_loss(_DummyClassifier(0), ds)
    assert loss == 0.0


def test_empirical_test_loss_all_wrong():
    X = np.zeros((10, 2))
    y = np.zeros(10)
    ds = Dataset(X, y)
    loss = empirical_test_loss(_DummyClassifier(1), ds)
    assert loss == 1.0


def test_empirical_test_loss_half():
    X = np.zeros((10, 2))
    y = np.array([0] * 5 + [1] * 5)
    ds = Dataset(X, y)
    loss = empirical_test_loss(_DummyClassifier(0), ds)
    assert loss == 0.5
    assert 0.0 <= loss <= 1.0


def test_empirical_test_loss_with_real_classifier():
    rng = np.random.default_rng(0)
    X_train = np.vstack([rng.normal(-3, 0.5, (20, 1)), rng.normal(3, 0.5, (20, 1))])
    y_train = np.array([0] * 20 + [1] * 20)
    clf = make_classifier("gaussian_nb")
    clf.fit(X_train, y_train)

    X_test = np.vstack([rng.normal(-3, 0.5, (20, 1)), rng.normal(3, 0.5, (20, 1))])
    y_test = np.array([0] * 20 + [1] * 20)
    loss = empirical_test_loss(clf, Dataset(X_test, y_test))
    assert 0.0 <= loss <= 1.0
    assert loss < 0.2  # well-separated classes
