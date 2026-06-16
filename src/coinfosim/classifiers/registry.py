"""
Classifier registry for CoInfoSim Sprint 1.

Provides factory functions that return *fresh, unfitted* scikit-learn
estimators for the three Sprint 1 classifiers:

- ``linear_svm``           -> Linear SVM (``SVC(kernel="linear")``)
- ``logistic_regression``  -> Logistic Regression
- ``gaussian_nb``          -> Gaussian Naive Bayes

No hyperparameter search and no probabilistic/RBF/ensemble models are
included in Sprint 1.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# A deterministic random_state for estimators that accept one.
_RANDOM_STATE = 0


def _make_linear_svm():
    return SVC(kernel="linear", random_state=_RANDOM_STATE)


def _make_logistic_regression():
    return LogisticRegression(max_iter=1000, random_state=_RANDOM_STATE)


def _make_gaussian_nb():
    return GaussianNB()


# Registry of factory functions. Each call returns a fresh unfitted estimator.
_FACTORIES: Dict[str, Callable[[], object]] = {
    "linear_svm": _make_linear_svm,
    "logistic_regression": _make_logistic_regression,
    "gaussian_nb": _make_gaussian_nb,
}

# Human-readable display labels for reports.
DISPLAY_LABELS: Dict[str, str] = {
    "linear_svm": "Linear SVM",
    "logistic_regression": "Logistic Regression",
    "gaussian_nb": "Gaussian Naive Bayes",
}

# Stable ordering of classifier keys for Sprint 1.
CLASSIFIER_KEYS: List[str] = ["linear_svm", "logistic_regression", "gaussian_nb"]


def available_classifiers() -> List[str]:
    """Return the registered classifier keys in a stable order."""
    return list(CLASSIFIER_KEYS)


def make_classifier(key: str):
    """Return a fresh, unfitted estimator for ``key``.

    Raises
    ------
    KeyError
        If ``key`` is not a registered classifier.
    """
    if key not in _FACTORIES:
        raise KeyError(
            f"unknown classifier {key!r}; available: {available_classifiers()}"
        )
    return _FACTORIES[key]()


def classifier_label(key: str) -> str:
    """Return the display label for a classifier ``key``."""
    if key not in DISPLAY_LABELS:
        raise KeyError(
            f"unknown classifier {key!r}; available: {available_classifiers()}"
        )
    return DISPLAY_LABELS[key]
