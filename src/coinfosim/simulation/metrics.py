"""
Empirical test-loss metric for CoInfoSim Sprint 1.

Sprint 1 uses a single performance metric: the empirical test loss, defined
as the misclassification rate on a fixed test set. Empirical train loss,
theoretical loss, and Bayes error are intentionally *not* implemented.
"""

from __future__ import annotations

import numpy as np

from coinfosim.samplers.dataset import Dataset


def empirical_test_loss(estimator, test_dataset: Dataset) -> float:
    """Return the misclassification rate of ``estimator`` on ``test_dataset``.

    Parameters
    ----------
    estimator:
        A fitted scikit-learn-style estimator exposing ``predict``.
    test_dataset:
        The test :class:`Dataset` (already restricted to the relevant
        channels if a subset is being evaluated).

    Returns
    -------
    float
        The misclassification rate ``mean(predicted_y != true_y)`` in ``[0, 1]``.
    """
    predictions = estimator.predict(test_dataset.X)
    predictions = np.asarray(predictions)
    return float(np.mean(predictions != test_dataset.y))
