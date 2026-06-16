"""
Minimal dataset container for CoInfoSim Sprint 1.

:class:`Dataset` stores a feature matrix ``X`` of shape ``(n_samples, d)``
and a target vector ``y`` of shape ``(n_samples,)``. It supports restricting
the features to a channel subset via :meth:`Dataset.select_channels`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


class Dataset:
    """Immutable-ish container for a feature matrix and target vector.

    Parameters
    ----------
    X:
        Feature matrix of shape ``(n_samples, d)``.
    y:
        Target vector of shape ``(n_samples,)``.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, ``y`` is not 1-D, or their sample counts differ.
    """

    def __init__(self, X, y) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D with shape (n_samples, d), got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be 1-D with shape (n_samples,), got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows; "
                f"got {X.shape[0]} and {y.shape[0]}"
            )

        self._X = X
        self._y = y

    @property
    def X(self) -> np.ndarray:
        """Feature matrix, shape ``(n_samples, d)``."""
        return self._X

    @property
    def y(self) -> np.ndarray:
        """Target vector, shape ``(n_samples,)``."""
        return self._y

    @property
    def n_samples(self) -> int:
        """Number of samples (rows)."""
        return self._X.shape[0]

    @property
    def d(self) -> int:
        """Number of channels (columns) currently present."""
        return self._X.shape[1]

    def select_channels(self, subset: Sequence[int]) -> "Dataset":
        """Return a new :class:`Dataset` restricted to the given channels.

        Parameters
        ----------
        subset:
            Sequence of zero-based channel indices (non-empty, in range).
        """
        idx = np.asarray(list(subset), dtype=int)
        if idx.size == 0:
            raise ValueError("subset must be non-empty")
        if np.any(idx < 0) or np.any(idx >= self._X.shape[1]):
            raise ValueError(
                f"subset indices must be in range [0, {self._X.shape[1]}); "
                f"got {tuple(subset)}"
            )
        return Dataset(self._X[:, idx], self._y)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Dataset(n_samples={self.n_samples}, d={self.d})"
