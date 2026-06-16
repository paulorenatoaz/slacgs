"""
Gaussian simulation model for CoInfoSim Sprint 1.

This module defines :class:`GaussianSimulationModel`, the new explicit
class-conditional Gaussian model used by the Sprint 1 CooperativeMonteCarlo
simulator. It is built in parallel to the legacy ``core.Model`` and does
*not* use the legacy sigma/rho parameter vector, ``LossType``, or
``channel_names``.

The model is initialized explicitly from per-class mean vectors and
covariance matrices::

    GaussianSimulationModel(
        means={0: mu0, 1: mu1},
        covariances={0: Sigma0, 1: Sigma1},
    )

Channels (features) are referenced internally by zero-based indices.
"""

from __future__ import annotations

from typing import Dict, Mapping, Sequence, Tuple

import numpy as np


class GaussianSimulationModel:
    """Explicit class-conditional multivariate Gaussian simulation model.

    Parameters
    ----------
    means:
        Mapping from class label to a mean vector of length ``d``.
    covariances:
        Mapping from class label to a ``(d, d)`` covariance matrix.

    The number of channels ``d`` and the set of class labels ``K`` are
    inferred from the inputs. All mean vectors and covariance matrices are
    stored internally as NumPy arrays of dtype ``float``.

    Raises
    ------
    ValueError
        If the model definition is inconsistent (missing means/covariances,
        wrong shapes, non-symmetric or non-positive-definite covariances).
    """

    def __init__(
        self,
        means: Mapping[int, Sequence[float]],
        covariances: Mapping[int, Sequence[Sequence[float]]],
    ) -> None:
        if not isinstance(means, Mapping) or not isinstance(covariances, Mapping):
            raise ValueError("means and covariances must be mappings keyed by class label")

        if len(means) == 0:
            raise ValueError("means must define at least one class")

        mean_labels = set(means.keys())
        cov_labels = set(covariances.keys())
        if mean_labels != cov_labels:
            raise ValueError(
                f"means and covariances must define the same class labels; "
                f"got means={sorted(mean_labels)} covariances={sorted(cov_labels)}"
            )

        if len(mean_labels) < 2:
            raise ValueError("Sprint 1 requires at least two classes")

        class_labels = sorted(mean_labels)

        # Infer d from the first mean vector.
        first_mean = np.asarray(means[class_labels[0]], dtype=float)
        if first_mean.ndim != 1 or first_mean.shape[0] == 0:
            raise ValueError("each mean must be a non-empty 1-D vector")
        d = int(first_mean.shape[0])

        stored_means: Dict[int, np.ndarray] = {}
        stored_covs: Dict[int, np.ndarray] = {}

        for label in class_labels:
            mean = np.asarray(means[label], dtype=float)
            cov = np.asarray(covariances[label], dtype=float)

            if mean.ndim != 1:
                raise ValueError(f"mean for class {label} must be 1-D, got shape {mean.shape}")
            if mean.shape[0] != d:
                raise ValueError(
                    f"mean for class {label} has length {mean.shape[0]}, expected {d}"
                )

            if cov.ndim != 2 or cov.shape != (d, d):
                raise ValueError(
                    f"covariance for class {label} must have shape ({d}, {d}), "
                    f"got {cov.shape}"
                )

            if not np.allclose(cov, cov.T, atol=1e-10):
                raise ValueError(f"covariance for class {label} is not symmetric")

            self._validate_positive_definite(cov, label)

            stored_means[label] = mean
            stored_covs[label] = cov

        self._class_labels: Tuple[int, ...] = tuple(class_labels)
        self._d = d
        self._means = stored_means
        self._covariances = stored_covs

    @staticmethod
    def _validate_positive_definite(cov: np.ndarray, label: int) -> None:
        """Validate that ``cov`` is positive definite via Cholesky."""
        try:
            np.linalg.cholesky(cov)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                f"covariance for class {label} is not positive definite"
            ) from exc

    @property
    def d(self) -> int:
        """Number of channels (features)."""
        return self._d

    @property
    def num_channels(self) -> int:
        """Alias for :attr:`d`."""
        return self._d

    @property
    def K(self) -> int:
        """Number of classes."""
        return len(self._class_labels)

    @property
    def num_classes(self) -> int:
        """Alias for :attr:`K`."""
        return len(self._class_labels)

    @property
    def class_labels(self) -> Tuple[int, ...]:
        """Sorted tuple of class labels."""
        return self._class_labels

    def mean(self, label: int) -> np.ndarray:
        """Return a copy of the mean vector for ``label``."""
        return self._means[label].copy()

    def covariance(self, label: int) -> np.ndarray:
        """Return a copy of the covariance matrix for ``label``."""
        return self._covariances[label].copy()

    def restrict_to_subset(
        self, subset: Sequence[int]
    ) -> "GaussianSimulationModel":
        """Return a new model restricted to the given channel ``subset``.

        Parameters
        ----------
        subset:
            Sequence of zero-based channel indices (non-empty, in range).
        """
        idx = self._validate_subset(subset)
        restricted_means = {
            label: self._means[label][idx] for label in self._class_labels
        }
        restricted_covs = {
            label: self._covariances[label][np.ix_(idx, idx)]
            for label in self._class_labels
        }
        return GaussianSimulationModel(restricted_means, restricted_covs)

    def _validate_subset(self, subset: Sequence[int]) -> np.ndarray:
        idx = np.asarray(list(subset), dtype=int)
        if idx.size == 0:
            raise ValueError("subset must be non-empty")
        if np.any(idx < 0) or np.any(idx >= self._d):
            raise ValueError(
                f"subset indices must be in range [0, {self._d}); got {tuple(subset)}"
            )
        if len(set(idx.tolist())) != idx.size:
            raise ValueError(f"subset must not contain duplicate channels; got {tuple(subset)}")
        return idx

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"GaussianSimulationModel(d={self._d}, "
            f"class_labels={self._class_labels})"
        )
