"""
Monte Carlo loss accumulator for CoInfoSim Sprint 1.

:class:`LossAccumulator` stores individual replication losses indexed by
``(n_per_class, subset, classifier_name, replication_id)`` and provides
summary statistics (mean, standard deviation, standard error) and the
number of completed replications for each ``n_per_class``.

Only empirical test loss is stored. There is no notion of train loss,
theoretical loss, or Bayes error.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

# A key identifying one (n_per_class, subset, classifier) cell.
CellKey = Tuple[int, Tuple[int, ...], str]


class LossAccumulator:
    """Accumulate per-replication empirical test losses and summarize them."""

    def __init__(self) -> None:
        # Map cell -> dict(replication_id -> loss).
        self._losses: Dict[CellKey, Dict[int, float]] = {}

    @staticmethod
    def _cell(n_per_class: int, subset, classifier_name: str) -> CellKey:
        return (int(n_per_class), tuple(subset), str(classifier_name))

    def add(
        self,
        n_per_class: int,
        subset,
        classifier_name: str,
        replication_id: int,
        loss: float,
    ) -> None:
        """Record one replication loss for a (n_per_class, subset, classifier) cell."""
        cell = self._cell(n_per_class, subset, classifier_name)
        self._losses.setdefault(cell, {})[int(replication_id)] = float(loss)

    def losses(self, n_per_class: int, subset, classifier_name: str) -> np.ndarray:
        """Return the recorded losses for a cell, ordered by replication id."""
        cell = self._cell(n_per_class, subset, classifier_name)
        rep_map = self._losses.get(cell, {})
        if not rep_map:
            return np.empty(0, dtype=float)
        ordered = [rep_map[r] for r in sorted(rep_map)]
        return np.asarray(ordered, dtype=float)

    def count(self, n_per_class: int, subset, classifier_name: str) -> int:
        """Number of replications recorded for a cell."""
        cell = self._cell(n_per_class, subset, classifier_name)
        return len(self._losses.get(cell, {}))

    def mean_loss(self, n_per_class: int, subset, classifier_name: str) -> float:
        """Mean test loss for a cell."""
        values = self.losses(n_per_class, subset, classifier_name)
        if values.size == 0:
            return float("nan")
        return float(np.mean(values))

    def std_loss(self, n_per_class: int, subset, classifier_name: str) -> float:
        """Sample standard deviation (ddof=1) of test losses for a cell.

        Returns ``0.0`` when fewer than two replications are present.
        """
        values = self.losses(n_per_class, subset, classifier_name)
        if values.size < 2:
            return 0.0
        return float(np.std(values, ddof=1))

    def standard_error(self, n_per_class: int, subset, classifier_name: str) -> float:
        """Standard error of the mean test loss for a cell.

        Returns ``0.0`` when fewer than two replications are present.
        """
        values = self.losses(n_per_class, subset, classifier_name)
        n = values.size
        if n < 2:
            return 0.0
        return float(np.std(values, ddof=1) / np.sqrt(n))

    def replications_completed(self, n_per_class: int) -> int:
        """Return the common replication count across all cells for ``n_per_class``.

        In Sprint 1 every (subset, classifier) pair shares the same number of
        replications for a given ``n_per_class``. This returns the maximum
        observed count (which equals the common count by construction).
        """
        counts = [
            len(rep_map)
            for cell, rep_map in self._losses.items()
            if cell[0] == int(n_per_class)
        ]
        return max(counts) if counts else 0

    def cells_for(self, n_per_class: int) -> List[CellKey]:
        """Return all recorded cells for a given ``n_per_class``."""
        return [cell for cell in self._losses if cell[0] == int(n_per_class)]

    def sample_sizes(self) -> List[int]:
        """Return the sorted list of recorded ``n_per_class`` values."""
        return sorted({cell[0] for cell in self._losses})
