"""
Standard-error stopping rule for CoInfoSim Sprint 1.

:class:`StandardErrorStoppingRule` decides when to stop accumulating Monte
Carlo replications for a given ``n_per_class``. It is the default Sprint 1
stopping rule and replaces the legacy mean-difference rule.

Decision logic (evaluated only at replication batch boundaries):

- Do not stop before ``min_replications`` replications are completed.
- Compute the observed 95% CI half-width ``1.96 * SE`` for every
  (subset, classifier) cell.
- Stop by *convergence* when the maximum observed half-width across all
  cells is ``<= ci_half_width_target``.
- Stop by *max budget* when ``max_replications`` is reached, regardless of
  convergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from coinfosim.results.accumulator import LossAccumulator

Z_95 = 1.96


@dataclass(frozen=True)
class StoppingDecision:
    """Outcome of a stopping-rule evaluation for one ``n_per_class``."""

    should_stop: bool
    reason: Optional[str]  # "converged", "max_budget", or None if not stopping
    replications: int
    max_ci_half_width: float


class StandardErrorStoppingRule:
    """Standard-error based Monte Carlo stopping rule.

    Parameters
    ----------
    min_replications:
        Minimum replications before stopping may occur.
    max_replications:
        Replication budget cap.
    ci_half_width_target:
        Target 95% CI half-width for convergence.
    z:
        Normal quantile for the CI (default 1.96 for ~95%).
    """

    def __init__(
        self,
        min_replications: int,
        max_replications: int,
        ci_half_width_target: float,
        z: float = Z_95,
    ) -> None:
        if min_replications < 2:
            raise ValueError("min_replications must be at least 2")
        if max_replications < min_replications:
            raise ValueError("max_replications must be >= min_replications")
        if ci_half_width_target <= 0:
            raise ValueError("ci_half_width_target must be positive")

        self.min_replications = int(min_replications)
        self.max_replications = int(max_replications)
        self.ci_half_width_target = float(ci_half_width_target)
        self.z = float(z)

    def max_ci_half_width(
        self,
        accumulator: LossAccumulator,
        n_per_class: int,
        cells: Sequence[Tuple[Sequence[int], str]],
    ) -> float:
        """Return the maximum observed CI half-width across all cells.

        ``cells`` is a sequence of ``(subset, classifier_name)`` pairs.
        """
        widths: List[float] = []
        for subset, clf in cells:
            se = accumulator.standard_error(n_per_class, subset, clf)
            widths.append(self.z * se)
        return max(widths) if widths else 0.0

    def evaluate(
        self,
        accumulator: LossAccumulator,
        n_per_class: int,
        cells: Sequence[Tuple[Sequence[int], str]],
    ) -> StoppingDecision:
        """Decide whether to stop at a batch boundary for ``n_per_class``.

        ``cells`` is a sequence of ``(subset, classifier_name)`` pairs that
        must all satisfy the CI target for convergence.
        """
        replications = accumulator.replications_completed(n_per_class)
        max_width = self.max_ci_half_width(accumulator, n_per_class, cells)

        if replications < self.min_replications:
            return StoppingDecision(
                should_stop=False,
                reason=None,
                replications=replications,
                max_ci_half_width=max_width,
            )

        if max_width <= self.ci_half_width_target:
            return StoppingDecision(
                should_stop=True,
                reason="converged",
                replications=replications,
                max_ci_half_width=max_width,
            )

        if replications >= self.max_replications:
            return StoppingDecision(
                should_stop=True,
                reason="max_budget",
                replications=replications,
                max_ci_half_width=max_width,
            )

        return StoppingDecision(
            should_stop=False,
            reason=None,
            replications=replications,
            max_ci_half_width=max_width,
        )
