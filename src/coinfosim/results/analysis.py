"""
Post-processing analysis for CoInfoSim Sprint 1.

Two summaries are computed from a
:class:`~coinfosim.results.accumulator.LossAccumulator`:

1. **Best subset** for each classifier and sample size::

       A*_f(n) = argmin_A  Lbar_{A, f}(n)

2. **Cooperative advantage threshold** ``N*`` between two subsets ``A`` and
   ``B`` for a classifier ``f``::

       N*(A, B; f) = min { n : Lbar_{B, f}(n) < Lbar_{A, f}(n) }

   i.e. the smallest sample size at which subset ``B`` strictly beats subset
   ``A``. If no such sample size exists, ``N*`` is undefined (``None``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

from coinfosim.results.accumulator import LossAccumulator
from coinfosim.simulation.subsets import subset_label
from coinfosim.classifiers.registry import classifier_label

Subset = Tuple[int, ...]


@dataclass(frozen=True)
class BestSubsetEntry:
    """Best subset for a (classifier, n_per_class) pair."""

    classifier: str
    n_per_class: int
    subset: Subset
    mean_loss: float


@dataclass(frozen=True)
class ThresholdResult:
    """Cooperative advantage threshold for ``B`` over ``A`` under classifier ``f``."""

    classifier: str
    subset_a: Subset
    subset_b: Subset
    n_star: Optional[int]  # None when no threshold is observed


def best_subset(
    accumulator: LossAccumulator,
    classifier_name: str,
    n_per_class: int,
    subsets: Sequence[Subset],
) -> BestSubsetEntry:
    """Return the subset with minimum mean test loss for a classifier and ``n``."""
    best: Optional[BestSubsetEntry] = None
    for subset in subsets:
        subset_t = tuple(subset)
        loss = accumulator.mean_loss(n_per_class, subset_t, classifier_name)
        if best is None or loss < best.mean_loss:
            best = BestSubsetEntry(
                classifier=classifier_name,
                n_per_class=n_per_class,
                subset=subset_t,
                mean_loss=loss,
            )
    assert best is not None  # subsets is non-empty by contract
    return best


def best_subset_rankings(
    accumulator: LossAccumulator,
    classifier_names: Sequence[str],
    sample_sizes: Sequence[int],
    subsets: Sequence[Subset],
) -> pd.DataFrame:
    """Return a DataFrame of best subsets per (classifier, n_per_class).

    Columns: ``classifier``, ``classifier_label``, ``n_per_class``,
    ``best_subset``, ``best_subset_label``, ``mean_loss``.
    """
    rows: List[dict] = []
    for clf in classifier_names:
        for n in sample_sizes:
            entry = best_subset(accumulator, clf, n, subsets)
            rows.append(
                {
                    "classifier": clf,
                    "classifier_label": classifier_label(clf),
                    "n_per_class": int(n),
                    "best_subset": entry.subset,
                    "best_subset_label": subset_label(entry.subset),
                    "mean_loss": entry.mean_loss,
                }
            )
    return pd.DataFrame(rows)


def cooperative_threshold(
    accumulator: LossAccumulator,
    classifier_name: str,
    subset_a: Subset,
    subset_b: Subset,
    sample_sizes: Sequence[int],
) -> ThresholdResult:
    """Return ``N*(A, B; f)`` = smallest ``n`` where ``B`` strictly beats ``A``.

    Sample sizes are evaluated in ascending order. ``n_star`` is ``None`` if
    ``B`` never strictly beats ``A`` over the provided sample sizes.
    """
    a = tuple(subset_a)
    b = tuple(subset_b)
    n_star: Optional[int] = None
    for n in sorted(sample_sizes):
        loss_a = accumulator.mean_loss(n, a, classifier_name)
        loss_b = accumulator.mean_loss(n, b, classifier_name)
        if loss_b < loss_a:
            n_star = int(n)
            break
    return ThresholdResult(
        classifier=classifier_name,
        subset_a=a,
        subset_b=b,
        n_star=n_star,
    )


def _best_single_subset(
    accumulator: LossAccumulator,
    classifier_name: str,
    sample_sizes: Sequence[int],
    subsets: Sequence[Subset],
) -> Subset:
    """Return the single-channel subset with lowest mean loss at the largest n."""
    singles = [s for s in subsets if len(s) == 1]
    n = max(sample_sizes)
    return best_subset(accumulator, classifier_name, n, singles).subset


def _best_pair_subset(
    accumulator: LossAccumulator,
    classifier_name: str,
    sample_sizes: Sequence[int],
    subsets: Sequence[Subset],
) -> Subset:
    """Return the two-channel subset with lowest mean loss at the largest n."""
    pairs = [s for s in subsets if len(s) == 2]
    n = max(sample_sizes)
    return best_subset(accumulator, classifier_name, n, pairs).subset


def standard_threshold_comparisons(
    accumulator: LossAccumulator,
    classifier_names: Sequence[str],
    sample_sizes: Sequence[int],
    subsets: Sequence[Subset],
) -> pd.DataFrame:
    """Compute the Sprint 1 standard cooperative-threshold comparisons.

    For each classifier, compute ``N*`` for:

    - best pair vs best single;
    - full subset vs best pair;
    - ``X1+X3`` vs ``X1``  (i.e. ``(0, 2)`` vs ``(0,)``);
    - ``X1+X2+X3`` vs ``X1+X2``  (i.e. ``(0, 1, 2)`` vs ``(0, 1)``).

    Returns a DataFrame with one row per (classifier, comparison).
    """
    full_subset = tuple(sorted(max(subsets, key=len)))
    rows: List[dict] = []

    for clf in classifier_names:
        best_single = _best_single_subset(accumulator, clf, sample_sizes, subsets)
        best_pair = _best_pair_subset(accumulator, clf, sample_sizes, subsets)

        comparisons = [
            ("best pair vs best single", best_single, best_pair),
            ("full subset vs best pair", best_pair, full_subset),
            ("X1+X3 vs X1", (0,), (0, 2)),
            ("X1+X2+X3 vs X1+X2", (0, 1), (0, 1, 2)),
        ]

        for name, a, b in comparisons:
            result = cooperative_threshold(accumulator, clf, a, b, sample_sizes)
            rows.append(
                {
                    "classifier": clf,
                    "classifier_label": classifier_label(clf),
                    "comparison": name,
                    "subset_a": a,
                    "subset_a_label": subset_label(a),
                    "subset_b": b,
                    "subset_b_label": subset_label(b),
                    "n_star": result.n_star,
                }
            )
    return pd.DataFrame(rows)
