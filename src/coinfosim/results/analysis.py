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
    """Cooperative advantage threshold for ``B`` over ``A`` under classifier ``f``.

    Two estimates are reported:

    - ``n_star_grid``: the smallest evaluated sample size at which ``B``
      strictly beats ``A`` (discrete grid threshold).
    - ``n_star_interp``: a linearly interpolated estimate of the crossing
      between the two consecutive sample sizes that bracket it. ``None`` when
      no crossing is observed, or equal to the first sample size when the very
      first evaluated point already favours ``B``.
    """

    classifier: str
    subset_a: Subset
    subset_b: Subset
    n_star_grid: Optional[int]
    n_star_interp: Optional[float]


def cooperative_threshold(
    accumulator: LossAccumulator,
    classifier_name: str,
    subset_a: Subset,
    subset_b: Subset,
    sample_sizes: Sequence[int],
) -> ThresholdResult:
    """Return the cooperative advantage threshold of ``B`` over ``A``.

    Define ``Delta(n) = L_A(n) - L_B(n)``; the cooperative subset ``B`` beats
    ``A`` when ``Delta(n) > 0``. Sample sizes are scanned in ascending order.

    - The grid threshold ``n_star_grid`` is the first ``n`` with ``Delta > 0``.
    - The interpolated threshold ``n_star_interp`` is estimated by linear
      interpolation between the consecutive points ``n_left`` (``Delta <= 0``)
      and ``n_right`` (``Delta > 0``) that bracket the crossing::

          n_star_interp = n_left
              + (0 - Delta(n_left)) * (n_right - n_left)
                / (Delta(n_right) - Delta(n_left))

    - If the first evaluated point already has ``Delta > 0``, the grid
      threshold is that first ``n`` and the interpolated value is set equal to
      it (no left bracket exists to interpolate from).
    - If no crossing occurs, both values are ``None``.
    """
    a = tuple(subset_a)
    b = tuple(subset_b)
    ordered = sorted(sample_sizes)

    deltas = [
        accumulator.mean_loss(n, a, classifier_name)
        - accumulator.mean_loss(n, b, classifier_name)
        for n in ordered
    ]

    n_star_grid: Optional[int] = None
    n_star_interp: Optional[float] = None

    for i, n in enumerate(ordered):
        if deltas[i] > 0:
            n_star_grid = int(n)
            if i == 0:
                # First evaluated point already favours B; no left bracket.
                n_star_interp = float(n)
            else:
                n_left = ordered[i - 1]
                n_right = n
                d_left = deltas[i - 1]
                d_right = deltas[i]
                denom = d_right - d_left
                if denom == 0:
                    n_star_interp = float(n_right)
                else:
                    n_star_interp = n_left + (0.0 - d_left) * (
                        n_right - n_left
                    ) / denom
            break

    return ThresholdResult(
        classifier=classifier_name,
        subset_a=a,
        subset_b=b,
        n_star_grid=n_star_grid,
        n_star_interp=n_star_interp,
    )


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


def best_subset_by_sample_size(
    accumulator: LossAccumulator,
    classifier_names: Sequence[str],
    sample_sizes: Sequence[int],
    subsets: Sequence[Subset],
) -> pd.DataFrame:
    """Return a matrix of winning subset labels: rows = ``n``, cols = classifiers.

    The index is ``n_per_class`` and each column (one per classifier, labelled
    with its display name) holds the label of the subset with the lowest mean
    test loss at that sample size.
    """
    data: Dict[str, List[str]] = {}
    index = [int(n) for n in sample_sizes]
    for clf in classifier_names:
        col = []
        for n in sample_sizes:
            entry = best_subset(accumulator, clf, n, subsets)
            col.append(subset_label(entry.subset))
        data[classifier_label(clf)] = col
    df = pd.DataFrame(data, index=index)
    df.index.name = "n_per_class"
    return df


def final_ranking_at_largest_n(
    accumulator: LossAccumulator,
    classifier_names: Sequence[str],
    sample_sizes: Sequence[int],
    subsets: Sequence[Subset],
) -> pd.DataFrame:
    """Rank all subsets by mean test loss at the largest evaluated ``n``.

    Returns a long DataFrame with one row per (classifier, subset). Columns:
    ``classifier``, ``classifier_label``, ``n_per_class``, ``rank``,
    ``subset``, ``subset_label``, ``mean_loss``, ``standard_error``.
    Rank 1 is the lowest mean loss (best) for each classifier.
    """
    n = max(sample_sizes)
    rows: List[dict] = []
    for clf in classifier_names:
        scored = [
            (
                tuple(s),
                accumulator.mean_loss(n, tuple(s), clf),
                accumulator.standard_error(n, tuple(s), clf),
            )
            for s in subsets
        ]
        scored.sort(key=lambda item: item[1])
        for rank, (subset_t, mean_loss, se) in enumerate(scored, start=1):
            rows.append(
                {
                    "classifier": clf,
                    "classifier_label": classifier_label(clf),
                    "n_per_class": int(n),
                    "rank": rank,
                    "subset": subset_t,
                    "subset_label": subset_label(subset_t),
                    "mean_loss": mean_loss,
                    "standard_error": se,
                }
            )
    return pd.DataFrame(rows)


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
                    "n_star_grid": result.n_star_grid,
                    "n_star_interp": result.n_star_interp,
                }
            )
    return pd.DataFrame(rows)
