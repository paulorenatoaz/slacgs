"""
Channel-subset enumeration and display labels for CoInfoSim Sprint 1.

Subsets are represented internally as sorted tuples of zero-based channel
indices. Display labels are one-based (``X1``, ``X2``, ...), joined with ``+``
for multi-channel subsets (e.g. ``X1+X3``).
"""

from __future__ import annotations

from itertools import combinations
from typing import List, Sequence, Tuple


def all_nonempty_subsets(d: int) -> List[Tuple[int, ...]]:
    """Return all non-empty subsets of ``range(d)`` as sorted index tuples.

    The ordering is by subset size first, then lexicographically. For
    ``d = 3`` this yields::

        (0,), (1,), (2,), (0, 1), (0, 2), (1, 2), (0, 1, 2)
    """
    if d <= 0:
        raise ValueError("d must be a positive integer")

    subsets: List[Tuple[int, ...]] = []
    for size in range(1, d + 1):
        for combo in combinations(range(d), size):
            subsets.append(tuple(combo))
    return subsets


def subset_label(subset: Sequence[int]) -> str:
    """Return the one-based display label for a channel ``subset``.

    Example: ``(0, 2) -> "X1+X3"``.
    """
    idx = list(subset)
    if len(idx) == 0:
        raise ValueError("subset must be non-empty")
    return "+".join(f"X{i + 1}" for i in idx)


def subset_labels(subsets: Sequence[Sequence[int]]) -> List[str]:
    """Return display labels for a sequence of subsets."""
    return [subset_label(s) for s in subsets]
