"""
Summary export for CoInfoSim Sprint 1 results.

Builds a tidy :class:`pandas.DataFrame` of per-cell summary statistics
(mean, std, standard error, replication count) from a
:class:`~coinfosim.results.accumulator.LossAccumulator`.
"""

from __future__ import annotations

from typing import List, Sequence

import numpy as np
import pandas as pd

from coinfosim.results.accumulator import LossAccumulator
from coinfosim.simulation.subsets import subset_label
from coinfosim.classifiers.registry import classifier_label


def summary_dataframe(
    accumulator: LossAccumulator,
    sample_sizes: Sequence[int],
    subsets: Sequence[Sequence[int]],
    classifier_names: Sequence[str],
) -> pd.DataFrame:
    """Return a tidy summary DataFrame for the given cells.

    Columns: ``n_per_class``, ``subset``, ``subset_label``,
    ``classifier``, ``classifier_label``, ``mean_loss``, ``std_loss``,
    ``standard_error``, ``replications``.
    """
    rows: List[dict] = []
    for n in sample_sizes:
        for subset in subsets:
            subset_t = tuple(subset)
            for clf in classifier_names:
                rows.append(
                    {
                        "n_per_class": int(n),
                        "subset": subset_t,
                        "subset_label": subset_label(subset_t),
                        "classifier": clf,
                        "classifier_label": classifier_label(clf),
                        "mean_loss": accumulator.mean_loss(n, subset_t, clf),
                        "std_loss": accumulator.std_loss(n, subset_t, clf),
                        "standard_error": accumulator.standard_error(n, subset_t, clf),
                        "replications": accumulator.count(n, subset_t, clf),
                    }
                )
    return pd.DataFrame(rows)


def hardest_cell_summary(
    accumulator: LossAccumulator,
    sample_sizes: Sequence[int],
    subsets: Sequence[Sequence[int]],
    classifier_names: Sequence[str],
    z: float = 1.96,
) -> pd.DataFrame:
    """Return, per ``n_per_class``, the (subset, classifier) cell with the
    largest CI half-width (``z * standard_error``).

    Columns: ``n_per_class``, ``subset_label``, ``classifier_label``,
    ``ci_half_width``, ``standard_error``, ``mean_loss``.
    This highlights the slowest-converging cell driving the stopping rule.
    """
    rows: List[dict] = []
    for n in sample_sizes:
        worst = None
        for subset in subsets:
            subset_t = tuple(subset)
            for clf in classifier_names:
                se = accumulator.standard_error(n, subset_t, clf)
                half = z * se
                if worst is None or half > worst["ci_half_width"]:
                    worst = {
                        "n_per_class": int(n),
                        "subset_label": subset_label(subset_t),
                        "classifier_label": classifier_label(clf),
                        "ci_half_width": half,
                        "standard_error": se,
                        "mean_loss": accumulator.mean_loss(n, subset_t, clf),
                    }
        if worst is not None:
            rows.append(worst)
    return pd.DataFrame(rows)
