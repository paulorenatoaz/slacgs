"""
Monte Carlo budget and execution-mode configuration for CoInfoSim Sprint 1.

Provides :class:`MonteCarloConfig` and :func:`get_mode_config` for the
``smoke``, ``fast``, and ``full`` execution modes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True)
class MonteCarloConfig:
    """Configuration for a Monte Carlo run.

    Attributes
    ----------
    mode:
        Execution-mode name (``smoke``, ``fast``, or ``full``).
    sample_sizes:
        Tuple of ``n_per_class`` values to evaluate.
    min_replications:
        Minimum replications before the stopping rule may trigger.
    max_replications:
        Maximum replications per ``n_per_class`` (budget cap).
    replication_batch_size:
        Number of replications between stopping-rule evaluations.
    test_samples_per_class:
        Size of the fixed test set per class.
    ci_half_width_target:
        Target 95% CI half-width for convergence.
    base_seed:
        Base random seed for reproducibility.
    """

    mode: str
    sample_sizes: Tuple[int, ...]
    min_replications: int
    max_replications: int
    replication_batch_size: int
    test_samples_per_class: int
    ci_half_width_target: float
    base_seed: int = 0

    def __post_init__(self) -> None:
        if len(self.sample_sizes) == 0:
            raise ValueError("sample_sizes must be non-empty")
        if any(n <= 0 for n in self.sample_sizes):
            raise ValueError("sample_sizes must be positive")
        if self.min_replications < 2:
            raise ValueError("min_replications must be at least 2")
        if self.max_replications < self.min_replications:
            raise ValueError("max_replications must be >= min_replications")
        if self.replication_batch_size <= 0:
            raise ValueError("replication_batch_size must be positive")
        if self.test_samples_per_class <= 0:
            raise ValueError("test_samples_per_class must be positive")
        if self.ci_half_width_target <= 0:
            raise ValueError("ci_half_width_target must be positive")


# Preset definitions for each execution mode.
_MODE_PRESETS: Dict[str, dict] = {
    "smoke": dict(
        sample_sizes=(2, 4, 8),
        min_replications=5,
        max_replications=20,
        replication_batch_size=5,
        test_samples_per_class=200,
        ci_half_width_target=0.05,
        base_seed=0,
    ),
    "fast": dict(
        sample_sizes=(2, 4, 8, 16, 32, 64),
        min_replications=30,
        max_replications=300,
        replication_batch_size=10,
        test_samples_per_class=1000,
        ci_half_width_target=0.01,
        base_seed=0,
    ),
    "full": dict(
        sample_sizes=(2, 4, 8, 16, 32, 64, 128, 256, 512),
        min_replications=100,
        max_replications=2000,
        replication_batch_size=20,
        test_samples_per_class=5000,
        ci_half_width_target=0.005,
        base_seed=0,
    ),
}

VALID_MODES = tuple(_MODE_PRESETS.keys())


def get_mode_config(mode: str) -> MonteCarloConfig:
    """Return the :class:`MonteCarloConfig` preset for ``mode``.

    Raises
    ------
    ValueError
        If ``mode`` is not one of ``smoke``, ``fast``, ``full``.
    """
    if mode not in _MODE_PRESETS:
        raise ValueError(
            f"unknown mode {mode!r}; valid modes: {list(VALID_MODES)}"
        )
    return MonteCarloConfig(mode=mode, **_MODE_PRESETS[mode])
