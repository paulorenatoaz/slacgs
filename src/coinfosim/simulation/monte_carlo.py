"""
Cooperative Monte Carlo simulator for CoInfoSim Sprint 1.

:class:`CooperativeMonteCarloSimulator` runs the Sprint 1 experiment loop::

    n_per_class -> replication -> subset -> classifier

For each ``n_per_class`` it accumulates empirical test-loss replications for
every (subset, classifier) pair, reusing one fixed test set, and applies the
standard-error stopping rule at replication batch boundaries.

The simulator computes empirical test loss only. It does *not* compute
empirical train loss, theoretical loss, or Bayes error.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from coinfosim.classifiers.registry import available_classifiers, make_classifier
from coinfosim.models.gaussian import GaussianSimulationModel
from coinfosim.results.accumulator import LossAccumulator
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler
from coinfosim.simulation.config import MonteCarloConfig
from coinfosim.simulation.metrics import empirical_test_loss
from coinfosim.simulation.stopping import StandardErrorStoppingRule
from coinfosim.simulation.subsets import all_nonempty_subsets


@dataclass
class StoppingInfo:
    """Final stopping status recorded for one ``n_per_class``."""

    n_per_class: int
    replications: int
    reason: str  # "converged" or "max_budget"
    max_ci_half_width: float


@dataclass
class SimulationResult:
    """Structured result returned by the cooperative Monte Carlo simulator."""

    model: GaussianSimulationModel
    config: MonteCarloConfig
    subsets: List[Tuple[int, ...]]
    classifier_names: List[str]
    accumulator: LossAccumulator
    stopping_info: Dict[int, StoppingInfo]
    runtime_seconds: float
    metadata: Dict[str, object] = field(default_factory=dict)

    @property
    def sample_sizes(self) -> List[int]:
        return list(self.config.sample_sizes)


class CooperativeMonteCarloSimulator:
    """Run the Sprint 1 cooperative Monte Carlo experiment."""

    def __init__(
        self,
        model: GaussianSimulationModel,
        config: MonteCarloConfig,
        subsets: Optional[Sequence[Sequence[int]]] = None,
        classifier_names: Optional[Sequence[str]] = None,
        stopping_rule: Optional[StandardErrorStoppingRule] = None,
        sampler: Optional[GaussianClassConditionalSampler] = None,
    ) -> None:
        self.model = model
        self.config = config

        if subsets is None:
            subsets = all_nonempty_subsets(model.d)
        self.subsets: List[Tuple[int, ...]] = [tuple(s) for s in subsets]

        if classifier_names is None:
            classifier_names = available_classifiers()
        self.classifier_names: List[str] = list(classifier_names)

        if stopping_rule is None:
            stopping_rule = StandardErrorStoppingRule(
                min_replications=config.min_replications,
                max_replications=config.max_replications,
                ci_half_width_target=config.ci_half_width_target,
            )
        self.stopping_rule = stopping_rule

        if sampler is None:
            sampler = GaussianClassConditionalSampler(
                model,
                base_seed=config.base_seed,
                test_samples_per_class=config.test_samples_per_class,
            )
        self.sampler = sampler

    def _cells(self) -> List[Tuple[Tuple[int, ...], str]]:
        return [
            (subset, clf)
            for subset in self.subsets
            for clf in self.classifier_names
        ]

    def run(self) -> SimulationResult:
        """Execute the full experiment and return a :class:`SimulationResult`."""
        start = time.time()
        accumulator = LossAccumulator()
        stopping_info: Dict[int, StoppingInfo] = {}
        cells = self._cells()

        # Fixed test set, generated once and reused everywhere.
        test_dataset = self.sampler.sample_test()
        # Pre-restrict the fixed test set per subset (test set is constant).
        test_by_subset = {
            subset: test_dataset.select_channels(subset) for subset in self.subsets
        }

        for n_per_class in self.config.sample_sizes:
            replication_id = 0
            last_decision = None

            while True:
                # Run one batch of replications.
                batch_end = replication_id + self.config.replication_batch_size
                while replication_id < batch_end:
                    train = self.sampler.sample_train(
                        n_per_class=n_per_class, replication_id=replication_id
                    )
                    for subset in self.subsets:
                        train_sub = train.select_channels(subset)
                        test_sub = test_by_subset[subset]
                        for clf_name in self.classifier_names:
                            estimator = make_classifier(clf_name)
                            estimator.fit(train_sub.X, train_sub.y)
                            loss = empirical_test_loss(estimator, test_sub)
                            accumulator.add(
                                n_per_class, subset, clf_name, replication_id, loss
                            )
                    replication_id += 1

                # Evaluate stopping rule at the batch boundary.
                last_decision = self.stopping_rule.evaluate(
                    accumulator, n_per_class, cells
                )
                if last_decision.should_stop:
                    break

            stopping_info[n_per_class] = StoppingInfo(
                n_per_class=n_per_class,
                replications=last_decision.replications,
                reason=last_decision.reason,
                max_ci_half_width=last_decision.max_ci_half_width,
            )

        runtime = time.time() - start

        metadata = {
            "mode": self.config.mode,
            "base_seed": self.config.base_seed,
            "test_samples_per_class": self.config.test_samples_per_class,
            "ci_half_width_target": self.config.ci_half_width_target,
            "min_replications": self.config.min_replications,
            "max_replications": self.config.max_replications,
            "replication_batch_size": self.config.replication_batch_size,
            "metric": "empirical_test_loss",
            "d": self.model.d,
            "class_labels": list(self.model.class_labels),
        }

        return SimulationResult(
            model=self.model,
            config=self.config,
            subsets=self.subsets,
            classifier_names=self.classifier_names,
            accumulator=accumulator,
            stopping_info=stopping_info,
            runtime_seconds=runtime,
            metadata=metadata,
        )
