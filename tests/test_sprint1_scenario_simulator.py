"""Tests for Sprint 1 scenario and cooperative Monte Carlo simulator (Checkpoint 4)."""

import numpy as np
import pytest

from coinfosim.scenarios.synthetic import (
    SCENARIO_1_NAME,
    SCENARIO_1_QUESTION,
    make_synthetic_scenario_1,
)
from coinfosim.simulation.config import MonteCarloConfig, get_mode_config
from coinfosim.simulation.monte_carlo import (
    CooperativeMonteCarloSimulator,
    SimulationResult,
)
from coinfosim.simulation.subsets import all_nonempty_subsets


# --- Scenario 1 ---------------------------------------------------------------

def test_scenario_1_construction():
    scenario = make_synthetic_scenario_1()
    assert scenario.name == SCENARIO_1_NAME
    assert scenario.question == SCENARIO_1_QUESTION
    assert scenario.d == 3
    assert scenario.model.K == 2
    assert np.allclose(scenario.model.mean(0), [-0.70, -0.55, -0.30])
    assert np.allclose(scenario.model.mean(1), [0.70, 0.55, 0.30])


def test_scenario_1_has_seven_subsets():
    scenario = make_synthetic_scenario_1()
    assert len(all_nonempty_subsets(scenario.d)) == 7


# --- End-to-end smoke simulation ----------------------------------------------

def _tiny_config():
    return MonteCarloConfig(
        mode="smoke",
        sample_sizes=(2, 4),
        min_replications=3,
        max_replications=6,
        replication_batch_size=3,
        test_samples_per_class=50,
        ci_half_width_target=0.05,
        base_seed=0,
    )


def test_end_to_end_smoke_simulation():
    scenario = make_synthetic_scenario_1()
    sim = CooperativeMonteCarloSimulator(scenario.model, _tiny_config())
    result = sim.run()

    assert isinstance(result, SimulationResult)
    assert result.sample_sizes == [2, 4]
    assert result.runtime_seconds >= 0.0

    # All seven subsets and three classifiers evaluated.
    assert len(result.subsets) == 7
    assert result.classifier_names == [
        "linear_svm",
        "logistic_regression",
        "gaussian_nb",
    ]

    # Every cell has recorded losses for every sample size.
    for n in result.sample_sizes:
        for subset in result.subsets:
            for clf in result.classifier_names:
                losses = result.accumulator.losses(n, subset, clf)
                assert losses.size >= 3
                assert np.all((losses >= 0.0) & (losses <= 1.0))


def test_simulation_shared_replication_count():
    scenario = make_synthetic_scenario_1()
    sim = CooperativeMonteCarloSimulator(scenario.model, _tiny_config())
    result = sim.run()
    for n in result.sample_sizes:
        counts = {
            result.accumulator.count(n, subset, clf)
            for subset in result.subsets
            for clf in result.classifier_names
        }
        assert len(counts) == 1  # all cells share the same replication count


def test_simulation_stopping_info_recorded():
    scenario = make_synthetic_scenario_1()
    sim = CooperativeMonteCarloSimulator(scenario.model, _tiny_config())
    result = sim.run()
    for n in result.sample_sizes:
        info = result.stopping_info[n]
        assert info.reason in {"converged", "max_budget"}
        assert info.replications >= 3


def test_simulation_metadata_metric_only_test_loss():
    scenario = make_synthetic_scenario_1()
    sim = CooperativeMonteCarloSimulator(scenario.model, _tiny_config())
    result = sim.run()
    assert result.metadata["metric"] == "empirical_test_loss"
    # No train-loss / theoretical / bayes keys anywhere in metadata.
    keys = " ".join(result.metadata.keys()).lower()
    assert "train" not in keys
    assert "theoretical" not in keys
    assert "bayes" not in keys


def test_simulation_fixed_test_set_reused():
    scenario = make_synthetic_scenario_1()
    sim = CooperativeMonteCarloSimulator(scenario.model, _tiny_config())
    # The sampler caches the fixed test set; calling twice returns the same object.
    t1 = sim.sampler.sample_test()
    t2 = sim.sampler.sample_test()
    assert t1 is t2
