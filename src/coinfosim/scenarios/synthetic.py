"""
Synthetic scenario definitions for CoInfoSim Sprint 1.

Only **Synthetic Scenario 1 — Simple Complementary Channel** is implemented
in this sprint.
"""

from __future__ import annotations

from dataclasses import dataclass

from coinfosim.models.gaussian import GaussianSimulationModel

SCENARIO_1_NAME = "Synthetic Scenario 1 — Simple Complementary Channel"
SCENARIO_1_QUESTION = (
    "When does an individually weaker channel improve classification by "
    "adding complementary information?"
)

# Scenario 1 model parameters.
_MU0 = [-0.70, -0.55, -0.30]
_MU1 = [0.70, 0.55, 0.30]
_SIGMA = [
    [1.00, 0.35, 0.05],
    [0.35, 1.00, 0.05],
    [0.05, 0.05, 1.00],
]


@dataclass(frozen=True)
class SyntheticScenario:
    """A named synthetic scenario bundling a model and its scientific question."""

    name: str
    question: str
    model: GaussianSimulationModel

    @property
    def d(self) -> int:
        return self.model.d


def make_synthetic_scenario_1() -> SyntheticScenario:
    """Construct Synthetic Scenario 1 — Simple Complementary Channel."""
    model = GaussianSimulationModel(
        means={0: _MU0, 1: _MU1},
        covariances={0: _SIGMA, 1: _SIGMA},
    )
    return SyntheticScenario(
        name=SCENARIO_1_NAME,
        question=SCENARIO_1_QUESTION,
        model=model,
    )
