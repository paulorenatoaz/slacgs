"""
CoInfoSim Sprint 1 models.

New simulation model classes for the CoInfoSim Sprint 1 implementation.
This package is independent of the legacy ``core`` (SLACGS-compatible)
sigma/rho parameter-vector model and must not depend on ``LossType``.
"""

from coinfosim.models.gaussian import GaussianSimulationModel

__all__ = ["GaussianSimulationModel"]
