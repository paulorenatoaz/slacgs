"""
CoInfoSim Sprint 1 samplers.

Dataset container and Gaussian class-conditional sampler used by the
Sprint 1 CooperativeMonteCarlo simulator.
"""

from coinfosim.samplers.dataset import Dataset
from coinfosim.samplers.gaussian import GaussianClassConditionalSampler

__all__ = ["Dataset", "GaussianClassConditionalSampler"]
