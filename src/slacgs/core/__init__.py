"""
SLACGS Core Module

Core simulation and modeling functionality for SLACGS.
Contains the main Model and Simulator classes.
"""

from .model import Model
from .simulator import Simulator
from .enumtypes import DictionaryType, LossType

__all__ = ['Model', 'Simulator', 'DictionaryType', 'LossType']
