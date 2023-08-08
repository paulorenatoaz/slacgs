from enum import Enum


class LossType(Enum):
  """Implemented Loss Functions

  Attributes:
    - THEORETICAL: estimated using probability theory
    - EMPIRICALTRAIN: estimated using empirical approach with train data
    - EMPIRICALTEST: estimated using empirical approach with test data

  """
  THEORETICAL = 'THEORETICAL'
  EMPIRICALTRAIN = 'EMPIRICAL_TRAIN'
  EMPIRICALTEST = 'EMPIRICAL_TEST'


class DictionaryType(Enum):
  """Implemented Dictionary Types

  Attributes:
    - LINEAR: dictionary with linear classifiers

  """

  LINEAR = 'LINEAR'
