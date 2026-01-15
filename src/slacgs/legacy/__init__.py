"""
SLACGS Legacy Module

DEPRECATED: This module contains deprecated functionality that is maintained
for backward compatibility only. These features will be removed in a future version.

Legacy Google Drive and Google Sheets integration.
Please use local file-based workflows instead.
"""

import warnings

warnings.warn(
    "The slacgs.legacy module contains deprecated functionality. "
    "Google Drive/Sheets integration is no longer actively maintained. "
    "Please use local file-based workflows with the CLI tools instead.",
    DeprecationWarning,
    stacklevel=2
)

# Legacy imports (with deprecation warnings in individual modules)
from .gdrive_client import *
from .gspread_client import *

__all__ = []  # Nothing exported by default - users must explicitly import from legacy
