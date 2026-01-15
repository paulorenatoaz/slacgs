"""Publishing utilities for slacgs.

Generates an index.html linking Scenario reports (HTML) and JSON data
intended to live on a dedicated gh-pages branch. Main stays code-only.
"""

from .publisher import publish_to_pages

__all__ = ['publish_to_pages']

