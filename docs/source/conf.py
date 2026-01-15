# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'slacgs'
copyright = '2023, Paulo R C Azevedo Filho'
author = 'Paulo R C Azevedo Filho'
release = '1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys
import sphinx_rtd_theme


# Add src directory to path for autodoc to find slacgs package
sys.path.insert(0, os.path.abspath('../../src'))
extensions = ['sphinx.ext.autodoc',
              "sphinx_rtd_theme",
              #'readthedocs-sphinx-ext',
              'sphinx.ext.napoleon',
              'sphinx.ext.mathjax']


# exclude_patterns = []
autoclass_content = 'both'


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output


html_theme_options = {
    "canonical_url": "",
    "analytics_id": "",  # Provided by Google in your dashboard, if you want to use Google Analytics
    "logo_only": False,  # If True, only the logo is displayed in the sidebar, not the project name
    "display_version": True,  # If True, the version number is displayed in the sidebar
    "prev_next_buttons_location": "both",  # Location of the "previous" and "next" buttons ("top", "bottom", "both", or None)
    "style_external_links": False,  # If True, external links are styled with an icon
    "vcs_pageview_mode": "",  # Changes how to view files when using display_github, display_gitlab, etc.
    "style_nav_header_background": "white",  # Changes the background color of the navigation header
    # Toc options
    "collapse_navigation": False,  # If True, the navigation entries are not expandable and you can only see one hierarchy level
    "sticky_navigation": True,  # If True, the sidebar stays fixed while scrolling
    "navigation_depth": 4,  # Indicate the depth of the headers shown in the sidebar (1 shows only top-level headers, 2 shows headers down to sub-headers, etc.)
    "includehidden": True,  # If True, "hidden" headers are included in the navigation
    "titles_only": False  # If True, only the titles of documents are included in the navigation, not the headers
}


html_theme = "sphinx_rtd_theme"
html_baseurl = 'https://paulorenatoaz.github.io/slacgs/'
# html_theme = 'readthedocs-sphinx-ext'
