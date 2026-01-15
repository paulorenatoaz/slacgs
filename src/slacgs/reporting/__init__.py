"""
SLACGS Reporting Module

Report generation and visualization functionality for SLACGS.
Contains Report, ReportData and report utility functions.
"""

from .report import Report
from .report_data import ReportData
from .report_utils import create_scenario_report

__all__ = ['Report', 'ReportData', 'create_scenario_report']
