"""
Visualization and Reporting Module for HFT Simulator

This module provides comprehensive visualization and reporting capabilities
for the HFT simulator, including real-time charts, performance dashboards,
and interactive analysis tools.

Components:
- dashboard.py: Main dashboard interface with real-time monitoring
- charts.py: Chart generation and plotting utilities
- reports.py: Report generation and export functionality
- web_interface.py: Web-based dashboard interface
"""

from .dashboard import Dashboard, DashboardConfig
from .charts import ChartGenerator, ChartType
from .reports import ReportGenerator, ReportType
from .web_interface import WebDashboard

__all__ = [
    'Dashboard',
    'DashboardConfig', 
    'ChartGenerator',
    'ChartType',
    'ReportGenerator',
    'ReportType',
    'WebDashboard'
]