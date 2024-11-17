# src/__init__.py
"""
Multivariate CCM Analysis Package
--------------------------------

This package provides tools for Multivariate Convergent Cross Mapping analysis
of time series data.
"""

from .analysis import run_ccm_analysis, print_summary
from .multivariate_ccm import MultivariateCCM
from .plotting import plot_ccm_results, plot_data_overview
from .utils import load_config, load_data, save_results

__version__ = '1.0.0'
__author__ = 'Your Name'

__all__ = [
    'run_ccm_analysis',
    'print_summary',
    'MultivariateCCM',
    'plot_ccm_results',
    'plot_data_overview',
    'load_config',
    'load_data',
    'save_results',
]