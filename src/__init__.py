"""
Multivariate CCM Analysis Package
--------------------------------

This package provides tools for Multivariate Convergent Cross Mapping analysis
of time series data.

Modules:
    analysis: Main analysis functions
    multivariate_ccm: Core CCM implementation
    plotting: Visualization functions
    utils: Utility functions for data handling

Example:
    from src.analysis import run_ccm_analysis
    from src.utils import load_config, load_data
"""

from . import analysis
from . import multivariate_ccm
from . import plotting
from . import utils

__version__ = '1.0.0'
__author__ = 'Your Name'

# Version information
VERSION_INFO = {
    'major': 1,
    'minor': 0,
    'patch': 0,
}

# Define what should be imported with 'from src import *'
__all__ = [
    'analysis',
    'multivariate_ccm',
    'plotting',
    'utils',
]

def get_version():
    """Return the version string."""
    return f"{VERSION_INFO['major']}.{VERSION_INFO['minor']}.{VERSION_INFO['patch']}"

def get_module_versions():
    """Return versions of key dependencies."""
    import pandas as pd
    import numpy as np
    import sklearn
    
    return {
        'pandas': pd.__version__,
        'numpy': np.__version__,
        'scikit-learn': sklearn.__version__
    }