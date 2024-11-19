"""
Multivariate Convergent Cross Mapping (CCM) Analysis Package

This package implements multivariate CCM analysis for time series data with parallel
processing capabilities, automated variable selection, and comprehensive visualization tools.

Key Features:
- Multivariate CCM implementation with optimized parallel processing
- Intelligent combination selection to reduce computational complexity
- Automated variable selection based on correlation analysis
- Progress tracking and comprehensive logging
- Visualization of results with customizable plots
- Support for both synthetic and real data
- Error handling and recovery mechanisms

Main Components:
- analysis: Core analysis functions and workflow management
- multivariate_ccm: Implementation of the multivariate CCM algorithm
- plotting: Visualization and plotting utilities
- utils: Helper functions and data management utilities
"""

from .analysis import run_ccm_analysis, print_summary
from .multivariate_ccm import MultivariateCCM
from .plotting import (
    plot_ccm_results,
    plot_data_overview,
    create_summary_plots,
    plot_data_chunk
)
from .utils import (
    load_config,
    load_data,
    save_results,
    generate_synthetic_data,
    setup_logger,
    select_optimal_combinations,
    evaluate_combination_importance,
    prepare_output_dirs
)

__version__ = '1.0.0'
__author__ = 'Sharath Sathish'
__email__ = 'your.email@example.com'
__description__ = 'Multivariate Convergent Cross Mapping (CCM) analysis package'

# Define what should be available to users when they import the package
__all__ = [
    # Main analysis functions
    'run_ccm_analysis',
    'print_summary',
    
    # Core CCM implementation
    'MultivariateCCM',
    
    # Plotting functions
    'plot_ccm_results',
    'plot_data_overview',
    'create_summary_plots',
    'plot_data_chunk',
    
    # Utility functions
    'load_config',
    'load_data',
    'save_results',
    'generate_synthetic_data',
    'setup_logger',
    'select_optimal_combinations',
    'evaluate_combination_importance',
    'prepare_output_dirs'
]

# Package metadata
metadata = {
    'name': 'ccmMul',
    'version': __version__,
    'author': __author__,
    'author_email': __email__,
    'description': __description__,
    'license': 'MIT',
    'python_requires': '>=3.8',
    'install_requires': [
        'numpy>=1.20.0',
        'pandas>=1.3.0',
        'scikit-learn>=0.24.0',
        'matplotlib>=3.3.0'
    ],
    'keywords': [
        'time series analysis',
        'convergent cross mapping',
        'causality analysis',
        'multivariate analysis',
        'data science'
    ],
    'classifiers': [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Mathematics'
    ]
}