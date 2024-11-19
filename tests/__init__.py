"""
Test suite for the Multivariate CCM Analysis Package.

This package contains comprehensive tests for all components of the CCM analysis package,
including unit tests, integration tests, and performance tests.

Test Structure:
- test_analysis.py: Tests for core analysis functions and workflow
- test_multivariate_ccm.py: Tests for CCM algorithm implementation
- test_plotting.py: Tests for visualization functions
- test_utils.py: Tests for utility functions and data management
- conftest.py: Shared test fixtures and configuration

Testing Framework:
- pytest for test execution and assertions
- pytest-cov for code coverage analysis
- tox for testing across multiple Python versions

Key Features Tested:
- Parallel processing implementation
- Data handling and validation
- Error handling and recovery
- Results generation and storage
- Visualization functionality
- Configuration management
- Performance and scalability
"""

import os
import sys

# Add the parent directory to the Python path to allow importing from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test suite metadata
__version__ = '1.0.0'
__author__ = 'Sharath Sathish'
__email__ = 'your.email@example.com'

# Test configuration
test_config = {
    'min_coverage': 80,  # Minimum required code coverage percentage
    'parallel_enabled': True,  # Enable parallel test execution
    'performance_threshold': {
        'small_dataset': 5,  # Maximum seconds for small dataset processing
        'large_dataset': 30   # Maximum seconds for large dataset processing
    },
    'supported_python_versions': ['3.8', '3.9', '3.10', '3.11']
}

# Define test categories for organization and selective execution
test_categories = {
    'unit': [
        'test_analysis.py',
        'test_multivariate_ccm.py',
        'test_plotting.py',
        'test_utils.py'
    ],
    'integration': [
        'test_analysis.py::test_run_ccm_analysis',
        'test_analysis.py::test_run_ccm_analysis_with_missing_data'
    ],
    'performance': [
        'test_multivariate_ccm.py::test_parallel_vs_serial_execution',
        'test_plotting.py::test_plot_data_overview_parallel_processing'
    ],
    'regression': [
        'test_multivariate_ccm.py::test_full_analysis',
        'test_analysis.py::test_run_ccm_analysis_output_files'
    ]
}

def get_test_dependencies():
    """Return list of required packages for testing."""
    return [
        'pytest>=6.0',
        'pytest-cov>=2.0',
        'pytest-xdist>=2.4',  # For parallel test execution
        'pytest-timeout>=2.1',  # For test timing
        'black>=22.0',
        'pylint>=2.8.0',
        'mypy>=0.900',
        'tox>=3.24.0'
    ]

def get_test_category(category):
    """Return list of tests for a specific category."""
    return test_categories.get(category, [])

def run_performance_checks():
    """Check if the test environment meets performance requirements."""
    import platform
    import multiprocessing
    
    system_info = {
        'os': platform.system(),
        'python_version': platform.python_version(),
        'processor': platform.processor(),
        'cpu_count': multiprocessing.cpu_count(),
        'memory': os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024.**3)
        if hasattr(os, 'sysconf') else None
    }
    
    # Define minimum requirements
    requirements = {
        'cpu_count': 2,
        'memory': 4  # GB
    }
    
    # Check requirements
    meets_requirements = (
        system_info['cpu_count'] >= requirements['cpu_count'] and
        (system_info['memory'] is None or 
         system_info['memory'] >= requirements['memory'])
    )
    
    return meets_requirements, system_info

# Initialize test environment
if __name__ == '__main__':
    # Print test environment information
    meets_req, sys_info = run_performance_checks()
    print("Test Environment Information:")
    print(f"OS: {sys_info['os']}")
    print(f"Python Version: {sys_info['python_version']}")
    print(f"CPU Count: {sys_info['cpu_count']}")
    print(f"Memory: {sys_info['memory']:.1f} GB" if sys_info['memory'] else "Memory: Unknown")
    print(f"Meets Requirements: {meets_req}")