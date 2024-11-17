# tests/test_plotting.py
import pytest
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plotting import plot_ccm_results, plot_data_overview
import tempfile
import os

@pytest.fixture(autouse=True)
def setup_test_dirs(sample_config):
    """Automatically create test directories before each test."""
    os.makedirs(sample_config['output']['plots_dir'], exist_ok=True)
    yield
    # Optional: Clean up after tests
    try:
        os.rmdir(sample_config['output']['plots_dir'])
    except OSError:
        pass  # Directory not empty or already deleted

def test_plot_ccm_results(sample_data, sample_config):
    # Create test results
    results = {
        'View': pd.DataFrame({
            'variables': [('y',), ('z',), ('y', 'z')],
            'rho': [0.8, 0.85, 0.95],
            'MAE': [0.2, 0.15, 0.1],
            'RMSE': [0.25, 0.2, 0.15]
        }),
        'best_combo': {
            'variables': ('y', 'z'),
            'rho': 0.95,
            'MAE': 0.1,
            'RMSE': 0.15
        },
        'predictions': {
            'time_indices': np.arange(25),
            'actual': np.random.randn(25),
            'predicted': np.random.randn(25)
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_config['output']['plots_dir'] = temp_dir
        filename = plot_ccm_results(results, 'x', sample_config, temp_dir, sample_data)
        assert os.path.exists(os.path.join(temp_dir, filename))

def test_plot_data_overview(sample_data, sample_config):
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_config['output']['plots_dir'] = temp_dir
        filename = plot_data_overview(sample_data, sample_config, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, filename))