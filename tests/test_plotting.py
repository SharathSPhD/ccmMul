# tests/test_plotting.py
import pytest
import matplotlib
matplotlib.use('Agg')  # Force non-interactive backend
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plotting import plot_ccm_results, plot_data_overview, create_summary_plots, plot_data_chunk
import tempfile
import os

@pytest.fixture(autouse=True)
def setup_test_dirs(sample_config):
    """Setup and cleanup test directories."""
    os.makedirs(sample_config['output']['plots_dir'], exist_ok=True)
    yield
    try:
        for file in os.listdir(sample_config['output']['plots_dir']):
            os.unlink(os.path.join(sample_config['output']['plots_dir'], file))
        os.rmdir(sample_config['output']['plots_dir'])
    except OSError:
        pass

def test_plot_ccm_results(sample_data, sample_config):
    """Test CCM results plotting functionality."""
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
    
    filename = plot_ccm_results(results, 'x', sample_config, 
                              sample_config['output']['plots_dir'], 
                              sample_data)
    
    assert os.path.exists(os.path.join(sample_config['output']['plots_dir'], filename))
    
    # Test with no valid results
    empty_results = {
        'View': None,
        'best_combo': None,
        'predictions': None
    }
    filename = plot_ccm_results(empty_results, 'x', sample_config,
                              sample_config['output']['plots_dir'],
                              sample_data)
    assert os.path.exists(os.path.join(sample_config['output']['plots_dir'], filename))

def test_plot_data_overview(sample_data, sample_config):
    """Test data overview plotting with various dataset sizes."""
    # Test with small dataset
    filename = plot_data_overview(sample_data, sample_config, 
                                sample_config['output']['plots_dir'])
    assert os.path.exists(os.path.join(sample_config['output']['plots_dir'], filename))
    
    # Test with large dataset
    large_data = pd.concat([sample_data] * 100, ignore_index=True)
    filename = plot_data_overview(large_data, sample_config,
                                sample_config['output']['plots_dir'])
    assert os.path.exists(os.path.join(sample_config['output']['plots_dir'], filename))

def test_plot_data_chunk():
    """Test data chunk plotting functionality."""
    # Create test data
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'datetime': pd.date_range('2023-01-01', periods=100)
    })
    
    # Test with datetime
    results = plot_data_chunk(data, ['x', 'y'], (0, 50), 'datetime')
    assert isinstance(results, dict)
    assert all(col in results for col in ['x', 'y'])
    assert all(len(results[col]) == 2 for col in results)
    
    # Test without datetime
    results = plot_data_chunk(data, ['x', 'y'], (0, 50))
    assert isinstance(results, dict)
    assert all(col in results for col in ['x', 'y'])
    assert all(len(results[col]) == 2 for col in results)

def test_create_summary_plots(sample_data, sample_config):
    """Test summary plots creation."""
    # Create test results
    results = {
        'x': {
            'best_combo': {
                'variables': ('y', 'z'),
                'rho': 0.95,
                'MAE': 0.1,
                'RMSE': 0.15
            }
        },
        'y': {
            'best_combo': {
                'variables': ('x', 'z'),
                'rho': 0.90,
                'MAE': 0.12,
                'RMSE': 0.18
            }
        },
        'z': {
            'best_combo': {
                'variables': ('x', 'y'),
                'rho': 0.85,
                'MAE': 0.15,
                'RMSE': 0.20
            }
        }
    }
    
    filename = create_summary_plots(results, sample_config,
                                  sample_config['output']['plots_dir'])
    assert os.path.exists(os.path.join(sample_config['output']['plots_dir'], filename))
    
    # Test with empty results
    empty_results = {}
    filename = create_summary_plots(empty_results, sample_config,
                                  sample_config['output']['plots_dir'])
    assert filename is None

def test_plot_ccm_results_with_datetime(sample_data, sample_config):
    """Test CCM results plotting with datetime index."""
    # Add datetime column
    sample_data['datetime'] = pd.date_range('2023-01-01', periods=len(sample_data))
    sample_config['data']['datetime_column'] = 'datetime'
    
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
    
    filename = plot_ccm_results(results, 'x', sample_config,
                              sample_config['output']['plots_dir'],
                              sample_data)
    assert os.path.exists(os.path.join(sample_config['output']['plots_dir'], filename))

def test_figure_properties():
    """Test matplotlib figure properties and cleanup."""
    plt.close('all')  # Close any existing figures
    
    # Create test data and results
    data = pd.DataFrame({
        'x': np.random.randn(100),
        'y': np.random.randn(100),
        'z': np.random.randn(100)
    })
    
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
    
    config = {
        'data': {'columns_to_keep': ['x', 'y', 'z']},
        'output': {'plots_dir': 'test_plots'}
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test plot_ccm_results
        plot_ccm_results(results, 'x', config, temp_dir, data)
        assert plt.get_fignums() == []  # Check that figures are closed
        
        # Test plot_data_overview
        plot_data_overview(data, config, temp_dir)
        assert plt.get_fignums() == []  # Check that figures are closed
        
        # Test create_summary_plots
        create_summary_plots({'x': results}, config, temp_dir)
        assert plt.get_fignums() == []  # Check that figures are closed

def test_plot_data_overview_parallel_processing(sample_data, sample_config):
    """Test data overview plotting with parallel processing."""
    # Create large dataset to trigger parallel processing
    large_data = pd.concat([sample_data] * 200, ignore_index=True)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with parallel processing enabled
        sample_config['analysis']['parallel']['enabled'] = True
        filename = plot_data_overview(large_data, sample_config, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, filename))
        
        # Test with parallel processing disabled
        sample_config['analysis']['parallel']['enabled'] = False
        filename = plot_data_overview(large_data, sample_config, temp_dir)
        assert os.path.exists(os.path.join(temp_dir, filename))

def test_plot_with_missing_data(sample_data, sample_config):
    """Test plotting functionality with missing data."""
    # Introduce some missing values
    data_with_nans = sample_data.copy()
    data_with_nans.loc[10:15, 'x'] = np.nan
    
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
        # Test all plotting functions with missing data
        filename1 = plot_ccm_results(results, 'x', sample_config, temp_dir, data_with_nans)
        filename2 = plot_data_overview(data_with_nans, sample_config, temp_dir)
        filename3 = create_summary_plots({'x': results}, sample_config, temp_dir)
        
        assert os.path.exists(os.path.join(temp_dir, filename1))
        assert os.path.exists(os.path.join(temp_dir, filename2))
        assert os.path.exists(os.path.join(temp_dir, filename3))