# tests/test_utils.py
import pytest
from src.utils import load_config, generate_synthetic_data, load_data, save_results
import tempfile
import os
import pandas as pd
import numpy as np

def test_load_config(temp_config_file, sample_config):
    loaded_config = load_config(temp_config_file)
    assert loaded_config == sample_config

def test_generate_synthetic_data(sample_config):
    data = generate_synthetic_data(sample_config['data']['synthetic_params'])
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == sample_config['data']['synthetic_params']['n_points']
    assert all(col in data.columns for col in ['datetime', 'x', 'y', 'z'])
    assert pd.api.types.is_datetime64_any_dtype(data['datetime'])

def test_load_data_synthetic(sample_config):
    data = load_data(sample_config)
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == sample_config['data']['synthetic_params']['n_points']
    assert all(col in data.columns for col in sample_config['data']['columns_to_keep'])

def test_save_results(sample_config):
    # Create test results
    results = {
        'x': {
            'best_combo': {
                'variables': ('y', 'z'),
                'rho': 0.95,
                'MAE': 0.1,
                'RMSE': 0.15
            },
            'predictions': {
                'time_indices': np.array([0, 1, 2]),
                'actual': np.array([1.0, 2.0, 3.0]),
                'predicted': np.array([1.1, 2.1, 3.1])
            },
            'View': pd.DataFrame({
                'variables': [('y',), ('z',), ('y', 'z')],
                'rho': [0.8, 0.85, 0.95],
                'MAE': [0.2, 0.15, 0.1],
                'RMSE': [0.25, 0.2, 0.15]
            })
        }
    }
    
    with tempfile.TemporaryDirectory() as temp_dir:
        sample_config['output']['results_dir'] = temp_dir
        saved_files = save_results(results, sample_config, timestamp='test')
        
        assert os.path.exists(os.path.join(temp_dir, saved_files['metrics_file']))
        assert os.path.exists(os.path.join(temp_dir, saved_files['predictions_file']))
        assert os.path.exists(os.path.join(temp_dir, saved_files['combinations_file']))