# tests/test_utils.py
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
import logging
from datetime import datetime
from src.utils import (
    load_config, 
    generate_synthetic_data, 
    load_data, 
    save_results, 
    setup_logger,
    select_optimal_combinations,
    evaluate_combination_importance,
    prepare_output_dirs,
    generate_synthetic_data_chunk,
    load_data_chunk,
    save_results_chunk
)

def test_load_config(temp_config_file, sample_config):
    """Test configuration loading functionality."""
    loaded_config = load_config(temp_config_file)
    assert loaded_config == sample_config
    
    # Test default parallel configuration
    del sample_config['analysis']['parallel']
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config, f)
        config_path = f.name
    
    loaded_config = load_config(config_path)
    assert 'parallel' in loaded_config['analysis']
    assert loaded_config['analysis']['parallel']['enabled'] is True
    os.unlink(config_path)

def test_generate_synthetic_data(sample_config):
    """Test synthetic data generation."""
    data = generate_synthetic_data(sample_config['data']['synthetic_params'])
    
    assert isinstance(data, pd.DataFrame)
    assert len(data) == sample_config['data']['synthetic_params']['n_points']
    assert all(col in data.columns for col in ['datetime', 'x', 'y', 'z'])
    assert pd.api.types.is_datetime64_any_dtype(data['datetime'])
    
    # Test with large dataset to trigger parallel processing
    large_config = sample_config.copy()
    large_config['data']['synthetic_params']['n_points'] = 20000
    large_data = generate_synthetic_data(large_config['data']['synthetic_params'])
    assert len(large_data) == 20000

def test_generate_synthetic_data_chunk():
    """Test synthetic data chunk generation."""
    params = {
        'n_points': 1000,
        'noise_level': 0.1
    }
    
    chunk_data = generate_synthetic_data_chunk(params, 0, 100)
    assert isinstance(chunk_data, pd.DataFrame)
    assert len(chunk_data) == 100
    assert all(col in chunk_data.columns for col in ['x', 'y', 'z'])

def test_load_data(sample_config, sample_data):
    """Test data loading functionality."""
    # Test with synthetic data
    data = load_data(sample_config)
    assert isinstance(data, pd.DataFrame)
    assert all(col in data.columns for col in sample_config['data']['columns_to_keep'])
    
    # Test with CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        sample_config['data']['type'] = 'file'
        sample_config['data']['file_path'] = f.name
        
        loaded_data = load_data(sample_config)
        assert isinstance(loaded_data, pd.DataFrame)
        assert all(col in loaded_data.columns 
                  for col in sample_config['data']['columns_to_keep'])
    
    os.unlink(f.name)

def test_load_data_chunk():
    """Test loading data chunks."""
    data = pd.DataFrame({
        'x': range(100),
        'y': range(100),
        'z': range(100)
    })
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        data.to_csv(f.name, index=False)
        chunk_data = load_data_chunk(f.name, (0, 50), ['x', 'y', 'z'])
        
        assert isinstance(chunk_data, pd.DataFrame)
        assert len(chunk_data) == 50
        assert all(col in chunk_data.columns for col in ['x', 'y', 'z'])
    
    os.unlink(f.name)

def test_save_results(sample_config):
    """Test results saving functionality."""
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
        saved_files = save_results(results, sample_config, 'test_timestamp')
        
        assert os.path.exists(os.path.join(temp_dir, saved_files['metrics_file']))
        assert os.path.exists(os.path.join(temp_dir, saved_files['predictions_file']))
        assert os.path.exists(os.path.join(temp_dir, saved_files['combinations_file']))

def test_save_results_chunk():
    """Test saving results chunks."""
    data = pd.DataFrame({
        'x': range(10),
        'y': range(10)
    })
    
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, 'test_chunk.csv')
        save_results_chunk((filepath, data, 'False'))
        assert os.path.exists(filepath)
        loaded_data = pd.read_csv(filepath)
        assert len(loaded_data) == len(data)

def test_select_optimal_combinations(sample_data):
    """Test optimal combinations selection."""
    combinations = select_optimal_combinations(
        sample_data, 'x', ['y', 'z'], max_combinations=100
    )
    
    assert isinstance(combinations, list)
    assert all(isinstance(combo, tuple) for combo in combinations)
    assert len(combinations) <= 100

def test_evaluate_combination_importance(sample_data):
    """Test combination importance evaluation."""
    importance = evaluate_combination_importance(
        sample_data, 'x', ('y', 'z')
    )
    
    assert isinstance(importance, float)
    assert 0 <= importance <= 1

def test_prepare_output_dirs(sample_config):
    """Test output directory preparation."""
    # Use temporary directory for testing
    with tempfile.TemporaryDirectory() as temp_dir:
        # Modify config to use temporary directory
        test_config = sample_config.copy()
        test_config['output'] = test_config['output'].copy()
        test_config['output']['plots_dir'] = os.path.join(temp_dir, 'plots')
        test_config['output']['results_dir'] = os.path.join(temp_dir, 'results')
        
        plots_dir, results_dir = prepare_output_dirs(test_config)
        
        assert os.path.exists(plots_dir)
        assert os.path.exists(results_dir)

@pytest.fixture(autouse=True)
def reset_logger():
    """Reset logger before and after each test."""
    # Clear any existing handlers
    logger = logging.getLogger('ccm_analysis')
    while logger.handlers:
        handler = logger.handlers.pop()
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.WARNING)
    yield
    # Clean up after test
    while logger.handlers:
        handler = logger.handlers.pop()
        handler.close()
    logger.propagate = True
    logger.setLevel(logging.WARNING)

def test_setup_logger(sample_config, tmp_path):
    """Test logger setup."""
    # Use temporary directory for logs
    log_dir = tmp_path / "logs"
    test_config = {
        'output': {
            'logs_dir': str(log_dir)
        }
    }
    
    # Setup logger
    logger = setup_logger(test_config)
    
    # Verify logger configuration
    assert logger.name == 'ccm_analysis'
    assert logger.level == logging.INFO
    assert len(logger.handlers) == 2
    assert any(isinstance(h, logging.FileHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    
    # Verify log file creation
    log_file = log_dir / "ccm_analysis.log"
    assert log_file.exists()
    
    # Verify handler configurations
    for handler in logger.handlers:
        assert handler.formatter is not None
        assert handler.level == 0  # Default level
        
    # Test logging
    test_message = "Test log message"
    logger.info(test_message)
    
    # Verify message was written to file
    with open(log_file) as f:
        log_content = f.read()
        assert test_message in log_content