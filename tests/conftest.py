# tests/conftest.py
import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import matplotlib
import logging
from datetime import datetime

# Force matplotlib to use non-interactive backend
matplotlib.use('Agg')

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_config():
    """Provide a sample configuration for testing."""
    return {
        "data": {
            "type": "synthetic",
            "file_path": "data/sample_data.csv",
            "datetime_column": "datetime",
            "columns_to_keep": ["x", "y", "z"],
            "synthetic_params": {
                "n_points": 100,
                "noise_level": 0.1
            }
        },
        "analysis": {
            "embedding_dimension": 3,
            "tau": -1,
            "train_size_ratio": 0.75,
            "num_surrogates": 100,
            "exclusion_radius": 0,
            "parallel": {
                "enabled": True,
                "max_workers": 2,
                "chunk_size": 1
            },
            "combinations": {
                "max_predictors": 3,
                "max_combinations": 100,
                "correlation_threshold": 0.7
            }
        },
        "output": {
            "plots_dir": "test_plots",
            "results_dir": "test_results",
            "logs_dir": "test_logs",
            "save_predictions": True,
            "filename_prefix": "test_ccm"
        }
    }

@pytest.fixture
def sample_data():
    """Generate sample time series data for testing."""
    t = np.linspace(0, 10*np.pi, 100)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'datetime': dates,
        'x': np.sin(t) + np.random.normal(0, 0.1, 100),
        'y': np.sin(t + np.pi/4) + np.random.normal(0, 0.1, 100),
        'z': np.sin(t + np.pi/2) + np.random.normal(0, 0.1, 100)
    })
    return data

@pytest.fixture
def temp_config_file(sample_config):
    """Create a temporary config file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config, f)
        config_path = f.name
    yield config_path
    os.unlink(config_path)

@pytest.fixture
def test_logger(sample_config):
    """Create a test logger instance."""
    os.makedirs(sample_config['output']['logs_dir'], exist_ok=True)
    log_file = os.path.join(sample_config['output']['logs_dir'], 'test_ccm_analysis.log')
    
    logger = logging.getLogger('test_ccm_analysis')
    logger.setLevel(logging.INFO)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Add handlers
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

@pytest.fixture
def test_timestamp():
    """Provide a fixed timestamp for testing."""
    return datetime.now().strftime('%Y%m%d_%H%M%S')