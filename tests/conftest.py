# tests/conftest.py
import os
import sys
import pytest
import pandas as pd
import numpy as np
import tempfile
import json
import matplotlib
# Force matplotlib to use non-interactive backend
matplotlib.use('Agg')

# Add the parent directory to the Python path so we can import the src package
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@pytest.fixture
def sample_config():
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
            "train_size_ratio": 0.75
        },
        "output": {
            "plots_dir": "test_plots",
            "results_dir": "test_results",
            "save_predictions": True,
            "filename_prefix": "test_ccm"
        }
    }

@pytest.fixture
def sample_data():
    # Generate simple test data
    t = np.linspace(0, 10*np.pi, 100)
    dates = pd.date_range(start='2023-01-01', periods=100, freq='h')
    data = pd.DataFrame({
        'datetime': dates,
        'x': np.sin(t),
        'y': np.sin(t + np.pi/4),
        'z': np.sin(t + np.pi/2)
    })
    return data

@pytest.fixture
def temp_config_file(sample_config):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(sample_config, f)
        config_path = f.name
    yield config_path
    os.unlink(config_path)
