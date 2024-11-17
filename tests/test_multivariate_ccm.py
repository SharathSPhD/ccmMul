# tests/test_multivariate_ccm.py

import pytest
from src.multivariate_ccm import MultivariateCCM
import pandas as pd
import numpy as np

@pytest.fixture
def sample_data():
    # Generate simple test data
    t = np.linspace(0, 10*np.pi, 100)
    data = pd.DataFrame({
        'x': np.sin(t),
        'y': np.sin(t + np.pi/4),
        'z': np.sin(t + np.pi/2)
    })
    return data

def test_multivariate_ccm_initialization(sample_data):
    mccm = MultivariateCCM(
        data=sample_data,
        columns=['x', 'y', 'z'],
        target='x'
    )
    
    assert mccm.target == 'x'
    assert set(mccm.predictors) == {'y', 'z'}
    assert mccm.config['embedding_dimension'] == 3

def test_create_embedding(sample_data):
    mccm = MultivariateCCM(
        data=sample_data,
        columns=['x', 'y'],
        target='x'
    )
    
    embedded_data, valid_indices = mccm.create_embedding(
        mccm.data_scaled,
        ['y'],
        E=2,
        tau=1
    )
    
    assert isinstance(embedded_data, pd.DataFrame)
    assert len(embedded_data.columns) == 2  # y and y_t1
    assert len(valid_indices) == len(embedded_data)

def test_full_analysis(sample_data):
    mccm = MultivariateCCM(
        data=sample_data,
        columns=['x', 'y', 'z'],
        target='x'
    )
    
    results = mccm.analyze()
    
    assert 'View' in results
    assert 'best_combo' in results
    assert 'predictions' in results
    assert results['best_combo'] is not None
    assert 'rho' in results['best_combo']