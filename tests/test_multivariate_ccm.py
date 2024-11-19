# tests/test_multivariate_ccm.py
import pytest
import pandas as pd
import numpy as np
from src.multivariate_ccm import MultivariateCCM
import time

@pytest.fixture
def mccm_instance(sample_data):
    """Create a basic MultivariateCCM instance for testing."""
    return MultivariateCCM(
        data=sample_data,
        columns=['x', 'y', 'z'],
        target='x'
    )

def test_initialization(sample_data):
    """Test MultivariateCCM initialization with various configurations."""
    # Test basic initialization
    mccm = MultivariateCCM(
        data=sample_data,
        columns=['x', 'y', 'z'],
        target='x'
    )
    assert mccm.target == 'x'
    assert set(mccm.predictors) == {'y', 'z'}
    assert mccm.config['embedding_dimension'] == 3
    
    # Test with datetime column
    data_with_datetime = sample_data.copy()
    data_with_datetime['datetime'] = pd.date_range('2023-01-01', periods=len(sample_data))
    mccm = MultivariateCCM(
        data=data_with_datetime,
        columns=['x', 'y', 'z'],  # Explicitly exclude datetime
        target='x'
    )
    # Check datetime handling
    assert hasattr(mccm, 'datetime_col')
    assert mccm.datetime_col is not None
    # Check that datetime is preserved in original data
    assert 'datetime' in mccm.data.columns
    # Check that datetime is NOT in scaled data
    assert 'datetime' not in mccm.data_scaled.columns
    # Check that only specified columns are used for scaling
    assert set(mccm.data_scaled.columns) == {'x', 'y', 'z'}

def test_create_embedding(mccm_instance):
    """Test the time-delay embedding creation."""
    data = pd.DataFrame({'x': range(10), 'y': range(10)})
    embedded_data, valid_indices = mccm_instance.create_embedding(
        data, ['x', 'y'], E=2, tau=-1
    )
    
    assert set(embedded_data.columns) == {'x', 'x_t1', 'y', 'y_t1'}
    assert len(embedded_data) == len(data) - 1
    assert all(idx in data.index for idx in valid_indices)

def test_find_nearest_neighbors(mccm_instance):
    """Test nearest neighbor finding functionality."""
    # Create simple test data
    embedding_matrix = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
    lib_indices = np.array([0, 1])
    pred_indices = np.array([2, 3])
    
    distances, indices = mccm_instance.find_nearest_neighbors(
        embedding_matrix, lib_indices, pred_indices, k=2
    )
    
    assert distances is not None
    assert indices is not None
    assert distances.shape == (2, 2)
    assert indices.shape == (2, 2)

def test_make_predictions(mccm_instance):
    """Test prediction generation from neighbors."""
    distances = np.array([[1.0, 2.0], [1.0, 2.0]])
    indices = np.array([[0, 1], [1, 2]])
    target_array = np.array([1.0, 2.0, 3.0])
    
    predictions = mccm_instance.make_predictions((distances, indices), target_array)
    
    assert len(predictions) == 2
    assert all(isinstance(p, float) for p in predictions)

def test_evaluate_combination(mccm_instance):
    """Test combination evaluation functionality."""
    metrics, predictions, actuals, time_indices = mccm_instance.evaluate_combination(['y'])
    
    assert isinstance(metrics, dict)
    assert all(key in metrics for key in ['rho', 'MAE', 'RMSE'])
    assert all(not np.isnan(metrics[key]) for key in metrics)
    
    if predictions is not None:
        assert len(predictions) == len(actuals)
        assert len(time_indices) == len(predictions)

def test_full_analysis(sample_data):
    """Test complete analysis workflow."""
    mccm = MultivariateCCM(
        data=sample_data,
        columns=['x', 'y', 'z'],
        target='x',
        config={
            'embedding_dimension': 2,
            'tau': -1,
            'train_size_ratio': 0.75,
            'num_surrogates': 10,
            'exclusion_radius': 0,
            'parallel': {
                'enabled': True,
                'max_workers': 2,
                'chunk_size': 1
            }
        }
    )
    
    results = mccm.analyze()
    
    assert 'View' in results
    assert 'best_combo' in results
    assert 'predictions' in results
    
    if results['best_combo'] is not None:
        assert all(key in results['best_combo'] for key in ['variables', 'rho', 'MAE', 'RMSE'])

def test_parallel_vs_serial_execution(sample_data):
    """Compare results between parallel and serial execution."""
    # Configure parallel instance
    mccm_parallel = MultivariateCCM(
        data=sample_data,
        columns=['x', 'y', 'z'],
        target='x',
        config={
            'embedding_dimension': 2,
            'tau': -1,
            'train_size_ratio': 0.75,
            'parallel': {'enabled': True, 'max_workers': 2}
        }
    )
    
    # Configure serial instance
    mccm_serial = MultivariateCCM(
        data=sample_data,
        columns=['x', 'y', 'z'],
        target='x',
        config={
            'embedding_dimension': 2,
            'tau': -1,
            'train_size_ratio': 0.75,
            'parallel': {'enabled': False}
        }
    )
    
    parallel_results = mccm_parallel.analyze()
    serial_results = mccm_serial.analyze()
    
    # Compare key metrics
    if parallel_results['best_combo'] and serial_results['best_combo']:
        assert np.isclose(
            parallel_results['best_combo']['rho'],
            serial_results['best_combo']['rho'],
            rtol=1e-10
        )

def test_error_handling(sample_data):
    """Test error handling in various scenarios."""
    # Test with invalid target
    with pytest.raises(ValueError):
        MultivariateCCM(sample_data, ['x', 'y', 'z'], 'invalid_target')
    
    # Test with empty data
    with pytest.raises(Exception):
        MultivariateCCM(pd.DataFrame(), ['x', 'y', 'z'], 'x')
    
    # Test with all NaN column
    invalid_data = sample_data.copy()
    invalid_data['x'] = np.nan
    mccm = MultivariateCCM(invalid_data, ['x', 'y', 'z'], 'y')
    results = mccm.analyze()
    assert results is not None

def test_compute_metrics(mccm_instance):
    """Test metric computation functionality."""
    actual = np.array([1.0, 2.0, 3.0, 4.0])
    predicted = np.array([1.1, 2.1, 3.1, 4.1])
    
    metrics = mccm_instance.compute_metrics(actual, predicted)
    
    assert 'rho' in metrics
    assert 'MAE' in metrics
    assert 'RMSE' in metrics
    assert all(not np.isnan(v) for v in metrics.values())
    assert metrics['rho'] <= 1.0 and metrics['rho'] >= -1.0
    assert metrics['MAE'] >= 0.0
    assert metrics['RMSE'] >= 0.0

def test_with_missing_data(sample_data):
    """Test analysis with missing data handling."""
    # Introduce some missing values
    data_with_nans = sample_data.copy()
    data_with_nans.loc[10:15, 'x'] = np.nan
    
    mccm = MultivariateCCM(
        data=data_with_nans,
        columns=['x', 'y', 'z'],
        target='y'  # Use 'y' as target since 'x' has NaNs
    )
    
    results = mccm.analyze()
    assert results is not None
    assert 'View' in results
    assert 'best_combo' in results