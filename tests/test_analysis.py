# tests/test_analysis.py
import pytest
import os
import numpy as np
import pandas as pd
from src.analysis import run_ccm_analysis, print_summary

@pytest.fixture(autouse=True)
def setup_test_dirs(sample_config):
    """Setup and cleanup test directories."""
    os.makedirs(sample_config['output']['plots_dir'], exist_ok=True)
    os.makedirs(sample_config['output']['results_dir'], exist_ok=True)
    os.makedirs(sample_config['output']['logs_dir'], exist_ok=True)
    yield
    # Cleanup
    for dir_path in [
        sample_config['output']['plots_dir'],
        sample_config['output']['results_dir'],
        sample_config['output']['logs_dir']
    ]:
        try:
            for file in os.listdir(dir_path):
                os.unlink(os.path.join(dir_path, file))
            os.rmdir(dir_path)
        except OSError:
            pass

def test_run_ccm_analysis(sample_data, sample_config, test_timestamp):
    """Test the main analysis function with sample data."""
    results = run_ccm_analysis(sample_data, sample_config, test_timestamp)
    
    # Basic validation
    assert isinstance(results, dict)
    assert all(target in results for target in sample_config['data']['columns_to_keep'])
    
    # Check result structure for each target
    for target, result in results.items():
        assert 'View' in result
        assert 'best_combo' in result
        assert 'predictions' in result
        
        if result['best_combo'] is not None:
            assert 'variables' in result['best_combo']
            assert 'rho' in result['best_combo']
            assert 'MAE' in result['best_combo']
            assert 'RMSE' in result['best_combo']
            
            # Validate metric ranges
            assert -1 <= result['best_combo']['rho'] <= 1
            assert result['best_combo']['MAE'] >= 0
            assert result['best_combo']['RMSE'] >= 0

def test_run_ccm_analysis_parallel_disabled(sample_data, sample_config, test_timestamp):
    """Test analysis with parallel processing disabled."""
    sample_config['analysis']['parallel']['enabled'] = False
    results = run_ccm_analysis(sample_data, sample_config, test_timestamp)
    assert isinstance(results, dict)
    assert all(target in results for target in sample_config['data']['columns_to_keep'])

def test_run_ccm_analysis_small_dataset(sample_data, sample_config, test_timestamp):
    """Test analysis with a very small dataset."""
    small_data = sample_data.head(10)
    results = run_ccm_analysis(small_data, sample_config, test_timestamp)
    assert isinstance(results, dict)
    assert all(target in results for target in sample_config['data']['columns_to_keep'])

def test_print_summary(capsys, sample_data, sample_config, test_timestamp):
    """Test the summary printing function."""
    results = run_ccm_analysis(sample_data, sample_config, test_timestamp)
    print_summary(results)
    
    captured = capsys.readouterr()
    assert "Analysis Summary" in captured.out
    
    # Check for each target's summary
    for target in sample_config['data']['columns_to_keep']:
        assert f"Target: {target}" in captured.out
        
        if results[target]['best_combo'] is not None:
            assert "Best predictors:" in captured.out
            assert "Correlation:" in captured.out
            assert "MAE:" in captured.out
            assert "RMSE:" in captured.out

def test_run_ccm_analysis_with_missing_data(sample_data, sample_config, test_timestamp):
    """Test analysis with some missing data."""
    # Introduce some NaN values
    sample_data.loc[10:15, 'x'] = np.nan
    results = run_ccm_analysis(sample_data, sample_config, test_timestamp)
    
    assert isinstance(results, dict)
    assert all(target in results for target in sample_config['data']['columns_to_keep'])

def test_run_ccm_analysis_output_files(sample_data, sample_config, test_timestamp):
    """Test that all expected output files are created."""
    run_ccm_analysis(sample_data, sample_config, test_timestamp)
    
    # Check for plot files
    assert os.path.exists(os.path.join(sample_config['output']['plots_dir'], 'data_overview.png'))
    assert os.path.exists(os.path.join(sample_config['output']['plots_dir'], 'analysis_summary.png'))
    
    for target in sample_config['data']['columns_to_keep']:
        assert os.path.exists(os.path.join(
            sample_config['output']['plots_dir'], 
            f'ccm_results_{target}.png'
        ))
    
    # Check for results files
    results_files = os.listdir(sample_config['output']['results_dir'])
    assert any(f.startswith(f"{sample_config['output']['filename_prefix']}_metrics_") for f in results_files)
    assert any(f.startswith(f"{sample_config['output']['filename_prefix']}_predictions_") for f in results_files)
    assert any(f.startswith(f"{sample_config['output']['filename_prefix']}_combinations_") for f in results_files)