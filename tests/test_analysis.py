# tests/test_analysis.py
import pytest
import os
from src.analysis import run_ccm_analysis, print_summary

@pytest.fixture(autouse=True)
def setup_test_dirs(sample_config):
    """Automatically create test directories before each test."""
    os.makedirs(sample_config['output']['plots_dir'], exist_ok=True)
    os.makedirs(sample_config['output']['results_dir'], exist_ok=True)
    yield
    # Optional: Clean up after tests
    try:
        os.rmdir(sample_config['output']['plots_dir'])
        os.rmdir(sample_config['output']['results_dir'])
    except OSError:
        pass  # Directory not empty or already deleted

def test_run_ccm_analysis(sample_data, sample_config):
    results = run_ccm_analysis(sample_data, sample_config)
    
    assert isinstance(results, dict)
    assert all(target in results for target in sample_config['data']['columns_to_keep'])
    assert all('best_combo' in result for result in results.values())

def test_print_summary(capsys, sample_data, sample_config):
    results = run_ccm_analysis(sample_data, sample_config)
    print_summary(results)
    
    captured = capsys.readouterr()
    assert "Analysis Summary" in captured.out
    assert "Target:" in captured.out