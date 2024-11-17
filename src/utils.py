# src/utils.py
"""Utility functions for CCM analysis."""
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional

def load_config(config_path: str = 'config/config.json') -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def generate_synthetic_data(params):
    """Generate synthetic test data with known relationships."""
    np.random.seed(42)
    n_points = params['n_points']
    noise_level = params['noise_level']
    
    t = np.linspace(0, 10*np.pi, n_points)
    x = np.sin(t) + np.random.normal(0, noise_level, n_points)
    y = np.sin(t + np.pi/4) + np.random.normal(0, noise_level, n_points)
    z = 0.3 * x + 0.5 * y + np.random.normal(0, noise_level, n_points)
    
    # Create DataFrame with datetime index
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='h')
    data = pd.DataFrame({
        'datetime': dates,
        'x': x,
        'y': y,
        'z': z
    })
    
    return data

def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load data based on configuration."""
    if config['data']['type'] == 'synthetic':
        data = generate_synthetic_data(config['data']['synthetic_params'])
    else:
        file_path = config['data']['file_path']
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        data = pd.read_csv(file_path)
        if config['data']['datetime_column'] in data.columns:
            data[config['data']['datetime_column']] = pd.to_datetime(
                data[config['data']['datetime_column']]
            )
    
    # Keep only specified columns
    columns_to_keep = config['data']['columns_to_keep']
    datetime_col = config['data']['datetime_column']
    
    if datetime_col in data.columns and datetime_col not in columns_to_keep:
        columns_to_keep = [datetime_col] + columns_to_keep
    
    data = data[columns_to_keep]
    
    return data

def save_results(results: Dict[str, Any], config: Dict[str, Any], 
                timestamp: Optional[str] = None) -> Dict[str, str]:
    """
    Save analysis results to CSV files.
    
    Parameters:
        results: dict of analysis results for each target
        config: configuration dictionary
        timestamp: optional timestamp for file naming
    """
    import pandas as pd
    import os
    from datetime import datetime
    
    # Create results directory
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Prepare filenames
    prefix = config['output']['filename_prefix']
    metrics_file = f"{prefix}_metrics_{timestamp}.csv"
    predictions_file = f"{prefix}_predictions_{timestamp}.csv"
    combinations_file = f"{prefix}_combinations_{timestamp}.csv"
    
    # Collect metrics for all targets
    metrics_data = []
    predictions_data = []
    combinations_data = []
    
    for target, result in results.items():
        if 'best_combo' in result and result['best_combo'] is not None:
            # Save metrics
            metrics_row = {
                'target': target,
                'best_predictors': ' & '.join(result['best_combo']['variables']),
                'rho': result['best_combo']['rho'],
                'MAE': result['best_combo']['MAE'],
                'RMSE': result['best_combo']['RMSE']
            }
            metrics_data.append(metrics_row)
            
            # Save predictions if available
            if 'predictions' in result and result['predictions'] is not None:
                pred_df = pd.DataFrame({
                    'target': target,
                    'time_index': result['predictions']['time_indices'],
                    'actual': result['predictions']['actual'],
                    'predicted': result['predictions']['predicted']
                })
                predictions_data.append(pred_df)
            
            # Save all combinations results
            if 'View' in result and result['View'] is not None:
                view_df = result['View'].copy()
                view_df['target'] = target
                view_df['variables'] = view_df['variables'].apply(' & '.join)
                combinations_data.append(view_df)
    
    # Save to CSV files
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_csv(os.path.join(results_dir, metrics_file), index=False)
        print(f"Saved metrics to {metrics_file}")
    
    if predictions_data and config['output']['save_predictions']:
        predictions_df = pd.concat(predictions_data, ignore_index=True)
        predictions_df.to_csv(os.path.join(results_dir, predictions_file), index=False)
        print(f"Saved predictions to {predictions_file}")
    
    if combinations_data:
        combinations_df = pd.concat(combinations_data, ignore_index=True)
        combinations_df.to_csv(os.path.join(results_dir, combinations_file), index=False)
        print(f"Saved combination results to {combinations_file}")
    
    return {
        'metrics_file': metrics_file,
        'predictions_file': predictions_file if predictions_data else None,
        'combinations_file': combinations_file
    }

def prepare_output_dirs(config):
    """Prepare output directories."""
    os.makedirs(config['output']['plots_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    return config['output']['plots_dir'], config['output']['results_dir']