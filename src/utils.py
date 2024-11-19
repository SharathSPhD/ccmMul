# src/utils.py
"""Utility functions for CCM analysis."""
import json
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from multiprocessing import Pool, cpu_count
import concurrent.futures
from functools import partial
import logging
from datetime import datetime

def setup_logger(config):
    """Setup logger with file and console handlers."""
    # Create logs directory if it doesn't exist
    os.makedirs(config['output']['logs_dir'], exist_ok=True)
    
    # Get or create logger
    logger = logging.getLogger('ccm_analysis')
    
    # Remove any existing handlers to avoid duplicates
    while logger.handlers:
        logger.handlers.pop()
    
    # Set logging level
    logger.setLevel(logging.INFO)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    # Create handlers
    log_file = os.path.join(config['output']['logs_dir'], 'ccm_analysis.log')
    file_handler = logging.FileHandler(log_file, mode='w')
    console_handler = logging.StreamHandler()
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def load_config(config_path: str = 'config/config.json') -> Dict[str, Any]:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Set default parallel processing configuration if not present
    if 'parallel' not in config.get('analysis', {}):
        config['analysis']['parallel'] = {
            'enabled': True,
            'max_workers': None,
            'chunk_size': 1
        }
    
    return config

def generate_synthetic_data_chunk(params: Dict[str, Any], 
                                start_idx: int, 
                                chunk_size: int) -> pd.DataFrame:
    """Generate a chunk of synthetic data."""
    t = np.linspace(
        start_idx * 2 * np.pi / params['n_points'],
        (start_idx + chunk_size) * 2 * np.pi / params['n_points'],
        chunk_size
    )
    
    x = np.sin(t) + np.random.normal(0, params['noise_level'], chunk_size)
    y = np.sin(t + np.pi/4) + np.random.normal(0, params['noise_level'], chunk_size)
    z = 0.3 * x + 0.5 * y + np.random.normal(0, params['noise_level'], chunk_size)
    
    return pd.DataFrame({'x': x, 'y': y, 'z': z})

def generate_synthetic_data(params: Dict[str, Any]) -> pd.DataFrame:
    """Generate synthetic test data with known relationships using parallel processing."""
    np.random.seed(42)
    n_points = params['n_points']
    
    # Use parallel processing for large datasets
    if n_points > 10000:
        n_workers = cpu_count()
        chunk_size = n_points // n_workers
        chunks = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for i in range(0, n_points, chunk_size):
                size = min(chunk_size, n_points - i)
                futures.append(
                    executor.submit(generate_synthetic_data_chunk, params, i, size)
                )
            
            chunks = [future.result() for future in futures]
        
        data = pd.concat(chunks, ignore_index=True)
    else:
        # Use direct generation for smaller datasets
        t = np.linspace(0, 10*np.pi, n_points)
        x = np.sin(t) + np.random.normal(0, params['noise_level'], n_points)
        y = np.sin(t + np.pi/4) + np.random.normal(0, params['noise_level'], n_points)
        z = 0.3 * x + 0.5 * y + np.random.normal(0, params['noise_level'], n_points)
        data = pd.DataFrame({'x': x, 'y': y, 'z': z})
    
    # Create DataFrame with datetime index
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='h')
    data['datetime'] = dates
    
    return data

def load_data_chunk(file_path: str, 
                   chunk_indices: Tuple[int, int], 
                   columns: list) -> pd.DataFrame:
    """Load a chunk of data from CSV file."""
    skiprows = chunk_indices[0]
    nrows = chunk_indices[1] - chunk_indices[0]
    return pd.read_csv(file_path, skiprows=skiprows, nrows=nrows, usecols=columns)

def load_data(config: Dict[str, Any]) -> pd.DataFrame:
    """Load data based on configuration with parallel processing for large files."""
    if config['data']['type'] == 'synthetic':
        return generate_synthetic_data(config['data']['synthetic_params'])
    
    file_path = config['data']['file_path']
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found: {file_path}")
    
    # Get file size and decide whether to use parallel processing
    file_size = os.path.getsize(file_path)
    large_file_threshold = 100 * 1024 * 1024  # 100 MB
    
    if file_size > large_file_threshold:
        # Count lines in file
        with open(file_path, 'r') as f:
            n_lines = sum(1 for _ in f)
        
        # Prepare columns list
        columns_to_keep = config['data']['columns_to_keep']
        datetime_col = config['data']['datetime_column']
        if datetime_col and datetime_col not in columns_to_keep:
            columns_to_keep = [datetime_col] + columns_to_keep
        
        # Calculate chunks
        n_workers = cpu_count()
        chunk_size = n_lines // n_workers
        chunks = []
        
        # Create chunks of line indices
        chunk_indices = [
            (i, min(i + chunk_size, n_lines)) 
            for i in range(0, n_lines, chunk_size)
        ]
        
        # Load chunks in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            load_chunk_partial = partial(
                load_data_chunk, 
                file_path, 
                columns=columns_to_keep
            )
            chunks = list(executor.map(load_chunk_partial, chunk_indices))
        
        data = pd.concat(chunks, ignore_index=True)
    else:
        # Load normally for smaller files
        data = pd.read_csv(file_path)
    
    # Process datetime column if present
    if config['data']['datetime_column'] in data.columns:
        data[config['data']['datetime_column']] = pd.to_datetime(
            data[config['data']['datetime_column']]
        )
    
    # Keep only specified columns
    columns_to_keep = config['data']['columns_to_keep']
    datetime_col = config['data']['datetime_column']
    if datetime_col in data.columns and datetime_col not in columns_to_keep:
        columns_to_keep = [datetime_col] + columns_to_keep
    
    return data[columns_to_keep]

def save_results_chunk(chunk_data: Tuple[str, pd.DataFrame, str]) -> None:
    """Save a chunk of results to CSV file."""
    filepath, data, index = chunk_data
    data.to_csv(filepath, index=index == 'True')

def save_results(results: Dict[str, Any], 
                config: Dict[str, Any], 
                timestamp: Optional[str] = None) -> Dict[str, str]:
    """Save analysis results to CSV files with parallel processing for large datasets."""
    results_dir = config['output']['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    prefix = config['output']['filename_prefix']
    metrics_file = f"{prefix}_metrics_{timestamp}.csv"
    predictions_file = f"{prefix}_predictions_{timestamp}.csv"
    combinations_file = f"{prefix}_combinations_{timestamp}.csv"
    
    metrics_data = []
    predictions_data = []
    combinations_data = []
    
    for target, result in results.items():
        if 'best_combo' in result and result['best_combo'] is not None:
            metrics_data.append({
                'target': target,
                'best_predictors': ' & '.join(result['best_combo']['variables']),
                'rho': result['best_combo']['rho'],
                'MAE': result['best_combo']['MAE'],
                'RMSE': result['best_combo']['RMSE']
            })
            
            if 'predictions' in result and result['predictions'] is not None:
                predictions_data.append(pd.DataFrame({
                    'target': target,
                    'time_index': result['predictions']['time_indices'],
                    'actual': result['predictions']['actual'],
                    'predicted': result['predictions']['predicted']
                }))
            
            if 'View' in result and result['View'] is not None:
                view_df = result['View'].copy()
                view_df['target'] = target
                view_df['variables'] = view_df['variables'].apply(' & '.join)
                combinations_data.append(view_df)
    
    # Save files in parallel for large datasets
    save_tasks = []
    
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        save_tasks.append((
            os.path.join(results_dir, metrics_file),
            metrics_df,
            'False'
        ))
    
    if predictions_data and config['output']['save_predictions']:
        predictions_df = pd.concat(predictions_data, ignore_index=True)
        save_tasks.append((
            os.path.join(results_dir, predictions_file),
            predictions_df,
            'False'
        ))
    
    if combinations_data:
        combinations_df = pd.concat(combinations_data, ignore_index=True)
        save_tasks.append((
            os.path.join(results_dir, combinations_file),
            combinations_df,
            'False'
        ))
    
    # Use parallel processing for saving if there are multiple files
    if len(save_tasks) > 1:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            executor.map(save_results_chunk, save_tasks)
    else:
        # Save sequentially for single file
        for task in save_tasks:
            save_results_chunk(task)
    
    return {
        'metrics_file': metrics_file if metrics_data else None,
        'predictions_file': predictions_file if predictions_data else None,
        'combinations_file': combinations_file if combinations_data else None
    }

def prepare_output_dirs(config: Dict[str, Any]) -> Tuple[str, str]:
    """Prepare output directories."""
    os.makedirs(config['output']['plots_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    return config['output']['plots_dir'], config['output']['results_dir']

# Add to utils.py

def select_optimal_combinations(data: pd.DataFrame, 
                              target: str,
                              predictors: list,
                              max_combinations: int = 10000) -> list:
    """
    Select most promising variable combinations using correlation analysis.
    
    Strategy:
    1. Calculate individual correlations with target
    2. Use correlation matrix to find complementary variables
    3. Build combinations prioritizing:
       - High correlation with target
       - Low correlation between predictors
       - Progressive addition of variables based on added value
    """
    # Calculate correlation matrix
    corr_matrix = data[predictors + [target]].corr()
    
    # Get correlations with target
    target_corr = abs(corr_matrix[target].drop(target))
    sorted_predictors = target_corr.sort_values(ascending=False).index.tolist()
    
    # Initialize combinations list
    selected_combinations = []
    
    # Add individual predictors (they might be important)
    selected_combinations.extend([(p,) for p in sorted_predictors[:5]])
    
    # Build pairs based on complementary information
    for i, p1 in enumerate(sorted_predictors):
        for p2 in sorted_predictors[i+1:]:
            # Check if predictors are not too correlated with each other
            if abs(corr_matrix.loc[p1, p2]) < 0.7:
                selected_combinations.append((p1, p2))
                
        if len(selected_combinations) >= max_combinations/2:
            break
    
    # Add triples for top correlated variables that aren't highly correlated
    top_predictors = sorted_predictors[:10]  # Limit to top 10 for triples
    for i, p1 in enumerate(top_predictors):
        for j, p2 in enumerate(top_predictors[i+1:], i+1):
            if abs(corr_matrix.loc[p1, p2]) >= 0.7:
                continue
            for p3 in top_predictors[j+1:]:
                if (abs(corr_matrix.loc[p1, p3]) < 0.7 and 
                    abs(corr_matrix.loc[p2, p3]) < 0.7):
                    selected_combinations.append((p1, p2, p3))
                    
        if len(selected_combinations) >= max_combinations:
            break
    
    return selected_combinations

def evaluate_combination_importance(data: pd.DataFrame,
                                 target: str,
                                 combination: tuple) -> float:
    """
    Evaluate importance of a variable combination using multiple criteria.
    """
    X = data[list(combination)]
    y = data[target]
    
    # Calculate combined correlation score
    corr_with_target = abs(data[list(combination)].corrwith(y))
    mean_corr = corr_with_target.mean()
    
    # Calculate inter-predictor correlations
    if len(combination) > 1:
        predictor_corr = abs(X.corr()).values
        np.fill_diagonal(predictor_corr, 0)
        redundancy = predictor_corr.mean()
        
        # Penalize for redundancy
        importance = mean_corr * (1 - redundancy)
    else:
        importance = mean_corr
        
    return importance