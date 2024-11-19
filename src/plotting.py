# src/plotting.py
"""Plotting functions for CCM analysis."""

import os
from typing import Dict, Any, Optional, List, Tuple
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import concurrent.futures
from multiprocessing import cpu_count

def plot_data_chunk(data: pd.DataFrame, 
                   columns: List[str], 
                   chunk_indices: Tuple[int, int],
                   datetime_col: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Plot a chunk of data and return the computed values."""
    results = {}
    chunk_data = data.iloc[chunk_indices[0]:chunk_indices[1]]
    
    for col in columns:
        if datetime_col:
            x = chunk_data[datetime_col].values
            y = chunk_data[col].values
        else:
            x = chunk_data.index.values
            y = chunk_data[col].values
        
        results[col] = (x, y)
    
    return results

def plot_ccm_results(results: Dict[str, Any], 
                    target: str, 
                    config: Dict[str, Any], 
                    save_dir: str, 
                    data: Optional[pd.DataFrame] = None) -> str:
    """Plot CCM analysis results with improved readability."""
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), dpi=100)
    
    if results['View'] is not None and len(results['View']) > 0:
        valid_results = results['View'].dropna(subset=['rho'])
        if len(valid_results) > 0:
            # Get top 20 results for clearer visualization
            top_results = valid_results.nlargest(20, 'rho')
            rhos = top_results['rho']
            combo_labels = [' & '.join(combo) for combo in top_results['variables']]
            
            # Create bar plot with improved formatting
            bars = ax1.bar(range(len(rhos)), rhos)
            ax1.set_xticks(range(len(combo_labels)))
            ax1.set_xticklabels(combo_labels, rotation=45, ha='right', fontsize=8)
            ax1.set_xlabel('Predictor Combinations (Top 20)', fontsize=10)
            ax1.set_ylabel('Correlation (ρ)', fontsize=10)
            ax1.set_title(f'Multivariate CCM Results for {target}', fontsize=12, pad=20)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom', fontsize=8)
    
    # Time series plot
    if 'predictions' in results and results['predictions'] is not None:
        pred_data = results['predictions']
        datetime_col = config['data'].get('datetime_column')
        
        if datetime_col and data is not None and datetime_col in data.columns:
            time_indices = pred_data['time_indices']
            dates = data[datetime_col].iloc[time_indices].values
            
            # Plot with improved styling
            ax2.plot(dates, pred_data['actual'], 'b-', 
                    label='Actual', alpha=0.7, linewidth=1.5)
            ax2.plot(dates, pred_data['predicted'], 'r--', 
                    label='Predicted', alpha=0.7, linewidth=1.5)
            
            split_date = dates[0]
            ax2.axvline(x=split_date, color='g', linestyle=':', 
                       label='Test Period Start', alpha=0.5)
            
            plt.setp(ax2.xaxis.get_majorticklabels(), 
                    rotation=45, ha='right', fontsize=8)
            ax2.set_xlabel('Date', fontsize=10)
        else:
            # Similar plotting for non-datetime index
            time_indices = np.arange(len(pred_data['actual']))
            ax2.plot(time_indices, pred_data['actual'], 'b-', 
                    label='Actual', alpha=0.7, linewidth=1.5)
            ax2.plot(time_indices, pred_data['predicted'], 'r--', 
                    label='Predicted', alpha=0.7, linewidth=1.5)
            ax2.axvline(x=0, color='g', linestyle=':', 
                       label='Test Period Start', alpha=0.5)
            ax2.set_xlabel('Time Index', fontsize=10)
        
        ax2.set_ylabel(f'{target} Value', fontsize=10)
        best_combo = ' & '.join(results['best_combo']['variables'])
        ax2.set_title(f'Best Prediction vs Actual for {target}\nPredictors: {best_combo}', 
                     fontsize=12, pad=20)
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
    
    # Improve overall layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save plot
    filename = f'ccm_results_{target}.png'
    plt.savefig(os.path.join(save_dir, filename), 
                bbox_inches='tight', 
                dpi=300)
    plt.close()
    
    return filename

def plot_data_overview(data: pd.DataFrame, 
                      config: Dict[str, Any], 
                      save_dir: str) -> str:
    """
    Plot overview of input time series data with parallel processing for large datasets.
    """
    os.makedirs(save_dir, exist_ok=True)
    columns = config['data']['columns_to_keep']
    datetime_col = config['data'].get('datetime_column')
    
    # Determine if parallel processing should be used based on data size
    large_dataset = len(data) * len(columns) > 1000000  # Threshold for large datasets
    
    if large_dataset:
        # Process data in chunks using parallel processing
        n_workers = min(cpu_count(), len(columns))
        chunk_size = len(data) // n_workers
        chunk_indices = [
            (i, min(i + chunk_size, len(data))) 
            for i in range(0, len(data), chunk_size)
        ]
        
        # Compute plot data in parallel
        plot_data = {}
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = []
            for indices in chunk_indices:
                future = executor.submit(
                    plot_data_chunk,
                    data,
                    columns,
                    indices,
                    datetime_col
                )
                futures.append(future)
            
            # Combine results from all chunks
            for future in concurrent.futures.as_completed(futures):
                chunk_results = future.result()
                for col in chunk_results:
                    if col not in plot_data:
                        plot_data[col] = []
                    plot_data[col].append(chunk_results[col])
    
        # Create optimized figure
        plt.figure(figsize=(12, 4 * len(columns)), dpi=100)
        
        for i, col in enumerate(columns, 1):
            plt.subplot(len(columns), 1, i)
            
            # Combine chunks efficiently
            x_data = np.concatenate([chunk[0] for chunk in plot_data[col]])
            y_data = np.concatenate([chunk[1] for chunk in plot_data[col]])
            
            # Plot with memory optimization
            if datetime_col:
                plt.plot_date(x_data, y_data, '-', label=col, lw=1)
                plt.setp(plt.gca().xaxis.get_majorticklabels(), 
                        rotation=45, ha='right')
                plt.xlabel('Date')
            else:
                plt.plot(x_data, y_data, '-', label=col, lw=1)
                plt.xlabel('Time Index')
            
            plt.ylabel(col)
            plt.title(f'Time Series: {col}')
            plt.grid(True)
            plt.legend()
    
    else:
        # Direct plotting for smaller datasets
        plt.figure(figsize=(12, 4 * len(columns)), dpi=100)
        
        for i, col in enumerate(columns, 1):
            plt.subplot(len(columns), 1, i)
            
            if datetime_col:
                plt.plot(data[datetime_col], data[col], '-', label=col, lw=1)
                plt.setp(plt.gca().xaxis.get_majorticklabels(), 
                        rotation=45, ha='right')
                plt.xlabel('Date')
            else:
                plt.plot(data.index, data[col], '-', label=col, lw=1)
                plt.xlabel('Time Index')
            
            plt.ylabel(col)
            plt.title(f'Time Series: {col}')
            plt.grid(True)
            plt.legend()
    
    plt.tight_layout()
    
    # Save plot with standard settings
    filename = 'data_overview.png'
    plt.savefig(os.path.join(save_dir, filename),
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    return filename

def create_summary_plots(results: Dict[str, Any], 
                        config: Dict[str, Any], 
                        save_dir: str) -> str:
    """
    Create summary plots for all analyses with performance metrics.
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Collect metrics across all targets
    metrics_data = {
        'target': [],
        'rho': [],
        'MAE': [],
        'RMSE': [],
        'n_predictors': []
    }
    
    for target, result in results.items():
        if 'best_combo' in result and result['best_combo'] is not None:
            metrics_data['target'].append(target)
            metrics_data['rho'].append(result['best_combo']['rho'])
            metrics_data['MAE'].append(result['best_combo']['MAE'])
            metrics_data['RMSE'].append(result['best_combo']['RMSE'])
            metrics_data['n_predictors'].append(len(result['best_combo']['variables']))
    
    if not metrics_data['target']:
        return None
    
    # Create summary plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), dpi=100)
    fig.suptitle('Analysis Summary Across All Targets', fontsize=14, y=0.95)
    
    # Plot 1: Correlation comparison
    x_pos = np.arange(len(metrics_data['target']))
    axes[0, 0].bar(x_pos, metrics_data['rho'])
    axes[0, 0].set_title('Best Correlation by Target')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(metrics_data['target'], rotation=45, ha='right')
    axes[0, 0].set_ylabel('Correlation (ρ)')
    axes[0, 0].set_ylim(0, 1.0)  # Correlation ranges from 0 to 1
    
    # Add value labels on bars
    for i, v in enumerate(metrics_data['rho']):
        axes[0, 0].text(i, v + 0.01, f'{v:.3f}', 
                       ha='center', va='bottom')
    
    # Plot 2: Error metrics comparison
    x = np.arange(len(metrics_data['target']))
    width = 0.35
    axes[0, 1].bar(x - width/2, metrics_data['MAE'], width, label='MAE')
    axes[0, 1].bar(x + width/2, metrics_data['RMSE'], width, label='RMSE')
    axes[0, 1].set_title('Error Metrics by Target')
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metrics_data['target'], rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].set_ylabel('Error Value')
    
    # Add value labels on bars
    for i, (mae, rmse) in enumerate(zip(metrics_data['MAE'], metrics_data['RMSE'])):
        axes[0, 1].text(i - width/2, mae + 0.01, f'{mae:.3f}', 
                       ha='center', va='bottom')
        axes[0, 1].text(i + width/2, rmse + 0.01, f'{rmse:.3f}', 
                       ha='center', va='bottom')
    
    # Plot 3: Number of predictors histogram
    max_predictors = max(metrics_data['n_predictors'])
    bins = np.arange(0.5, max_predictors + 1.5, 1)
    axes[1, 0].hist(metrics_data['n_predictors'], bins=bins, 
                    rwidth=0.8, align='mid')
    axes[1, 0].set_title('Distribution of Predictor Count')
    axes[1, 0].set_xlabel('Number of Predictors')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_xticks(range(1, max_predictors + 1))
    
    # Plot 4: Correlation vs Number of predictors scatter
    scatter = axes[1, 1].scatter(metrics_data['n_predictors'], 
                                metrics_data['rho'], 
                                alpha=0.6)
    axes[1, 1].set_title('Correlation vs Number of Predictors')
    axes[1, 1].set_xlabel('Number of Predictors')
    axes[1, 1].set_ylabel('Correlation (ρ)')
    axes[1, 1].set_ylim(0, 1.0)
    axes[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Add target labels to scatter points
    for i, txt in enumerate(metrics_data['target']):
        axes[1, 1].annotate(txt, 
                           (metrics_data['n_predictors'][i], 
                            metrics_data['rho'][i]),
                           xytext=(5, 5), 
                           textcoords='offset points')
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Add grid to all plots
    for ax in axes.flat:
        ax.grid(True, linestyle='--', alpha=0.7)
    
    # Save plot with standard settings
    filename = 'analysis_summary.png'
    plt.savefig(os.path.join(save_dir, filename),
                bbox_inches='tight',
                dpi=300)
    plt.close()
    
    return filename