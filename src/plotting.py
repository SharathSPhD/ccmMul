import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

def plot_ccm_results(results, target, config, save_dir, data=None):
    """
    Plot and save CCM analysis results.
    
    Parameters:
    -----------
    results : dict
        Analysis results for the target variable
    target : str
        Name of target variable
    config : dict
        Configuration dictionary
    save_dir : str
        Directory to save plots
    data : pandas.DataFrame, optional
        Original data with datetime column if available
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot 1: Bar plot of correlations
    if results['View'] is not None and len(results['View']) > 0:
        valid_results = results['View'].dropna(subset=['rho'])
        if len(valid_results) > 0:
            rhos = valid_results['rho']
            combo_labels = [' & '.join(combo) for combo in valid_results['variables']]
            
            bars = ax1.bar(range(len(rhos)), rhos)
            ax1.set_xticks(range(len(combo_labels)))
            ax1.set_xticklabels(combo_labels, rotation=45, ha='right')
            ax1.set_xlabel('Predictor Combinations')
            ax1.set_ylabel('Correlation (ρ)')
            ax1.set_title(f'Multivariate CCM Results for {target}')
            ax1.grid(True)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}',
                        ha='center', va='bottom')
    
    # Plot 2: Time series comparison
    if 'predictions' in results and results['predictions'] is not None:
        pred_data = results['predictions']
        
        # Use datetime for x-axis if available
        datetime_col = config['data'].get('datetime_column')
        if datetime_col and data is not None and datetime_col in data.columns:
            # Get datetime values for the prediction period
            time_indices = pred_data['time_indices']
            dates = data[datetime_col].iloc[time_indices].values
            
            # Plot with dates
            ax2.plot(dates, pred_data['actual'], 'b-', 
                    label='Actual', alpha=0.6, linewidth=2)
            ax2.plot(dates, pred_data['predicted'], 'r--', 
                    label='Predicted', alpha=0.8, linewidth=2)
            
            # Add train/test split line
            split_date = dates[0]  # First prediction date
            ax2.axvline(x=split_date, color='g', linestyle=':', label='Test Period Start')
            
            # Format x-axis
            plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
            ax2.set_xlabel('Date')
            
        else:
            # Use index for x-axis if no datetime
            time_indices = np.arange(len(pred_data['actual']))
            
            ax2.plot(time_indices, pred_data['actual'], 'b-', 
                    label='Actual', alpha=0.6, linewidth=2)
            ax2.plot(time_indices, pred_data['predicted'], 'r--', 
                    label='Predicted', alpha=0.8, linewidth=2)
            
            # Add train/test split line
            ax2.axvline(x=0, color='g', linestyle=':', label='Test Period Start')
            ax2.set_xlabel('Time Index')
        
        ax2.set_ylabel(f'{target} Value')
        best_combo = ' & '.join(results['best_combo']['variables'])
        ax2.set_title(f'Best Prediction vs Actual for {target}\nPredictors: {best_combo}')
        ax2.legend()
        ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    filename = f'ccm_results_{target}.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    
    return filename

def plot_data_overview(data, config, save_dir):
    """
    Plot overview of input time series data.
    
    Parameters:
    -----------
    data : pandas.DataFrame
        Input data with time series
    config : dict
        Configuration dictionary
    save_dir : str
        Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    columns = config['data']['columns_to_keep']
    datetime_col = config['data'].get('datetime_column')
    
    plt.figure(figsize=(12, 4 * len(columns)))
    
    for i, col in enumerate(columns, 1):
        plt.subplot(len(columns), 1, i)
        
        if datetime_col and datetime_col in data.columns:
            plt.plot(data[datetime_col], data[col], label=col)
            plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=45, ha='right')
            plt.xlabel('Date')
        else:
            plt.plot(data.index, data[col], label=col)
            plt.xlabel('Time Index')
            
        plt.ylabel(col)
        plt.title(f'Time Series: {col}')
        plt.grid(True)
        plt.legend()
    
    plt.tight_layout()
    
    # Save plot
    filename = 'data_overview.png'
    plt.savefig(os.path.join(save_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()
    
    return filename