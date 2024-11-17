# src/analysis.py
"""Analysis module for CCM calculations."""

import os
from typing import Dict, Any
import pandas as pd

from .multivariate_ccm import MultivariateCCM
from .plotting import plot_ccm_results, plot_data_overview
from .utils import save_results

def run_ccm_analysis(data: 'pd.DataFrame', config: Dict[str, Any], 
                    timestamp: str = None) -> Dict[str, Any]:
    """Run CCM analysis for all variables."""
    print("Multivariate CCM Analysis")
    print("========================")
    
    os.makedirs(config['output']['plots_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    print("\nCreating data overview plot...")
    overview_file = plot_data_overview(data, config, config['output']['plots_dir'])
    print(f"Data overview saved as {overview_file}")
    
    results = {}
    columns = config['data']['columns_to_keep']
    
    for target in columns:
        print(f"\nAnalyzing {target} as target variable...")
        mccm = MultivariateCCM(
            data=data,
            columns=columns,
            target=target,
            config=config['analysis']
        )
        target_results = mccm.analyze()
        plot_file = plot_ccm_results(
            target_results, 
            target, 
            config, 
            config['output']['plots_dir'], 
            data=data
        )
        print(f"Plots saved as {plot_file}")
        results[target] = target_results
    
    saved_files = save_results(results, config, timestamp)
    print("\nSaved results files:")
    for file_type, filename in saved_files.items():
        if filename:
            print(f"{file_type}: {filename}")
    
    return results

def print_summary(results: Dict[str, Any]) -> None:
    """Print summary of all analyses."""
    print("\nAnalysis Summary:")
    print("================")
    
    for target, result in results.items():
        print(f"\nTarget: {target}")
        if 'best_combo' in result and result['best_combo'] is not None:
            print("Best predictors:", result['best_combo']['variables'])
            print(f"Correlation: {result['best_combo']['rho']:.3f}")
            print(f"MAE: {result['best_combo']['MAE']:.3f}")
            print(f"RMSE: {result['best_combo']['RMSE']:.3f}")
        else:
            print("No valid results found.")