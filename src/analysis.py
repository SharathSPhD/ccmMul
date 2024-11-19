# src/analysis.py
"""Analysis module for CCM calculations."""

import os
import time
from typing import Dict, Any
import pandas as pd
from multiprocessing import cpu_count

from .multivariate_ccm import MultivariateCCM
from .plotting import plot_ccm_results, plot_data_overview, create_summary_plots
from .utils import save_results

def run_ccm_analysis(data: pd.DataFrame, config: Dict[str, Any], 
                    timestamp: str = None) -> Dict[str, Any]:
    """Run CCM analysis for all variables with parallel processing support."""
    print("Multivariate CCM Analysis")
    print("========================")
    
    start_time = time.time()
    
    # Create output directories
    os.makedirs(config['output']['plots_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    
    # Log parallel processing settings
    parallel_config = config['analysis'].get('parallel', {})
    if parallel_config.get('enabled', True):
        n_workers = parallel_config.get('max_workers') or cpu_count()
        print(f"\nParallel processing enabled with {n_workers} workers")
    else:
        print("\nParallel processing disabled")
    
    print("\nCreating data overview plot...")
    overview_file = plot_data_overview(data, config, config['output']['plots_dir'])
    print(f"Data overview saved as {overview_file}")
    
    results = {}
    columns = config['data']['columns_to_keep']
    
    total_analyses = len(columns)
    for i, target in enumerate(columns, 1):
        print(f"\nAnalyzing {target} as target variable ({i}/{total_analyses})...")
        analysis_start = time.time()
        
        mccm = MultivariateCCM(
            data=data,
            columns=columns,
            target=target,
            config=config['analysis']
        )
        target_results = mccm.analyze()
        
        analysis_time = time.time() - analysis_start
        print(f"Analysis time for {target}: {analysis_time:.2f} seconds")
        
        plot_file = plot_ccm_results(
            target_results, 
            target, 
            config, 
            config['output']['plots_dir'], 
            data=data
        )
        print(f"Plots saved as {plot_file}")
        results[target] = target_results
    
    # Create and save summary plots
    print("\nCreating summary plots...")
    summary_file = create_summary_plots(results, config, config['output']['plots_dir'])
    if summary_file:
        print(f"Summary plots saved as {summary_file}")
    
    # Save results
    saved_files = save_results(results, config, timestamp)
    print("\nSaved results files:")
    for file_type, filename in saved_files.items():
        if filename:
            print(f"{file_type}: {filename}")
    
    total_time = time.time() - start_time
    print(f"\nTotal analysis time: {total_time:.2f} seconds")
    
    return results

def print_summary(results: Dict[str, Any]) -> None:
    """Print summary of all analyses."""
    print("\nAnalysis Summary:")
    print("================")
    
    for target, result in results.items():
        print(f"\nTarget: {target}")
        if 'best_combo' in result and result['best_combo'] is not None:
            variables = result['best_combo']['variables']
            if isinstance(variables, tuple):
                variables = ' & '.join(variables)
            print("Best predictors:", variables)
            print(f"Correlation: {result['best_combo']['rho']:.3f}")
            print(f"MAE: {result['best_combo']['MAE']:.3f}")
            print(f"RMSE: {result['best_combo']['RMSE']:.3f}")
        else:
            print("No valid results found.")