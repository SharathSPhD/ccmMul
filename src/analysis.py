from .multivariate_ccm import MultivariateCCM
from .plotting import plot_ccm_results

def run_ccm_analysis(data, config, timestamp=None):
    """Run CCM analysis for all variables."""
    from .utils import save_results
    
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
        
        # Run analysis
        target_results = mccm.analyze()
        
        # Create plots
        plot_file = plot_ccm_results(
            target_results, target, config, config['output']['plots_dir']
        )
        print(f"Plots saved as {plot_file}")
        
        results[target] = target_results
    
    # Save numerical results
    saved_files = save_results(results, config, timestamp)
    print("\nSaved results files:")
    for file_type, filename in saved_files.items():
        if filename:
            print(f"{file_type}: {filename}")
    
    return results

def print_summary(results):
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