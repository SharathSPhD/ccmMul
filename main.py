#!/usr/bin/env python3

from src.utils import load_config, load_data, prepare_output_dirs
from src.analysis import run_ccm_analysis, print_summary

def main():
    """Main function to run the analysis."""
    print("Multivariate CCM Analysis")
    print("========================")
    
    try:
        # Load configuration
        config = load_config()
        
        # Prepare output directories
        prepare_output_dirs(config)
        
        # Load data
        print("\nLoading data...")
        data = load_data(config)
        print(f"Loaded data with shape: {data.shape}")
        
        # Run analysis
        print("\nRunning CCM analysis...")
        results = run_ccm_analysis(data, config)
        
        # Print summary
        print_summary(results)
        
    except Exception as e:
        print(f"\nError in analysis: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()