# main.py

import sys
import time
from multiprocessing import cpu_count
from src.utils import load_config, load_data, prepare_output_dirs, setup_logger
from src.analysis import run_ccm_analysis, print_summary
from datetime import datetime

def main():
    start_time = time.time()
    
    try:
        # Load configuration
        config = load_config()
        
        # Setup logger
        logger = setup_logger(config)
        logger.info("Multivariate CCM Analysis Started")
        
        # Log parallel processing settings
        parallel_config = config['analysis'].get('parallel', {})
        if parallel_config.get('enabled', True):
            n_workers = parallel_config.get('max_workers') or cpu_count()
            logger.info(f"Parallel processing enabled with {n_workers} workers")
        else:
            logger.info("Parallel processing disabled")
        
        # Prepare output directories
        prepare_output_dirs(config)
        
        # Load data
        logger.info("Loading data...")
        data = load_data(config)
        logger.info(f"Loaded data with shape: {data.shape}")
        
        # Generate timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Run analysis
        logger.info("Running CCM analysis...")
        results = run_ccm_analysis(data, config, timestamp)
        
        # Print summary
        print_summary(results)
        
        # Log execution time
        end_time = time.time()
        total_time = end_time - start_time
        logger.info(f"Total execution time: {total_time:.2f} seconds")
        
        # Log results location
        logger.info("\nResults can be found in:")
        logger.info(f"Plots directory: {config['output']['plots_dir']}")
        logger.info(f"Results directory: {config['output']['results_dir']}")
        logger.info(f"Logs directory: {config['output']['logs_dir']}")
        
    except KeyboardInterrupt:
        logger.error("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error in analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == '__main__':
    main()