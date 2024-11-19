# Multivariate CCM Analysis

A Python package for Multivariate Convergent Cross Mapping (CCM) analysis of time series data with parallel processing capabilities.

## Features

- Multivariate CCM implementation with optimized parallel processing
- Intelligent combination selection to reduce computational complexity
- Automated variable selection based on correlation analysis
- Progress tracking and logging
- Visualization of results with customizable plots
- Support for both synthetic and real data
- Error handling and recovery mechanisms

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SharathSPhD/multivariate-ccm.git
cd multivariate-ccm
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv ccmenv
source ccmenv/bin/activate  # Linux/Mac
# or
ccmenv\Scripts\activate  # Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

The analysis is controlled through a `config.json` file with the following structure:

```json
{
    "data": {
        "type": "file",  // or "synthetic"
        "file_path": "data/sample_data.csv",
        "datetime_column": "datetime",
        "columns_to_keep": ["var1", "var2", "var3"],
        "synthetic_params": {
            "n_points": 1000,
            "noise_level": 0.1
        }
    },
    "analysis": {
        "embedding_dimension": 3,
        "tau": -1,
        "train_size_ratio": 0.75,
        "num_surrogates": 100,
        "exclusion_radius": 0,
        "parallel": {
            "enabled": true,
            "max_workers": null,  // null uses all available cores
            "chunk_size": 1
        }
    },
    "output": {
        "plots_dir": "plots",
        "results_dir": "results",
        "logs_dir": "logs",
        "save_predictions": true,
        "filename_prefix": "ccm_analysis"
    }
}
```

## Usage

1. Prepare your data:
   - Use CSV format with datetime column
   - Ensure columns are properly formatted
   - Place data file in the data directory

2. Configure analysis:
   - Modify config.json to match your data structure
   - Set desired analysis parameters
   - Configure output directories

3. Run the analysis:
```bash
python main.py
```

4. Check results:
   - Analysis results are saved in the specified results directory
   - Plots are generated in the plots directory
   - Logs are stored in the logs directory

## Output Files

The analysis generates several output files:
- `ccm_analysis_metrics_{timestamp}.csv`: Contains correlation metrics for all combinations
- `ccm_analysis_predictions_{timestamp}.csv`: Contains predictions vs actual values
- `ccm_analysis_combinations_{timestamp}.csv`: Details of all variable combinations tested
- Plot files: 
  - `data_overview.png`: Overview of input time series
  - `ccm_results_{target}.png`: Results for each target variable
  - `analysis_summary.png`: Summary of all analyses

## Performance Considerations

- Parallel processing is enabled by default and uses all available cores
- For large datasets, adjust chunk_size in config for better performance
- Memory usage scales with the number of combinations being evaluated
- Progress is reported during analysis for monitoring

## Limitations

- Maximum number of predictors is limited to control computational complexity
- Parallel processing may be memory-intensive for very large datasets
- Some combinations may result in invalid calculations (reported in logs)

## Error Handling

The package includes comprehensive error handling:
- Invalid combinations are skipped and logged
- Process errors are caught and reported
- Analysis continues even if some combinations fail
- Full error logs are maintained

## Logging

- All operations are logged to `logs/ccm_analysis.log`
- Log file is overwritten with each run
- Both console and file logging are enabled
- Errors and warnings are clearly marked

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{multivariate_ccm,
  author = {Sharath Sathish},
  title = {Multivariate CCM Analysis},
  year = {2024},
  url = {https://github.com/SharathSPhD/multivariate-ccm}
}
```