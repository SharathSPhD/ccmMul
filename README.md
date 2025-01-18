# Multivariate CCM Analysis

A Python package for Multivariate Convergent Cross Mapping (CCM) analysis of time series data. This implementation combines traditional CCM with multivariate analysis capabilities to identify causal relationships between multiple time series.

## Features

- Multivariate CCM analysis
- Support for both synthetic and real data
- Time series prediction with validation
- Automated visualization of results
- Comprehensive metric calculations
- Flexible configuration system

## Project Structure

```
ccmMul/
├── config/
│   └── config.json         # Configuration parameters
├── data/
│   └── sample_data.csv     # Sample or user data
├── src/
│   ├── __init__.py        # Package initialization
│   ├── analysis.py        # Main analysis functions
│   ├── multivariate_ccm.py # Core CCM implementation
│   ├── plotting.py        # Visualization functions
│   └── utils.py           # Utility functions
├── plots/                  # Generated plots
├── results/               # Analysis results
├── README.md
├── LICENSE
└── main.py               # Main execution script
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ShrathSPhD/ccmMul.git
cd ccmMul
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Configuration

Modify `config/config.json` to customize the analysis:

```json
{
    "data": {
        "type": "synthetic",  // or "file"
        "file_path": "data/sample_data.csv",
        "datetime_column": "datetime",
        "columns_to_keep": ["x", "y", "z"]
    },
    "analysis": {
        "embedding_dimension": 3,
        "tau": -1,
        "train_size_ratio": 0.75
    }
}
```

## Usage

1. For synthetic data:
```python
python main.py
```

2. For your own data:
- Place your CSV file in the `data/` directory
- Update `config.json` with appropriate settings
- Run `python main.py`

## Output

The analysis generates three types of output:

1. Plots (`plots/` directory):
   - Correlation bar plots for variable combinations
   - Time series plots of actual vs predicted values

2. Results (`results/` directory):
   - Metrics summary (correlations, MAE, RMSE)
   - Detailed predictions
   - Variable combination analysis

3. Console output:
   - Analysis progress
   - Summary statistics
   - File save locations

## Example Output

```
Multivariate CCM Analysis
========================
Analyzing x as target variable...
Best combination results:
Variables: ('y', 'z')
Correlation (rho): 0.976
MAE: 0.167
RMSE: 0.216
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Requirements

- Python 3.8+
- numpy
- pandas
- scikit-learn
- matplotlib

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@software{ccmMul2024,
  author = Sharath S,
  title = {Multivariate CCM Analysis},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/SharathSPhD/ccmMul}
}
```



Project Link: [https://github.com/ShrathSPhD/ccmMul](https://github.com/SharathSPhD/ccmMul)
