import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from multiprocessing import Pool, cpu_count
from itertools import combinations
from functools import partial
from .utils import select_optimal_combinations

class MultivariateCCM:
    """Multivariate Convergent Cross Mapping for time series prediction."""
    
    def __init__(self, data, columns=None, target=None, config=None):
        """Initialize MultivariateCCM with configuration parameters."""
        self.data = data.copy()
        
        # Set default config if none provided first (as it doesn't depend on data)
        self.config = config or {
            'embedding_dimension': 3,
            'tau': -1,
            'train_size_ratio': 0.75,
            'num_surrogates': 100,
            'exclusion_radius': 0,
            'parallel': {
                'enabled': True,
                'max_workers': None,
                'chunk_size': 1
            }
        }
        
        # Set columns first (before datetime handling)
        self.columns = columns or [col for col in data.columns 
                                if col not in ['datetime', 'time']]
        
        # Set target and predictors
        self.target = target or self.columns[0]
        if self.target in self.columns:
            self.predictors = [col for col in self.columns if col != self.target]
        else:
            raise ValueError(f"Target {self.target} not found in columns")
        
        # Store datetime column if present
        self.datetime_col = None
        if 'datetime' in self.data.columns:
            self.datetime_col = self.data['datetime'].copy()
            self.data = self.data.drop('datetime', axis=1)
        elif 'time' in self.data.columns:
            self.datetime_col = self.data['time'].copy()
            self.data = self.data.drop('time', axis=1)
        
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        # Initialize results storage
        self.results = {
            'View': None,
            'Predictions': None,
            'best_combo': None,
        }
        
        # Scale only numeric columns (using self.columns which excludes datetime)
        self.scaler = StandardScaler()
        self.data_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.data[self.columns]),
            columns=self.columns
        )
        
        # Add datetime back if it was present
        if self.datetime_col is not None:
            self.data['datetime'] = self.datetime_col

    def create_embedding(self, data, columns, E, tau):
        """Create time-delay embedding for multiple variables."""
        embedded_data = pd.DataFrame()
        
        for col in columns:
            for i in range(E):
                shift = -i * tau
                col_name = f"{col}_t{i}" if i > 0 else col
                embedded_data[col_name] = data[col].shift(shift)
        
        # Remove rows with NaN from embedding
        return embedded_data.dropna(), embedded_data.dropna().index

    def find_nearest_neighbors(self, embedding_matrix, lib_indices, pred_indices, k):
        """Find k nearest neighbors using KDTree."""
        if len(lib_indices) == 0 or len(pred_indices) == 0:
            return None, None
            
        tree = KDTree(embedding_matrix[lib_indices])
        distances, indices = tree.query(embedding_matrix[pred_indices], k=k)
        
        # Map indices back to original library indices
        mapped_indices = lib_indices[indices]
        
        return distances, mapped_indices

    def make_predictions(self, neighbors, target_array):
        """Make predictions using exponential weighting of neighbors."""
        if neighbors[0] is None or neighbors[1] is None:
            return np.array([])
            
        distances, indices = neighbors
        
        # Compute weights using exponential decay
        min_distances = np.fmax(distances[:, 0:1], 1e-6)
        weights = np.exp(-distances / min_distances)
        weight_sum = weights.sum(axis=1, keepdims=True)
        
        # Get target values for neighbors and compute weighted predictions
        neighbor_targets = target_array[indices]
        predictions = (weights * neighbor_targets).sum(axis=1) / weight_sum.ravel()
        
        return predictions

    def evaluate_combination(self, predictor_cols, E=None, tau=None):
        """Evaluate a specific combination of predictors."""
        E = E or self.config['embedding_dimension']
        tau = tau or self.config['tau']
        
        # Create embedding for these predictors
        embedded_data, valid_indices = self.create_embedding(
            self.data_scaled, predictor_cols, E, tau
        )
        
        if len(embedded_data) == 0:
            return {'rho': np.nan, 'MAE': np.nan, 'RMSE': np.nan}, None, None, None
        
        # Convert to numpy array for efficient computation
        embedding_matrix = embedded_data.values
        target_array = self.data_scaled[self.target].values[valid_indices]
        
        # Split into training and testing sets
        total_points = len(embedding_matrix)
        train_size = int(total_points * self.config['train_size_ratio'])
        
        lib_indices = np.arange(train_size)
        pred_indices = np.arange(train_size, total_points)
        
        if len(pred_indices) == 0:
            return {'rho': np.nan, 'MAE': np.nan, 'RMSE': np.nan}, None, None, None
        
        # Find nearest neighbors and make predictions
        neighbors = self.find_nearest_neighbors(
            embedding_matrix, lib_indices, pred_indices, k=E+1
        )
        predictions = self.make_predictions(neighbors, target_array)
        
        # Get actual values for test set
        actual = target_array[pred_indices]
        metrics = self.compute_metrics(actual, predictions)
        
        # Inverse transform predictions and actual values
        predictions_original = self.scaler.inverse_transform(
            np.zeros((len(predictions), len(self.columns)))
        )[:, self.columns.index(self.target)]
        predictions_original += predictions * self.scaler.scale_[self.columns.index(self.target)]
        predictions_original += self.scaler.mean_[self.columns.index(self.target)]
        
        actual_original = self.scaler.inverse_transform(
            np.zeros((len(actual), len(self.columns)))
        )[:, self.columns.index(self.target)]
        actual_original += actual * self.scaler.scale_[self.columns.index(self.target)]
        actual_original += self.scaler.mean_[self.columns.index(self.target)]
        
        time_indices = valid_indices[pred_indices]
        
        return metrics, predictions_original, actual_original, time_indices

    def evaluate_combination_wrapper(self, combo):
        """Wrapper function for parallel processing."""
        return self.evaluate_combination(list(combo))

    def compute_metrics(self, actual, predicted):
        """Compute performance metrics."""
        if len(actual) == 0 or len(predicted) == 0:
            return {'rho': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
            
        # Remove any NaN values
        mask = ~np.isnan(actual) & ~np.isnan(predicted)
        actual = actual[mask]
        predicted = predicted[mask]
        
        if len(actual) < 2:
            return {'rho': np.nan, 'MAE': np.nan, 'RMSE': np.nan}
        
        # Compute metrics
        rho = np.corrcoef(actual, predicted)[0, 1]
        mae = np.mean(np.abs(actual - predicted))
        rmse = np.sqrt(np.mean((actual - predicted) ** 2))
        
        return {'rho': rho, 'MAE': mae, 'RMSE': rmse}

    def analyze(self):
        """Perform full multivariate CCM analysis with improved error handling."""
        print(f"\nAnalyzing target: {self.target}")
        print(f"Predictors: {self.predictors}")
        
        try:
            # Get optimal combinations 
            all_combinations = select_optimal_combinations(
                data=self.data,
                target=self.target,
                predictors=self.predictors,
                max_combinations=10000  # Hard-coded default or could be added to config['analysis']
            )
            
            total_combinations = len(all_combinations)
            print(f"\nSelected {total_combinations} optimal combinations for evaluation")
            
            all_results = []
            best_predictions = None
            best_actuals = None
            best_time_indices = None
            best_rho = float('-inf')

            if self.config['parallel']['enabled']:
                n_workers = min(self.config['parallel'].get('max_workers') or cpu_count(), 8)
                chunk_size = max(5, total_combinations // (n_workers * 4))
                print(f"Using parallel processing with {n_workers} workers and chunk size {chunk_size}")
                
                with Pool(processes=n_workers, maxtasksperchild=100) as pool:
                    results_iterator = pool.imap(
                        self.evaluate_combination_wrapper,
                        all_combinations,
                        chunksize=chunk_size
                    )
                    
                    # Process results with error handling
                    for i, result in enumerate(results_iterator, 1):
                        try:
                            metrics, predictions, actuals, time_indices = result
                            combo = all_combinations[i-1]
                            
                            if i % max(1, total_combinations//20) == 0:
                                print(f"\rProgress: {i}/{total_combinations} combinations evaluated ({(i/total_combinations)*100:.1f}%)", end="")
                            
                            result_dict = {
                                'variables': combo,
                                'rho': metrics['rho'],
                                'MAE': metrics['MAE'],
                                'RMSE': metrics['RMSE']
                            }
                            all_results.append(result_dict)
                            
                            if metrics['rho'] > best_rho and not np.isnan(metrics['rho']):
                                best_rho = metrics['rho']
                                best_predictions = predictions
                                best_actuals = actuals
                                best_time_indices = time_indices
                                
                        except Exception as e:
                            print(f"\nError processing combination {i}: {str(e)}")
                            continue
                
                pool.close()
                pool.join()
                
            else:
                # Serial processing
                for i, combo in enumerate(all_combinations, 1):
                    if i % max(1, total_combinations//20) == 0:
                        print(f"\rProgress: {i}/{total_combinations} combinations evaluated ({(i/total_combinations)*100:.1f}%)", end="")
                    
                    metrics, predictions, actuals, time_indices = self.evaluate_combination(list(combo))
                    result = {
                        'variables': combo,
                        'rho': metrics['rho'],
                        'MAE': metrics['MAE'],
                        'RMSE': metrics['RMSE']
                    }
                    all_results.append(result)
                    
                    if metrics['rho'] > best_rho and not np.isnan(metrics['rho']):
                        best_rho = metrics['rho']
                        best_predictions = predictions
                        best_actuals = actuals
                        best_time_indices = time_indices

            print("\n\nCreating results summary...")
            
            if all_results:
                valid_results = [r for r in all_results if not np.isnan(r['rho'])]
                if valid_results:
                    self.results['View'] = pd.DataFrame(valid_results)
                    self.results['View'] = self.results['View'].sort_values('rho', ascending=False)
                    self.results['best_combo'] = self.results['View'].iloc[0].to_dict()
                    self.results['predictions'] = {
                        'predicted': best_predictions,
                        'actual': best_actuals,
                        'time_indices': best_time_indices
                    }
                    
                    print("\nBest combination results:")
                    print(f"Variables: {self.results['best_combo']['variables']}")
                    print(f"Correlation (rho): {self.results['best_combo']['rho']:.3f}")
                    print(f"MAE: {self.results['best_combo']['MAE']:.3f}")
                    print(f"RMSE: {self.results['best_combo']['RMSE']:.3f}")
                else:
                    print("\nNo valid results found.")
            else:
                print("\nNo valid results found.")
                
        except Exception as e:
            print(f"\nError in analysis: {str(e)}")
            raise
            
        return self.results