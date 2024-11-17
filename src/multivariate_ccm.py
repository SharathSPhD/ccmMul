import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class MultivariateCCM:
    """Multivariate Convergent Cross Mapping for time series prediction."""
    
    def __init__(self, data, columns=None, target=None, config=None):
        """Initialize MultivariateCCM with configuration parameters."""
        self.data = data.copy()
        
        # Remove datetime and time columns if present
        if 'datetime' in self.data.columns:
            self.data = self.data.drop('datetime', axis=1)
        if 'time' in self.data.columns:
            self.data = self.data.drop('time', axis=1)
            
        # Reset index
        self.data = self.data.reset_index(drop=True)
        
        # Set default config if none provided
        self.config = config or {
            'embedding_dimension': 3,
            'tau': -1,
            'train_size_ratio': 0.75,
            'num_surrogates': 100,
            'exclusion_radius': 0
        }
        
        # Set columns and target
        self.columns = columns or list(self.data.columns)
        self.target = target or self.columns[0]
        if self.target in self.columns:
            self.predictors = [col for col in self.columns if col != self.target]
        else:
            raise ValueError(f"Target {self.target} not found in columns")
        
        # Initialize results storage
        self.results = {
            'View': None,
            'Predictions': None,
            'best_combo': None,
        }
        
        # Scale the data
        self.scaler = StandardScaler()
        self.data_scaled = pd.DataFrame(
            self.scaler.fit_transform(self.data),
            columns=self.data.columns
        )

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
        """Perform full multivariate CCM analysis."""
        print(f"\nAnalyzing target: {self.target}")
        print(f"Predictors: {self.predictors}")
        
        all_results = []
        best_predictions = None
        best_actuals = None
        best_time_indices = None
        best_rho = float('-inf')
        
        # Evaluate each combination of predictors
        from itertools import combinations
        for r in range(1, len(self.predictors) + 1):
            for combo in combinations(self.predictors, r):
                print(f"\nEvaluating combination: {combo}")
                metrics, predictions, actuals, time_indices = self.evaluate_combination(list(combo))
                
                result = {
                    'variables': combo,
                    'rho': metrics['rho'],
                    'MAE': metrics['MAE'],
                    'RMSE': metrics['RMSE']
                }
                all_results.append(result)
                
                print(f"Metrics: rho={metrics['rho']:.3f}, MAE={metrics['MAE']:.3f}, RMSE={metrics['RMSE']:.3f}")
                
                # Store best predictions
                if metrics['rho'] > best_rho:
                    best_rho = metrics['rho']
                    best_predictions = predictions
                    best_actuals = actuals
                    best_time_indices = time_indices
        
        # Create View DataFrame
        self.results['View'] = pd.DataFrame(all_results)
        self.results['View'] = self.results['View'].sort_values('rho', ascending=False)
        
        # Store best combination and predictions
        if len(all_results) > 0 and not np.isnan(self.results['View']['rho'].iloc[0]):
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
        
        return self.results
