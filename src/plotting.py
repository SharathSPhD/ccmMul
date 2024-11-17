import matplotlib.pyplot as plt
import numpy as np
import os

def plot_ccm_results(results, target, config, save_dir):
    """Plot and save CCM analysis results."""
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