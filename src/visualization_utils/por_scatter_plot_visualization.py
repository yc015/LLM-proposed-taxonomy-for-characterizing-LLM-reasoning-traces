from src.model_name_translator import model_name_decoder
import numpy as np
import matplotlib.pyplot as plt


def plot_dimension_comparison(coder, split="train", figsize=(12, 12), min_distance=0.05):
    """
    Create a scatter plot comparing average dimension values between two models.
    Uses simple distance-based text positioning to avoid overlaps.
    
    Parameters:
    -----------
    coder : object
        Coder object containing model options and code occurrence data
    split : str, default="train"
        Data split to use ("train", "eval", etc.)
    figsize : tuple, default=(12, 12)
        Figure size as (width, height)
    min_distance : float, default=0.02
        Minimum distance between text labels (as fraction of axis range)
    
    Returns:
    --------
    fig, ax : matplotlib figure and axis objects
    """
    
    # Extract data
    models = coder.model_options
    vectors = coder.code_occurrence_overall[split]["vectors"]
    vectors_model_1 = np.array([vector for key, vector in vectors.items() if models[0] in key])
    vectors_model_2 = np.array([vector for key, vector in vectors.items() if models[1] in key])
    dimension_to_code = coder.code_occurrence_overall[split]["code_order"]
    
    # Calculate average values for each dimension across both models
    avg_model_1 = np.mean(vectors_model_1, axis=0)
    avg_model_2 = np.mean(vectors_model_2, axis=0)
    
    # Create the scatter plot
    fig, ax = plt.subplots(figsize=figsize)
    scatter = ax.scatter(avg_model_2, avg_model_1, alpha=0.7, s=60)
    
    # Calculate axis ranges for distance calculations
    x_range = np.max(avg_model_2) - np.min(avg_model_2)
    y_range = np.max(avg_model_1) - np.min(avg_model_1)
    
    # Show only one label for close dots
    positions = list(zip(avg_model_2, avg_model_1))
    shown_positions = []
    
    for i, (x, y) in enumerate(positions):
        # Check if this position is too close to any already shown position
        show_label = True
        
        for prev_x, prev_y, _ in shown_positions:
            dx = abs(x - prev_x) / x_range if x_range > 0 else 0
            dy = abs(y - prev_y) / y_range if y_range > 0 else 0
            
            if dx < min_distance and dy < min_distance:
                show_label = False
                break
        
        # Only show label if position is not too close to existing ones
        if show_label:
            shown_positions.append((x, y, dimension_to_code[i]))
            ax.annotate(dimension_to_code[i], 
                       (x, y), 
                       xytext=(5, 5),
                       textcoords='offset points', 
                       fontsize=8, 
                       alpha=0.8)
    
    # Set labels
    ax.set_xlabel(f'Average Dimension Value - {model_name_decoder[models[1]]}', fontsize=12)
    ax.set_ylabel(f'Average Dimension Value - {model_name_decoder[models[0]]}', fontsize=12)
    
    # Add a diagonal reference line (y = x)
    min_val = min(min(avg_model_1), min(avg_model_2))
    max_val = max(max(avg_model_1), max(avg_model_2))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.3, label='Equally occurred')
    
    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Add grid and legend
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    
    # Print statistics
    print(f"Model 1 ({model_name_decoder[models[0]]}) - Mean: {np.mean(avg_model_1):.4f}, Std: {np.std(avg_model_1):.4f}")
    print(f"Model 2 ({model_name_decoder[models[1]]}) - Mean: {np.mean(avg_model_2):.4f}, Std: {np.std(avg_model_2):.4f}")
    print(f"Number of dimensions: {len(dimension_to_code)}")
    
    return fig, ax

# Example usage:
# fig, ax = plot_dimension_comparison(coder, split="train", figsize=(12, 12))
# plt.show()