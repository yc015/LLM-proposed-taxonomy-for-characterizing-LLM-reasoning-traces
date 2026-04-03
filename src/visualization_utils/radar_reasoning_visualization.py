#!/usr/bin/env python3
"""
Radar Chart Visualization for Reasoning Characteristics
This script creates radar charts showing model performance across the most predictive reasoning dimensions.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder


def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.
    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)
            # Add white background boxes to labels for better readability
            for label in self.get_xticklabels():
                label.set_bbox(dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="lightgray"))

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta


def find_most_predictive_dimensions(vectors, labels, code_names, n_dims=8):
    """Find the most predictive dimensions using F-statistic for continuous features."""
    # Encode labels as integers for F-test
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    # Calculate F-statistics for each dimension
    f_scores, _ = f_classif(vectors, labels_encoded)
    
    # Find top predictive dimensions
    top_indices = np.argsort(f_scores)[-n_dims:][::-1]
    
    most_predictive_codes = []
    for idx in top_indices:
        score = f_scores[idx]
        code_name = code_names[idx]
        most_predictive_codes.append((idx, code_name, score))
    
    print(f"Top {n_dims} most predictive reasoning dimensions:")
    for i, (idx, code_name, score) in enumerate(most_predictive_codes):
        print(f"  {i+1}. {code_name} (F-score: {score:.3f})")
    
    return most_predictive_codes


def calculate_model_averages(vectors, labels, most_predictive_codes):
    """Calculate average frequency for each model on the most predictive dimensions."""
    unique_labels = np.unique(labels)
    model_averages = {}
    
    for label in unique_labels:
        mask = np.array(labels) == label
        model_vectors = vectors[mask]
        
        # Extract values for the most predictive dimensions
        avg_values = []
        for idx, code_name, score in most_predictive_codes:
            avg_freq = np.mean(model_vectors[:, idx])
            avg_values.append(avg_freq)
        
        model_averages[label] = avg_values
        
    return model_averages


def create_radar_visualization(vectors, labels, code_names, n_dims=8, frame='polygon', eval_vector=None, train_label="train", eval_label="eval"):
    """Create radar charts showing model performance across most predictive dimensions.
    
    Parameters:
    -----------
    vectors : numpy.ndarray
        Training vectors
    labels : list
        Training labels
    code_names : list
        Names of all dimensions
    n_dims : int
        Number of dimensions to visualize
    frame : str
        Frame style ('circle' or 'polygon')
    eval_vector : numpy.ndarray, optional
        Single evaluation vector to overlay on the charts
    train_label : str
        Label for training data in legend (default: "train")
    eval_label : str
        Label for evaluation data in legend (default: "eval")
    """
    
    # Find most predictive dimensions
    most_predictive_codes = find_most_predictive_dimensions(vectors, labels, code_names, n_dims)
    
    # Calculate model averages
    model_averages = calculate_model_averages(vectors, labels, most_predictive_codes)
    
    # Extract eval values if provided (no averaging needed - single vector)
    eval_values = None
    if eval_vector is not None:
        eval_vals = []
        for idx, code_name, score in most_predictive_codes:
            eval_vals.append(eval_vector[idx])
        eval_values = eval_vals
    
    # Prepare theta for radar chart
    theta = radar_factory(n_dims, frame=frame)
    
    # Create spoke labels (truncate long names for readability)
    spoke_labels = [code_name[:25] + "..." if len(code_name) > 25 else code_name 
                   for _, code_name, _ in most_predictive_codes]
    
    # Set up colors for models
    unique_labels = list(model_averages.keys())
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_labels)))
    model_colors = dict(zip(unique_labels, colors))
    
    # Create subplot layout based on number of models
    n_models = len(unique_labels)
    if n_models == 1:
        fig, axs = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
        axs = [axs]  # Make it iterable
    elif n_models == 2:
        fig, axs = plt.subplots(figsize=(16, 8), nrows=1, ncols=2,
                               subplot_kw=dict(projection='radar'))
    elif n_models <= 4:
        fig, axs = plt.subplots(figsize=(16, 12), nrows=2, ncols=2,
                               subplot_kw=dict(projection='radar'))
        axs = axs.flatten()
    else:
        # For more than 4 models, create a larger grid
        rows = int(np.ceil(n_models / 3))
        fig, axs = plt.subplots(figsize=(18, 6*rows), nrows=rows, ncols=3,
                               subplot_kw=dict(projection='radar'))
        axs = axs.flatten()
    
    fig.subplots_adjust(wspace=0.25, hspace=0.30, top=0.90, bottom=0.05)
    
    # Calculate global maximum across all models (based solely on training data) for consistent scaling
    all_values = [vals for vals in model_averages.values()]
    global_max = max([max(vals) for vals in all_values]) * 1.2
    grid_levels = np.linspace(0, global_max, 5)
    
    # Plot each model on its own radar chart
    for i, (model_label, avg_values) in enumerate(model_averages.items()):
        if i >= len(axs):
            break
            
        ax = axs[i]
        color = model_colors[model_label]
        
        # Set radial grid lines with consistent global scale
        ax.set_rgrids(grid_levels, [f'{val:.3f}' for val in grid_levels])
        
        # Plot the model's performance (training data)
        ax.plot(theta, avg_values, color=color, linewidth=3, label=train_label)
        ax.fill(theta, avg_values, facecolor=color, alpha=0.25)
        
        # Overlay eval data if provided
        if eval_values is not None:
            ax.plot(theta, eval_values, color='tab:red', linewidth=3, 
                   linestyle='--', marker='o', markersize=6, label=eval_label)
        
        # Set variable labels
        ax.set_varlabels(spoke_labels)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)
        
        # Set title
        ax.set_title(f'{model_label}', weight='bold', size='large', 
                    position=(0.5, 1.1), horizontalalignment='center')
        
        # Set y-axis limits to ensure consistent scaling across subplots
        ax.set_ylim(0, global_max)
    
    # Hide any unused subplots
    for i in range(len(model_averages), len(axs)):
        axs[i].set_visible(False)
    
    # Add overall title
    fig.suptitle(f'Reasoning Characteristics: Top {n_dims} Most Predictive Dimensions', 
                 fontsize=16, weight='bold', y=0.95)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n=== RADAR CHART VISUALIZATION SUMMARY ===")
    print(f"Total train samples: {len(vectors)}")
    if eval_vector is not None:
        print(f"Eval data: Single vector provided")
    print(f"Models compared: {len(unique_labels)}")
    print(f"Dimensions visualized: {n_dims}")
    print(f"Feature dimensions total: {vectors.shape[1]}")
    
    for model_label, avg_values in model_averages.items():
        print(f"\n{model_label} - {train_label.upper()} Average frequencies:")
        for (idx, code_name, score), avg_val in zip(most_predictive_codes, avg_values):
            print(f"  {code_name[:40]}: {avg_val:.4f}")
    
    if eval_values is not None:
        print(f"\n{eval_label.upper()} - Values:")
        for (idx, code_name, score), eval_val in zip(most_predictive_codes, eval_values):
            print(f"  {code_name[:40]}: {eval_val:.4f}")
    
    # Add eval_values to the return dict for backward compatibility
    return_dict = model_averages.copy()
    if eval_values is not None:
        return_dict[eval_label] = eval_values
    
    return fig, axs, most_predictive_codes, return_dict


def load_coder_and_visualize(coder_path, n_dims=8):
    """Load the coder and create radar visualization."""
    print(f"Loading coder from: {coder_path}")
    
    # Load the coder
    with open(coder_path, 'rb') as f:
        coder = pickle.load(f)
    
    # Extract the data using the provided code
    vectors = coder._apply_comparative_normalization_to_training_data(coder.code_occurrence_overall['train'])
    labels = coder.code_occurrence_overall['train']['labels']
    code_name_across_dimensions = coder.code_occurrence_overall['train']['code_order']
    
    print(f"Data loaded successfully!")
    print(f"Vectors shape: {vectors.shape}")
    print(f"Number of labels: {len(labels)}")
    print(f"Unique models: {np.unique(labels)}")
    
    # Create the radar visualization
    return create_radar_visualization(vectors, labels, code_name_across_dimensions, n_dims)


if __name__ == "__main__":
    # Example usage - modify the path to your coder file
    coder_path = "path/to/your/coder.pkl"  # Update this path
    
    # You can adjust the number of dimensions to visualize (default: 8)
    n_dimensions = 8
    
    try:
        fig, axs, predictive_codes, model_avgs = load_coder_and_visualize(coder_path, n_dimensions)
        print("Visualization complete!")
    except FileNotFoundError:
        print(f"Coder file not found at: {coder_path}")
        print("Please update the coder_path variable with the correct path to your .pkl file")
    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check that your coder file is valid and contains the expected data structure")


# ============================================================================
# NOTEBOOK USAGE EXAMPLES
# ============================================================================

def example_notebook_usage():
    """
    Example usage for Jupyter notebooks - run this after loading your coder object.
    
    # Step 1: Load your data (assuming you already have a coder object)
    vectors = coder._apply_comparative_normalization_to_training_data(coder.code_occurrence_overall['train'])
    labels = coder.code_occurrence_overall['train']['labels']
    code_name_across_dimensions = coder.code_occurrence_overall['train']['code_order']
    
    # Step 2: Create radar visualization with 8 dimensions (default)
    fig, axs, predictive_codes, model_avgs = create_radar_visualization(
        vectors, labels, code_name_across_dimensions, n_dims=8
    )
    
    # Step 3: Or try different numbers of dimensions
    # For 6 dimensions:
    fig, axs, predictive_codes, model_avgs = create_radar_visualization(
        vectors, labels, code_name_across_dimensions, n_dims=6
    )
    
    # For 10 dimensions:
    fig, axs, predictive_codes, model_avgs = create_radar_visualization(
        vectors, labels, code_name_across_dimensions, n_dims=10
    )
    
    # Step 4: With evaluation data overlay (NEW!)
    # If you have a single evaluation vector:
    eval_vector = coder._apply_comparative_normalization_to_training_data(coder.code_occurrence_overall['eval'])[0]  # First vector
    fig, axs, predictive_codes, model_avgs = create_radar_visualization(
        vectors, labels, code_name_across_dimensions, n_dims=8, eval_vector=eval_vector
    )
    # Now model_avgs will contain 'eval' key with the evaluation values
    
    # Step 5: With custom labels
    fig, axs, predictive_codes, model_avgs = create_radar_visualization(
        vectors, labels, code_name_across_dimensions, n_dims=8, 
        eval_vector=eval_vector, train_label="training", eval_label="validation"
    )
    
    # Step 6: Access the results
    print("Most predictive codes:", predictive_codes)
    print("Model averages:", model_avgs)
    # If eval_vector was provided, model_avgs['eval'] contains the eval values
    """
    pass


def example_direct_function_calls():
    """
    Example of calling functions directly with your data.
    Use this if you want more control over the process.
    """
    # Assuming you have vectors, labels, code_name_across_dimensions
    
    # Step 1: Find most predictive dimensions
    # most_predictive_codes = find_most_predictive_dimensions(
    #     vectors, labels, code_name_across_dimensions, n_dims=8
    # )
    
    # Step 2: Calculate model averages
    # model_averages = calculate_model_averages(vectors, labels, most_predictive_codes)
    
    # Step 3: Create visualization
    # fig, axs, predictive_codes, model_avgs = create_radar_visualization(
    #     vectors, labels, code_name_across_dimensions, n_dims=8, frame='polygon'
    # )
    
    # You can also change the frame style:
    # frame='circle' for circular radar charts
    # frame='polygon' for polygonal radar charts (default)
    pass


# Quick reference for common usage patterns:
"""
QUICK USAGE GUIDE:

1. FOR JUPYTER NOTEBOOKS (with existing coder object):
   vectors = coder._apply_comparative_normalization_to_training_data(coder.code_occurrence_overall['train'])
   labels = coder.code_occurrence_overall['train']['labels']
   code_name_across_dimensions = coder.code_occurrence_overall['train']['code_order']
   
   # Basic usage:
   fig, axs, predictive_codes, model_avgs = create_radar_visualization(
       vectors, labels, code_name_across_dimensions, n_dims=8
   )
   
   # With evaluation data overlay:
   eval_vector = coder._apply_comparative_normalization_to_training_data(coder.code_occurrence_overall['eval'])[0]
   fig, axs, predictive_codes, model_avgs = create_radar_visualization(
       vectors, labels, code_name_across_dimensions, n_dims=8, eval_vector=eval_vector
   )
   
   # With custom labels:
   fig, axs, predictive_codes, model_avgs = create_radar_visualization(
       vectors, labels, code_name_across_dimensions, n_dims=8, 
       eval_vector=eval_vector, train_label="training", eval_label="validation"
   )

2. FOR STANDALONE SCRIPT:
   python radar_reasoning_visualization.py
   (Remember to update the coder_path in the script)

3. CUSTOMIZATION OPTIONS:
   - n_dims: Number of dimensions to visualize (default: 8)
   - frame: 'circle' or 'polygon' (default: 'polygon')
   - eval_vector: Optional single evaluation vector to overlay (default: None)
   - train_label: Label for training data in legend (default: "train")
   - eval_label: Label for evaluation data in legend (default: "eval")
   
4. RETURN VALUES:
   - fig, axs: Matplotlib figure and axes objects
   - predictive_codes: List of (index, code_name, f_score) tuples
   - model_avgs: Dictionary mapping model names to average frequencies
                 (includes eval_label key if eval_vector was provided)
""" 