#!/usr/bin/env python3
"""
Visualization script for intervention experiment statistics.
Reads success rates from saved statistics files and creates a grouped bar chart.
"""

import os
import re
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from collections import defaultdict

def extract_success_rates(stats_file_path):
    """
    Extract full and partial success rates from a statistics file.
    
    Returns:
        tuple: (full_success_rate, partial_success_rate) as floats
    """
    try:
        with open(stats_file_path, 'r') as f:
            content = f.read()
            
        # Extract full success rate
        full_match = re.search(r'Full Success Rate \(thinking\): \d+/\d+ \(([0-9.]+)\)', content)
        full_rate = float(full_match.group(1)) if full_match else 0.0
        
        # Extract partial success rate  
        partial_match = re.search(r'Partial Success Rate \(answer only\): \d+/\d+ \(([0-9.]+)\)', content)
        partial_rate = float(partial_match.group(1)) if partial_match else 0.0
        
        return full_rate, partial_rate
        
    except Exception as e:
        print(f"Error reading {stats_file_path}: {e}")
        return 0.0, 0.0

def scan_intervention_experiments(base_dir, print_results=False, ignore_models=[], ignore_datasets=[]):
    """
    Scan the outputs directory for intervention experiment results.
    
    Returns:
        dict: {dataset_name: {model_name: (full_rate, partial_rate)}}
    """
    results = defaultdict(dict)
    
    if not os.path.exists(base_dir):
        print(f"Base directory {base_dir} does not exist")
        return results
    
    # Look for directories ending with "_intervention_experiment"
    for item in os.listdir(base_dir):
        item_path = os.path.join(base_dir, item)
        if os.path.isdir(item_path) and item.endswith("_intervention_experiment"):
            dataset_name = item.replace("_intervention_experiment", "")

            if dataset_name in ignore_datasets:
                continue
            # Look for model subdirectories
            for model_dir in os.listdir(item_path):
                if model_dir in ignore_models:
                    continue
                model_path = os.path.join(item_path, model_dir)
                if os.path.isdir(model_path):
                    stats_file = os.path.join(model_path, f"intervention_experiment_statistics_{model_dir}.txt")
                    
                    if os.path.exists(stats_file):
                        full_rate, partial_rate = extract_success_rates(stats_file)
                        results[dataset_name][model_dir] = (full_rate, partial_rate)
                        if print_results:
                            print(f"Found {dataset_name}/{model_dir}: Full={full_rate:.3f}, Partial={partial_rate:.3f}")
    
    return results

def create_visualization(results, if_stacked=False, if_plot_partial=True, figsize=(20, 6)):
    """
    Create a grouped bar chart visualization.
    
    Args:
        results: Dictionary containing the experiment results
        if_stacked: If True, stack 'in answer' bars on top of 'in thinking' bars
    """
    if not results:
        print("No data found to visualize")
        return
    
    # Prepare data for plotting
    datasets = list(results.keys())
    all_models = set()
    for dataset_models in results.values():
        all_models.update(dataset_models.keys())
    all_models = sorted(list(all_models))
    
    # Set up the plot
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate bar positions
    x = np.arange(len(datasets))
    width = 0.4
    
    # Extract data for each model
    full_success_data = []
    partial_success_data = []
    
    for dataset in datasets:
        dataset_full = []
        dataset_partial = []
        
        for model in all_models:
            if model in results[dataset]:
                full_rate, partial_rate = results[dataset][model]
                dataset_full.append(full_rate)
                dataset_partial.append(partial_rate)
            else:
                dataset_full.append(0)
                dataset_partial.append(0)
        
        full_success_data.append(dataset_full)
        partial_success_data.append(dataset_partial)
    
    # Convert to numpy arrays and reshape for plotting
    full_success_data = np.array(full_success_data)
    partial_success_data = np.array(partial_success_data)
    
    # Create bars for each model - same color family with different intensities
    # Base colors for each model
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(all_models) + 1))
    
    # Create darker and lighter versions for each model
    colors_full = []    # Darker colors for full success
    colors_partial = []  # Lighter colors for partial success
    
    for base_color in base_colors:
        # Convert to RGB
        rgb = base_color[:3]
        
        # Create darker version (multiply by 0.7)
        dark_rgb = tuple(c * 1 for c in rgb)
        colors_full.append(dark_rgb + (1,))  # Add alpha
        
        # Create lighter version (interpolate with white)
        light_rgb = tuple(c * 0.6 + 0.4 for c in rgb)  # 40% original + 60% white
        colors_partial.append(light_rgb + (0.9,))  # Add alpha
    
    if if_stacked:
        width *= 2
        bar_width = width / len(all_models)
        
        for i, model in enumerate(all_models):
            # Position bars for this model (centered)
            x_pos = x + i * bar_width + bar_width/2
            
            # Plot stacked bars
            full_heights = full_success_data[:, i].copy()
            partial_heights = partial_success_data[:, i].copy()
            
            # Bottom bars (Target in thinking)
            zero_mask_full = full_heights == 0
            ax.bar(x_pos, full_heights, bar_width, 
                   label=f'{model}' if i == 0 else "", 
                   color=colors_full[i])
            
            # Plot stub bars for zero thinking values
            if np.any(zero_mask_full):
                stub_height = 0.01
                ax.bar(x_pos[zero_mask_full], np.full(np.sum(zero_mask_full), stub_height), 
                       bar_width, color='none', edgecolor=colors_full[i], linewidth=1.5, 
                       linestyle='--', alpha=0.7)
            
            # Top bars (Target in answer) - stacked on top
            if if_plot_partial:
                zero_mask_partial = partial_heights == 0
                # For stacking, use the full_heights as bottom
                ax.bar(x_pos, partial_heights, bar_width, bottom=full_heights, hatch='//', edgecolor='white',
                    color=colors_partial[i])
                
                # Plot stub bars for zero answer values (stacked on top of thinking)
                if np.any(zero_mask_partial):
                    stub_height = 0.01
                    ax.bar(x_pos[zero_mask_partial], np.full(np.sum(zero_mask_partial), stub_height), 
                        bar_width, bottom=full_heights[zero_mask_partial], 
                        color='none', edgecolor=colors_partial[i], linewidth=1.5, 
                        linestyle='--', alpha=0.7)
    else:
        # Original side-by-side mode
        bar_width = width / len(all_models)
        
        for i, model in enumerate(all_models):
            # Position bars for this model
            x_pos_full = x - width/2 + i * bar_width + bar_width/2
            x_pos_partial = x + width/2 + i * bar_width + bar_width/2 + 0.05
            
            # Plot full success bars (Target in think)
            full_heights = full_success_data[:, i].copy()
            zero_mask_full = full_heights == 0
            
            # Plot normal bars for non-zero values
            ax.bar(x_pos_full, full_heights, bar_width, 
                   label=f'{model} - Target in think' if i == 0 else "", 
                   color=colors_full[i])
            
            # Plot stub bars for zero values
            if np.any(zero_mask_full):
                stub_height = 0.01  # Small visible stub
                ax.bar(x_pos_full[zero_mask_full], np.full(np.sum(zero_mask_full), stub_height), 
                       bar_width, color='none', edgecolor=colors_full[i], linewidth=1.5, 
                       linestyle='--', alpha=0.7)
            
            # Plot partial success bars (Target in answer)
            partial_heights = partial_success_data[:, i].copy()
            zero_mask_partial = partial_heights == 0
            
            # Plot normal bars for non-zero values
            ax.bar(x_pos_partial, partial_heights, bar_width,
                   label=f'{model} - Target in answer' if i == 0 else "",
                   color=colors_partial[i])
            
            # Plot stub bars for zero values
            if np.any(zero_mask_partial):
                stub_height = 0.01  # Small visible stub
                ax.bar(x_pos_partial[zero_mask_partial], np.full(np.sum(zero_mask_partial), stub_height), 
                       bar_width, color='none', edgecolor=colors_partial[i], linewidth=1.5, 
                       linestyle='--', alpha=0.7)
    
    # Add value labels on top of bars
    for i, model in enumerate(all_models):
        if if_stacked:
            if if_plot_partial:
                x_pos = x + i * bar_width + bar_width/2
                
                for j in range(len(datasets)):
                    # Label for total value (at top of stacked bar)
                    total_height = full_success_data[j, i] + partial_success_data[j, i]
                    if total_height == 0:
                        y_total = 0.03  # Above stub bars
                    else:
                        y_total = total_height + 0.01
                    ax.text(x_pos[j], y_total, 
                        f'{total_height:.2f}', 
                        ha='center', va='bottom', fontsize=9)
            else:
                x_pos = x + i * bar_width + bar_width/2
                
                for j in range(len(datasets)):
                    # Label for total value (at top of stacked bar)
                    total_height = full_success_data[j, i]
                    if total_height == 0:
                        y_total = 0.03  # Above stub bars
                    else:
                        y_total = total_height + 0.01
                    ax.text(x_pos[j], y_total, 
                        f'{total_height:.2f}', 
                        ha='center', va='bottom', fontsize=9)
                
        else:
            x_pos_full = x - width/2 + i * bar_width + bar_width/2
            x_pos_partial = x + width/2 + i * bar_width + bar_width/2 + 0.05
            
            for j in range(len(datasets)):
                # Label for full success bars
                if full_success_data[j, i] == 0:
                    # For zero values, place label above the stub bar
                    y_offset = 0.03
                else:
                    y_offset = max(0.02, full_success_data[j, i] + 0.01)
                ax.text(x_pos_full[j], y_offset, 
                       f'{full_success_data[j, i]:.2f}', 
                       ha='center', va='bottom', fontsize=8)
                
                # Label for partial success bars
                if partial_success_data[j, i] == 0:
                    # For zero values, place label above the stub bar
                    y_offset = 0.03
                else:
                    y_offset = max(0.02, partial_success_data[j, i] + 0.01)
                ax.text(x_pos_partial[j], y_offset, 
                       f'{partial_success_data[j, i]:.2f}', 
                       ha='center', va='bottom', fontsize=8)
    
    # Customize the plot
    ax.set_ylabel('Success Rate')
    ax.set_title("Success Rate of Intervening Models's Reasoning")
    ax.set_xticks(x + width/2)
    ax.set_xticklabels(datasets, ha='center')
    
    # Set y-axis limits to provide space for value labels
    if if_stacked:
        # For stacked mode, consider the sum of both values
        max_value = np.max(full_success_data + partial_success_data)
    else:
        max_value = max(np.max(full_success_data), np.max(partial_success_data))
    ax.set_ylim(0, max_value * 1.15)
    
    # Create custom legend
    legend_elements = []
    
    # Add model entries (using base colors)
    for i, model in enumerate(all_models):
        legend_elements.append(plt.Rectangle((0,0),1,1, fc=base_colors[i], label=f'{model}'))
    
    # Add separator line
    legend_elements.append(Line2D([0], [0], color='none', label=''))
    
    # Add pattern explanation entries
    if if_stacked:
        legend_elements.append(plt.Rectangle((0,0),1,1, fc='gray', alpha=1.0, label='Target in thinking (bottom, darker)'))
        legend_elements.append(plt.Rectangle((0,0),1,1, fc='lightgray', alpha=0.9, label='Target in answer (top, lighter)'))
    else:
        legend_elements.append(plt.Rectangle((0,0),1,1, fc='gray', alpha=1.0, label='Target in thinking (darker)'))
        legend_elements.append(plt.Rectangle((0,0),1,1, fc='lightgray', alpha=0.9, label='Target in answer (lighter)'))
    # legend_elements.append(plt.Rectangle((0,0),1,1, fc='none', edgecolor='gray', linestyle='--', alpha=0.7, label='Zero values (dashed outline)'))
    
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(0.98, 1), frameon=False)
    
    # Remove grid and customize spines
    ax.grid(False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Set font weight to normal (not bold)
    plt.rcParams['font.weight'] = 'normal'
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # Save the plot
    output_path = '/n/home04/yidachen/reasoning_characteristics/intervention_experiment_visualization.png'
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    # Show the plot
    plt.show()

def main(if_stacked=False):
    """Main function to run the visualization.
    
    Args:
        if_stacked: If True, create stacked bars instead of side-by-side bars
    """
    base_dir = "/n/home04/yidachen/reasoning_characteristics/outputs"
    
    print("Scanning for intervention experiment results...")
    results = scan_intervention_experiments(base_dir)
    
    if not results:
        print("No intervention experiment results found.")
        return
    
    print(f"\nFound results for {len(results)} datasets:")
    for dataset, models in results.items():
        print(f"  {dataset}: {len(models)} models")
    
    mode = "stacked" if if_stacked else "side-by-side"
    print(f"\nCreating visualization in {mode} mode...")
    create_visualization(results, if_stacked=if_stacked)

if __name__ == "__main__":
    # Run with side-by-side bars (default)
    main()
    
    # Uncomment the line below to use stacked bars instead
    # main(if_stacked=True)