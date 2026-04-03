import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter, defaultdict
import numpy as np
import re
from src.model_name_translator import model_name_decoder


def convert_to_sequence_of_reasonings(annotation_text, codebook_keys):
    """
    Extract reasoning behavior names from annotation text in order of appearance.
    
    Args:
        annotation_text (str): The annotation document text
        codebook_keys (list): List of possible reasoning behavior names from coder.codebook.keys()
    
    Returns:
        list: Reasoning behaviors in order of appearance
    """
    
    # Find all text within square brackets
    bracket_pattern = r'\[([^\]]+)\]'
    matches = re.findall(bracket_pattern, annotation_text)
    
    # Filter matches to only include those that are in codebook_keys
    reasoning_behaviors = []
    for match in matches:
        if match in codebook_keys:
            reasoning_behaviors.append(match)
    
    return reasoning_behaviors


def convert_bor_annotations_to_sor(coder, split, correct_ids, wrong_ids, model_of_interest):
    correct_reasoning_sors = []
    for idx in correct_ids:
        if idx not in coder.training_logs[split]:
            continue
        model_order = coder.training_logs[split][idx]["data"]["labels"]
        model_actual_name_order = [model_name_decoder[model_name] for model_name in model_order]
        model_idx = model_actual_name_order.index(model_of_interest)
        a_or_b = "AB"[model_idx]
        
        both_annotation = coder.training_logs[split][idx]['output_classification']
        extracted_annotation = both_annotation[both_annotation.find(f"=== Annotation for OUTPUT {a_or_b}"):
            both_annotation.rfind(f"-----End of OUTPUT {a_or_b}")]
        correct_reasoning_sors.append(convert_to_sequence_of_reasonings(extracted_annotation, coder.codebook.keys()))

    wrong_reasoning_sors = []
    for idx in wrong_ids:
        if idx not in coder.training_logs[split]:
            continue
        model_order = coder.training_logs[split][idx]["data"]["labels"]
        model_actual_name_order = [model_name_decoder[model_name] for model_name in model_order]
        model_idx = model_actual_name_order.index(model_of_interest)
        a_or_b = "AB"[model_idx]
        
        both_annotation = coder.training_logs[split][idx]['output_classification']
        extracted_annotation = both_annotation[both_annotation.find(f"=== Annotation for OUTPUT {a_or_b}"):
            both_annotation.rfind(f"-----End of OUTPUT {a_or_b}")]
        wrong_reasoning_sors.append(convert_to_sequence_of_reasonings(extracted_annotation, coder.codebook.keys()))

    return correct_reasoning_sors, wrong_reasoning_sors


def aggregate_sequences_fixed_length(sequences, target_length):
    """
    Aggregate sequences to a fixed target length.
    
    Args:
        sequences (list): List of reasoning sequences
        target_length (int): Target length for all sequences
    
    Returns:
        list: Aggregated sequences
    """
    aggregated = []
    
    for seq in sequences:
        if len(seq) == 0:
            continue
            
        if len(seq) <= target_length:
            # If sequence is shorter than target, pad with None or keep as is
            aggregated.append(seq + [None] * (target_length - len(seq)))
        else:
            # Divide sequence into target_length bins
            bin_size = len(seq) / target_length
            agg_seq = []
            
            for i in range(target_length):
                start_idx = int(i * bin_size)
                end_idx = int((i + 1) * bin_size)
                
                # Get the most common behavior in this bin
                bin_behaviors = seq[start_idx:end_idx]
                if bin_behaviors:
                    # Use most common behavior in the bin, or first if tie
                    most_common = Counter(bin_behaviors).most_common(1)[0][0]
                    agg_seq.append(most_common)
                else:
                    agg_seq.append(None)
            
            aggregated.append(agg_seq)
    
    return aggregated

def aggregate_sequences_fixed_interval(sequences, interval_size):
    """
    Aggregate sequences using fixed interval size.
    
    Args:
        sequences (list): List of reasoning sequences
        interval_size (int): Number of behaviors to aggregate into one
    
    Returns:
        list: Aggregated sequences
    """
    aggregated = []
    
    for seq in sequences:
        if len(seq) == 0:
            continue
            
        agg_seq = []
        for i in range(0, len(seq), interval_size):
            chunk = seq[i:i + interval_size]
            if chunk:
                # Use most common behavior in the chunk
                most_common = Counter(chunk).most_common(1)[0][0]
                agg_seq.append(most_common)
        
        aggregated.append(agg_seq)
    
    return aggregated



def compress_consecutive_behaviors(sequence):
    """
    Compress consecutive repeated behaviors into single occurrences.
    
    Parameters:
    -----------
    sequence : list
        List of behaviors/codes, e.g., [A, B, B, C, A, A, A, A, C, D]
    
    Returns:
    --------
    list
        Compressed sequence, e.g., [A, B, C, A, C, D]
    """
    if not sequence:
        return sequence
    
    compressed = [sequence[0]]  # Start with first element
    
    for i in range(1, len(sequence)):
        if sequence[i] != sequence[i-1]:  # Only add if different from previous
            compressed.append(sequence[i])
    
    return compressed

def analyze_reasoning_sequences(correct_reasoning_sors, wrong_reasoning_sors, 
                              aggregation_method='fixed_length', 
                              target_length=20, 
                              interval_size=15, compressed=False):
    """
    Analyze and visualize reasoning sequences for correct vs wrong answers with aggregation.
    
    Args:
        correct_reasoning_sors (list): List of reasoning sequences for correct answers
        wrong_reasoning_sors (list): List of reasoning sequences for wrong answers
        aggregation_method (str): 'fixed_length', 'fixed_interval', or 'none'
        target_length (int): Target length for fixed_length aggregation
        interval_size (int): Interval size for fixed_interval aggregation
    """
    
    print("=== Original Statistics ===")
    print(f"Number of correct reasoning traces: {len(correct_reasoning_sors)}")
    print(f"Number of wrong reasoning traces: {len(wrong_reasoning_sors)}")

    # Apply compression preprocessing if requested
    if compressed:
        print("Applying consecutive behavior compression...")
        correct_reasoning_sors = [compress_consecutive_behaviors(seq) for seq in correct_reasoning_sors]
        wrong_reasoning_sors = [compress_consecutive_behaviors(seq) for seq in wrong_reasoning_sors]
    
    original_correct_lengths = [len(seq) for seq in correct_reasoning_sors]
    original_wrong_lengths = [len(seq) for seq in wrong_reasoning_sors]
    
    print(f"Original correct traces - Avg length: {np.mean(original_correct_lengths):.2f}, Range: {min(original_correct_lengths)}-{max(original_correct_lengths)}")
    print(f"Original wrong traces - Avg length: {np.mean(original_wrong_lengths):.2f}, Range: {min(original_wrong_lengths)}-{max(original_wrong_lengths)}")
    
    # Apply aggregation
    if aggregation_method == 'fixed_length':
        print(f"\nApplying fixed length aggregation to {target_length} positions...")
        correct_agg = aggregate_sequences_fixed_length(correct_reasoning_sors, target_length)
        wrong_agg = aggregate_sequences_fixed_length(wrong_reasoning_sors, target_length)
        max_position = target_length
    elif aggregation_method == 'fixed_interval':
        print(f"\nApplying fixed interval aggregation with interval size {interval_size}...")
        correct_agg = aggregate_sequences_fixed_interval(correct_reasoning_sors, interval_size)
        wrong_agg = aggregate_sequences_fixed_interval(wrong_reasoning_sors, interval_size)
        max_position = max(len(seq) for seq in correct_agg + wrong_agg)
    else:
        print("\nNo aggregation applied...")
        correct_agg = correct_reasoning_sors
        wrong_agg = wrong_reasoning_sors
        max_position = min(20, max(max(original_correct_lengths), max(original_wrong_lengths)))
    
    # Print aggregated statistics
    agg_correct_lengths = [len(seq) for seq in correct_agg]
    agg_wrong_lengths = [len(seq) for seq in wrong_agg]
    
    print(f"\nAggregated correct traces - Avg length: {np.mean(agg_correct_lengths):.2f}, Range: {min(agg_correct_lengths)}-{max(agg_correct_lengths)}")
    print(f"Aggregated wrong traces - Avg length: {np.mean(agg_wrong_lengths):.2f}, Range: {min(agg_wrong_lengths)}-{max(agg_wrong_lengths)}")
    
    # Analyze aggregated sequences
    correct_behaviors = [behavior for seq in correct_agg for behavior in seq if behavior is not None]
    wrong_behaviors = [behavior for seq in wrong_agg for behavior in seq if behavior is not None]
    
    correct_counter = Counter(correct_behaviors)
    wrong_counter = Counter(wrong_behaviors)
    
    # Get all unique behaviors
    all_behaviors = set(correct_behaviors + wrong_behaviors)
    
    # Create visualizations
    fig, axes = plt.subplots(3, 3, figsize=(36, 27))
    
    # Plot 1: Original vs Aggregated sequence length distribution
    axes[0, 0].hist(original_correct_lengths, alpha=0.7, label='Original Correct', bins=30, color='lightgreen')
    axes[0, 0].hist(original_wrong_lengths, alpha=0.7, label='Original Wrong', bins=30, color='lightcoral')
    axes[0, 0].set_xlabel('Sequence Length')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Original Sequence Length Distribution')
    axes[0, 0].legend()
    axes[0, 0].set_yscale('log')  # Log scale for better visualization
    
    # Plot 2: Aggregated sequence lengths
    axes[0, 1].hist(agg_correct_lengths, alpha=0.7, label='Agg. Correct', bins=20, color='green')
    axes[0, 1].hist(agg_wrong_lengths, alpha=0.7, label='Agg. Wrong', bins=20, color='red')
    axes[0, 1].set_xlabel('Aggregated Sequence Length')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Aggregated Sequence Length Distribution')
    axes[0, 1].legend()
    
    # Plot 3: Overall behavior frequency comparison
    behavior_comparison = []
    for behavior in all_behaviors:
        correct_freq = correct_counter.get(behavior, 0) / len(correct_behaviors) if correct_behaviors else 0
        wrong_freq = wrong_counter.get(behavior, 0) / len(wrong_behaviors) if wrong_behaviors else 0
        behavior_comparison.append({
            'behavior': behavior,
            'correct_freq': correct_freq,
            'wrong_freq': wrong_freq,
            'difference': correct_freq - wrong_freq
        })
    
    behavior_df = pd.DataFrame(behavior_comparison)
    behavior_df = behavior_df.sort_values('difference', ascending=False)
    
    # Show top 15 behaviors with largest differences
    top_behaviors_df = behavior_df.head(15)
    y_pos = np.arange(len(top_behaviors_df))
    axes[0, 2].barh(y_pos, top_behaviors_df['difference'], 
                    color=['green' if x > 0 else 'red' for x in top_behaviors_df['difference']])
    axes[0, 2].set_yticks(y_pos)
    axes[0, 2].set_yticklabels([b[:25] + '...' if len(str(b)) > 25 else str(b) for b in top_behaviors_df['behavior']], fontsize=8)
    axes[0, 2].set_xlabel('Frequency Difference (Correct - Wrong)')
    axes[0, 2].set_title('Top Reasoning Behavior Frequency Differences')
    axes[0, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 4: Position-based analysis
    position_analysis = defaultdict(lambda: {'correct': 0, 'wrong': 0})
    
    for seq in correct_agg:
        for pos, behavior in enumerate(seq[:max_position]):
            if behavior is not None:
                position_analysis[pos]['correct'] += 1
    
    for seq in wrong_agg:
        for pos, behavior in enumerate(seq[:max_position]):
            if behavior is not None:
                position_analysis[pos]['wrong'] += 1
    
    positions = list(range(min(max_position, 20)))  # Limit to 20 for visualization
    correct_counts = [position_analysis[pos]['correct'] for pos in positions]
    wrong_counts = [position_analysis[pos]['wrong'] for pos in positions]
    
    x = np.arange(len(positions))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.7)
    axes[1, 0].bar(x + width/2, wrong_counts, width, label='Wrong', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Position in Aggregated Sequence')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title(f'Reasoning Steps by Position (First {len(positions)})')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels([f'{i+1}' for i in positions])
    axes[1, 0].legend()
    
    # Plot 5: Heatmap of behavior by position (top behaviors) - Correct
    top_behaviors = [item[0] for item in correct_counter.most_common(15)]
    viz_positions = min(max_position, 15)  # Limit positions for readability
    
    position_behavior_matrix = np.zeros((len(top_behaviors), viz_positions))
    
    for seq in correct_agg:
        for pos, behavior in enumerate(seq[:viz_positions]):
            if behavior in top_behaviors:
                behavior_idx = top_behaviors.index(behavior)
                position_behavior_matrix[behavior_idx, pos] += 1
    
    # Normalize by number of sequences
    position_behavior_matrix = position_behavior_matrix / len(correct_agg)
    
    sns.heatmap(position_behavior_matrix, 
                xticklabels=[f'Pos {i+1}' for i in range(viz_positions)],
                yticklabels=[str(b)[:20] + '...' if len(str(b)) > 20 else str(b) for b in top_behaviors],
                ax=axes[1, 1], cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Frequency'})
    axes[1, 1].set_title('Reasoning Behavior Heatmap - Correct Answers')
    
    # Plot 6: Heatmap for wrong answers
    position_behavior_matrix_wrong = np.zeros((len(top_behaviors), viz_positions))
    
    for seq in wrong_agg:
        for pos, behavior in enumerate(seq[:viz_positions]):
            if behavior in top_behaviors:
                behavior_idx = top_behaviors.index(behavior)
                position_behavior_matrix_wrong[behavior_idx, pos] += 1
    
    position_behavior_matrix_wrong = position_behavior_matrix_wrong / len(wrong_agg)
    
    sns.heatmap(position_behavior_matrix_wrong,
                xticklabels=[f'Pos {i+1}' for i in range(viz_positions)],
                yticklabels=[str(b)[:20] + '...' if len(str(b)) > 20 else str(b) for b in top_behaviors],
                ax=axes[1, 2], cmap='YlOrRd', annot=True, fmt='.2f', cbar_kws={'label': 'Frequency'})
    axes[1, 2].set_title('Reasoning Behavior Heatmap - Wrong Answers')
    
    # Plot 7: Difference heatmap (Correct - Wrong)
    difference_matrix = position_behavior_matrix - position_behavior_matrix_wrong
    
    sns.heatmap(difference_matrix,
                xticklabels=[f'Pos {i+1}' for i in range(viz_positions)],
                yticklabels=[str(b)[:20] + '...' if len(str(b)) > 20 else str(b) for b in top_behaviors],
                ax=axes[2, 0], cmap='RdBu_r', center=0, annot=True, fmt='.2f',
                cbar_kws={'label': 'Frequency Difference'})
    axes[2, 0].set_title('Behavior Frequency Difference (Correct - Wrong)')
    
    # Plot 8: Transition analysis
    correct_transitions = []
    for seq in correct_reasoning_sors:
        clean_seq = [b for b in seq if b is not None]
        for i in range(len(clean_seq) - 1):
            correct_transitions.append((clean_seq[i], clean_seq[i + 1]))
    
    wrong_transitions = []
    for seq in wrong_reasoning_sors:
        clean_seq = [b for b in seq if b is not None]
        for i in range(len(clean_seq) - 1):
            wrong_transitions.append((clean_seq[i], clean_seq[i + 1]))
    
    correct_transition_counter = Counter(correct_transitions)
    wrong_transition_counter = Counter(wrong_transitions)
    
    # Transition difference analysis
    all_transitions = set(list(correct_transition_counter.keys())[:20] + list(wrong_transition_counter.keys())[:20])
    transition_comparison = []
    
    for trans in all_transitions:
        correct_freq = correct_transition_counter.get(trans, 0) / len(correct_transitions) if correct_transitions else 0
        wrong_freq = wrong_transition_counter.get(trans, 0) / len(wrong_transitions) if wrong_transitions else 0
        transition_comparison.append({
            'transition': f"{str(trans[0])[:12]}... -> {str(trans[1])[:12]}..." if len(str(trans[0])) > 12 or len(str(trans[1])) > 12 else f"{trans[0]} -> {trans[1]}",
            'difference': correct_freq - wrong_freq,
            'correct_freq': correct_freq,
            'wrong_freq': wrong_freq
        })
    
    transition_df = pd.DataFrame(transition_comparison)
    # show both the top 15 transitions and the top 15 transitions with positive difference
    # and the least 15 transitions with negative difference
    transition_df_positive = transition_df[transition_df['difference'] > 0].head(10)
    transition_df_negative = transition_df[transition_df['difference'] < 0].head(10)
    transition_df = pd.concat([transition_df_positive, transition_df_negative])
    transition_df = transition_df.sort_values('difference', ascending=False)
    
    y_pos = np.arange(len(transition_df))
    
    axes[2, 1].barh(y_pos, transition_df['difference'], 
                    color=['green' if x > 0 else 'red' for x in transition_df['difference']])
    axes[2, 1].set_yticks(y_pos)
    axes[2, 1].set_yticklabels(transition_df['transition'], fontsize=7)
    axes[2, 1].set_xlabel('Frequency Difference (Correct - Wrong)')
    axes[2, 1].set_title('Top Reasoning Transitions Difference')
    axes[2, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 9: Behavior diversity over positions
    position_diversity = {'correct': [], 'wrong': []}
    
    for pos in range(min(max_position, 15)):
        # Calculate unique behaviors at each position
        correct_behaviors_at_pos = []
        wrong_behaviors_at_pos = []
        
        for seq in correct_agg:
            if pos < len(seq) and seq[pos] is not None:
                correct_behaviors_at_pos.append(seq[pos])
        
        for seq in wrong_agg:
            if pos < len(seq) and seq[pos] is not None:
                wrong_behaviors_at_pos.append(seq[pos])
        
        position_diversity['correct'].append(len(set(correct_behaviors_at_pos)))
        position_diversity['wrong'].append(len(set(wrong_behaviors_at_pos)))
    
    positions_div = range(len(position_diversity['correct']))
    axes[2, 2].plot(positions_div, position_diversity['correct'], 'g-o', label='Correct', linewidth=2)
    axes[2, 2].plot(positions_div, position_diversity['wrong'], 'r-o', label='Wrong', linewidth=2)
    axes[2, 2].set_xlabel('Position in Sequence')
    axes[2, 2].set_ylabel('Number of Unique Behaviors')
    axes[2, 2].set_title('Reasoning Diversity by Position')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed statistics
    print(f"\n=== Aggregated Analysis Results ===")
    print(f"Aggregation method: {aggregation_method}")
    if aggregation_method == 'fixed_length':
        print(f"Target length: {target_length}")
    elif aggregation_method == 'fixed_interval':
        print(f"Interval size: {interval_size}")
    
    print(f"\nTop 5 Reasoning Transitions - CORRECT:")
    for (from_behavior, to_behavior), count in correct_transition_counter.most_common(5):
        print(f"  {str(from_behavior)[:30]}... -> {str(to_behavior)[:30]}...: {count}")
    
    print(f"\nTop 5 Reasoning Transitions - WRONG:")
    for (from_behavior, to_behavior), count in wrong_transition_counter.most_common(5):
        print(f"  {str(from_behavior)[:30]}... -> {str(to_behavior)[:30]}...: {count}")
    
    return behavior_df, correct_counter, wrong_counter, correct_agg, wrong_agg


def create_reasoning_heatmaps(correct_reasoning_sors, wrong_reasoning_sors, 
                             aggregation_method='fixed_length', 
                             target_length=20, 
                             interval_size=15,
                             top_behaviors=20,
                             figsize=(36, 12),
                              compressed=False):
    """
    Create only the heatmap visualizations for reasoning sequence analysis.
    
    Args:
        correct_reasoning_sors (list): List of reasoning sequences for correct answers
        wrong_reasoning_sors (list): List of reasoning sequences for wrong answers
        aggregation_method (str): 'fixed_length', 'fixed_interval', or 'none'
        target_length (int): Target length for fixed_length aggregation
        interval_size (int): Interval size for fixed_interval aggregation
        top_behaviors (int): Number of top behaviors to show in heatmap
        figsize (tuple): Figure size for the plot
    
    Returns:
        tuple: (behavior_matrix_correct, behavior_matrix_wrong, difference_matrix, behavior_labels, position_labels)
    """
    # Apply compression preprocessing if requested
    if compressed:
        print("Applying consecutive behavior compression...")
        correct_reasoning_sors = [compress_consecutive_behaviors(seq) for seq in correct_reasoning_sors]
        wrong_reasoning_sors = [compress_consecutive_behaviors(seq) for seq in wrong_reasoning_sors]
    
    print("=== Heatmap Analysis ===")
    print(f"Number of correct reasoning traces: {len(correct_reasoning_sors)}")
    print(f"Number of wrong reasoning traces: {len(wrong_reasoning_sors)}")
    
    original_correct_lengths = [len(seq) for seq in correct_reasoning_sors]
    original_wrong_lengths = [len(seq) for seq in wrong_reasoning_sors]
    
    print(f"Original sequence lengths - Correct: {np.mean(original_correct_lengths):.1f}±{np.std(original_correct_lengths):.1f}, Wrong: {np.mean(original_wrong_lengths):.1f}±{np.std(original_wrong_lengths):.1f}")
    
    # Apply aggregation
    if aggregation_method == 'fixed_length':
        print(f"Applying fixed length aggregation to {target_length} positions...")
        correct_agg = aggregate_sequences_fixed_length(correct_reasoning_sors, target_length)
        wrong_agg = aggregate_sequences_fixed_length(wrong_reasoning_sors, target_length)
        max_position = target_length
        position_labels = [f'Pos {i+1}' for i in range(max_position)]
    elif aggregation_method == 'fixed_interval':
        print(f"Applying fixed interval aggregation with interval size {interval_size}...")
        correct_agg = aggregate_sequences_fixed_interval(correct_reasoning_sors, interval_size)
        wrong_agg = aggregate_sequences_fixed_interval(wrong_reasoning_sors, interval_size)
        max_position = max(len(seq) for seq in correct_agg + wrong_agg)
        position_labels = [f'Chunk {i+1}' for i in range(max_position)]
    else:
        print("No aggregation applied...")
        correct_agg = correct_reasoning_sors
        wrong_agg = wrong_reasoning_sors
        max_position = min(30, max(max(original_correct_lengths), max(original_wrong_lengths)))
        position_labels = [f'Pos {i+1}' for i in range(max_position)]
    
    # Get aggregated statistics
    agg_correct_lengths = [len(seq) for seq in correct_agg]
    agg_wrong_lengths = [len(seq) for seq in wrong_agg]
    
    print(f"Aggregated sequence lengths - Correct: {np.mean(agg_correct_lengths):.1f}±{np.std(agg_correct_lengths):.1f}, Wrong: {np.mean(agg_wrong_lengths):.1f}±{np.std(agg_wrong_lengths):.1f}")
    print(f"Heatmap dimensions: {top_behaviors} behaviors × {max_position} positions")
    
    # Get all behaviors and find top ones
    correct_behaviors = [behavior for seq in correct_agg for behavior in seq if behavior is not None]
    wrong_behaviors = [behavior for seq in wrong_agg for behavior in seq if behavior is not None]
    
    correct_counter = Counter(correct_behaviors)
    wrong_counter = Counter(wrong_behaviors)
    
    # Get top behaviors based on combined frequency
    all_behavior_counts = Counter(correct_behaviors + wrong_behaviors)
    top_behavior_names = [item[0] for item in all_behavior_counts.most_common(top_behaviors)]
    
    # Create behavior frequency matrices
    correct_matrix = np.zeros((len(top_behavior_names), max_position))
    wrong_matrix = np.zeros((len(top_behavior_names), max_position))
    
    # Fill correct matrix
    for seq in correct_agg:
        for pos, behavior in enumerate(seq[:max_position]):
            if behavior is not None and behavior in top_behavior_names:
                behavior_idx = top_behavior_names.index(behavior)
                correct_matrix[behavior_idx, pos] += 1
    
    # Fill wrong matrix
    for seq in wrong_agg:
        for pos, behavior in enumerate(seq[:max_position]):
            if behavior is not None and behavior in top_behavior_names:
                behavior_idx = top_behavior_names.index(behavior)
                wrong_matrix[behavior_idx, pos] += 1
    
    # Normalize by number of sequences
    correct_matrix = correct_matrix / len(correct_agg)
    wrong_matrix = wrong_matrix / len(wrong_agg)
    
    # Calculate difference matrix
    difference_matrix = correct_matrix - wrong_matrix
    
    # Create behavior labels (truncate long names)
    behavior_labels = []
    for behavior in top_behavior_names:
        behavior_str = str(behavior)
        if len(behavior_str) > 40:
            behavior_labels.append(behavior_str[:37] + '...')
        else:
            behavior_labels.append(behavior_str)
    
    # Create the heatmaps
    fig, axes = plt.subplots(3, 1, figsize=figsize)
    
    # Correct sequences heatmap
    sns.heatmap(correct_matrix,
                xticklabels=position_labels,
                yticklabels=behavior_labels,
                ax=axes[0], 
                cmap='YlOrRd', 
                annot=False,  # Turn off annotations for cleaner look with many cells
                cbar_kws={'label': 'Frequency'},
                vmin=0)
    axes[0].set_title('Correct Sequences', fontsize=14)
    axes[0].set_xlabel('Position in Sequence', fontsize=12)
    axes[0].set_ylabel('Reasoning Behaviors', fontsize=12)
    
    # Wrong sequences heatmap
    sns.heatmap(wrong_matrix,
                xticklabels=position_labels,
                yticklabels=behavior_labels,
                ax=axes[1], 
                cmap='YlOrRd', 
                annot=False,
                cbar_kws={'label': 'Frequency'},
                vmin=0)
    axes[1].set_title('Wrong Sequences', fontsize=14)
    axes[1].set_xlabel('Position in Sequence', fontsize=12)
    axes[1].set_ylabel('')  # Remove y-label for middle plot
    
    # Difference heatmap
    max_abs_diff = np.max(np.abs(difference_matrix))
    sns.heatmap(difference_matrix,
                xticklabels=position_labels,
                yticklabels=behavior_labels,
                ax=axes[2], 
                cmap='RdBu_r', 
                center=0, 
                annot=False,
                cbar_kws={'label': 'Frequency Difference'},
                vmin=-max_abs_diff,
                vmax=max_abs_diff)
    axes[2].set_title('Difference (Correct - Wrong)', fontsize=14)
    axes[2].set_xlabel('Position in Sequence', fontsize=12)
    axes[2].set_ylabel('')  # Remove y-label for right plot
    
    # Adjust layout and tick parameters
    for ax in axes:
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Print some summary statistics
    print(f"\n=== Summary Statistics ===")
    print(f"Most distinctive behaviors (highest absolute difference):")
    
    # Calculate max absolute difference for each behavior
    behavior_max_diff = []
    for i, behavior in enumerate(top_behavior_names):
        max_abs_diff_for_behavior = np.max(np.abs(difference_matrix[i, :]))
        avg_diff = np.mean(difference_matrix[i, :])
        behavior_max_diff.append({
            'behavior': behavior,
            'max_abs_diff': max_abs_diff_for_behavior,
            'avg_diff': avg_diff,
            'correct_freq': np.mean(correct_matrix[i, :]),
            'wrong_freq': np.mean(wrong_matrix[i, :])
        })
    
    # Sort by max absolute difference
    behavior_max_diff.sort(key=lambda x: x['max_abs_diff'], reverse=True)
    
    for i, item in enumerate(behavior_max_diff[:10]):
        behavior_name = item['behavior']
        if len(str(behavior_name)) > 50:
            behavior_name = str(behavior_name)[:47] + '...'
        print(f"{i+1:2d}. {behavior_name}")
        print(f"     Max |diff|: {item['max_abs_diff']:.3f}, Avg diff: {item['avg_diff']:.3f}")
        print(f"     Correct freq: {item['correct_freq']:.3f}, Wrong freq: {item['wrong_freq']:.3f}")
    
    return correct_matrix, wrong_matrix, difference_matrix, behavior_labels, position_labels


# def analyze_reasoning_sequences(correct_reasoning_sors, wrong_reasoning_sors):
#     """
#     Analyze and visualize reasoning sequences for correct vs wrong answers.
    
#     Args:
#         correct_reasoning_sors (list): List of reasoning sequences for correct answers
#         wrong_reasoning_sors (list): List of reasoning sequences for wrong answers
#     """
    
#     # 1. Basic statistics
#     print("=== Basic Statistics ===")
#     print(f"Number of correct reasoning traces: {len(correct_reasoning_sors)}")
#     print(f"Number of wrong reasoning traces: {len(wrong_reasoning_sors)}")
    
#     correct_lengths = [len(seq) for seq in correct_reasoning_sors]
#     wrong_lengths = [len(seq) for seq in wrong_reasoning_sors]
    
#     print(f"Correct traces - Avg length: {np.mean(correct_lengths):.2f}, Range: {min(correct_lengths)}-{max(correct_lengths)}")
#     print(f"Wrong traces - Avg length: {np.mean(wrong_lengths):.2f}, Range: {min(wrong_lengths)}-{max(wrong_lengths)}")
    
#     # 2. Reasoning behavior frequency analysis
#     correct_behaviors = [behavior for seq in correct_reasoning_sors for behavior in seq]
#     wrong_behaviors = [behavior for seq in wrong_reasoning_sors for behavior in seq]
    
#     correct_counter = Counter(correct_behaviors)
#     wrong_counter = Counter(wrong_behaviors)
    
#     # Get all unique behaviors
#     all_behaviors = set(correct_behaviors + wrong_behaviors)
    
#     # 3. Create visualizations
#     fig, axes = plt.subplots(2, 3, figsize=(24, 12))
    
#     # Plot 1: Sequence length distribution
#     axes[0, 0].hist(correct_lengths, alpha=0.7, label='Correct', bins=20, color='green')
#     axes[0, 0].hist(wrong_lengths, alpha=0.7, label='Wrong', bins=20, color='red')
#     axes[0, 0].set_xlabel('Sequence Length')
#     axes[0, 0].set_ylabel('Frequency')
#     axes[0, 0].set_title('Distribution of Reasoning Sequence Lengths')
#     axes[0, 0].legend()
    
#     # Plot 2: Overall behavior frequency comparison
#     behavior_comparison = []
#     for behavior in all_behaviors:
#         correct_freq = correct_counter.get(behavior, 0) / len(correct_behaviors) if correct_behaviors else 0
#         wrong_freq = wrong_counter.get(behavior, 0) / len(wrong_behaviors) if wrong_behaviors else 0
#         behavior_comparison.append({
#             'behavior': behavior,
#             'correct_freq': correct_freq,
#             'wrong_freq': wrong_freq,
#             'difference': correct_freq - wrong_freq
#         })
    
#     behavior_df = pd.DataFrame(behavior_comparison)
#     behavior_df = behavior_df.sort_values('difference', ascending=False)
    
#     y_pos = np.arange(len(behavior_df))
#     axes[0, 1].barh(y_pos, behavior_df['difference'], color=['green' if x > 0 else 'red' for x in behavior_df['difference']])
#     axes[0, 1].set_yticks(y_pos)
#     axes[0, 1].set_yticklabels(behavior_df['behavior'], fontsize=8)
#     axes[0, 1].set_xlabel('Frequency Difference (Correct - Wrong)')
#     axes[0, 1].set_title('Reasoning Behavior Frequency Difference')
#     axes[0, 1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
#     # Plot 3: Position-based analysis (first 10 positions)
#     max_position = min(10, max(max(correct_lengths), max(wrong_lengths)))
#     position_analysis = defaultdict(lambda: {'correct': 0, 'wrong': 0})
    
#     for seq in correct_reasoning_sors:
#         for pos, behavior in enumerate(seq[:max_position]):
#             position_analysis[pos]['correct'] += 1
    
#     for seq in wrong_reasoning_sors:
#         for pos, behavior in enumerate(seq[:max_position]):
#             position_analysis[pos]['wrong'] += 1
    
#     positions = list(range(max_position))
#     correct_counts = [position_analysis[pos]['correct'] for pos in positions]
#     wrong_counts = [position_analysis[pos]['wrong'] for pos in positions]
    
#     x = np.arange(len(positions))
#     width = 0.35
    
#     axes[0, 2].bar(x - width/2, correct_counts, width, label='Correct', color='green', alpha=0.7)
#     axes[0, 2].bar(x + width/2, wrong_counts, width, label='Wrong', color='red', alpha=0.7)
#     axes[0, 2].set_xlabel('Position in Sequence')
#     axes[0, 2].set_ylabel('Count')
#     axes[0, 2].set_title('Reasoning Steps by Position (First 10)')
#     axes[0, 2].set_xticks(x)
#     axes[0, 2].set_xticklabels([f'{i+1}' for i in positions])
#     axes[0, 2].legend()
    
#     # Plot 4: Heatmap of behavior by position (top behaviors)
#     top_behaviors = [item[0] for item in correct_counter.most_common(10)]
    
#     position_behavior_matrix = np.zeros((len(top_behaviors), max_position))
    
#     for seq in correct_reasoning_sors:
#         for pos, behavior in enumerate(seq[:max_position]):
#             if behavior in top_behaviors:
#                 behavior_idx = top_behaviors.index(behavior)
#                 position_behavior_matrix[behavior_idx, pos] += 1
    
#     sns.heatmap(position_behavior_matrix / len(correct_reasoning_sors), 
#                 xticklabels=[f'Pos {i+1}' for i in range(max_position)],
#                 yticklabels=[b[:20] + '...' if len(b) > 20 else b for b in top_behaviors],
#                 ax=axes[1, 0], cmap='YlOrRd', annot=True, fmt='.2f')
#     axes[1, 0].set_title('Reasoning Behavior Heatmap - Correct Answers')
    
#     # Plot 5: Similar heatmap for wrong answers
#     position_behavior_matrix_wrong = np.zeros((len(top_behaviors), max_position))
    
#     for seq in wrong_reasoning_sors:
#         for pos, behavior in enumerate(seq[:max_position]):
#             if behavior in top_behaviors:
#                 behavior_idx = top_behaviors.index(behavior)
#                 position_behavior_matrix_wrong[behavior_idx, pos] += 1
    
#     sns.heatmap(position_behavior_matrix_wrong / len(wrong_reasoning_sors),
#                 xticklabels=[f'Pos {i+1}' for i in range(max_position)],
#                 yticklabels=[b[:20] + '...' if len(b) > 20 else b for b in top_behaviors],
#                 ax=axes[1, 1], cmap='YlOrRd', annot=True, fmt='.2f')
#     axes[1, 1].set_title('Reasoning Behavior Heatmap - Wrong Answers')
    
#     # Plot 6: Transition analysis (most common 2-grams)
#     correct_transitions = []
#     for seq in correct_reasoning_sors:
#         for i in range(len(seq) - 1):
#             correct_transitions.append((seq[i], seq[i + 1]))
    
#     wrong_transitions = []
#     for seq in wrong_reasoning_sors:
#         for i in range(len(seq) - 1):
#             wrong_transitions.append((seq[i], seq[i + 1]))
    
#     correct_transition_counter = Counter(correct_transitions)
#     wrong_transition_counter = Counter(wrong_transitions)
    
#     # Show top 10 transitions for each
#     print("\n=== Top 5 Reasoning Transitions ===")
#     print("CORRECT:")
#     for (from_behavior, to_behavior), count in correct_transition_counter.most_common(5):
#         print(f"  {from_behavior} -> {to_behavior}: {count}")
    
#     print("WRONG:")
#     for (from_behavior, to_behavior), count in wrong_transition_counter.most_common(5):
#         print(f"  {from_behavior} -> {to_behavior}: {count}")
    
#     # Create transition difference plot
#     all_transitions = set(correct_transitions + wrong_transitions)
#     transition_comparison = []
    
#     for transition in list(correct_transition_counter.most_common(10)) + list(wrong_transition_counter.most_common(10)):
#         trans = transition[0]
#         if trans not in [tc['transition'] for tc in transition_comparison]:  # Avoid duplicates
#             correct_freq = correct_transition_counter.get(trans, 0) / len(correct_transitions) if correct_transitions else 0
#             wrong_freq = wrong_transition_counter.get(trans, 0) / len(wrong_transitions) if wrong_transitions else 0
#             transition_comparison.append({
#                 'transition': f"{trans[0][:15]}... -> {trans[1][:15]}..." if len(trans[0]) > 15 or len(trans[1]) > 15 else f"{trans[0]} -> {trans[1]}",
#                 'difference': correct_freq - wrong_freq
#             })
    
#     transition_df = pd.DataFrame(transition_comparison)
#     transition_df = transition_df.sort_values('difference', ascending=False).head(10)
    
#     y_pos = np.arange(len(transition_df))
#     axes[1, 2].barh(y_pos, transition_df['difference'], 
#                     color=['green' if x > 0 else 'red' for x in transition_df['difference']])
#     axes[1, 2].set_yticks(y_pos)
#     axes[1, 2].set_yticklabels(transition_df['transition'], fontsize=6)
#     axes[1, 2].set_xlabel('Frequency Difference (Correct - Wrong)')
#     axes[1, 2].set_title('Top Reasoning Transitions Difference')
#     axes[1, 2].axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
#     plt.tight_layout()
#     plt.show()
    
#     return behavior_df, correct_counter, wrong_counter