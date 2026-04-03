#!/usr/bin/env python3
"""
Simple script to analyze intervention experiment results and reproduce the final statistics.

Usage:
    python analyze_intervention_results.py <output_directory>

Example:
    python analyze_intervention_results.py intervention_outputs_all/summary_expand_intervention/gpqa/Qwen3-14B_claude_abc123
"""

import os
import sys
import json
import pickle
from pathlib import Path
import re
import regex
from src.eval.answer_extraction import extract_gpqa_answer


def analyze_results(output_dir):
    """Analyze intervention results from output directory"""
    output_path = Path(output_dir)
    
    if not output_path.exists():
        print(f"Error: Directory {output_dir} does not exist")
        return
    
    # Find all question directories
    question_dirs = [d for d in output_path.iterdir() 
                    if d.is_dir() and d.name.startswith('question_')]
    
    if not question_dirs:
        print(f"Error: No question directories found in {output_dir}")
        return
    
    print(f"Found {len(question_dirs)} question directories")
    
    # Initialize counters
    total_questions = 0
    correct_before_intervention = 0
    correct_after_intervention = 0
    successful_interventions = 0  # incorrect -> correct
    unsuccessful_interventions = 0  # correct -> incorrect
    
    # Handle "covered_only_wrong" case - need to account for originally correct questions
    correct_ids_count = 0
    if "covered_only_wrong" in output_dir:
        # Extract model name from output directory path to load correct_ids
        # Expected format: .../model_name_...covered_only_wrong...
        path_parts = str(output_dir).split('/')
        for part in path_parts:
            if "covered_only_wrong" in part:
                # Extract model name from the folder name (before the first underscore before "covered_only_wrong")
                parts = part.split('_')
                model_short_name = parts[0]  # This should be the model name
                break
        
        # Load correct_ids to account for questions that were originally correct
        if output_dir.split("/")[-2].startswith("Qwen3") or output_dir.split("/")[-2].startswith("Phi-4") or output_dir.split("/")[-2].startswith("QwQ"):
            backlog_dir = f"/n/home04/yidachen/reasoning_characteristics/backlog/unfinished_thinking/gpqa-Qwen3-Style-Prompt/"
        else:
            backlog_dir = f"/n/home04/yidachen/reasoning_characteristics/backlog/unfinished_thinking/gpqa/"
        correct_ids = set()
        correct_file = os.path.join(backlog_dir, f"correct_ids_{model_short_name}.pkl")
        if os.path.exists(correct_file):
            with open(correct_file, "rb") as f:
                correct_ids.update(pickle.load(f))
            correct_ids_count = len(correct_ids)
            print(f"Found {correct_ids_count} originally correct questions to add to counts")

    # Analyze each question
    for question_dir in question_dirs:
        try:
            # Load metadata
            metadata_file = question_dir / "metadata.json"
            if not metadata_file.exists():
                print(f"Warning: No metadata.json found in {question_dir}")
                continue
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load re-expand reasoning (intervention result)
            reexpand_file = question_dir / "re_expand_reasoning.txt"
            if not reexpand_file.exists():
                print(f"Warning: No re_expand_reasoning.txt found in {question_dir}")
                continue
                
            with open(reexpand_file, 'r') as f:
                reexpand_text = f.read()
            
            # Get answers and metadata
            original_extracted_answer = metadata['original_extracted_answer']
            question_id = metadata['question_id']
            model_name = metadata['model_name']
            model_id = model_name.split("/")[-1]
            
            # Load choice order from the correct path
            if output_dir.split("/")[-2].startswith("Qwen3") or output_dir.split("/")[-2].startswith("Phi-4") or output_dir.split("/")[-2].startswith("QwQ"):
                choice_order_path = f"/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/gpqa-gpqa_diamond_Qwen3_Style_Prompt/{model_id}/choice_order_gpqa-gpqa_diamond_question_id_{question_id}_{model_id}.pkl"
            else:
                choice_order_path = f"/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/gpqa-gpqa_diamond/{model_id}/choice_order_gpqa-gpqa_diamond_question_id_{question_id}_{model_id}.pkl"
            with open(choice_order_path, "rb") as infile:
                choice_order = pickle.load(infile)
            correct_answer_idx = choice_order.index("Correct Answer")
            
            # Extract intervention result answer
            intervention_answer = extract_gpqa_answer(reexpand_text)
            
            # Calculate correctness
            was_correct_before = original_extracted_answer == correct_answer_idx
            is_correct_after = intervention_answer == correct_answer_idx
            
            # Update counters
            total_questions += 1
            
            if was_correct_before:
                correct_before_intervention += 1
                
            if is_correct_after:
                correct_after_intervention += 1
                
            # Track intervention effects
            if not was_correct_before and is_correct_after:
                successful_interventions += 1
            elif was_correct_before and not is_correct_after:
                unsuccessful_interventions += 1
                
        except Exception as e:
            print(f"Error processing {question_dir}: {e}")
            continue
    
    # Add correct_ids_count if this was a "covered_only_wrong" experiment
    if "covered_only_wrong" in output_dir:
        correct_before_intervention += correct_ids_count
        correct_after_intervention += correct_ids_count
        total_questions += correct_ids_count
    
    # Print results (matching the original format)
    print(f"\n{'-'*50}")
    print(f"FINAL RESULTS:")
    print(f"Total questions processed: {total_questions}")
    print(f"Correct before intervention: {correct_before_intervention}/{total_questions}")
    print(f"Correct after intervention: {correct_after_intervention}/{total_questions}")
    print(f"Improvement: {correct_after_intervention - correct_before_intervention}")
    print(f"Successful interventions (incorrect -> correct): {successful_interventions}")
    print(f"Unsuccessful interventions (correct -> incorrect): {unsuccessful_interventions}")
    
    if total_questions > 0:
        print(f"Accuracy before: {correct_before_intervention/total_questions:.3f}")
        print(f"Accuracy after: {correct_after_intervention/total_questions:.3f}")
    
    print(f"Output directory analyzed: {output_dir}")
    print(f"{'-'*50}")
    
    return {
        'total_questions': total_questions,
        'correct_before': correct_before_intervention,
        'correct_after': correct_after_intervention,
        'successful_interventions': successful_interventions,
        'unsuccessful_interventions': unsuccessful_interventions,
        'accuracy_before': correct_before_intervention/total_questions if total_questions > 0 else 0,
        'accuracy_after': correct_after_intervention/total_questions if total_questions > 0 else 0
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_intervention_results.py <output_directory>")
        print("\nExample:")
        print("python analyze_intervention_results.py intervention_outputs_all/summary_expand_intervention/gpqa/Qwen3-14B_claude_abc123")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    analyze_results(output_dir)


if __name__ == "__main__":
    main()