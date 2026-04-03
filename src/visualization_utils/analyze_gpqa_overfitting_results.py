#!/usr/bin/env python3
"""
Simple script to analyze GPQA overfitting test results and reproduce the final statistics.

Usage:
    python analyze_gpqa_overfitting_results.py <output_directory>

Example:
    python analyze_gpqa_overfitting_results.py intervention_outputs_all/overfitting_test_with_only_choices/gpqa/Qwen3-14B_overfitting_test_with_only_choices_abc123_seed_42
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
    """Analyze GPQA overfitting test results from output directory"""
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
    
    # Initialize counters (matching the original script exactly)
    correct_before_intervention = 0
    correct_after_intervention = 0
    successful_interventions = 0  # incorrect before -> correct after
    unsuccessful_interventions = 0  # correct before -> incorrect after
    valid_answer = 0
    valid_answer_correct_after_intervention = 0
    tot = 0

    # Analyze each question
    for question_dir in sorted(question_dirs, key=lambda x: int(x.name.split('_')[1])):
        try:
            # Load metadata
            metadata_file = question_dir / "metadata.json"
            if not metadata_file.exists():
                print(f"Warning: No metadata.json found in {question_dir}")
                continue
                
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            # Load intervened reasoning (intervention result)
            intervened_file = question_dir / "intervened_reasoning.txt"
            if not intervened_file.exists():
                print(f"Warning: No intervened_reasoning.txt found in {question_dir}")
                continue
                
            with open(intervened_file, 'r') as f:
                intervened_reasoning = f.read()
            
            # Get question info from metadata
            question_id = metadata['question_id']
            model_name = metadata['model_name']
            model_id = model_name.split("/")[-1]
            correct_answer = metadata['correct_answer']
            
            # Load choice order from the correct path
            choice_order_path = f"/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/gpqa-gpqa_diamond_Qwen3_Style_Prompt/{model_id}/choice_order_gpqa-gpqa_diamond_question_id_{question_id}_{model_id}.pkl"
            with open(choice_order_path, "rb") as infile:
                choice_order = pickle.load(infile)
            correct_answer_idx = choice_order.index("Correct Answer")
            
            # Load original response to get original answer
            original_response_path = f"/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/gpqa-gpqa_diamond_Qwen3_Style_Prompt/{model_id}/gpqa-gpqa_diamond_question_id_{question_id}_{model_id}.txt"
            with open(original_response_path, "r") as infile:
                original_response = infile.read()
                original_answer = original_response[original_response.find("### ANSWER"):]
                extracted_original_answer = extract_gpqa_answer(original_answer)
            
            # Extract intervention result answer
            intervention_answer = extract_gpqa_answer(intervened_reasoning)
            
            # Calculate correctness (matching original logic exactly)
            was_correct_before = extracted_original_answer == correct_answer_idx
            is_correct_after = intervention_answer == correct_answer_idx
            
            # Update counters (following original script logic)
            if was_correct_before:
                correct_before_intervention += 1

            # Check if answer is valid (0, 1, 2, or 3)
            if intervention_answer in [0, 1, 2, 3]:
                valid_answer += 1
                if is_correct_after:
                    valid_answer_correct_after_intervention += 1

            if is_correct_after:
                correct_after_intervention += 1                
                # Track successful intervention (incorrect -> correct)
                if not was_correct_before:
                    successful_interventions += 1
            else:
                # Track unsuccessful intervention (correct -> incorrect)
                if was_correct_before:
                    unsuccessful_interventions += 1
            
            tot += 1
                
        except Exception as e:
            print(f"Error processing {question_dir}: {e}")
            continue
    
    # Print results (matching the original format exactly)
    print(f"\n{'-'*50}")
    print(f"FINAL RESULTS:")
    print(f"Total questions processed: {tot}")
    print(f"Correct before intervention: {correct_before_intervention}/{tot}")
    print(f"Correct after intervention: {correct_after_intervention}/{tot}")
    print(f"Successful with only choices: {successful_interventions}")
    print(f"Unsuccessful with only choices: {unsuccessful_interventions}")
    print(f"Correct after interventio (only count valid answer): {valid_answer_correct_after_intervention}/{valid_answer}")
    print(f"Accuracy after intervention (only count valid answer): {valid_answer_correct_after_intervention/valid_answer:.3f}")
    if tot > 0:
        print(f"Accuracy before: {correct_before_intervention/tot:.3f}")
        print(f"Accuracy after: {correct_after_intervention/tot:.3f}")
    print(f"Output saved to: {output_dir}")
    print(f"{'-'*50}")
    
    return {
        'total_questions': tot,
        'correct_before': correct_before_intervention,
        'correct_after': correct_after_intervention,
        'successful_interventions': successful_interventions,
        'unsuccessful_interventions': unsuccessful_interventions,
        'valid_answer': valid_answer,
        'valid_answer_correct_after_intervention': valid_answer_correct_after_intervention,
        'accuracy_before': correct_before_intervention/tot if tot > 0 else 0,
        'accuracy_after': correct_after_intervention/tot if tot > 0 else 0,
        'accuracy_after_valid_only': valid_answer_correct_after_intervention/valid_answer if valid_answer > 0 else 0
    }


def main():
    if len(sys.argv) != 2:
        print("Usage: python analyze_gpqa_overfitting_results.py <output_directory>")
        print("\nExample:")
        print("python analyze_gpqa_overfitting_results.py intervention_outputs_all/overfitting_test_with_only_choices/gpqa/Qwen3-14B_overfitting_test_with_only_choices_abc123_seed_42")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    analyze_results(output_dir)


if __name__ == "__main__":
    main()
