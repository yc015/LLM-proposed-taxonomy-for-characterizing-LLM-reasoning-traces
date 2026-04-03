import os
import re
import pickle
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from src.prompt_dataset import load_reasoning_traces, shuffle_outputs_and_labels
from src.eval.answer_extraction import extract_boxed_answers


def parse_folder_name(folder_name: str) -> Tuple:
    """Parse folder name to extract model_name, intervention_type, and run_number."""
    pattern = r'(.+)-(inst-intervened|no-intervened)-run-(\d+)'
    match = re.match(pattern, folder_name)
    if match:
        model_name = match.group(1)
        intervention_type = match.group(2)
        run_number = int(match.group(3))
        return model_name, intervention_type, run_number
    return None, None, None


def extract_gpqa_answer(answer: str) -> int:
    """Extract answer choice from GPQA format."""
    answer_loc = answer.lower().rfind("the correct answer is") + len("the correct answer is ")
    answer = answer[answer_loc: answer_loc + 3]
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    elif "C" in answer:
        return 2
    elif "D" in answer:
        return 3
    return 4

def extract_gpqa_answer_boxed(answer):
    # answer_loc = answer.lower().rfind("the correct answer is") + len("the correct answer is ")
    # answer = answer[answer_loc: answer_loc + 3]
    answer = extract_boxed_answers(answer)
    if "A" in answer:
        return 0
    elif "B" in answer:
        return 1
    elif "C" in answer:
        return 2
    elif "D" in answer:
        return 3
    return 4


def check_answer_correctness(answer_text: str, choice_order_file: str) -> bool:
    """Check if the extracted answer is correct based on choice order file."""
    if not os.path.exists(choice_order_file):
        return False
        
    with open(choice_order_file, "rb") as infile:
        choice_order = pickle.load(infile)
    
    if "boxed{" in choice_order_file:
        extract_answer = extract_gpqa_answer_boxed(answer_text)
    else:
        extract_answer = extract_gpqa_answer(answer_text)
    
    if extract_answer < len(choice_order) and choice_order[extract_answer] == "Correct Answer":
        return True
    return False


def create_paired_dataset(base_folder: str, 
                         model_name: str,
                         labels: Dict[str, str] = None,
                         shuffle_pairs: bool = True,
                         max_pairs_per_question: int = None) -> List[Dict[str, Any]]:
    """
    Create a paired dataset of correct and wrong reasoning for the same questions from a specific model.
    
    Args:
        base_folder: Path to the intervention output folder
        model_name: Specific model name to create pairs from (e.g., "claude-3-5-sonnet-20241022")
        labels: Dictionary mapping correctness to labels 
                (default: {"correct": "stronger model", "wrong": "weaker model"})
        shuffle_pairs: Whether to shuffle the pairs using the existing shuffle function
        max_pairs_per_question: Maximum number of correct/wrong pairs to generate per question.
                               If None, generates max(correct_responses, wrong_responses) pairs.
    
    Returns:
        List of dataset items with paired outputs and labels
    """
    if labels is None:
        labels = {"correct": "stronger-model", "wrong": "weaker-model"}
    
    # Load all reasoning traces using existing function
    reasoning_traces = load_reasoning_traces(base_folder)
    
    # Group traces by question, separating correct and wrong answers for the specific model
    question_responses = defaultdict(lambda: {"correct": [], "wrong": []})
    
    for folder_name, traces in reasoning_traces.items():
        parsed_model_name, intervention_type, run_number = parse_folder_name(folder_name)
        
        # Filter for specific model and no-intervened runs only
        if parsed_model_name is None or intervention_type != "no-intervened" or parsed_model_name != model_name:
            continue
            
        for question_id, trace in traces.items():
            answer_text = trace.get('answer', '')
            
            if len(answer_text) < 1:
                continue
                
            # Check correctness using choice order file
            choice_order_file = f"{base_folder}/{folder_name}/choice_order_gpqa-gpqa_diamond_question_id_{question_id}_{folder_name}.pkl"
            
            is_correct = check_answer_correctness(answer_text, choice_order_file)
            
            response_data = {
                'thinking': trace['thinking'],
                'answer': answer_text,
                'question': trace['question'],
                'model_name': parsed_model_name,
                'run_number': run_number,
                'question_id': question_id
            }
            
            if is_correct:
                question_responses[question_id]["correct"].append(response_data)
            else:
                question_responses[question_id]["wrong"].append(response_data)
    
    # Create paired dataset
    outputs = []
    output_labels = []
    ids = []
    questions = []
    
    for question_id, responses in question_responses.items():
        correct_responses = responses["correct"]
        wrong_responses = responses["wrong"]
        
        if not correct_responses or not wrong_responses:
            continue
            
        # Generate max(correct, wrong) pairs
        if max_pairs_per_question is None:
            max_pairs = max(len(correct_responses), len(wrong_responses))
        else:
            max_pairs = min(max_pairs_per_question, max(len(correct_responses), len(wrong_responses)))
        
        for i in range(max_pairs):
            # Sample responses with replacement if needed
            correct_idx = i % len(correct_responses)
            wrong_idx = i % len(wrong_responses)
            
            correct_response = correct_responses[correct_idx]
            wrong_response = wrong_responses[wrong_idx]
            
            # Create pair (correct first, then wrong)
            output_pair = [correct_response['thinking'], wrong_response['thinking']]
            label_pair = [labels["correct"], labels["wrong"]]
            
            outputs.append(output_pair)
            output_labels.append(label_pair)
            ids.append(question_id)
            questions.append(correct_response['question'])  # Same question for both
    
    # Use existing shuffle function if requested
    if shuffle_pairs and outputs:
        shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions = shuffle_outputs_and_labels(
            outputs, output_labels, ids, questions, balance_order=False
        )
    else:
        shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions = outputs, output_labels, ids, questions
    
    # Create final dataset
    dataset = []
    for single_outputs, single_labels, single_id, single_question in zip(
        shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions
    ):
        dataset_item = {
            "outputs": single_outputs,
            "labels": single_labels,
            "id": single_id,
            "question": single_question
        }
        dataset.append(dataset_item)
    
    return dataset


# if __name__ == "__main__":
#     # Example usage
#     base_folder = "intervention_output/gpqa-gpqa_diamond/"
    
#     # Create dataset with default labels
#     dataset = create_paired_dataset(base_folder)
    
#     print(f"Created dataset with {len(dataset)} paired examples")
    
#     # Example with custom labels
#     custom_labels = {"correct": "expert", "wrong": "novice"}
#     custom_dataset = create_paired_dataset(base_folder, custom_labels)
    
#     print(f"Created custom dataset with {len(custom_dataset)} paired examples")
    
#     # Show example
#     if dataset:
#         print("\nExample dataset item:")
#         print(f"Question ID: {dataset[0]['id']}")
#         print(f"Labels: {dataset[0]['labels']}")
#         print(f"First output (first 100 chars): {dataset[0]['outputs'][0][:100]}...")
#         print(f"Second output (first 100 chars): {dataset[0]['outputs'][1][:100]}...") 