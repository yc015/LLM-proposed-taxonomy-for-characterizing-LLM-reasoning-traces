# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocess the GPQA dataset to parquet format using r1 format and functions.
Creates train/test split by using gpqa_main minus gpqa_diamond as train set,
and gpqa_diamond as test set.
"""

import argparse
import os
import random
from functools import partial

from datasets import load_dataset

from verl.utils.hdfs_io import copy, makedirs


def example_map_fn(example, idx, process_fn, data_source, ability, split):
    """Map function from r1 data processing"""
    question, solution = process_fn(example)
    data = {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": split, "index": idx},
    }
    return data


def build_gpqa_train_test_datasets(randomize_order=True, qwen3_style=False, phi4_style=False, use_intervention=False, intervention_prompt=""):
    """Build GPQA dataset using r1 format with proper train/test split"""
    import random

    GPQA_QUERY_TEMPLATE = (
        "Answer the following multiple choice question. The last line of your response should be of the following "
        "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
        "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
    )

    def process_gpqa(example):
        choices = [example["Incorrect Answer 1"], example["Incorrect Answer 2"], example["Incorrect Answer 3"]]
        if randomize_order:
            random.shuffle(choices)
        gold_index = random.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])
        query_prompt = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )
        gold_choice = "ABCD"[gold_index]
        return query_prompt, gold_choice

    def process_gpqa_qwen3_style(example, idx):
        """Process GPQA in qwen3-style format with pre-generated permutations"""
        question = example["Question"]
        choices = [
            ("Correct Answer", example["Correct Answer"]),
            ("Incorrect Answer 1", example["Incorrect Answer 1"]),
            ("Incorrect Answer 2", example["Incorrect Answer 2"]),
            ("Incorrect Answer 3", example["Incorrect Answer 3"])
        ]
        
        # Use pre-generated permutation for this question based on index
        # Generate consistent permutation using index as seed
        local_random = random.Random(idx)
        new_order = choices.copy()
        local_random.shuffle(new_order)
        
        choices_dict = {label: choice for label, choice in choices}
        choices = [(label, choices_dict[label]) for label, choice in new_order]
        
        # Create the question prompt
        question_prompt = f"{question}\n\n"
        for letter, (_, choice) in zip(["A", "B", "C", "D"], choices):
            choice = choice.strip().strip('\n')
            question_prompt += f"{letter}: {choice}\n"
        
        # Add intervention prompt if enabled
        if use_intervention and intervention_prompt:
            question_prompt += f'\n{intervention_prompt}\n\n'

        question_prompt += '\nPlease reason step by step, and put your final answer within \\boxed{}.\nPlease only provide the letter of the answer in the box.'
        
        # Find which letter corresponds to the correct answer
        gold_choice = None
        for letter, (label, _) in zip(["A", "B", "C", "D"], choices):
            if label == "Correct Answer":
                gold_choice = letter
                break
        
        return question_prompt, gold_choice

    def process_gpqa_phi4_style(example, idx):
        """Process GPQA in Phi-4 style format with intervention in system prompt"""
        question = example["Question"]
        choices = [
            ("Correct Answer", example["Correct Answer"]),
            ("Incorrect Answer 1", example["Incorrect Answer 1"]),
            ("Incorrect Answer 2", example["Incorrect Answer 2"]),
            ("Incorrect Answer 3", example["Incorrect Answer 3"])
        ]
        
        # Use pre-generated permutation for this question based on index
        # Generate consistent permutation using index as seed
        local_random = random.Random(idx)
        new_order = choices.copy()
        local_random.shuffle(new_order)
        
        choices_dict = {label: choice for label, choice in choices}
        choices = [(label, choices_dict[label]) for label, choice in new_order]
        
        # Create the question prompt (without intervention - it goes in system)
        question_prompt = f"{question}\n\n"
        for letter, (_, choice) in zip(["A", "B", "C", "D"], choices):
            choice = choice.strip().strip('\n')
            question_prompt += f"{letter}: {choice}\n"
        
        question_prompt += '\nPlease reason step by step, and put your final answer within \\boxed{}.\nPlease only provide the letter of the answer in the box.'
        
        # Find which letter corresponds to the correct answer
        gold_choice = None
        for letter, (label, _) in zip(["A", "B", "C", "D"], choices):
            if label == "Correct Answer":
                gold_choice = letter
                break
        
        return question_prompt, gold_choice

    data_source = "Idavidrein/gpqa"
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)

    # Load both splits
    print("Loading gpqa_main split...")
    gpqa_main = load_dataset(data_source, "gpqa_main", split="train")
    print(f"Loaded {len(gpqa_main)} examples from gpqa_main")
    
    print("Loading gpqa_diamond split...")
    gpqa_diamond = load_dataset(data_source, "gpqa_diamond", split="train")
    print(f"Loaded {len(gpqa_diamond)} examples from gpqa_diamond")
    
    # Create a set of questions in gpqa_diamond for filtering
    diamond_questions = set(gpqa_diamond["Question"])
    print(f"Found {len(diamond_questions)} unique questions in gpqa_diamond")
    
    # Filter gpqa_main to exclude questions that are in gpqa_diamond
    def is_not_in_diamond(example):
        return example["Question"] not in diamond_questions
    
    gpqa_train = gpqa_main.filter(is_not_in_diamond)
    print(f"After filtering out gpqa_diamond questions, train set has {len(gpqa_train)} examples")
    
    # Use gpqa_diamond as test set
    gpqa_test = gpqa_diamond
    print(f"Test set has {len(gpqa_test)} examples")
    
    # Process both datasets
    def create_map_fn(split_name):
        if phi4_style:
            def map_fn_phi4(example, idx):
                question, solution = process_gpqa_phi4_style(example, idx)
                # Create Phi-4 style system prompt with optional intervention
                if intervention_prompt:
                    system_content = (
                        "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves "
                        "thoroughly exploring questions through a systematic thinking process before providing the final precise "
                        "and accurate solutions. In particular, you should follow the guidelines below when exploring the questions "
                        f"{intervention_prompt} Please structure your response into two main sections: Thought and Solution using "
                        "the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, "
                        "detail your reasoning process in steps. Each step should include detailed considerations such as analysing "
                        "questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current "
                        "steps, refining any errors, and revisiting previous steps. In the Solution section, based on various "
                        "attempts, explorations, and reflections from the Thought section, systematically present the final solution "
                        "that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary "
                        "steps needed to reach the conclusion. Put your final answer within \\boxed{}."
                    )
                else:
                    system_content = (
                        "You are Phi, a language model trained by Microsoft to help users. Your role as an assistant involves "
                        "thoroughly exploring questions through a systematic thinking process before providing the final precise "
                        "and accurate solutions. Please structure your response into two main sections: Thought and Solution using "
                        "the specified format: <think> {Thought section} </think> {Solution section}. In the Thought section, "
                        "detail your reasoning process in steps. Each step should include detailed considerations such as analysing "
                        "questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current "
                        "steps, refining any errors, and revisiting previous steps. In the Solution section, based on various "
                        "attempts, explorations, and reflections from the Thought section, systematically present the final solution "
                        "that you deem correct. The Solution section should be logical, accurate, and concise and detail necessary "
                        "steps needed to reach the conclusion. Put your final answer within \\boxed{}."
                    )
                data = {
                    "data_source": data_source,
                    "prompt": [
                        {"role": "system", "content": system_content},
                        {"role": "user", "content": question}
                    ],
                    "ability": "reasoning",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {"split": split_name, "index": idx, "format": "phi4-style"},
                }
                return data
            return map_fn_phi4
        elif qwen3_style:
            def map_fn_qwen3(example, idx):
                question, solution = process_gpqa_qwen3_style(example, idx)
                data = {
                    "data_source": data_source,
                    "prompt": [{"role": "user", "content": question}],
                    "ability": "reasoning",
                    "reward_model": {"style": "rule", "ground_truth": solution},
                    "extra_info": {"split": split_name, "index": idx, "format": "qwen3-style"},
                }
                return data
            return map_fn_qwen3
        else:
            return partial(
                example_map_fn, process_fn=process_gpqa, data_source=data_source, ability="reasoning", split=split_name
            )
    
    # Map train dataset
    train_map_fn = create_map_fn("train")
    train_dataset = gpqa_train.map(train_map_fn, with_indices=True, remove_columns=gpqa_train.column_names)
    
    # Map test dataset
    test_map_fn = create_map_fn("test")
    test_dataset = gpqa_test.map(test_map_fn, with_indices=True, remove_columns=gpqa_test.column_names)
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None, help="Model ID for intervention prompt (required if --use_intervention)")
    parser.add_argument("--randomize_order", action="store_true", default=True, 
                       help="Randomize the order of multiple choice answers")
    parser.add_argument("--qwen3_style", action="store_true", default=False,
                       help="Use qwen3-style format with boxed answers and pre-generated permutations")
    parser.add_argument("--phi4_style", action="store_true", default=False,
                       help="Use Phi-4 style format with structured thinking prompt in system message")
    parser.add_argument("--use_intervention", action="store_true", default=False,
                       help="Use intervention prompt in questions")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy results to")

    args = parser.parse_args()

    # Validate arguments
    if args.use_intervention and not args.model_id:
        raise ValueError("--model_id is required when --use_intervention is enabled")
    if args.phi4_style and args.use_intervention and not args.model_id:
        raise ValueError("--model_id is required when --phi4_style is enabled with intervention")

    # Set random seed for reproducible shuffling
    random.seed(args.seed)

    # Load intervention prompt if needed
    intervention_prompt = ""
    if args.use_intervention and args.model_id:
        ds_id = "Idavidrein/gpqa"
        split = "gpqa_main"  # Use gpqa_main for intervention prompt path
        intervention_path = f"/n/home04/yidachen/reasoning_characteristics/finetuning-intervention-prompt/{ds_id.split('/')[-1]}-gpqa_diamond/{args.model_id.split('/')[-1]}-intervention-prompt.txt"
        
        if not os.path.exists(intervention_path):
            raise FileNotFoundError(f"Intervention prompt file not found: {intervention_path}")
        
        with open(intervention_path, "r") as f:
            intervention_prompt = f.read()
        print(f"Loaded intervention prompt from {intervention_path}")
    elif args.phi4_style and not args.use_intervention:
        print("Using Phi-4 style without intervention prompt")

    # Build the datasets using r1 format
    train_dataset, test_dataset = build_gpqa_train_test_datasets(
        randomize_order=args.randomize_order, 
        qwen3_style=args.qwen3_style, 
        phi4_style=args.phi4_style,
        use_intervention=args.use_intervention,
        intervention_prompt=intervention_prompt
    )
    
    print(f"Created train dataset with {len(train_dataset)} examples")
    print(f"Created test dataset with {len(test_dataset)} examples")

    # Create output directory path
    base_name = "gpqa"
    if args.use_intervention:
        if not args.model_id:
            raise ValueError("Model ID required for intervention mode")
        model_name = args.model_id.split('/')[-1]
        dir_name = f"{base_name}-intervention-{model_name}" 
    else:
        dir_name = f"{base_name}"
    
    local_dir = os.path.expanduser(f"~/data/{dir_name}")
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    print(f"Output directory: {local_dir}")

    # Save both splits
    format_suffix = "_qwen3" if args.qwen3_style else "_phi4" if args.phi4_style else ""
    train_filename = f"gpqa_train{format_suffix}.parquet"
    test_filename = f"gpqa_test{format_suffix}.parquet"
    
    train_dataset.to_parquet(os.path.join(local_dir, train_filename))
    test_dataset.to_parquet(os.path.join(local_dir, test_filename))
    
    print(f"Saved train dataset to {os.path.join(local_dir, train_filename)}")
    print(f"Saved test dataset to {os.path.join(local_dir, test_filename)}")

    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        makedirs(args.hdfs_dir)
        copy(src=local_dir, dst=args.hdfs_dir)
        print(f"Copied dataset to HDFS: {args.hdfs_dir}")

    # Print dataset statistics
    print(f"\nDataset Statistics:")
    print(f"  Train examples: {len(train_dataset)}")
    print(f"  Test examples: {len(test_dataset)}")
    print(f"  Total examples: {len(train_dataset) + len(test_dataset)}")
    print(f"  Train/Test ratio: {len(train_dataset)/(len(train_dataset) + len(test_dataset)):.1%}/{len(test_dataset)/(len(train_dataset) + len(test_dataset)):.1%}")

    # Print a sample to verify format
    print("\nSample train data format:")
    sample = train_dataset[0]
    for key, value in sample.items():
        if key == "prompt":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")
    
    print("\nSample test data format:")
    sample = test_dataset[0]
    for key, value in sample.items():
        if key == "prompt":
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value}")