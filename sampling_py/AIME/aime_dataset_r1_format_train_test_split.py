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
Preprocess the AIME dataset to parquet format using r1 format and functions.
Creates train/test split by using AIME_1983_2024 as train set,
and aime_2025 as test set.
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


def build_aime_train_test_datasets(use_intervention=False, intervention_prompt="", filter_year=2000):
    """Build AIME dataset using r1 format with proper train/test split"""

    AIME_QUERY_TEMPLATE = "{question}\n\nPlease end your solution with Answer: $\\boxed{{number}}$ where number is the numerical answer without unit."

    def process_aime_train(example):
        """Process AIME training data from di-zhang-fdu/AIME_1983_2024"""
        question = example["Question"]
        answer = str(example["Answer"])  # Convert to string in case it's numeric
        
        # Create the question prompt
        if use_intervention and intervention_prompt:
            query_prompt = f"{question}\n\n{intervention_prompt}\n\n{AIME_QUERY_TEMPLATE.split('{question}')[1]}"
        else:
            query_prompt = AIME_QUERY_TEMPLATE.format(question=question)
        
        return query_prompt, answer

    def process_aime_test(example):
        """Process AIME test data from yentinglin/aime_2025"""
        question = example["problem"]
        solution = str(example["solution"])  # Convert to string in case it's numeric
        
        # Create the question prompt
        if use_intervention and intervention_prompt:
            query_prompt = f"{question}\n\n{intervention_prompt}\n\n{AIME_QUERY_TEMPLATE.split('{question}')[1]}"
        else:
            query_prompt = AIME_QUERY_TEMPLATE.format(question=question)
        
        return query_prompt, solution

    # Load training dataset
    train_data_source = "di-zhang-fdu/AIME_1983_2024"
    print(f"Loading the {train_data_source} dataset from huggingface...", flush=True)
    aime_train = load_dataset(train_data_source, split="train")
    print(f"Loaded {len(aime_train)} examples from {train_data_source}")
    
    # Filter training data by year if specified
    if filter_year is not None:
        def year_filter(example):
            return example["Year"] >= filter_year
        
        original_size = len(aime_train)
        aime_train = aime_train.filter(year_filter)
        filtered_size = len(aime_train)
        print(f"After filtering for years >= {filter_year}: {filtered_size} examples (removed {original_size - filtered_size})")
    
    # Load test dataset
    test_data_source = "yentinglin/aime_2025"
    print(f"Loading the {test_data_source} dataset from huggingface...", flush=True)
    aime_test = load_dataset(test_data_source, split="train")
    print(f"Loaded {len(aime_test)} examples from {test_data_source}")
    
    # Process training dataset
    train_map_fn = partial(
        example_map_fn, 
        process_fn=process_aime_train, 
        data_source="AIME", 
        ability="reasoning", 
        split="train"
    )
    train_dataset = aime_train.map(train_map_fn, with_indices=True, remove_columns=aime_train.column_names)
    
    # Process test dataset
    test_map_fn = partial(
        example_map_fn, 
        process_fn=process_aime_test, 
        data_source="AIME", 
        ability="reasoning", 
        split="test"
    )
    test_dataset = aime_test.map(test_map_fn, with_indices=True, remove_columns=aime_test.column_names)
    
    return train_dataset, test_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default=None, help="Model ID for intervention prompt (required if --use_intervention)")
    parser.add_argument("--use_intervention", action="store_true", default=False,
                       help="Use intervention prompt in questions")
    parser.add_argument("--filter_year", default=2010, type=int, help="Filter training data to include only questions from this year onwards (default: 2000)")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy results to")

    args = parser.parse_args()

    # Validate arguments
    if args.use_intervention and not args.model_id:
        raise ValueError("--model_id is required when --use_intervention is enabled")

    # Set random seed for reproducible shuffling
    random.seed(args.seed)

    # Load intervention prompt if needed
    intervention_prompt = ""
    if args.use_intervention:
        # Use aime as the dataset name for intervention prompt path
        ds_name = "aime"
        intervention_path = f"/n/home04/yidachen/reasoning_characteristics/finetuning-intervention-prompt/{ds_name}/{args.model_id.split('/')[-1]}-intervention-prompt.txt"
        
        if not os.path.exists(intervention_path):
            raise FileNotFoundError(f"Intervention prompt file not found: {intervention_path}")
        
        with open(intervention_path, "r") as f:
            intervention_prompt = f.read()
        print(f"Loaded intervention prompt from {intervention_path}")

    # Build the datasets using r1 format
    train_dataset, test_dataset = build_aime_train_test_datasets(
        use_intervention=args.use_intervention,
        intervention_prompt=intervention_prompt,
        filter_year=args.filter_year
    )
    
    print(f"Created train dataset with {len(train_dataset)} examples")
    print(f"Created test dataset with {len(test_dataset)} examples")

    # Create output directory path
    base_name = "aime"
    if args.use_intervention:
        if not args.model_id:
            raise ValueError("Model ID required for intervention mode")
        model_name = args.model_id.split('/')[-1]
        dir_name = f"{base_name}-intervention-train-test-split-{model_name}"
    else:
        dir_name = f"{base_name}-train-test-split"
    
    local_dir = os.path.expanduser(f"~/data/{dir_name}")
    
    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)
    print(f"Output directory: {local_dir}")

    # Save both splits
    train_filename = "aime_train.parquet"
    test_filename = "aime_test.parquet"
    
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
    print(f"\nFinal Dataset Statistics:")
    print(f"  Train examples: {len(train_dataset)} (from di-zhang-fdu/AIME_1983_2024, filtered for years >= {args.filter_year})")
    print(f"  Test examples: {len(test_dataset)} (from yentinglin/aime_2025)")
    print(f"  Total examples: {len(train_dataset) + len(test_dataset)}")
    print(f"  Data source field: AIME")

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