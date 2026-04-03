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
Preprocess the GPQA dataset to parquet format using r1 format and functions
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


def build_gpqa_dataset(split_name="gpqa_main", randomize_order=True, qwen3_style=False):
    """Build GPQA dataset using r1 format"""
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

    # Load the specific split directly, just like in r1
    dataset = load_dataset(data_source, split_name, split="train")
    
    if qwen3_style:
        # Use qwen3-style processing with index-based permutations
        def map_fn_qwen3(example, idx):
            question, solution = process_gpqa_qwen3_style(example, idx)
            data = {
                "data_source": data_source,
                "prompt": [{"role": "user", "content": question}],
                "ability": "reasoning",
                "reward_model": {"style": "rule", "ground_truth": solution},
                "extra_info": {"split": "train", "index": idx, "format": "qwen3-style"},
            }
            return data
        
        dataset = dataset.map(map_fn_qwen3, with_indices=True, remove_columns=dataset.column_names)
    else:
        # Use original processing
        map_fn = partial(
            example_map_fn, process_fn=process_gpqa, data_source=data_source, ability="reasoning", split="train"
        )
        dataset = dataset.map(map_fn, with_indices=True, remove_columns=dataset.column_names)
    
    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="~/data/gpqa")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--split", default="gpqa_main", choices=["gpqa_main", "gpqa_diamond"], 
                       help="Dataset split to use")
    parser.add_argument("--randomize_order", action="store_true", default=True, 
                       help="Randomize the order of multiple choice answers")
    parser.add_argument("--qwen3_style", action="store_true", default=False,
                       help="Use qwen3-style format with boxed answers and pre-generated permutations")
    parser.add_argument("--train_test_split", action="store_true", default=True,
                       help="Split the data into train/test (80-20)")
    parser.add_argument("--test_ratio", default=0.2, type=float,
                       help="Ratio of data to use for test split")
    parser.add_argument("--seed", default=42, type=int, help="Random seed for reproducibility")

    args = parser.parse_args()

    # Set random seed for reproducible shuffling
    random.seed(args.seed)

    # Build the dataset using r1 format
    dataset = build_gpqa_dataset(split_name=args.split, randomize_order=args.randomize_order, qwen3_style=args.qwen3_style)
    print(f"Loaded {len(dataset)} examples from {args.split}")

    local_dir = os.path.expanduser(args.local_dir)
    hdfs_dir = args.hdfs_dir

    # Create local directory if it doesn't exist
    os.makedirs(local_dir, exist_ok=True)

    if args.train_test_split:
        # Split into train/test
        shuffled_dataset = dataset.shuffle(seed=args.seed)
        test_size = int(len(shuffled_dataset) * args.test_ratio)
        train_size = len(shuffled_dataset) - test_size
        
        train_dataset = shuffled_dataset.select(range(train_size))
        test_dataset = shuffled_dataset.select(range(train_size, len(shuffled_dataset)))
        
        print(f"Split dataset into {len(train_dataset)} train and {len(test_dataset)} test examples")
        
        # Update split info in extra_info for each dataset
        def update_split_info(example, split_name):
            example["extra_info"]["split"] = split_name
            return example
        
        train_dataset = train_dataset.map(lambda x: update_split_info(x, "train"))
        test_dataset = test_dataset.map(lambda x: update_split_info(x, "test"))
        
        # Save both splits
        format_suffix = "_qwen3" if args.qwen3_style else ""
        train_filename = f"{args.split}_train{format_suffix}.parquet"
        test_filename = f"{args.split}_test{format_suffix}.parquet"
        
        train_dataset.to_parquet(os.path.join(local_dir, train_filename))
        test_dataset.to_parquet(os.path.join(local_dir, test_filename))
        
        print(f"Saved train dataset to {os.path.join(local_dir, train_filename)}")
        print(f"Saved test dataset to {os.path.join(local_dir, test_filename)}")
    else:
        # Process and save as single file
        format_suffix = "_qwen3" if args.qwen3_style else ""
        output_filename = f"{args.split}{format_suffix}.parquet"
        dataset.to_parquet(os.path.join(local_dir, output_filename))
        print(f"Saved processed dataset to {os.path.join(local_dir, output_filename)}")

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
        print(f"Copied dataset to HDFS: {hdfs_dir}")

    # Print a sample to verify format
    print("\nSample data format:")
    if args.train_test_split:
        sample = train_dataset[0]
    else:
        sample = dataset[0]
    for key, value in sample.items():
        print(f"  {key}: {value}") 