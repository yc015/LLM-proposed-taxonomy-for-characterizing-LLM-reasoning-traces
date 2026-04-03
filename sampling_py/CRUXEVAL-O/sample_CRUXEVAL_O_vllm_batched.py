# vLLM version of CRUXEVAL_O batched sampling script

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import pickle
import os
from tqdm.auto import tqdm
import argparse
import random


parser = argparse.ArgumentParser(description="Generate responses using a specified model with vLLM.")
parser.add_argument('--model_id', type=str, required=True, help='The model ID to use for generation.')
parser.add_argument('--do_sample', type=bool, default=True, help='Whether to use sampling; use greedy decoding otherwise.')
parser.add_argument('--temperature', type=float, default=0.8, help='The value used to module the next token probabilities.')
parser.add_argument('--max_new_tokens', type=int, default=16384, help='The maximum number of new tokens to generate.')
parser.add_argument('--top_k', type=int, default=50, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
parser.add_argument('--top_p', type=float, default=0.95, help='If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.')
parser.add_argument('--batch_size', type=int, default=64, help='The number of samples to process in a batch.')
parser.add_argument('--min_p', type=float, default=0)
parser.add_argument('--tensor_parallel_size', type=int, default=2, help='Number of GPUs for vLLM tensor parallelism.')
parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization for VLLM.')
parser.add_argument('--download_dir', type=str, default='/n/holylabs/LABS/wattenberg_lab/Lab/pretrained_models/', help='Download directory for model weights.')
parser.add_argument('--output_base_dir', type=str, default='outputs', help='Base directory for output files.')
parser.add_argument('--backlog_base_dir', type=str, default='backlog/unfinished_thinking', help='Base directory for backlog files.')
args = parser.parse_args()

# Task Overall Instruction
overall_inst = """Based on the given Python code, which may contain errors, complete the assert statement with the output when executing the code on the given test case. Do not output any extra information, even if the function is incorrect or incomplete.
\n
"""

# Load the dataset

ds_id = "cruxeval-org/cruxeval"
split = None
ds = load_dataset(ds_id, split)
split = "O"

# Define the model and tokenizer using vLLM

model_id = args.model_id
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Initialize VLLM model
# Initialize VLLM model
llm = LLM(
    model=model_id,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=args.gpu_memory_utilization,
    trust_remote_code=True,
    download_dir=args.download_dir,
    max_model_len=None,  # Use model's default max length
)

# Create sampling parameters
sampling_params = SamplingParams(
    temperature=args.temperature if args.do_sample else 0.0,
    top_p=args.top_p,
    top_k=args.top_k,
    max_tokens=args.max_new_tokens,
    min_p=args.min_p,
    stop_token_ids=None,
)

if "Qwen2.5" in model_id or "math-7b-rl" in model_id:
    think_stop_id = -123
elif "Seed" in model_id:
    think_stop_id = tokenizer.vocab["think"]
else:
    think_stop_id = tokenizer.vocab["</think>"]

# Define the output folder

dataset_name = ds_id.split("/")[-1]
if split:
    dataset_name += f"-{split}"
model_name = model_id.split("/")[-1]
output_folder = os.path.join(args.output_base_dir, dataset_name, model_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load unfinished IDs
backlog_dir = os.path.join(args.backlog_base_dir, ds_id.split('/')[-1])
unfinished_ids = set()
unfinished_file = os.path.join(backlog_dir, f"unfinished_thoughts_{model_name}.pkl")
if os.path.exists(unfinished_file):
    with open(unfinished_file, "rb") as f:
        unfinished_ids.update(pickle.load(f))

# Loop through the dataset in batches
for i in tqdm(range(0, len(ds['test']), args.batch_size)):
    # Get the current batch of prompts
    batch_prompts = []
    batch_indices = []
    batch_formatted_prompts = []
    
    for j in range(args.batch_size):
        if i + j >= len(ds['test']):
            break
        idx = i + j
        question_id = ds['test']['id'][idx]
        filename = f"{dataset_name}_question_id_{question_id}_{model_name}.txt"
        
        # Check if the output file already exists
        if os.path.exists(os.path.join(output_folder, filename)) and question_id not in unfinished_ids:
            continue

        question_prompt = overall_inst
        question_prompt += f"""
[PYTHON]
{ds['test']['code'][idx]}
[/PYTHON]

assert f({ds['test']['input'][idx]}) == ??

In your final output, surround your answer with no additional words, with [ANSWER] and [/ANSWER] tags. Your answer should be [ANSWER] [Expected outputs] [/ANSWER]"""

        batch_prompts.append(question_prompt)
        batch_indices.append(ds['test']['id'][idx])
        
        # Format prompt for VLLM
        messages = [{"role": "user", "content": question_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        batch_formatted_prompts.append(formatted_prompt)

    # Skip if no new prompts to process
    if not batch_formatted_prompts:
        continue

    # Generate responses using VLLM
    generated_outputs = llm.generate(batch_formatted_prompts, sampling_params)

    for k, output in enumerate(generated_outputs):
        generated_text = output.outputs[0].text
        generated_tokens = output.outputs[0].token_ids
        
        # Parse thinking content
        if think_stop_id in generated_tokens:
            try:
                index = len(generated_tokens) - generated_tokens[::-1].index(think_stop_id)
            except ValueError:
                index = 0
        else:
            index = len(generated_tokens)
            
        thinking_content = tokenizer.decode(generated_tokens[:index], skip_special_tokens=True).strip("\n")
        content = tokenizer.decode(generated_tokens[index:], skip_special_tokens=True).strip("\n")
        question_id = batch_indices[k]
        filename = f"{dataset_name}_question_id_{question_id}_{model_name}.txt"
        
        with open(os.path.join(output_folder, filename), "w") as f:
            f.write(f"### QUESTION: {batch_prompts[k]}\n\n")
            f.write(f"### THINKING: {thinking_content}\n\n")
            f.write(f"### ANSWER: {content}")
            
        with open(os.path.join(output_folder, f"raw_output_{dataset_name}_question_id_{question_id}_{model_name}.txt"), "w") as f:
            # For VLLM, we save the complete prompt + generated text
            raw_output = batch_formatted_prompts[k] + generated_text
            f.write(f"{raw_output}\n")

        # Save the generated tokens to a pkl file
        output_file = f"output_ids_{dataset_name}_question_id_{question_id}_{model_name}.pkl"
        with open(os.path.join(output_folder, output_file), "wb") as f:
            pickle.dump(generated_tokens, f) 