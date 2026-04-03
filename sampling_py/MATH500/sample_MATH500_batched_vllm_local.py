# Load model directly

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import torch
from datasets import load_dataset
import pickle
import os
from tqdm.auto import tqdm
import argparse


parser = argparse.ArgumentParser(description="Generate responses using a local model or HuggingFace model.")
parser.add_argument('--model_path', type=str, required=True, help='The local path to the model folder or HuggingFace model ID.')
parser.add_argument('--from_hf', action='store_true', help='Load model from HuggingFace instead of local path.')
parser.add_argument('--do_sample', type=bool, default=True, help='Whether to use sampling; use greedy decoding otherwise.')
parser.add_argument('--temperature', type=float, default=0.8, help='The value used to module the next token probabilities.')
parser.add_argument('--max_new_tokens', type=int, default=16384, help='The maximum number of new tokens to generate.')
parser.add_argument('--top_k', type=int, default=50, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
parser.add_argument('--top_p', type=float, default=0.95, help='If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.')
parser.add_argument('--batch_size', type=int, default=250, help='The number of samples to process in a batch.')
parser.add_argument('--min_p', type=float, default=0)
parser.add_argument('--tensor_parallel_size', type=int, default=2, help='Number of GPUs to use for tensor parallelism.')
parser.add_argument('--gpu_memory_utilization', type=float, default=0.9, help='GPU memory utilization for VLLM.')
parser.add_argument('--download_dir', type=str, default='/n/holylabs/LABS/wattenberg_lab/Lab/pretrained_models/', help='Download directory for model weights.')
parser.add_argument('--output_base_dir', type=str, default='outputs', help='Base directory for output files.')
args = parser.parse_args()

# Load the dataset
ds_id = "HuggingFaceH4/MATH-500"
ds = load_dataset(ds_id)

# Define the model and tokenizer
model_path = args.model_path

# Handle HuggingFace vs local model loading
if args.from_hf:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # For HuggingFace models, add download_dir for caching
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
        download_dir=args.download_dir,
        max_model_len=None,
    )
else:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # For local models, no download_dir needed
    llm = LLM(
        model=model_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        dtype="auto",
        max_model_len=None,
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

# Define the output folder
dataset_name = ds_id.split("/")[-1]

if args.from_hf:
    # For HuggingFace models, use the model ID as model name
    model_name = model_path.split("/")[-1] if "/" in model_path else model_path
    output_folder = os.path.join(args.output_base_dir, dataset_name, model_name)
else:
    # For local models, create custom naming: qwen25_{step}_actor
    if "step_" in model_path:
        # Extract step number from path like "global_step_10" or "step_10"
        step_start = model_path.find("step_") + 5
        step_part = model_path[step_start:]
        # Find the end of the step number (next slash or end of string)
        step_end = step_part.find("/")
        if step_end == -1:
            step_number = step_part
        else:
            step_number = step_part[:step_end]
        
        model_name = f"qwen25_{step_number}_actor"
        output_folder = os.path.join(args.output_base_dir, dataset_name, model_name)
    else:
        # Fallback if no step found
        model_name = "qwen25_actor"
        output_folder = os.path.join(args.output_base_dir, dataset_name, model_name)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Loop through the dataset in batches
for i in range(0, len(ds["test"]), args.batch_size):
    # Get the current batch of prompts
    batch_prompts = []
    batch_indices = []
    batch_formatted_prompts = []
    
    for j in range(args.batch_size):
        if i + j >= len(ds["test"]):
            break
        idx = i + j
        question_id = idx
        filename = f"{dataset_name}_question_id_{question_id}_{model_name}.txt"
        
        # Check if the output file already exists
        if os.path.exists(os.path.join(output_folder, filename)):
            continue
        
        question_prompt = ds["test"]['problem'][idx] + "\n\nPlease end your solution with Answer: $\\boxed{number}$ where number is the numerical answer without unit."
        batch_prompts.append(question_prompt)
        batch_indices.append(idx)
        
        # Format prompt for VLLM
        messages = [{"role": "user", "content": question_prompt}]
        formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_formatted_prompts.append(formatted_prompt)

    # Skip if no new prompts to process
    if not batch_formatted_prompts:
        continue

    # Generate responses using VLLM
    generated_outputs = llm.generate(batch_formatted_prompts, sampling_params)

    for k, output in enumerate(generated_outputs):
        generated_text = output.outputs[0].text
        question_id = batch_indices[k]
        filename = f"{dataset_name}_question_id_{question_id}_{model_name}.txt"
        
        # Save the complete generated response
        with open(os.path.join(output_folder, filename), "w") as f:
            f.write(f"### QUESTION: {batch_prompts[k]}\n\n")
            f.write(f"### THINKING: {generated_text}\n\n")
            f.write(f"### ANSWER: ")  