import os
import torch
from mistral_common.protocol.instruct.messages import SystemMessage, UserMessage
from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from vllm import LLM, SamplingParams
from datasets import load_dataset
import argparse
from huggingface_hub import hf_hub_download

parser = argparse.ArgumentParser(description="Generate responses using a specified model.")
parser.add_argument('--model_id', type=str, required=True, help='The model ID to use for generation.')
parser.add_argument('--do_sample', type=bool, default=True, help='Whether to use sampling; use greedy decoding otherwise.')
parser.add_argument('--temperature', type=float, default=0.8, help='The value used to module the next token probabilities.')
parser.add_argument('--max_new_tokens', type=int, default=16384, help='The maximum number of new tokens to generate.')
parser.add_argument('--top_k', type=int, default=50, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
parser.add_argument('--top_p', type=float, default=0.95, help='If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.')
parser.add_argument('--batch_size', type=int, default=64, help='The number of samples to process in a batch.')
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
model_id = args.model_id
SYSTEM_PROMPT_file = hf_hub_download(repo_id=model_id, filename="SYSTEM_PROMPT.txt")
with open(SYSTEM_PROMPT_file, "r") as file:
    SYSTEM_PROMPT = file.read()

# Initialize VLLM model
llm = LLM(
    model=model_id,
    tensor_parallel_size=args.tensor_parallel_size,
    gpu_memory_utilization=args.gpu_memory_utilization,
    trust_remote_code=True,
    download_dir=args.download_dir,
    max_model_len=32768,  # Use model's default max length
    tokenizer_mode="mistral",
    config_format="mistral",
    load_format="mistral",
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
model_name = model_id.split("/")[-1]
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
        
        # Use original prompt structure with boxed answer format
        question_prompt = ds["test"][idx]['problem'] + "\n\nPlease end your solution with Answer: $\\boxed{number}$ where number is the numerical answer without unit."
        batch_prompts.append(question_prompt)
        batch_indices.append(idx)
        
        # Create message using Mistral format
        message = ChatCompletionRequest(
            messages=[
                SystemMessage(content=SYSTEM_PROMPT),
                UserMessage(content=question_prompt),
            ],
        )
        
        # Tokenize the message
        batch_formatted_prompts.append(f"[SYSTEM_PROMPT]{SYSTEM_PROMPT}[/SYSTEM_PROMPT][INST]{question_prompt}[/INST]")

    # Skip if no new prompts to process
    if not batch_formatted_prompts:
        continue

    # Generate responses using VLLM
    generated_outputs = llm.generate(batch_formatted_prompts, sampling_params)

    for k, output in enumerate(generated_outputs):
        generated_text = output.outputs[0].text
        
        # Extract thinking content (assuming <think> tags are used)
        start_index = generated_text.find("<think>")
        end_index = generated_text.find("</think>")
        if start_index != -1 and end_index != -1:
            thinking_content = generated_text[start_index + 7:end_index].strip("\n")
            answer_content = generated_text[end_index + 8:].strip("\n")
        else:
            thinking_content = ""
            answer_content = generated_text.strip("\n")
            
        question_id = batch_indices[k]
        filename = f"{dataset_name}_question_id_{question_id}_{model_name}.txt"
        
        with open(os.path.join(output_folder, filename), "w") as f:
            f.write(f"### QUESTION: {batch_prompts[k]}\n\n")
            f.write(f"### THINKING: {thinking_content}\n\n")
            f.write(f"### ANSWER: {answer_content}")
            
        with open(os.path.join(output_folder, f"raw_output_{dataset_name}_question_id_{question_id}_{model_name}.txt"), "w") as f:
            # For VLLM, we save the complete prompt + generated text
            raw_output = batch_formatted_prompts[k] + generated_text
            f.write(f"{raw_output}\n")