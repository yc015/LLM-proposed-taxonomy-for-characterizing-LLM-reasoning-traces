import os 

# HF
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

# Torch
import torch

# Sklearn
from sklearn.model_selection import train_test_split

# Utility
import os
import argparse
import random
import numpy as np

np.random.seed(0)
random.seed(10)

# Coder - import both versions
from src.coder_few_shot import CoderFewShot
from src.coder_prior_knowledge import CoderPriorKnowledge
# from src.experimental.coder_two_stage_vllm import CoderTwoStageVLLM
from src.prompt_dataset import load_reasoning_traces, shuffle_outputs_and_labels

parser = argparse.ArgumentParser(description="Generate responses using a specified model.")
parser.add_argument('--coder_model_id', type=str, default="meta-llama/Llama-3.3-70B-Instruct", help='The model ID to use for coding.')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to the reasoning dataset.')
parser.add_argument('--temperature', type=float, default=0.6, help='The value used to module the next token probabilities.')
parser.add_argument('--max_new_tokens', type=int, default=32768, help='The maximum number of new tokens to generate.')
parser.add_argument('--top_k', type=int, default=50, help='The number of highest probability vocabulary tokens to keep for top-k-filtering.')
parser.add_argument('--top_p', type=float, default=0.95, help='If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation.')
parser.add_argument('--min_p', type=float, default=0)
parser.add_argument('--frequency_penalty', type=float, default=0.0)
parser.add_argument('--compared_models', type=str, default="Qwen3-14B,Phi-4-reasoning-plus", help="names of models, separate by comma without space")
parser.add_argument('--think_mode', type=str, default="no", help="Whether to let the coder model to think or not (if coder is a thinking model).")
parser.add_argument('--think_budget', type=int, default=0, help="Max number of words to be generated in the thinking.")
parser.add_argument('--global_patience', type=int, default=15, help="Max number of no-update run before exiting the training.")
parser.add_argument('--patience', type=int, default=2, help="Max number of update trial on a single example.")
parser.add_argument('--num_warmup', type=int, default=1, help="Number of warmup samples.")
parser.add_argument('--num_train', type=int, default=49, help="Number of training samples.")
parser.add_argument('--num_test', type=int, default=200, help="Number of testing samples.")
parser.add_argument('--output_path', type=str, default=None)
parser.add_argument('--job_id', type=str, required=True, help='Job ID for output file naming')
parser.add_argument('--vllm', action='store_true', help='Use vLLM version of the coder for faster inference (enables batched evaluation and optimized multi-GPU inference)')
parser.add_argument('--two_stage', action='store_true', help='Use two-stage classification approach with rule-based annotation followed by classification')
parser.add_argument('--multi_gpus', type=int, default=8, help='Number of GPUs to use for vLLM tensor parallelism (only applies when --vllm is used)')
parser.add_argument('--batch_size', type=int, default=50, help='Batch size for evaluation (only applies when --vllm is used)')
parser.add_argument('--evaluation_method', type=str, default="generative", help='Method used in evaluation: ["generative", "hard_vote", "soft_vote", "naive_bayes"]')
parser.add_argument('--run_type', type=str, default="Normal", help='Run type: ["Normal", "VML"]')
parser.add_argument('--max_rule', type=int, default=40, help='Max dimension of the codebook')

parser.add_argument('--max_train_samples', type=int, default=320, help='Max number of training samples')
parser.add_argument('--accumulation_size', type=int, default=10, help='Number of samples to accumulate before attempting updates')
parser.add_argument('--sampling_training', action='store_true', help='Use sampling in training')
parser.add_argument('--num_shots', type=int, default=0, help='Number of few shot examples (pair of reasoning); 0 means prior knowledge baseline')
parser.add_argument('--seed', type=int, default=10, help='Seed for reproducibility')
args = parser.parse_args()

torch.manual_seed(args.seed)

model_options = args.compared_models.split(",")

if len(model_options) <= 0:
    print("No model options provided. Exiting...")
    exit()

dataset_folder = args.dataset_path.split('/')[-1].split('.')[0]  # Assuming the dataset file has an extension
if args.output_path is None:
    output_folder = f"/n/home04/yidachen/reasoning_characteristics/coder_ckpt/{dataset_folder}-baseline"
    intermediate_output_folder = f"/n/home04/yidachen/reasoning_characteristics/coder_ckpt/{dataset_folder}-baseline/intermediate_ckpt"
    
    # Create the directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    if not os.path.exists(intermediate_output_folder):
        os.makedirs(intermediate_output_folder, exist_ok=True)
else:
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    output_folder = args.output_path
    if not os.path.exists(os.path.join(args.output_path, "intermediate_ckpt")):
        os.makedirs(os.path.join(args.output_path, "intermediate_ckpt"), exist_ok=True)

    intermediate_output_folder = os.path.join(args.output_path, "intermediate_ckpt")
    print("Use provided output folder:", args.output_path)

# Construct the output path with datetime
if args.num_shots <= 0:
    coder_type = "prior_knowledge"
else:
    coder_type = f"{args.num_shots:d}_shots"

if args.sampling_training:
    print("Only Presence of Reasoning coder supports sampling style training")
    coder_type = "por"

args.output_path = (
    f"{output_folder}/baseline-coder-{coder_type}-{args.coder_model_id.split('/')[-1]}"
    f"-compare-{'_'.join(model_options)}-on-{dataset_folder}-{args.job_id}.pkl"
)
intermediate_output_path = (
    f"{intermediate_output_folder}/baseline-coder-{coder_type}-{args.coder_model_id.split('/')[-1]}"
    f"-compare-{'_'.join(model_options)}-on-{dataset_folder}-{args.job_id}.pkl"
)

print("Use default output path:", args.output_path)


warmup_num = args.num_warmup
train_num = args.num_train
test_num = args.num_test

# Define the base folder for reasoning traces
base_folder = args.dataset_path

# Coder Sampling parameters:
sampling_parameters = {'temperature': args.temperature, 'do_sample': True, 'max_new_tokens': args.max_new_tokens, 'top_p': args.top_p, 'frequency_penalty': args.frequency_penalty}


enabled_thinking = True if args.think_mode.lower() == "yes" else False
if enabled_thinking:
    print("Enable thinking mode")

# Choose coder class based on --vllm and --two_stage flags
if coder_type == "prior_knowledge":
    print("BASELINE: Zero shot prior knowledge coder")
    CoderClass = CoderPriorKnowledge
    sampling_parameters = None  # vLLM uses SamplingParams internally
elif args.num_shots > 0:
    print(f"BASELINE: {args.num_shots} shot coder")
    CoderClass = CoderFewShot
    # For vLLM, we need to pass sampling parameters differently
    sampling_parameters = None  # vLLM uses SamplingParams internally

### Instantiate the Coder Model
if "llama" in args.coder_model_id.lower():
    max_input_tokens = 131000
else:
    max_input_tokens = 24576
print("Max input tokens", max_input_tokens)

initial_code_example = None
min_num_train_samples = 0

model_name_translator = {
                        # No translation needed. Allow model to use its prior knowledge in judgement.
                         }

for model_option in model_options:
    if model_option not in list(model_name_translator.keys()):
        model_name_translator[model_option] = model_option

model_options = [model_name_translator[model_option] for model_option in model_options]

model_name_decoder = {value: key for key, value in model_name_translator.items()}

code_inst = None
correction_inst = None

coder = CoderClass(args.coder_model_id, 
                   model_options, 
                   think_mode=enabled_thinking,
                   cache_dir="/n/holylabs/LABS/wattenberg_lab/Lab/pretrained_models/",
                   think_budget=args.think_budget,
                   multi_gpus=args.multi_gpus if args.vllm else 8,
                   sampling_parameters=sampling_parameters,
                   max_input_tokens=max_input_tokens,
                   global_patience=args.global_patience,
                   patience=args.patience,
                   initial_code_example=initial_code_example,
                   evaluation_method="generative", # Evaluation method must be generative for zero or few shot baseline
                   code_inst=code_inst,
                   correction_inst=correction_inst,
                   max_rule=args.max_rule)

coder.num_few_shot_examples = args.num_shots

# Must be trained for these number of samples before exiting the training.
coder.min_num_train_samples = min_num_train_samples
coder.max_train_samples = args.max_train_samples
coder.stop_criteria = ["max_rules", "global_patience", "max_train_samples"]

if args.run_type == "VML":
    coder.batch_update = True
    coder.batch_update_size = 4
else:
    coder.batch_update = False
    coder.batch_update_size = 1

# Load reasoning traces
reasoning_traces = load_reasoning_traces(base_folder)

for model_name, traces in reasoning_traces.items():
    print("---------------")
    print(f"Model: {model_name}")
    print(f"Number of traces: {len(traces)}")
    for question_id, trace in traces.items():
        question = trace.get('question', '')
        thinking = trace.get('thinking', '')
        print(f"Question ID: {question_id}")
        print(f"Question: {question[:100]}")
        print(f"Reasoning Trace: {thinking[:100]}\n")
        break
    print("Empty Reasoning Traces:")
    for question_id, trace in traces.items():
        if len(trace.get('thinking', '')) < 1:
            print(question_id)
    print("\n")


# ### Run Coder

# Note: sampling_parameters are now handled in the coder initialization
# For vLLM: uses internal SamplingParams
# For standard: uses the dict format
if not args.vllm:
    # Only set sampling_parameters for standard coder
    coder.sampling_parameters = sampling_parameters


# ### Reasoning Dataset
# Generate few_shot_outputs and few_shot_labels
model_codename_0 = model_name_decoder[coder.model_options[0]]
model_codename_1 = model_name_decoder[coder.model_options[1]]
num_samples = len(reasoning_traces[model_codename_0])

all_idx = list(reasoning_traces[model_codename_0].keys())

all_idx = sorted(all_idx)

outputs = [
    [reasoning_traces[model_codename_0][idx]['thinking'], 
    reasoning_traces[model_codename_1][idx]['thinking']]
    for idx in all_idx
]

questions = [
    reasoning_traces[model_codename_0][idx]['question']
    for idx in all_idx
]

labels = [
    [coder.model_options[0], coder.model_options[1]]
    for idx in all_idx
]

ids = [idx for idx in all_idx]

# Shuffle the few_shot_outputs and few_shot_labels
shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions = shuffle_outputs_and_labels(outputs, labels, ids, questions)

dataset = [{"outputs": single_outputs, "labels": single_labels, "id": single_id, "question": single_question} for single_outputs, single_labels, single_id, single_question in zip(shuffled_outputs, shuffled_labels, shuffled_ids, shuffled_questions)]

warmup_dataset, remaining_dataset, _, remaining_labels = train_test_split(dataset, shuffled_labels, test_size=len(dataset) - warmup_num, random_state=42)
train_dataset, eval_dataset, _, _ = train_test_split(remaining_dataset, remaining_labels, test_size=test_num, train_size=train_num, random_state=42, stratify=remaining_labels)

print(f"Warmup dataset size: {len(warmup_dataset)}")
print(f"Train dataset size: {len(train_dataset)}")
print(f"Eval dataset size: {len(eval_dataset)}")

# dataset = dataset[:warmup_num + train_num + test_num]

for i in range(10):
    print(dataset[i]["labels"])
    print()


# ### Warm up the coder
coder.warm_start(warmup_dataset, ckpt_path=None)

# ### Training

coder.train(train_dataset, ckpt_path=None)

# ### Evaluation
attr_path = intermediate_output_path
attr_path = intermediate_output_path[:intermediate_output_path.rfind(".pkl")] + "_evaled_attr.pkl"
# Use batching for evaluation when vLLM is enabled for better performance
if args.vllm:
    print("Using batched evaluation for vLLM")
    coder.eval(eval_dataset, batched=True, batch_size=args.batch_size, ckpt_path=attr_path)
else:
    print("Using sequential evaluation for standard coder")
    coder.eval(eval_dataset, batched=False, ckpt_path=attr_path)
coder.save_coder(args.output_path)
