import os 

# Torch
import torch

# Sklearn
from sklearn.model_selection import train_test_split

# Utility
import pickle
import os
import argparse
import random
from src.initial_codebook import INITIAL_CODEBOOK, INITIAL_CODEBOOK_VML

from src.prompts.vml_inst import VML_CODE_INST, VML_CORRECTION_INST
from src import model_name_translator

import numpy as np

torch.manual_seed(10)
np.random.seed(0)
random.seed(10)

# Coder - import both versions
from src.coder import Coder
from src.coder_vllm import CoderVLLM
from src.coder_por import CoderPOR
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

parser.add_argument('--extend_with_nan', action='store_true', help='Use nan for extension')
parser.add_argument('--imputation_method', type=str, default="mean", help='Options: "mean", "median", "most_frequent", "constant", "knn", "iterative"')
parser.add_argument('--use_initial_codebook', type=str, default="yes", help='Use initial codebook')
parser.add_argument('--fixed_codebook_baseline', type=str, default="no", help='Fix taxonomy baseline')
parser.add_argument('--seed', type=int, default=-1, help='Seed for random number generator')
parser.add_argument('--split_path', type=str, default=None, help='Path to the split file')
args = parser.parse_args()

if args.seed > 0:
    torch.manual_seed(args.seed)


model_options = args.compared_models.split(",")

if len(model_options) <= 0:
    print("No model options provided. Exiting...")
    exit()

dataset_folder = args.dataset_path.split('/')[-1].split('.')[0]  # Assuming the dataset file has an extension
if args.fixed_codebook_baseline == "yes":
    fixed_codebook_baseline = "-fixed-codebook-baseline"
else:
    fixed_codebook_baseline = ""

if args.run_type == "VML":
    run_type = "-VML"
else:
    run_type = ""

if args.output_path is None:
    output_folder = f"/n/home04/yidachen/reasoning_characteristics/coder_ckpt/{dataset_folder}-PoR{fixed_codebook_baseline}{run_type}"
    intermediate_output_folder = f"/n/home04/yidachen/reasoning_characteristics/coder_ckpt/{dataset_folder}-PoR{fixed_codebook_baseline}{run_type}/intermediate_ckpt"
    
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
if args.two_stage:
    coder_type = "two-stage-vllm"
elif args.vllm:
    coder_type = "por"
else:
    coder_type = "standard"

if args.sampling_training:
    print("Only Presence of Reasoning coder supports sampling style training")
    coder_type = "por"

args.output_path = (
    f"{output_folder}/coder-{coder_type}-{args.coder_model_id.split('/')[-1]}"
    f"-compare-{'_'.join(model_options)}-on-{dataset_folder}-{args.job_id}.pkl"
)
intermediate_output_path = (
    f"{intermediate_output_folder}/coder-{coder_type}-{args.coder_model_id.split('/')[-1]}"
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
if args.sampling_training:
    print("Only Presence of Reasoning coder supports sampling style training")
    print("Using vLLM version of the coder for faster inference")
    CoderClass = CoderPOR
    sampling_parameters = None  # vLLM uses SamplingParams internally
elif args.run_type == "VML":
    print("Using VML version of the coder")
    CoderClass = CoderPOR
    sampling_parameters = None  # vLLM uses SamplingParams internally
elif args.vllm:
    print("Using vLLM version of the coder for faster inference")
    CoderClass = CoderVLLM
    # For vLLM, we need to pass sampling parameters differently
    sampling_parameters = None  # vLLM uses SamplingParams internally
else:
    print("Using standard transformers version of the coder")
    CoderClass = Coder

### Instantiate the Coder Model
if "llama" in args.coder_model_id.lower():
    max_input_tokens = 73728
else:
    max_input_tokens = 24576
print("Max input tokens", max_input_tokens)

initial_code_example = None
min_num_train_samples = 100
if "cruxeval" in args.dataset_path.lower() or "execution" in args.dataset_path.lower():
    min_num_train_samples = 300
elif "gpqa" in args.dataset_path.lower():
    min_num_train_samples = 100
elif "math" in args.dataset_path.lower():
    min_num_train_samples = 200


if args.run_type == "VML":
    # Use original model names for VML
    for model_option in model_options:
        model_name_translator[model_option] = model_option

for model_option in model_options:
    if model_option not in list(model_name_translator.keys()):
        model_name_translator[model_option] = model_option

model_options = [model_name_translator[model_option] for model_option in model_options]

model_name_decoder = {value: key for key, value in model_name_translator.items()}

if args.run_type == "VML":
    print("Run VML baseline")
    code_inst = VML_CODE_INST
    correction_inst = VML_CORRECTION_INST
    initial_code_example = None
else:
    code_inst = None
    correction_inst = None

if args.use_initial_codebook == "yes" or args.fixed_codebook_baseline == "yes":
    initial_codebook = INITIAL_CODEBOOK
else:
    initial_codebook = None

evaluation_method = args.evaluation_method
if args.run_type == "VML":
    evaluation_method = "generative"
    initial_code_example = INITIAL_CODEBOOK_VML
    max_input_tokens = 122880
    # if "aime" in args.dataset_path:
        # max_input_tokens = 122880
    min_num_train_samples = 50
    initial_codebook = None

for model_name in model_options:
    if "Seed" in model_name:
        min_num_train_samples = 400

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
                   evaluation_method=evaluation_method,
                   code_inst=code_inst,
                   correction_inst=correction_inst,
                   max_rule=args.max_rule,
                   codebook=initial_codebook)
        
if args.fixed_codebook_baseline:
    print("Predefined Taxonomy Baseline: No Updates to the Codebook.")
    coder.no_update_to_codebook = args.fixed_codebook_baseline == "yes"

# Must be trained for these number of samples before exiting the training.
coder.min_num_train_samples = min_num_train_samples
coder._extend_with_nan = args.extend_with_nan
coder._imputation_method = args.imputation_method
coder.max_train_samples = args.max_train_samples
coder.stop_criteria = ["max_rules", "global_patience", "max_train_samples"]

if args.run_type == "VML":
    coder.run_type = "VML"
    coder.batch_update = True
    if "aime" in args.dataset_path.lower():
        coder.batch_update_size = 1
    elif "gpqa" in args.dataset_path.lower():
        coder.batch_update_size = 2
    else:
        coder.batch_update_size = 6
    coder.patience = 1
else:
    coder.run_type = "Normal"
    coder.batch_update = False
    coder.batch_update_size = 1

print(f"BATCH UPDATE SIZE: {coder.batch_update_size}")

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
# all_idx = sorted(all_idx)

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

if args.split_path is not None and os.path.exists(args.split_path):
    print(f"Use predefined split at {args.split_path}")
    with open(args.split_path, "rb") as f:
        splits = pickle.load(f)
        warmup_qid = splits["warmup_qid"]
        train_qid = splits["train_qid"]
        eval_qid = splits["eval_qid"]
        random.shuffle(train_qid)
        warmup_dataset = []
        train_dataset = []
        eval_dataset = []
        for qid in warmup_qid:
            for data in dataset:
                if data["id"] == qid:
                    warmup_dataset.append(data)
        for qid in train_qid:
            for data in dataset:
                if data["id"] == qid:
                    train_dataset.append(data)
        for qid in eval_qid:
            for data in dataset:
                if data["id"] == qid:
                    eval_dataset.append(data)
else:
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

attr_path = intermediate_output_path
attr_path = intermediate_output_path[:intermediate_output_path.rfind(".pkl")] + "_warmup_attr.pkl"
if not args.fixed_codebook_baseline == "yes":
    coder.warm_start(warmup_dataset, ckpt_path=attr_path)

# ### Training
attr_path = intermediate_output_path
attr_path = intermediate_output_path[:intermediate_output_path.rfind(".pkl")] + "_trained_attr.pkl"

if args.sampling_training:
    print("Train with sampled observations")
    coder.train(train_dataset, 
                ckpt_path=attr_path, 
                accumulate_observation_training=True,
                accumulation_size=args.accumulation_size,
                sampling_training=args.sampling_training,
                batch_size=args.accumulation_size)
else:
    coder.train(train_dataset, ckpt_path=attr_path)

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
