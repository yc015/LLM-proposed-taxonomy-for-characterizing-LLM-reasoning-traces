#!/bin/bash

# Accept compared_models as command line argument, with default fallback
compared_models_1="${1:-Qwen-14B,QwQ-32B}"
compared_models_2="$2"

# vLLM Configuration
coder_model_id="meta-llama/Llama-3.3-70B-Instruct" # Qwen/Qwen3-32B, meta-llama/Llama-3.3-70B-Instruct
think_mode="no" # yes, no
dataset_path="/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/execution-O" # "/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/aime_24_25"
# Define the base folder for output files
base_folder="/n/home04/yidachen/reasoning_characteristics/training_logs/Baseline-execution-O-vllm" # "/n/home04/yidachen/reasoning_characteristics/training_logs/aime_24_25-vllm"
# Ensure the base folder exists
mkdir -p "$base_folder"

# Extract parts for the output file name
model_id=$(basename "$coder_model_id")
dataset_name=$(basename "$dataset_path")
think_budget=0 # 600, 0
job_id=${SLURM_JOB_ID}

warmup_size=1
train_size=382
eval_size=95
temperature=0.6
max_new_tokens=32768
top_p=0.95
top_k=45

num_shots=5

# vLLM specific parameters
multi_gpus=4  # Use all 8 GPUs for vLLM tensor parallelism
batch_size=50  # Increased batch size for better vLLM throughput

echo "Starting vLLM Coder Training on AIME dataset"
echo "Model: $coder_model_id"
echo "Dataset: $dataset_path"
echo "Compared Models: $compared_models_1"
echo "Multi-GPUs: $multi_gpus"
echo "Batch Size: $batch_size"
echo "Job ID: $job_id"

# ----
# Main comparison using input argument
# ----

compared_models_name_1=$(echo "$compared_models_1" | tr ',' '-')
output_file_1="${base_folder}/${model_id}_${dataset_name}_${compared_models_name_1}_${job_id}.txt"

# Use all 8 GPUs if no second comparison, otherwise use first 4 GPUs
if [[ -z "$compared_models_2" ]]; then
    cuda_devices_1="0,1,2,3"
    multi_gpus_1=4
else
    cuda_devices_1="0,1,2,3"
    multi_gpus_1=4
fi

# Comparison 2 - using second argument (only if not empty)
if [[ -n "$compared_models_2" ]]; then
    # If we have two comparisons, run first in background and second in foreground
    CUDA_VISIBLE_DEVICES=$cuda_devices_1 python coder_baseline_zero_few_shots.py \
        --coder_model_id "$coder_model_id" \
        --dataset_path "$dataset_path" \
        --compared_models "$compared_models_1" \
        --think_mode "$think_mode" \
        --think_budget "$think_budget" \
        --num_warmup "$warmup_size" \
        --num_train "$train_size" \
        --num_test "$eval_size" \
        --job_id "${job_id}_1" \
        --vllm \
        --multi_gpus "$multi_gpus_1" \
        --batch_size "$batch_size" \
        --temperature "$temperature" \
        --max_new_tokens "$max_new_tokens" \
        --top_p "$top_p" \
        --num_shots "$num_shots" \
        --top_k "$top_k" | tee -a "$output_file_1" &

    compared_models_name_2=$(echo "$compared_models_2" | tr ',' '-')
    output_file_2="${base_folder}/${model_id}_${dataset_name}_${compared_models_name_2}_${job_id}.txt"

    # Run second comparison in foreground to keep SLURM job alive
    CUDA_VISIBLE_DEVICES=4,5,6,7 python coder_baseline_zero_few_shots.py \
        --coder_model_id "$coder_model_id" \
        --dataset_path "$dataset_path" \
        --compared_models "$compared_models_2" \
        --think_mode "$think_mode" \
        --think_budget "$think_budget" \
        --num_warmup "$warmup_size" \
        --num_train "$train_size" \
        --num_test "$eval_size" \
        --job_id "${job_id}_1" \
        --vllm \
        --multi_gpus "$multi_gpus" \
        --batch_size "$batch_size" \
        --temperature "$temperature" \
        --max_new_tokens "$max_new_tokens" \
        --top_p "$top_p" \
        --num_shots "$num_shots" \
        --top_k "$top_k" | tee -a "$output_file_2" & wait
else
    echo "No second comparison provided, running only first comparison"
    # Run single comparison in foreground to keep SLURM job alive
    CUDA_VISIBLE_DEVICES=$cuda_devices_1 python coder_baseline_zero_few_shots.py \
        --coder_model_id "$coder_model_id" \
        --dataset_path "$dataset_path" \
        --compared_models "$compared_models_1" \
        --think_mode "$think_mode" \
        --think_budget "$think_budget" \
        --num_warmup "$warmup_size" \
        --num_train "$train_size" \
        --num_test "$eval_size" \
        --job_id "${job_id}_1" \
        --vllm \
        --multi_gpus "$multi_gpus_1" \
        --batch_size "$batch_size" \
        --temperature "$temperature" \
        --max_new_tokens "$max_new_tokens" \
        --top_p "$top_p" \
        --num_shots "$num_shots" \
        --top_k "$top_k" | tee -a "$output_file_1"
fi

echo "vLLM coder training job completed!"
echo "Results saved in: $base_folder" 