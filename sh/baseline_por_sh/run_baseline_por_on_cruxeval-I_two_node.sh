#!/bin/bash

#SBATCH --job-name=BaselinePoRCRUXVLLMCoder
#SBATCH --gres=gpu:8        # vLLM benefits from multiple GPUs
#SBATCH --time=0-04:00:00    # run for one day
#SBATCH --cpus-per-task=16    # change as needed
 
 
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/coding/vllm-llama3.3-coder-on-CRUX-I-%j.log

# Accept compared_models as command line argument, with default fallback
compared_models_1="${1:-Qwen-14B,QwQ-32B}"
compared_models_2="$2"

source activate rifeval

# vLLM Configuration
coder_model_id="meta-llama/Llama-3.3-70B-Instruct" # Qwen/Qwen3-32B, meta-llama/Llama-3.3-70B-Instruct
think_mode="no" # yes, no
dataset_path="/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/cruxeval-I" # "/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/aime_24_25"
# Define the base folder for output files
base_folder="/n/home04/yidachen/reasoning_characteristics/training_logs/Baseline-cruxeval-I-vllm" # "/n/home04/yidachen/reasoning_characteristics/training_logs/aime_24_25-vllm"
# Ensure the base folder exists
mkdir -p "$base_folder"

# Extract parts for the output file name
model_id=$(basename "$coder_model_id")
dataset_name=$(basename "$dataset_path")
think_budget=0 # 600, 0
job_id=${SLURM_JOB_ID}

warmup_size=1
train_size=639
eval_size=160
temperature=0.6
max_new_tokens=32768
top_p=0.95
top_k=50

# Define list of num_shots values to test
num_shots_list=(0 1 3 5 7 10)

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
echo "Num shots values: ${num_shots_list[*]}"

# ----
# Main comparison using input argument
# ----

compared_models_name_1=$(echo "$compared_models_1" | tr ',' '-')

# Use all 8 GPUs if no second comparison, otherwise use first 4 GPUs
if [[ -z "$compared_models_2" ]]; then
    cuda_devices_1="0,1,2,3"
    multi_gpus_1=4
else
    cuda_devices_1="0,1,2,3"
    multi_gpus_1=4
fi

# Loop through each num_shots value
for num_shots in "${num_shots_list[@]}"; do
    echo "Running experiment with num_shots=$num_shots"
    mkdir -p "$base_folder/shots_${num_shots}"
    
    # Comparison 2 - using second argument (only if not empty)
    if [[ -n "$compared_models_2" ]]; then
        # If we have two comparisons, run first in background and second in foreground
        output_file_1="${base_folder}/shots_${num_shots}/${model_id}_${dataset_name}_${compared_models_name_1}_shots${num_shots}_${job_id}.txt"
        
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
        output_file_2="${base_folder}/shots_${num_shots}/${model_id}_${dataset_name}_${compared_models_name_2}_shots${num_shots}_${job_id}.txt"

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
            --top_k "$top_k" | tee -a "$output_file_2"
        
        # Wait for background job to complete before next iteration
        wait
    else
        echo "No second comparison provided, running only first comparison"
        output_file_1="${base_folder}/shots_${num_shots}/${model_id}_${dataset_name}_${compared_models_name_1}_shots${num_shots}_${job_id}.txt"
        
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
    
    echo "Completed experiment with num_shots=$num_shots"
done

echo "vLLM coder training job completed!"
echo "Results saved in: $base_folder" 