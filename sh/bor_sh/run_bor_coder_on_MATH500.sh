#!/bin/bash

#SBATCH --job-name=MATHBORCoder
#SBATCH --gres=gpu:8        # BOR with vLLM uses tensor parallelism across multiple GPUs
#SBATCH --time=0-12:00:00    # run for 12 hours (BOR with vLLM should be faster than sequential processing)
#SBATCH --cpus-per-task=16    # change as needed
#SBATCH --partition gpu 
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/coding/bor-llama3.3-coder-on-MATH-%j.log

# Accept compared_models as command line argument, with default fallback
compared_models_1="${1:-Qwen3-14B,Phi-4-reasoning-plus}"
compared_models_2="$2"

source activate rifeval

# BOR Configuration
coder_model_id="meta-llama/Llama-3.3-70B-Instruct" # Qwen/Qwen3-32B, meta-llama/Llama-3.3-70B-Instruct
think_mode="no" # yes, no
dataset_path="/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/MATH-500" # "/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/aime_24_25"
# Define the base folder for output files
base_folder="/n/home04/yidachen/reasoning_characteristics/training_logs/MATH-500-bor" # "/n/home04/yidachen/reasoning_characteristics/training_logs/aime_24_25-bor"
# Ensure the base folder exists

# Extract parts for the output file name
model_id=$(basename "$coder_model_id")
dataset_name=$(basename "$dataset_path")
think_budget=0 # 600, 0
job_id=${SLURM_JOB_ID}

warmup_size=1
train_size=399
eval_size=100
temperature=0.6
max_new_tokens=32768
top_p=0.95
evaluation_method="euclidean_similarity"  # BOR-specific methods: multinomial_nb, cosine_similarity, euclidean_similarity, knn, logistic_regression
global_patience=15
top_k=50
run_type="Normal" # VML, Normal
max_rule=30

max_train_samples=800
normalization_method="comparative" # L1 used for cross models
imputation_method="knn"
averaged_eval=False
num_reruns=5
reject_inconsistent_codes=False
use_initial_codebook="no"
seed=-1

# BOR specific parameters
batch_size=40  # Smaller batch size for BOR (used for accumulation and evaluation)
accumulation_size=40  # Number of samples to accumulate before attempting updates
fixed_codebook_baseline="no"

# vLLM specific parameters (now that CoderBOR inherits from CoderVLLM)
multi_gpus=4  # Number of GPUs for vLLM tensor parallelism

# Build conditional flags
averaged_eval_flag=""
if [[ "$averaged_eval" == "True" ]]; then
    averaged_eval_flag="--averaged_eval"
fi

if [[ "$use_initial_codebook" == "yes" ]]; then
    base_folder="${base_folder}/use_initial_codebook"
fi

mkdir -p "$base_folder"

reject_inconsistent_codes_flag=""
if [[ "$reject_inconsistent_codes" == "True" ]]; then
    reject_inconsistent_codes_flag="--reject_inconsistent_codes"
fi

echo "Starting BOR Coder Training on MATH dataset"
echo "Model: $coder_model_id"
echo "Dataset: $dataset_path"
echo "Compared Models: $compared_models_1"
echo "Batch Size: $batch_size"
echo "Accumulation Size: $accumulation_size"
echo "Evaluation Method: $evaluation_method"
echo "Multi GPUs: $multi_gpus"
echo "Job ID: $job_id"

# ----
# Main comparison using input argument
# ----

compared_models_name_1=$(echo "$compared_models_1" | tr ',' '-')
output_file_1="${base_folder}/${model_id}_${dataset_name}_${normalization_method}_${compared_models_name_1}_${job_id}.txt"

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
    CUDA_VISIBLE_DEVICES=$cuda_devices_1 python coder_bor_full_training_procedure.py \
        --coder_model_id "$coder_model_id" \
        --dataset_path "$dataset_path" \
        --compared_models "$compared_models_1" \
        --think_mode "$think_mode" \
        --think_budget "$think_budget" \
        --num_warmup "$warmup_size" \
        --num_train "$train_size" \
        --num_test "$eval_size" \
        --job_id "${job_id}_1" \
        --batch_size "$batch_size" \
        --accumulation_size "$accumulation_size" \
        --temperature "$temperature" \
        --max_new_tokens "$max_new_tokens" \
        --top_p "$top_p" \
        --evaluation_method "$evaluation_method" \
        --global_patience "$global_patience" \
        --run_type "$run_type" \
        --max_rule "$max_rule" \
        --top_k "$top_k" \
        --vllm \
        --extend_with_nan \
        --sampling_training \
        --max_train_samples "$max_train_samples" \
        --normalize \
        --normalization_method "$normalization_method" \
        --imputation_method "$imputation_method" \
        --multi_gpus "$multi_gpus_1" \
        --num_reruns "$num_reruns" \
        $averaged_eval_flag \
        --use_initial_codebook "$use_initial_codebook" \
        --fixed_codebook_baseline "$fixed_codebook_baseline" \
        --seed "$seed" \
        $reject_inconsistent_codes_flag | tee -a "$output_file_1" &

    compared_models_name_2=$(echo "$compared_models_2" | tr ',' '-')
    output_file_2="${base_folder}/${model_id}_${dataset_name}_${normalization_method}_${compared_models_name_2}_${job_id}.txt"

    # Run second comparison in foreground to keep SLURM job alive
    CUDA_VISIBLE_DEVICES=4,5,6,7 python coder_bor_full_training_procedure.py \
        --coder_model_id "$coder_model_id" \
        --dataset_path "$dataset_path" \
        --compared_models "$compared_models_2" \
        --think_mode "$think_mode" \
        --think_budget "$think_budget" \
        --num_warmup "$warmup_size" \
        --num_train "$train_size" \
        --num_test "$eval_size" \
        --job_id "${job_id}_2" \
        --batch_size "$batch_size" \
        --accumulation_size "$accumulation_size" \
        --temperature "$temperature" \
        --max_new_tokens "$max_new_tokens" \
        --top_p "$top_p" \
        --evaluation_method "$evaluation_method" \
        --global_patience "$global_patience" \
        --run_type "$run_type" \
        --max_rule "$max_rule" \
        --top_k "$top_k" \
        --vllm \
        --extend_with_nan \
        --sampling_training \
        --max_train_samples "$max_train_samples" \
        --normalize \
        --normalization_method "$normalization_method" \
        --imputation_method "$imputation_method" \
        --multi_gpus "$multi_gpus" \
        --num_reruns "$num_reruns" \
        $averaged_eval_flag \
        --use_initial_codebook "$use_initial_codebook" \
        --fixed_codebook_baseline "$fixed_codebook_baseline" \
        --seed "$seed" \
        $reject_inconsistent_codes_flag | tee -a "$output_file_2" & wait
else
    echo "No second comparison provided, running only first comparison"
    # Run single comparison in foreground to keep SLURM job alive
    CUDA_VISIBLE_DEVICES=$cuda_devices_1 python coder_bor_full_training_procedure.py \
        --coder_model_id "$coder_model_id" \
        --dataset_path "$dataset_path" \
        --compared_models "$compared_models_1" \
        --think_mode "$think_mode" \
        --think_budget "$think_budget" \
        --num_warmup "$warmup_size" \
        --num_train "$train_size" \
        --num_test "$eval_size" \
        --job_id "${job_id}_1" \
        --batch_size "$batch_size" \
        --accumulation_size "$accumulation_size" \
        --temperature "$temperature" \
        --max_new_tokens "$max_new_tokens" \
        --top_p "$top_p" \
        --evaluation_method "$evaluation_method" \
        --global_patience "$global_patience" \
        --run_type "$run_type" \
        --max_rule "$max_rule" \
        --top_k "$top_k" \
        --vllm \
        --extend_with_nan \
        --sampling_training \
        --max_train_samples "$max_train_samples" \
        --normalize \
        --normalization_method "$normalization_method" \
        --imputation_method "$imputation_method" \
        --multi_gpus "$multi_gpus_1" \
        --num_reruns "$num_reruns" \
        $averaged_eval_flag \
        --use_initial_codebook "$use_initial_codebook" \
        --fixed_codebook_baseline "$fixed_codebook_baseline" \
        --seed "$seed" \
        $reject_inconsistent_codes_flag | tee -a "$output_file_1"
fi

# ----
# To run the following comparisons, uncomment the relevant section below.
# ----

# # Comparison with different BOR evaluation methods
# compared_models_2="Qwen3-14B,QwQ-32B"
# compared_models_name_2=$(echo "$compared_models_2" | tr ',' '-')
# output_file_2="${base_folder}/${model_id}_${dataset_name}_${compared_models_name_2}_cosine_${job_id}.txt"
#
# CUDA_VISIBLE_DEVICES=0 python coder_bor_full_training_procedure.py \
#     --coder_model_id "$coder_model_id" \
#     --dataset_path "$dataset_path" \
#     --compared_models "$compared_models_2" \
#     --think_mode "$think_mode" \
#     --think_budget "$think_budget" \
#     --num_warmup 1 \
#     --num_train 200 \
#     --num_test 600 \
#     --job_id "${job_id}_2" \
#     --batch_size 8 \
#     --accumulation_size 10 \
#     --evaluation_method "cosine_similarity" \
#     --temperature 0.6 \
#     --max_new_tokens 32768 \
#     --top_p 0.95 \
#     --top_k 50 | tee "$output_file_2"
#
# # Comparison with KNN evaluation
# compared_models_3="AceReason-Nemotron-14B,Phi-4-reasoning-plus"
# compared_models_name_3=$(echo "$compared_models_3" | tr ',' '-')
# output_file_3="${base_folder}/${model_id}_${dataset_name}_${compared_models_name_3}_knn_${job_id}.txt"
#
# CUDA_VISIBLE_DEVICES=0 python coder_bor_full_training_procedure.py \
#     --coder_model_id "$coder_model_id" \
#     --dataset_path "$dataset_path" \
#     --compared_models "$compared_models_3" \
#     --think_mode "$think_mode" \
#     --think_budget "$think_budget" \
#     --num_warmup 1 \
#     --num_train 200 \
#     --num_test 600 \
#     --job_id "${job_id}_3" \
#     --batch_size 8 \
#     --accumulation_size 15 \
#     --evaluation_method "knn" \
#     --temperature 0.6 \
#     --max_new_tokens 32768 \
#     --top_p 0.95 \
#     --top_k 50 | tee "$output_file_3"

echo "BOR coder training job completed!"
echo "Results saved in: $base_folder" 