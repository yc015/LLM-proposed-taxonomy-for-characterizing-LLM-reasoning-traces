#!/bin/bash

#SBATCH --job-name=PORMultiDataset
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --time=0-20:00:00
#SBATCH --cpus-per-task=8
#SBATCH --partition seas_gpu
#SBATCH --mem=196GB
#SBATCH --output=/n/home04/yidachen/reasoning_characteristics/bash_output/multi/por-multidataset-%j.log

compared_models="${1:-Qwen3-14B,Magistral-Small-2506}"
dataset_paths="${2:-/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/MATH-500,/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/cruxeval-O,/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/gpqa-gpqa_diamond,/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/execution-O,/n/holylabs/LABS/wattenberg_lab/Users/yidachen/coders/outputs/aime_24_25}"

source activate rifeval

coder_model_id="meta-llama/Llama-3.3-70B-Instruct"
think_mode="no"
think_budget=0
job_id=${SLURM_JOB_ID}

num_warmup=1
num_train=0.8
num_test=0.2
temperature=0.6
max_new_tokens=32768
top_p=0.95
top_k=50
global_patience=40
patience=2
max_rule=20
max_train_samples=1200
accumulation_size=40
batch_size=40
evaluation_method="generative"
run_type="Normal"
imputation_method="knn"
extend_with_nan_flag="--extend_with_nan"
sampling_training_flag="--sampling_training"
multi_gpus=4
reject_inconsistent_codes_flag=""
averaged_eval_flag=""
num_reruns=5
log_dir="/n/home04/yidachen/reasoning_characteristics/training_logs/por-multidataset"
checkpoint_dir="/n/home04/yidachen/reasoning_characteristics/coder_ckpt/por-multidataset"
use_initial_codebook="no"
fixed_codebook_baseline="no"
model_id=$(basename "$coder_model_id")
dataset_tag=$(echo "$dataset_paths" | tr '/,' '_')
log_file="${log_dir}/${model_id}_${job_id}.txt"

mkdir -p "$log_dir"
mkdir -p "$checkpoint_dir"

echo "Running POR multi-dataset concatenated training"
echo "Datasets: $dataset_paths"

python coder_por_multi_dataset_concat_training.py \
    --coder_model_id "$coder_model_id" \
    --dataset_paths "$dataset_paths" \
    --compared_models "$compared_models" \
    --think_mode "$think_mode" \
    --think_budget "$think_budget" \
    --num_warmup "$num_warmup" \
    --num_train "$num_train" \
    --num_test "$num_test" \
    --temperature "$temperature" \
    --max_new_tokens "$max_new_tokens" \
    --top_p "$top_p" \
    --top_k "$top_k" \
    --global_patience "$global_patience" \
    --patience "$patience" \
    --max_rule "$max_rule" \
    --max_train_samples "$max_train_samples" \
    --accumulation_size "$accumulation_size" \
    --batch_size "$batch_size" \
    --evaluation_method "$evaluation_method" \
    --run_type "$run_type" \
    --imputation_method "$imputation_method" \
    --multi_gpus "$multi_gpus" \
    --num_reruns "$num_reruns" \
    --use_initial_codebook "$use_initial_codebook" \
    --fixed_codebook_baseline "$fixed_codebook_baseline" \
    --job_id "$job_id" \
    $extend_with_nan_flag \
    $sampling_training_flag \
    $reject_inconsistent_codes_flag \
    $averaged_eval_flag \
    --output_dir "$checkpoint_dir" | tee -a "$log_file"
