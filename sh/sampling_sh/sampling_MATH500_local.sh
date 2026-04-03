#!/bin/bash

#SBATCH --job-name=LocalQwenMathSampling
#SBATCH --gres=gpu:2        # uncomment only if/as needed
#SBATCH --time=0-04:00:00    # run for one day
#SBATCH --cpus-per-task=4    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/sampling/local-qwen-math-sampling-%j.log

# ===========================================
# CONFIGURATION - Edit these variables as needed
# ===========================================

# Set to "local" for local model path, "hf" for HuggingFace model ID
MODE="local"

# Model path or HuggingFace model ID
# MODEL_PATH_OR_ID="/n/holylabs/LABS/wattenberg_lab/Users/yidachen/grpo-checkpoints/qwen2.5_grpo/global_step_40/actor/huggingface"
MODEL_PATH_OR_ID="/n/holylabs/LABS/wattenberg_lab/Users/yidachen/grpo-checkpoints/qwen2.5_grpo/global_step_5/actor/huggingface"
# MODEL_PATH_OR_ID="Qwen/Qwen2.5-3B"
# Example configurations:
# For local model:
# MODE="local"
# MODEL_PATH_OR_ID="/path/to/your/local/model"

# For HuggingFace model:
# MODE="hf"  
# MODEL_PATH_OR_ID="Qwen/Qwen2.5-7B-Instruct"

# ===========================================

source activate verl

# Run the sampling script based on the mode
if [ "$MODE" == "local" ]; then
    echo "Running with local model: $MODEL_PATH_OR_ID"
    python sampling_py/sample_MATH500_batched_vllm_local.py \
        --model_path "$MODEL_PATH_OR_ID" \
        --temperature 0.6 \
        --top_p 0.95 \
        --min_p 0 \
        --top_k 20 \
        --batch_size 250 \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.9
elif [ "$MODE" == "hf" ]; then
    echo "Running with HuggingFace model: $MODEL_PATH_OR_ID"
    python sampling_py/sample_MATH500_batched_vllm_local.py \
        --model_path "$MODEL_PATH_OR_ID" \
        --from_hf \
        --temperature 0.6 \
        --top_p 0.95 \
        --min_p 0 \
        --top_k 20 \
        --batch_size 250 \
        --tensor_parallel_size 2 \
        --gpu_memory_utilization 0.9
else
    echo "Error: Invalid mode '$MODE'. Use 'local' or 'hf'"
    exit 1
fi 