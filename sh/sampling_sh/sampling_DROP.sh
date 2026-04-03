#!/bin/bash

#SBATCH --job-name=QwQSamplingReasoningModels
#SBATCH --partition=learn  # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=1-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/QwQ-DROP-sampling-%j.log

source activate rifeval
## python sampling_py/sample_DROP_batched.py --model_id "nvidia/AceReason-Nemotron-14B" --temperature 0.6 --top_p 0.95
python sampling_py/sample_DROP_batched.py --model_id "Qwen/QwQ-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20
## python sampling_py/sample_DROP_batched.py --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --temperature 0.6 --top_p 0.95
## python sampling_py/sample_DROP_batched.py --model_id "microsoft/Phi-4-reasoning-plus" --temperature 0.8 --top_p 0.95 --top_k 50
## python sampling_py/sample_DROP_batched.py --model_id "Qwen/Qwen3-14B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20