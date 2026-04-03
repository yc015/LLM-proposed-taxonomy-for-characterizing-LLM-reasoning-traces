#!/bin/bash

#SBATCH --job-name=DSSamplingReasoningModels
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=1-00:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/sampling/DS-ARC-AGI-sampling-%j.log

source activate rifeval

## python sampling_py/sample_Arc-AGI-1.py --model_id "nvidia/AceReason-Nemotron-14B" --temperature 0.6 --top_p 0.95 --max_new_tokens 16384
## python sampling_py/sample_Arc-AGI-1.py --model_id "Qwen/QwQ-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 16384
python sampling_py/sample_Arc-AGI-1.py --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --temperature 0.6 --top_p 0.95 --max_new_tokens 16384
## python sampling_py/sample_Arc-AGI-1.py --model_id "microsoft/Phi-4-reasoning-plus" --temperature 0.8 --top_p 0.95 --top_k 50 --max_new_tokens 16384
## python sampling_py/sample_Arc-AGI-1.py --model_id "Qwen/Qwen3-14B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 16384
## python sampling_py/sample_Arc-AGI-1_magistral.py --model_id "mistralai/Magistral-Small-2506" --temperature 0.7 --top_p 0.95 --min_p 0 --max_new_tokens 16384