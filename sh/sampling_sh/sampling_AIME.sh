#!/bin/bash

#SBATCH --job-name=MagistralSamplingReasoningModels
#SBATCH --partition=learn  # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:2        # uncomment only if/as needed
#SBATCH --mem-per-cpu=128G
#SBATCH --time=0-06:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/sampling/Magistral-AIME-24-sampling-%j.log

source activate rifeval

# python sampling_py/sample_AIME_24_vllm.py --model_id "Qwen/Qwen3-0.6B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
# python sampling_py/sample_AIME_24_vllm.py --model_id "Qwen/Qwen3-1.7B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
# python sampling_py/sample_AIME_24_vllm.py --model_id "Qwen/Qwen3-4B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
# python sampling_py/sample_AIME_24_vllm.py --model_id "Qwen/Qwen3-8B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
# python sampling_py/sample_AIME_24_vllm.py --model_id "Qwen/Qwen3-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768

# python sampling_py/sample_AIME_25_vllm.py --model_id "Qwen/Qwen3-0.6B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_vllm.py --model_id "Qwen/Qwen3-1.7B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_vllm.py --model_id "Qwen/Qwen3-4B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_vllm.py --model_id "Qwen/Qwen3-8B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_vllm.py --model_id "Qwen/Qwen3-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768

## python sampling_py/sample_AIME_24_vllm.py --model_id "nvidia/AceReason-Nemotron-14B" --temperature 0.6 --top_p 0.95 --max_new_tokens 32768
## python sampling_py/sample_AIME_24_vllm.py --model_id "Qwen/QwQ-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
## python sampling_py/sample_AIME_24_vllm.py --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --temperature 0.6 --top_p 0.95 --max_new_tokens 32768
## python sampling_py/sample_AIME_24_vllm.py --model_id "microsoft/Phi-4-reasoning-plus" --temperature 0.8 --top_p 0.95 --top_k 50 --max_new_tokens 32768
# python sampling_py/sample_AIME_24_vllm.py --model_id "microsoft/Phi-4-reasoning" --temperature 0.8 --top_p 0.95 --top_k 50 --max_new_tokens 32768
## python sampling_py/sample_AIME_24_vllm.py --model_id "Qwen/Qwen3-14B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
## python sampling_py/sample_AIME_24_magistral.py --model_id "mistralai/Magistral-Small-2506" --temperature 0.7 --top_p 0.95 --min_p 0 --max_new_tokens 32768
# python sampling_py/sample_AIME_24_magistral_vllm.py --model_id "mistralai/Magistral-Small-2506" --temperature 0.7 --top_p 0.95 --min_p 0 --max_new_tokens 32768

## python sampling_py/sample_AIME_25_vllm.py --model_id "nvidia/AceReason-Nemotron-14B" --temperature 0.6 --top_p 0.95 --max_new_tokens 32768
## python sampling_py/sample_AIME_25_vllm.py --model_id "Qwen/QwQ-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
## python sampling_py/sample_AIME_25_vllm.py --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --temperature 0.6 --top_p 0.95 --max_new_tokens 32768
## python sampling_py/sample_AIME_25_vllm.py --model_id "microsoft/Phi-4-reasoning-plus" --temperature 0.8 --top_p 0.95 --top_k 50 --max_new_tokens 32768
# python sampling_py/sample_AIME_25_vllm.py --model_id "microsoft/Phi-4-reasoning" --temperature 0.8 --top_p 0.95 --top_k 50 --max_new_tokens 32768
## python sampling_py/sample_AIME_25_vllm.py --model_id "Qwen/Qwen3-14B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20 --max_new_tokens 32768
## python sampling_py/sample_AIME_25_magistral.py --model_id "mistralai/Magistral-Small-2506" --temperature 0.7 --top_p 0.95 --min_p 0 --max_new_tokens 32768
python sampling_py/sample_AIME_25_magistral_vllm.py --model_id "mistralai/Magistral-Small-2506" --temperature 0.7 --top_p 0.95 --min_p 0 --max_new_tokens 32768