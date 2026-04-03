#!/bin/bash

#SBATCH --job-name=MagistralMathSamplingReasoningModels
#SBATCH --account=genai_interns
#SBATCH --partition=genai_interns  # change as needed, e.g., lowpri on some clusters
#SBATCH --qos=genai_interns
#SBATCH --gres=gpu:2        # uncomment only if/as needed
#SBATCH --time=1-00:00:00    # run for one day
#SBATCH --cpus-per-task=4    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/sampling/Magistral-math-sampling-%j.log

source activate rifeval
## python sampling_py/sample_MATH500_batched.py --model_id "nvidia/AceReason-Nemotron-14B" --temperature 0.6 --top_p 0.95
## python sampling_py/sample_MATH500_batched.py --model_id "Qwen/QwQ-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20
## python sampling_py/sample_MATH500_batched.py --model_id "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" --temperature 0.6 --top_p 0.95
## python sampling_py/sample_MATH500_batched.py --model_id "microsoft/Phi-4-reasoning-plus" --temperature 0.8 --top_p 0.95 --top_k 50
# python sampling_py/sample_MATH500_batched_vllm.py --model_id "microsoft/Phi-4-reasoning" --temperature 0.8 --top_p 0.95 --top_k 50
# python sampling_py/sample_MATH500_batched.py --model_id "Qwen/Qwen3-14B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20
# python sampling_py/sample_MATH500_batched_magistral_vllm.py --model_id "mistralai/Magistral-Small-2506" --temperature 0.7 --top_p 0.95 --min_p 0

python sampling_py/MATH500/sample_MATH500_batched_vllm.py --model_id "ByteDance-Seed/Seed-Coder-8B-Reasoning" --temperature 0.6 --top_p 0.95 --top_k 50
# python sampling_py/sample_MATH500_batched_vllm.py --model_id "Qwen/Qwen2.5-Math-7B-Instruct" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20
# python sampling_py/sample_MATH500_batched_vllm.py --model_id "deepseek-ai/deepseek-math-7b-rl" --temperature 0.7 --top_p 0.95 --min_p 0

# python sampling_py/sample_MATH500_batched_vllm.py --model_id "Qwen/Qwen3-0.6B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20
# python sampling_py/sample_MATH500_batched_vllm.py --model_id "Qwen/Qwen3-1.7B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20
# python sampling_py/sample_MATH500_batched_vllm.py --model_id "Qwen/Qwen3-4B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20
# python sampling_py/sample_MATH500_batched_vllm.py --model_id "Qwen/Qwen3-8B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20
# python sampling_py/sample_MATH500_batched_vllm.py --model_id "Qwen/Qwen3-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 20