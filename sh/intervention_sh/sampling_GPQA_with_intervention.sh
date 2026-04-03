#!/bin/bash

#SBATCH --job-name=MagistralGPQASamplingReasoningModels
#SBATCH --partition=learn  # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:2        # uncomment only if/as needed
#SBATCH --time=0-04:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/sampling/Magistral-gpqa-sampling-%j.log

source activate rifeval

# python sampling_py/GPQA/sample_GPQA_with_intervention_multi_run_vllm.py --model_id "Qwen/Qwen3-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 50 --max_new_tokens 32768 --tensor_parallel_size 2 --use_intervention

python sampling_py/GPQA/sample_GPQA_with_intervention_multi_run_vllm_Qwen3.py --model_id "Qwen/Qwen3-32B" --temperature 0.6 --top_p 0.95 --min_p 0 --top_k 50 --max_new_tokens 32768 --tensor_parallel_size 2 --use_intervention