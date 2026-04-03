#!/bin/bash

#SBATCH --job-name=MagistralGPQASamplingReasoningModels
#SBATCH --partition=learn  # change as needed, e.g., lowpri on some clusters
#SBATCH --gres=gpu:2        # uncomment only if/as needed
#SBATCH --time=0-04:00:00    # run for one day
#SBATCH --cpus-per-task=2    # change as needed
## %j is the job id, %u is the user id
#SBATCH --output=/n/home04/yidachen/bash_output/sampling/Magistral-gpqa-sampling-%j.log

source activate rifeval

job_id=${SLURM_JOB_ID}
intervene_model_name="${1:-Qwen/Qwen3-0.6B}"
seed=42
summarize_model_name="Qwen/Qwen3-14B"

output_file_name="/data/n/home04/yidachen/reasoning_characteristics/intervention_logs/gpqa/gpqa-summarize-then-expand-intervention-${intervene_model_name}-${summarize_model_name}-${job_id}.log"

CUDA_VISIBLE_DEVICES=0,1 python notebooks/intervention_experiments/prompt_level_intervention/gpqa_summarize_then_expand_vllm_two_stage.py --model_name $intervene_model_name --enable_thinking --job_id $job_id --intervene_only_on_wrong --seed $seed --summarize_model_name $summarize_model_name | tee -a "$output_file_name"