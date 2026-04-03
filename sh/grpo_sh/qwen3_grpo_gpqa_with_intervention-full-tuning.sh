#!/bin/bash

#SBATCH --job-name=TrainQwen3GRPO-GPQA-intervened
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=3-00:00:00    # run for one day
#SBATCH --cpus-per-task=32    # change as needed
 
 
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/%u/reasoning_characteristics/grpo-training-logs/intervened-qwen3-grpo-gpqa-training-full-tuning-%j.log

source activate verl

style="qwen3"
model_path="Qwen/Qwen3-8B" # meta-llama/Llama-3.1-8B-Instruct, Qwen/Qwen3-8B, microsoft/phi-4
model_id=$(basename "$model_path")
max_response_length=32768 # 32768, 29696 for base models, 8192 for llama3.1, 13312 for phi-4
rollout_n=8
job_id=${SLURM_JOB_ID}
with_keyword="_with_keyword"

# Data paths - GPQA dataset
python /n/home04/yidachen/reasoning_characteristics/sampling_py/GPQA/gpqa_dataset_r1_format_train_test_split.py --model_id $model_path --qwen3_style --use_intervention --randomize_order

# gpqa_train_path=$HOME/data/gpqa-intervention-50-50-split/gpqa_diamond_train_qwen3.parquet
# gpqa_test_path=$HOME/data/gpqa-intervention-50-50-split/gpqa_diamond_test_qwen3.parquet
gpqa_train_path=$HOME/data/gpqa-intervention-${model_id}/gpqa_train_${style}.parquet
gpqa_test_path=$HOME/data/gpqa-intervention-${model_id}/gpqa_test_${style}.parquet
base_folder="/n/home04/yidachen/reasoning_characteristics/grpo-training-logs/${model_id}-training-logs/${model_id}-intervention${with_keyword}"
output_file="${base_folder}/intervened-${model_id}-grpo-gpqa-training_${job_id}${with_keyword}-full-log.txt"

train_files="['$gpqa_train_path']"
test_files="['$gpqa_test_path']"

train_batch_size=16

export WANDB_API_KEY=f7872f2a34788eaca92242c34f8d0ac1bd25cb87

# GRPO training script for Qwen2.5-7B with GPQA - Memory Optimized
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size="$train_batch_size" \
    data.max_prompt_length=2048 \
    data.max_response_length="$max_response_length" \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="$model_path" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
    actor_rollout_ref.rollout.n="$rollout_n" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=recipe/r1/reward_score_qwen3.py \
    custom_reward_function.name=reward_func${with_keyword} \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="reasoning_step_reward_grpo_gpqa_${model_id}" \
    trainer.experiment_name="system-prompt-intervention-${model_id}_grpo_gpqa_training-${job_id}-bs-${train_batch_size}-n-${rollout_n}${with_keyword}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="/n/holylabs/LABS/wattenberg_lab/Users/yidachen/grpo-checkpoints/system-prompt-intervention-${model_id}_grpo_gpqa-${job_id}-bs-${train_batch_size}" \
    trainer.default_hdfs_dir=null \
    trainer.max_actor_ckpt_to_keep=2 \
    trainer.max_critic_ckpt_to_keep=2 | tee -a "$output_file"  $@ 