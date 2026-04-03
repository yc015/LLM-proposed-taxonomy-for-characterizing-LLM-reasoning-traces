#!/bin/bash

#SBATCH --job-name=TrainQwen2.5GRPO-GPQA
#SBATCH --gres=gpu:8        # uncomment only if/as needed
#SBATCH --time=1-00:00:00    # run for one day
#SBATCH --cpus-per-task=8    # change as needed
 
 
## %j is the job id, %u is the user id
#SBATCH --output=/data/home/%u/reasoning_characteristics/qwen25-grpo-gpqa-training-%j.log

# Data paths - GPQA dataset
gpqa_train_path=$HOME/data/gpqa-intervention-math/gpqa_diamond_train_qwen3.parquet
gpqa_test_path=$HOME/data/gpqa-intervention-math/gpqa_diamond_test_qwen3.parquet

train_files="['$gpqa_train_path']"
test_files="['$gpqa_test_path']"

job_id=${SLURM_JOB_ID}

export WANDB_API_KEY=f7872f2a34788eaca92242c34f8d0ac1bd25cb87

# GRPO training script for Qwen2.5-7B with GPQA - Memory Optimized
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=16 \
    data.max_prompt_length=2048 \
    data.max_response_length=24576 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.shuffle=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.model.lora_rank=64 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_num_batched_tokens=131072 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='system_interv_grpo_gpqa_qwen3_8b' \
    trainer.experiment_name="system-prompt-intervention-qwen3_8b_grpo_gpqa_training-math-${job_id}" \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    trainer.total_epochs=15 \
    trainer.default_local_dir="/n/holylabs/LABS/wattenberg_lab/Users/yidachen/grpo-checkpoints/system-prompt-intervention-qwen3_grpo_gpqa-${job_id}" \
    trainer.default_hdfs_dir=null \
    trainer.max_actor_ckpt_to_keep=3 \
    trainer.max_critic_ckpt_to_keep=3 $@ 