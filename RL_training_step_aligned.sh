#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export CUDA_VISIBLE_DEVICES=0,1,2,3
source ~/miniforge3/etc/profile.d/conda.sh
conda activate rrec

NVIDIA_NCCL_LIB="$(python3 -c 'import nvidia.nccl; import os; print(os.path.join(os.path.dirname(nvidia.nccl.__file__), "lib"))' 2>/dev/null || true)"
if [ -n "${NVIDIA_NCCL_LIB}" ] && [ -d "${NVIDIA_NCCL_LIB}" ]; then
    export LD_LIBRARY_PATH="${NVIDIA_NCCL_LIB}:${LD_LIBRARY_PATH:-}"
fi

n_gpus_per_node=4
nnodes=1
experiment_name="Office_Products_stage3_step_aligned_Qwen3-1.7B"
stage2_checkpoint="./output_dir/Office_Products_stage2_step_aligned_reasoning_activation_Qwen3-1.7B/final_checkpoint"
item_info_path="${SCRIPT_DIR}/data/Amazon/info/Office_Products_5_2016-10-2018-11.txt"

mkdir -p ./logs

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo_step_aligned \
    data.train_files=./data/Amazon/rec_reasoning_verl/Office_Products/train.parquet \
    data.val_files=./data/Amazon/rec_reasoning_verl/Office_Products/test.parquet \
    data.train_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path="${stage2_checkpoint}" \
    actor_rollout_ref.actor.optim.lr=5e-7 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=16 \
    +actor_rollout_ref.rollout.step_aligned_reasoning=True \
    +actor_rollout_ref.rollout.sid_item_info_path="${item_info_path}" \
    +actor_rollout_ref.rollout.step_aligned_think_chunk_tokens=256 \
    +actor_rollout_ref.rollout.step_aligned_min_tokens_per_block=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=step_aligned \
    +reward_model.reward_kwargs.item_info_path="${item_info_path}" \
    +reward_model.reward_kwargs.match_reward=1.0 \
    +reward_model.reward_kwargs.format_reward=0.0 \
    +reward_model.reward_kwargs.require_exact_think_blocks=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='RecRL_Reasoning' \
    trainer.experiment_name="${experiment_name}" \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 "$@"
