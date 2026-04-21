#!/bin/bash
# Qwen2.5-32B-Instruct, explicit **V2 prompt** alias.
# Uses the shared v2 prompt pipeline (`prompt_xs_bucket_v2`) with backend=qwen32.
# Procedure kept parallel to the old Qwen32 setup: 12 rolls, max_mem_gib=70, batch_size=1, grad_accum=16.

#SBATCH -J xs_roll_q32_v2
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -o /finance_ML/zhanghaohan/rl_remodeled/may_fix/logs/slurm-xs-roll-qwen32b-v2-%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/env_py310

cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/scripts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "PIPELINE_V2 prompt=prompt_xs_bucket_v2 backend=qwen32 job=${SLURM_JOB_ID:-local} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-all}"

python3 -u roll_xs_bucket_factor_qwen32b_v2_roll.py \
  --start_train_month 2023-01 \
  --n_rolls 12 \
  --max_mem_gib 70 \
  --batch_size 1 \
  --grad_accum 16

echo "done"
