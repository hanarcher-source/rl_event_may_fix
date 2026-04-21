#!/bin/bash
# Qwen2.5-32B-Instruct, explicit **V1 prompt** alias.
# V1 == the existing no-date prompt path (`roll_xs_bucket_factor_qwen32b_nodate_roll.py`).
# Same procedure as the previously used run: 12 rolls, max_mem_gib=70, batch_size=1, grad_accum=16.

#SBATCH -J xs_roll_q32_v1
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -o /finance_ML/zhanghaohan/rl_remodeled/may_fix/logs/slurm-xs-roll-qwen32b-v1-%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/scripts
echo "VARIANT=qwen32_v1_prompt job=${SLURM_JOB_ID:-local} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-all}"

python3 -u roll_xs_bucket_factor_qwen32b_v1_roll.py \
  --start_train_month 2023-01 \
  --n_rolls 12 \
  --max_mem_gib 70 \
  --batch_size 1 \
  --grad_accum 16

echo "done"
