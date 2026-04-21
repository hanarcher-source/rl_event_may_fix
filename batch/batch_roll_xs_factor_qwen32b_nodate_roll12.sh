#!/bin/bash
# Qwen2.5-32B-Instruct: same rolling factor job as batch_roll_xs_factor_qwen32b.sh, but
# `roll_xs_bucket_factor_qwen32b_nodate_roll.py` → no calendar line in prompt (pipeline_nodate).
# Results run_id: qwen32_xs_roll12_nodate_<timestamp>

#SBATCH -J xs_roll_q32_nod12
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -o /finance_ML/zhanghaohan/rl_remodeled/may_fix/logs/slurm-xs-roll-qwen32b-nodate-roll12-%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/scripts
echo "VARIANT=qwen32_xs_roll_nodate job=${SLURM_JOB_ID:-local} host=$(hostname) gpus=${CUDA_VISIBLE_DEVICES:-all}"

python3 -u roll_xs_bucket_factor_qwen32b_nodate_roll.py \
  --start_train_month 2023-01 \
  --n_rolls 12 \
  --max_mem_gib 70 \
  --batch_size 1 \
  --grad_accum 16

echo "done"
