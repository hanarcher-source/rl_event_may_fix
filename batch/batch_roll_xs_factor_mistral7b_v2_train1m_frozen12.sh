#!/bin/bash
# Mistral-7B-Instruct, v2 prompt (same as roll_xs_bucket_factor_pipeline_v2 --backend mistral):
# train **one** month only (roll 1 calendar: train 2023-01, val 2023-02), then **frozen** weights and
# OOS for 12 test months via **combo_cache** (one LM forward per unique combo_key across all OOS).
# Comparable defaults to slurm-xs-roll12-mistral-v2-*: start_train_month=2023-01, n_rolls=12.

#SBATCH -J xs_mistral_v2_t1_frz12
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o /finance_ML/zhanghaohan/rl_remodeled/may_fix/logs/slurm-xs-mistral-v2-train1m-frozen12-%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/env_py310

cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/scripts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "VARIANT=v2_train1month_frozen_oos12 oos_infer=combo_cache backend=mistral job=${SLURM_JOB_ID:-local} host=$(hostname)"

# Training uses v2 defaults for mistral: BATCH_SIZE=4 GRAD_ACCUM=4 (same as rolling job 54652).
python3 -u roll_xs_bucket_factor_v2_train1m_frozen_oos12.py \
  --backend mistral \
  --start_train_month 2023-01 \
  --n_rolls 12 \
  --oos_infer combo_cache \
  --infer_batch_size 32

echo "done"
