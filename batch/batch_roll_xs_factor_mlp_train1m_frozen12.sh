#!/bin/bash
#SBATCH -J xs_mlp_t1_frz12
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 12:00:00
#SBATCH -o /finance_ML/zhanghaohan/rl_remodeled/may_fix/logs/slurm-xs-mlp-train1m-frozen12-%j.out
#SBATCH --reservation=finai

# MLP: train first month only, frozen OOS 12 months via combo_cache (unique test_month+combo_key).

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/env_py310

cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/scripts
echo "VARIANT=mlp_train1month_frozen_oos12 oos_infer=combo_cache job=${SLURM_JOB_ID:-local} host=$(hostname)"

python3 -u roll_xs_bucket_factor_mlp_train1m_frozen_oos12.py \
  --start_train_month 2023-01 \
  --n_rolls 12 \
  --oos_infer combo_cache \
  --infer_batch 4096

echo "done"
