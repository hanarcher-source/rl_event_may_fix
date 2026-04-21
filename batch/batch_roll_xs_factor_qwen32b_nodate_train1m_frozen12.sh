#!/bin/bash
# Qwen2.5-32B-Instruct, **V1 no-date** prompt (`roll_xs_bucket_factor_pipeline_nodate`): train one month,
# frozen OOS for 12 test months via combo_cache. Request multiple GPUs + --multi_gpu_auto for HF sharding.

#SBATCH -J xs_q32_nod_t1_frz12
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH -t 48:00:00
#SBATCH -o /finance_ML/zhanghaohan/rl_remodeled/may_fix/logs/slurm-xs-qwen32b-nodate-train1m-frozen12-%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/env_py310

cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/scripts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "VARIANT=nodate_train1month_frozen_oos12 backend=qwen32 oos_infer=combo_cache job=${SLURM_JOB_ID:-local} host=$(hostname)"

python3 -u roll_xs_bucket_factor_qwen32b_nodate_train1m_frozen_oos12.py \
  --multi_gpu_auto \
  --max_mem_gib 70 \
  --start_train_month 2023-01 \
  --n_rolls 12 \
  --oos_infer combo_cache \
  --infer_batch_size 32

echo "done"
