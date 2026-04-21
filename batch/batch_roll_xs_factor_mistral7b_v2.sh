#!/bin/bash
# V2 prompt (prompt_xs_bucket_v2). 1 GPU. Outputs: results/<backend>_v2_xs_roll12_* / logs / mappings

#SBATCH -J xs_roll12_mistral_v2
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 48:00:00
#SBATCH -o /finance_ML/zhanghaohan/rl_remodeled/may_fix/logs/slurm-xs-roll12-mistral-v2-%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/env_py310

cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/scripts
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
echo "PIPELINE_V2 prompt=prompt_xs_bucket_v2 backend=mistral job=${SLURM_JOB_ID:-local} host=$(hostname)"

python3 -u roll_xs_bucket_factor_pipeline_v2.py \
  --backend mistral \
  --start_train_month 2023-01 \
  --n_rolls 12

echo "done"
