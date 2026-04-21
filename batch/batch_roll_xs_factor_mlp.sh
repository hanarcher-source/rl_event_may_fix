#!/bin/bash
# 12-month stitched factor table (trivial MLP baseline): same roll as LLM pipeline.

#SBATCH -J xs_roll_mlp
#SBATCH -p batch
#SBATCH -N 1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o /finance_ML/zhanghaohan/rl_remodeled/may_fix/logs/slurm-xs-roll-mlp-%j.out
#SBATCH --reservation=finai

set -euo pipefail

source /finance_ML/zhanghaohan/conda_env/etc/profile.d/conda.sh
conda activate /finance_ML/zhanghaohan/mycondaenv

cd /finance_ML/zhanghaohan/rl_remodeled/may_fix/scripts
echo "host=$(hostname) job=${SLURM_JOB_ID:-local}"
python3 -u roll_xs_bucket_factor_mlp.py --start_train_month 2023-01 --n_rolls 12
echo "done"
