#!/bin/bash
# Train the TinyStories LM on one H200.
# Submit FROM THE REPOSITORY ROOT:
#     sbatch scripts/slurm/train.sbatch
# Extra arguments are forwarded to scripts/train.py, e.g.
#     sbatch scripts/slurm/train.sbatch --lr 1e-3 --run-name ts-lr1e-3
#SBATCH --job-name=a1-train-ts
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=runs/train-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "node=$(hostname)  gpu=${CUDA_VISIBLE_DEVICES:-?}  start=$(date -Is)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python scripts/train.py --device cuda "$@"

echo "end=$(date -Is)"
