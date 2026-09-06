#!/bin/bash
# Encode a raw corpus into a flat uint16 token array.
# Submit FROM THE REPOSITORY ROOT; every argument is forwarded to
# cs336_basics/encode_corpus.py, so this file is corpus-agnostic:
#
#   sbatch slurm/encode_corpus.sbatch \
#       --input data/raw/TinyStoriesV2-GPT4-train.txt \
#       --vocab data/tokenizer/tinystories_vocab.pkl \
#       --merges data/tokenizer/tinystories_merges.pkl \
#       --output data/tokens/ts_train.npy
#
# OpenWebText is ~5x larger; raise the wall clock at submit time, which
# overrides the #SBATCH line below:
#
#   sbatch --time=08:00:00 --job-name=a1-encode-owt \
#       slurm/encode_corpus.sbatch --input data/raw/owt_train.txt ...
#SBATCH --job-name=a1-encode
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=runs/encode-%j.log

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

echo "node=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}  start=$(date -Is)"

uv run python -m cs336_basics.encode_corpus "$@"

echo "end=$(date -Is)"
