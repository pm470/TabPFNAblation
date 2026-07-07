#!/bin/bash
#SBATCH --job-name=tabpfn_gen_data
#SBATCH --output=logs/gen_data_%j.out
#SBATCH --error=logs/gen_data_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Ensure logs directory exists
mkdir -p logs

# Load modules
module purge
module load compiler/gnu 2>/dev/null || true

# Load environment variables if .env exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Fallback to the cluster workspace path if not defined in .env
WORKSPACE_DIR="${WORKSPACE_DIR:-/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data}"
DATA_DIR="${WORKSPACE_DIR}/data"
mkdir -p "$DATA_DIR"

export PATH="$HOME/.local/bin:$PATH"

# Install uv if it's not installed in the cluster home directory
if ! command -v uv &>/dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

echo "Generating synthetic prior data to ${DATA_DIR}/200k_3000x45_10.h5"
if [ -f "${DATA_DIR}/200k_3000x45_10.h5" ]; then
    echo "Data file already exists. Skipping generation."
    exit 0
fi

UV_INDEX_STRATEGY="unsafe-best-match" uv run python scripts/generate_prior.py \
    --output_file "${DATA_DIR}/200k_3000x45_10.h5" \
    --num_datasets 200000 \
    --max_seq_len 3000 \
    --max_features 45 \
    --max_classes 10

echo "Generation completed."
