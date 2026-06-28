#!/bin/bash
#SBATCH --job-name=tabpfn_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0-2  # Array for seeds 0, 1, 2

# Ensure logs directory exists
mkdir -p logs

# Load modules (using nvidia-hpc-sdk for GPU support)
module purge
module load compiler/gnu toolkit/nvidia-hpc-sdk/25.1 2>/dev/null || true

# Load environment variables
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Fallback to the cluster workspace path if not defined in .env
WORKSPACE_DIR="${WORKSPACE_DIR:-/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data}"
DATA_DIR="${WORKSPACE_DIR}/data"
RESULTS_DIR="${WORKSPACE_DIR}/results"

mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ACTIVATION="gelu"

echo "Starting training for activation=${ACTIVATION}, seed=${SEED}"
echo "Using workspace: ${WORKSPACE_DIR}"

UV_INDEX_STRATEGY="unsafe-best-match" uv run python train_ablation.py \
    --activation "$ACTIVATION" \
    --seed "$SEED" \
    --num_steps 2500 \
    --eval_every 25 \
    --checkpoint_every 250 \
    --data_file "${DATA_DIR}/300k_150x5_2.h5" \
    --output_dir "${RESULTS_DIR}"

echo "Training completed."
