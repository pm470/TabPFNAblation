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

# Ensure logs directory exists in the code repo
mkdir -p logs

# Load environment variables (e.g., WORKSPACE_DIR) if .env exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
fi

# Fallback to the cluster workspace path if not defined in .env
WORKSPACE_DIR="${WORKSPACE_DIR:-/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data}"
DATA_DIR="${WORKSPACE_DIR}/data"
RESULTS_DIR="${WORKSPACE_DIR}/results"

# Ensure workspace directories exist
mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

# Activate environment (adjust path if needed for BwUniClust3.0)
source .venv/bin/activate || true

# Map array index to seed
SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ACTIVATION="gelu"

echo "Starting training for activation=${ACTIVATION}, seed=${SEED}"
echo "Using workspace: ${WORKSPACE_DIR}"

python train_ablation.py \
    --activation "$ACTIVATION" \
    --seed "$SEED" \
    --num_steps 2500 \
    --eval_every 25 \
    --checkpoint_every 250 \
    --data_file "${DATA_DIR}/300k_150x5_2.h5" \
    --output_dir "${RESULTS_DIR}"

echo "Training completed."
