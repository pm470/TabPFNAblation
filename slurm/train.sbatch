#!/bin/bash
#SBATCH --job-name=tabpfn_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0  # Array for seed 0 (testing)

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

# Use the SLURM array task ID directly as the seed (0-9)
SEED=${SLURM_ARRAY_TASK_ID:-0}
ACTIVATION="${ACTIVATION:-gelu}"
GATED_UNRESTRICTED="${GATED_UNRESTRICTED:-0}"

EXTRA_FLAGS=""
if [ "$GATED_UNRESTRICTED" == "1" ] || [ "$GATED_UNRESTRICTED" == "true" ]; then
  DIR_NAME="${ACTIVATION}_unrestricted"
  EXTRA_FLAGS="--gated-unrestricted"
else
  DIR_NAME="${ACTIVATION}"
fi

echo "Starting training for activation=${ACTIVATION} (dir=${DIR_NAME}), seed=${SEED}"
echo "Using workspace: ${WORKSPACE_DIR}"

FINAL_CHECKPOINT="${RESULTS_DIR}/${DIR_NAME}/seed_${SEED}/checkpoints/final.pt"
if [ -f "$FINAL_CHECKPOINT" ]; then
  echo "Seed ${SEED} for ${DIR_NAME} is already fully trained. Skipping."
  exit 0
fi

UV_INDEX_STRATEGY="unsafe-best-match" uv run python -u run_experiment.py \
  --activation "$ACTIVATION" \
  $EXTRA_FLAGS \
  --seed "$SEED" \
  --num_steps 5000 \
  --batch_size 8 \
  --lr 1e-3 \
  --eval_every 250 \
  --checkpoint_every 250 \
  --max_seq_len 3000 \
  --max_features 45 \
  --max_classes 10 \
  --data_file "${DATA_DIR}/200k_3000x45_10.h5" \
  --output_dir "${RESULTS_DIR}"

echo "Training completed."
