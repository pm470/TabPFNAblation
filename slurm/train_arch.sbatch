#!/bin/bash
#SBATCH --job-name=tabpfn_arch
#SBATCH --output=logs/arch_%A_%a.out
#SBATCH --error=logs/arch_%A_%a.err
#SBATCH --time=00:30:00
#SBATCH --partition=gpu_a100_short
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --array=0  # Override via submit_arch_sweep.sh

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
RESULTS_DIR="${WORKSPACE_DIR}/results_arch"

mkdir -p "$DATA_DIR"
mkdir -p "$RESULTS_DIR"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Required env vars (set by submit_arch_sweep.sh):
#   ACTIVATION      - activation function name (gelu, swiglu, bilinear)
#   SWEEP_AXIS      - sweep axis label (layers, hidden, embedding)
#   EMBEDDING_SIZE  - embedding dimension
#   MLP_HIDDEN_SIZE - MLP hidden dimension
#   NUM_LAYERS      - number of transformer layers

# Use the SLURM array task ID directly as the seed (0-2)
SEED=${SLURM_ARRAY_TASK_ID:-0}

# Validate required env vars
: "${ACTIVATION:?ACTIVATION env var is required}"
: "${SWEEP_AXIS:?SWEEP_AXIS env var is required}"
: "${EMBEDDING_SIZE:?EMBEDDING_SIZE env var is required}"
: "${MLP_HIDDEN_SIZE:?MLP_HIDDEN_SIZE env var is required}"
: "${NUM_LAYERS:?NUM_LAYERS env var is required}"

echo "Starting arch sweep: axis=${SWEEP_AXIS}, activation=${ACTIVATION}, E=${EMBEDDING_SIZE}, H=${MLP_HIDDEN_SIZE}, L=${NUM_LAYERS}, seed=${SEED}"
echo "Using workspace: ${WORKSPACE_DIR}"

ARCH_TAG="e${EMBEDDING_SIZE}_h${MLP_HIDDEN_SIZE}_l${NUM_LAYERS}"
FINAL_CHECKPOINT="${RESULTS_DIR}/${SWEEP_AXIS}/${ACTIVATION}/${ARCH_TAG}/seed_${SEED}/checkpoints/final.pt"
if [ -f "$FINAL_CHECKPOINT" ]; then
  echo "Already fully trained. Skipping."
  exit 0
fi

UV_INDEX_STRATEGY="unsafe-best-match" uv run python -u run_experiment.py \
  --activation "$ACTIVATION" \
  --seed "$SEED" \
  --num_steps 10000 \
  --batch_size 4 \
  --accumulation_steps 2 \
  --gradient-checkpointing \
  --lr 1e-3 \
  --eval_every 250 \
  --checkpoint_every 250 \
  --max_seq_len 3000 \
  --max_features 45 \
  --max_classes 10 \
  --embedding_size "$EMBEDDING_SIZE" \
  --mlp_hidden_size "$MLP_HIDDEN_SIZE" \
  --num_layers "$NUM_LAYERS" \
  --arch-sweep "$SWEEP_AXIS" \
  --data_file "${DATA_DIR}/200k_3000x45_10.h5" \
  --output_dir "${RESULTS_DIR}"

echo "Training completed."
