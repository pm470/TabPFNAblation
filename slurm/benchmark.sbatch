#!/bin/bash
#SBATCH --job-name=tabpfn_bench
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
#SBATCH --array=0  # Array for seed 0 (testing)

# Ensure logs directory exists
mkdir -p logs

# Load modules
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
RESULTS_DIR="${WORKSPACE_DIR}/results"

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

SEEDS=(0)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ACTIVATION="gelu"

# The checkpoint is loaded from the workspace results directory
CHECKPOINT="${RESULTS_DIR}/${ACTIVATION}/seed_${SEED}/checkpoints/step_30000.pt"

echo "Starting TabArena Benchmark for ${CHECKPOINT}"
echo "Using workspace: ${WORKSPACE_DIR}"

UV_INDEX_STRATEGY="unsafe-best-match" uv run python scripts/benchmark_model.py \
  --checkpoint "$CHECKPOINT" \
  --activation "$ACTIVATION" \
  --subset "lite" \
  --n_ensemble 8

echo "Benchmarking completed."
