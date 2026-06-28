#!/bin/bash
#SBATCH --job-name=tabpfn_bench
#SBATCH --output=logs/bench_%A_%a.out
#SBATCH --error=logs/bench_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --cpus-per-task=8
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
RESULTS_DIR="${WORKSPACE_DIR}/results"

# Activate environment
source .venv/bin/activate || true

SEEDS=(0 1 2)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ACTIVATION="gelu"

# The checkpoint is loaded from the workspace results directory
CHECKPOINT="${RESULTS_DIR}/${ACTIVATION}/seed_${SEED}/checkpoints/final.pt"

echo "Starting TabArena Benchmark for ${CHECKPOINT}"
echo "Using workspace: ${WORKSPACE_DIR}"

# Benchmark output will naturally go into the checkpoint's parent run directory in the workspace
python benchmark_model.py \
    --checkpoint "$CHECKPOINT" \
    --activation "$ACTIVATION" \
    --subset "classification" \
    --n_ensemble 8

echo "Benchmarking completed."
