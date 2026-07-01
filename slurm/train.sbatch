#!/bin/bash
#SBATCH --job-name=tabpfn_train
#SBATCH --output=logs/train_%A_%a.out
#SBATCH --error=logs/train_%A_%a.err
#SBATCH --time=16:00:00
#SBATCH --partition=gpu_a100_il
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

SEEDS=(0)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
ACTIVATION="gelu"

echo "Starting training for activation=${ACTIVATION}, seed=${SEED}"
echo "Using workspace: ${WORKSPACE_DIR}"

# Safely rename the existing activation folder if it exists.
# We use a lock directory tagged with the Slurm Job ID so that only the 
# very first array task performs the rename, preventing race conditions.
LOCK_DIR="${RESULTS_DIR}/.lock_${ACTIVATION}_${SLURM_ARRAY_JOB_ID}"
if mkdir "$LOCK_DIR" 2>/dev/null; then
    if [ -d "${RESULTS_DIR}/${ACTIVATION}" ]; then
        RUN=1
        while [ -d "${RESULTS_DIR}/${ACTIVATION}_run${RUN}" ]; do
            ((RUN++))
        done
        mv "${RESULTS_DIR}/${ACTIVATION}" "${RESULTS_DIR}/${ACTIVATION}_run${RUN}"
        echo "Renamed existing ${ACTIVATION} folder to ${ACTIVATION}_run${RUN}"
    fi
fi

# Ensure all array tasks wait a moment for the rename to complete across the network file system
sleep 3

UV_INDEX_STRATEGY="unsafe-best-match" uv run python run_experiment.py \
    --activation "$ACTIVATION" \
    --seed "$SEED" \
    --num_steps 100000 \
    --eval_every 5000 \
    --checkpoint_every 10000 \
    --data_file "${DATA_DIR}/300k_500x50_10.h5" \
    --output_dir "${RESULTS_DIR}"

echo "Training completed."
