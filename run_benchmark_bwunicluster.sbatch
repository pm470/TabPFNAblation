#!/bin/bash
#SBATCH --job-name=tabpfn_tabarena
#SBATCH --output=logs/tabpfn_tabarena_%j.out
#SBATCH --error=logs/tabpfn_tabarena_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1

# Load modules (using nvidia-hpc-sdk for GPU support)
module purge
module load compiler/gnu toolkit/nvidia-hpc-sdk/25.1

# Record the start time
START_TIME=$(date +%s)
echo "========================================"
echo "Starting TabArena Benchmark on node $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU IDs allocated: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo "========================================"

# Change to the directory from which the job was submitted
cd "$SLURM_SUBMIT_DIR" || exit

# Ensure ~/.local/bin is in PATH for uv
export PATH="$HOME/.local/bin:$PATH"

# Install uv if it's not installed in the cluster home directory
if ! command -v uv &>/dev/null; then
  echo "uv not found, installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi

# Remove local AMD ROCm config & lockfile before running so uv downloads NVIDIA/CUDA torch
rm -f uv.toml uv.lock

# Run the experiment using uv to manage dependencies and python version
echo "Running baseline benchmark..."
UV_INDEX_STRATEGY="unsafe-best-match" uv run python run_experiment.py --benchmark tabarena --activation gelu --seed 0

echo "Benchmark complete."

# Record the end time
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "End time: $(date)"
echo "Total execution time: $DURATION seconds."
