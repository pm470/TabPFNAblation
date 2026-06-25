#!/bin/bash
#SBATCH --job-name=tabpfn_tabarena
#SBATCH --output=tabpfn_tabarena_%j.out
#SBATCH --error=tabpfn_tabarena_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1

# Load modules (using nvidia-hpc-sdk for GPU support)
module purge
module load compiler/gnu toolkit/nvidia-hpc-sdk/25.1

echo "Starting TabArena Benchmark on node $HOSTNAME"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Change to the directory from which the job was submitted
cd $SLURM_SUBMIT_DIR

# Install uv if it's not installed in the cluster home directory
if ! command -v uv &> /dev/null
then
    echo "uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
fi

# Run the experiment using uv to manage dependencies and python version
echo "Running baseline benchmark..."
uv run python run_experiment.py --benchmark tabarena --activation gelu --seed 0

echo "Benchmark complete."
