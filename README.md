# TabPFN Activation Function Ablation Study

Investigating whether modern activation functions (SwiGLU, GeGLU, Mish, etc.) can improve [nanoTabPFN](https://github.com/automl/nanoTabPFN) compared to the GELU baseline.

## Setup

Requires [mise](https://mise.jdx.dev/) and a ROCm-compatible AMD GPU.

```bash
mise install                # Python 3.11 + uv
uv sync --extra dev         # all dependencies
```

Download the prior data dump (~990MB) from [figshare](https://figshare.com/s/63fc1ada93e42e388e63) into the project root.

## Usage

```bash
# Single experiment (fast local breast_cancer eval)
python run_experiment.py --activation gelu --seed 0

# Single experiment with full TabArena evaluation at the end
python run_experiment.py --activation gelu --seed 0 --benchmark tabarena

# All seed × activation combos (default breast_cancer)
bash run_all.sh

# Plot results
python plot_results.py

# Tests & linting
uv run pytest -v
uv run ruff check .
```

Results are written to `results/<activation>/seed_<N>/` with `config.json`, `metrics.jsonl`, and `checkpoints/`.

## Cluster specific commands (bwUniCluster)

To run and monitor experiments on the bwUniCluster, use the following SLURM commands based on common workflow:

```bash
# Check for available idle nodes
sinfo_t_idle

# Submit the benchmark job
sbatch run_benchmark_bwunicluster.sbatch

# Check the status of your jobs
sacct -u $USER

# Check expected start times for pending jobs
squeue --start -u $USER

# View the output and error logs (in the logs directory)
cd logs
cat tabpfn_tabarena_<JOBID>.out
tail -f tabpfn_tabarena_<JOBID>.err
```

# Running TabArena Benchmark on bwUniCluster 3.0

Follow these steps to transfer your code, data, and run the TabArena baseline benchmark on the bwUniCluster.

## 1. Connect to bwUniCluster

Make sure you are connected to the university VPN or eduroam network.

## 2. Transfer Code and Data

From your local machine, copy the project directory and the data file to the cluster:

```bash
# Copy the code repository (excluding virtual environments and caches)
rsync -avz --exclude '.venv' --exclude '__pycache__' --exclude '.git' ~/code/TabPFNAblation <username>@uc3.scc.kit.edu:~/TabPFNAblation

# Copy the large data file (if not already copied or not inside the repo)
# Adjust the path if 300k_150x5_2.h5 is located elsewhere.
rsync -avz ~/code/TabPFNAblation/300k_150x5_2.h5 <username>@uc3.scc.kit.edu:~/TabPFNAblation/
```

## 3. Submit the Benchmark Job

Log into the cluster:

```bash
ssh -Y <username>@uc3.scc.kit.edu
```

Navigate to your code directory and submit the batch job:

```bash
cd ~/TabPFNAblation
sbatch run_benchmark_bwunicluster.sbatch
```

### Checking Job Status

You can monitor your job's progress by running:

```bash
squeue -u <username>
```

The output of the job will be written to a file named `tabpfn_tabarena_<JOBID>.out` and errors to `tabpfn_tabarena_<JOBID>.err`. You can view the output as it runs:

```bash
tail -f tabpfn_tabarena_<JOBID>.out
```

## 4. Retrieve Results

Once the job is completed, you can copy the `results` directory back to your local machine:

```bash
scp -r <username>@uc3.scc.kit.edu:~/TabPFNAblation/results ~/code/TabPFNAblation/cluster_results
```

## Notes

- The SLURM script `run_benchmark_bwunicluster.sbatch` is currently configured to use the `gpu_a100_il` partition which might have access restrictions or be named differently (like `gpu_4` or `gpu_8` depending on full node allocations). If you encounter an "Invalid partition" error, check `sinfo` for available GPU partitions and update the `#SBATCH --partition=` line.
- The `tabarena_eval.py` script has been updated to use the full suite of TabArena datasets.
