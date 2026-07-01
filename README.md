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

## Running on the Cluster (bwUniCluster 3.0)

To run and monitor experiments on the bwUniCluster, follow these steps:

### 1. Connect & Setup

Make sure you are connected to the university VPN or eduroam network.
First, set up your cluster connection details in a `.env` file (copy from `.env.example`).

### 2. Transfer Code and Data

Use the provided sync script to transfer your code to the cluster:

```bash
bash scripts/sync.sh
```

*Note:* The sync script excludes the large prior data dump (`300k_150x5_2.h5`). You will need to manually copy it once to your cluster directory if you haven't already:

```bash
rsync -avz --progress ~/code/TabPFNAblation/300k_150x5_2.h5 <username>@uc3.scc.kit.edu:~/TabPFNAblation/
```

### 3. Submit Jobs

Log into the cluster and submit your SLURM jobs from the `slurm` directory:

```bash
ssh <username>@uc3.scc.kit.edu
cd ~/TabPFNAblation

# Example: Submit a benchmark job
sbatch slurm/benchmark.sbatch
```

### 4. Monitor & Manage Jobs

Use standard SLURM commands to monitor your jobs:

```bash
# Check for available idle nodes
sinfo_t_idle

# Check the status of your jobs
squeue -u $USER

# Check expected start times for pending jobs
squeue --start -u $USER

# View the output and error logs
# (Assuming your job writes to logs/ directory, e.g., logs/tabpfn_tabarena_<JOBID>.out)
tail -f logs/tabpfn_tabarena_<JOBID>.out
```

### 5. Retrieve Results

Once the job is completed, you can sync the `results` directory back to your local machine:

```bash
rsync -avz <username>@uc3.scc.kit.edu:~/TabPFNAblation/results/ ~/code/TabPFNAblation/results/
```
