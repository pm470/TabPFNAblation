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
