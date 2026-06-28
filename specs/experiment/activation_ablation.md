---
id: EXP-001
title: "Modular Training & Benchmarking Pipeline"
status: implemented
module: "train_ablation.py, benchmark_model.py, sbatch_train.sh, sbatch_benchmark.sh"
last_synced: 2026-06-28
---

# Modular Training & Benchmarking Pipeline

## User Story

As a researcher, I want a modular, cluster-ready pipeline that separates data loading, model pretraining, and TabArena benchmarking into distinct scripts. This allows me to pretrain foundation models using SLURM arrays and benchmark them in parallel, overcoming the 33-hour sequential evaluation bottleneck.

## Acceptance Criteria

### Pretraining CLI Interface (`train_ablation.py`)

- [x] AC-1: `--activation` (str, default `"gelu"`) — activation function name.
- [x] AC-2: `--data_file` (str, default `"300k_150x5_2.h5"`) — path to the HDF5 prior dump.
- [x] AC-3: `--seed` (int, default `0`) — random seed.
- [x] AC-4: `--num_steps` (int, default `2500`) — number of training steps.
- [x] AC-5: `--batch_size` (int, default `32`) — batch size.
- [x] AC-6: `--lr` (float, default `4e-3`) — learning rate.
- [x] AC-7: `--eval_every` (int, default `25`) — metrics logging interval (inline evaluation is disabled for speed).
- [x] AC-8: `--checkpoint_every` (int, default `250`) — step-based checkpoint interval.
- [x] AC-9: `--checkpoint_every_minutes` (float, default `10.0`) — time-based checkpoint interval.
- [x] AC-10: `--output_dir` (str, default `"results"`) — base output directory.
- [x] AC-11: Model architecture args: `--embedding_size` (96), `--num_attention_heads` (4), `--mlp_hidden_size` (192), `--num_layers` (3), `--num_outputs` (10).

### Output Directory Structure

- [x] AC-12: Run directory is `{output_dir}/{activation}/seed_{seed}/`.
- [x] AC-13: `config.json` and `metrics.jsonl` are saved here.
- [x] AC-14: Checkpoint directory is `{run_dir}/checkpoints/` containing `step_XXXXX.pt` and `final.pt`.
- [x] AC-15: TabArena evaluations are output to `{run_dir}/tabarena_exp/`.

### Benchmarking CLI Interface (`benchmark_model.py`)

- [x] AC-16: `--checkpoint` (str, required) — path to the trained `model.pt` file.
- [x] AC-17: `--subset` (str, default `"classification"`) — TabArena subset to evaluate on.
- [x] AC-18: `--n_ensemble` (int, default `8`) — number of NanoTabPFN ensemble iterations.
- [x] AC-19: Architecture args to instantiate the model before loading weights (`--activation`, `--embedding_size`, etc.).

### Cluster Integration (SLURM)

- [x] AC-20: Scripts `.env` aware, extracting `WORKSPACE_DIR` if present to cleanly separate code from heavy data.
- [x] AC-21: `sbatch_train.sh` uses `#SBATCH --array=0-2` to run seeds in parallel.
- [x] AC-22: `sbatch_benchmark.sh` reads checkpoints from `WORKSPACE_DIR` and evaluates them in parallel arrays.
