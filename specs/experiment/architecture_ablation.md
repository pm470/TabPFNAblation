---
id: EXP-002
title: "Architectural Ablation Study (Depth, Width, Embedding)"
status: implemented
module: "train_arch.sbatch, submit_arch_sweep.sh, plot_results.py, experiment_utils.py"
last_synced: 2026-07-27
---

# Architectural Ablation Study (Depth, Width, Embedding)

## User Story

As a researcher, I want to evaluate how the top-performing activations (Bilinear and SwiGLU) scale compared to the GELU baseline as model architecture parameters change. I conduct one-at-a-time sweeps over layers, hidden width, and embedding size, holding the other parameters at a new baseline with a 2× hidden/embedding ratio.

## Acceptance Criteria

### Experiment Design

- [x] AC-1: Three activations are evaluated: GELU (baseline), SwiGLU, Bilinear.
- [x] AC-2: Three sweep axes: layers, hidden width, embedding size.
- [x] AC-3: 3 seeds per configuration (seeds 0, 1, 2).
- [x] AC-4: New baseline architecture: embedding=128, hidden=256, layers=3, heads=4 (hidden/embedding ratio = 2×).
- [x] AC-5: Attention heads fixed at 4 throughout all configurations.

### Sweep Grid

- [x] AC-6: Layer sweep values: [1, 2, 3, 4, 6] with E=128, H=256 fixed.
- [x] AC-7: Hidden width sweep values: [64, 128, 256, 384, 512] with E=128, L=3 fixed.
- [x] AC-8: Embedding size sweep values: [64, 128, 192, 256] with H=256, L=3 fixed.
- [x] AC-9: Total configurations: 14 architecture points × 3 activations × 3 seeds = 126 jobs.

### Training Configuration

- [x] AC-10: Batch size = 4 (micro-batch), gradient accumulation = 2 steps (effective batch = 8, matching Phase 1).
- [x] AC-11: Gradient checkpointing always enabled.
- [x] AC-12: num_steps = 10000 (forward passes), yielding 5000 optimizer steps with accumulation=2.
- [x] AC-13: Learning rate = 1e-3 (same as Phase 1).
- [x] AC-14: Prior data: same H5 file as Phase 1.
- [x] AC-15: Evaluation: quick eval only (3 proxy datasets). No TabArena in the sweep.

### Results Directory Structure

- [x] AC-16: Results stored in `results_arch/{sweep_axis}/{activation}/e{E}_h{H}_l{L}/seed_{N}/`.
- [x] AC-17: Each run directory contains `config.json`, `metrics.jsonl`, and `checkpoints/`.
- [x] AC-18: The `experiment_utils.py` `setup_experiment` function supports `--arch-sweep` argument to generate the new directory layout.

### SLURM Infrastructure

- [x] AC-19: `slurm/train_arch.sbatch` accepts env vars: `ACTIVATION`, `EMBEDDING_SIZE`, `MLP_HIDDEN_SIZE`, `NUM_LAYERS`, `SWEEP_AXIS`.
- [x] AC-20: `slurm/submit_arch_sweep.sh` loops over the full grid, checks for `final.pt` to skip completed runs, and submits only incomplete jobs.
- [x] AC-21: Short GPU queue (30 min walltime). Manual resubmission for jobs that don't complete.
- [x] AC-22: `submit_arch_sweep.sh` supports `--dry-run` flag and accepts sweep axis argument (`layers`, `hidden`, `embedding`, `all`).

### Plotting

- [x] AC-23: `plot_results.py` gains a `load_arch_results()` function that reads from `results_arch/` directory structure.
- [x] AC-24: Three new line plots: one per sweep axis (X=parameter value, Y=ROC-AUC mean, lines=activations, shaded error bands from seed std).
- [x] AC-25: Plots saved as `arch_layers_scaling.{png,svg}`, `arch_hidden_scaling.{png,svg}`, `arch_embedding_scaling.{png,svg}`.
- [x] AC-26: CLI supports `--arch` flag to generate architecture ablation plots, reading from `results_arch/`.
- [x] AC-27: Embedding scaling plot uses linear x-axis. Hidden scaling plot uses log2 x-axis.

## Notes

- The new baseline (H=256) differs from Phase 1 (H=192). The Phase 1 results remain in `results/` and are not affected.
- Gradient checkpointing and accumulation ensure all configs fit in 40GB A100 VRAM.
- The layer=3, hidden=256, embedding=128 configuration appears in all three sweeps (as the baseline point).
