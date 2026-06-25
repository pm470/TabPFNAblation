# TabPFN Activation Function Ablation Study

## Project Context

This is a CS Master's research project (DLL course) investigating whether modern activation functions (e.g., SwiGLU, GeGLU, Mish) can improve tabular foundation models like TabPFN compared to the GELU baseline.

## Hypothesis

Modern activation functions might yield statistically significant improvements compared to the GELU baseline on the TabArena benchmark.

## Codebase

- Based on **nanoTabPFN** (<https://github.com/automl/nanoTabPFN>) — a simplified ~500 LOC reimplementation of TabPFNv2.
- `model.py` and `prior.py` are from upstream nanoTabPFN. `prior.py` is excluded from ruff linting.
- `train.py` has been enhanced with full seed control, checkpoint saving, and structured JSONL metric logging.
- `run_experiment.py` is the CLI experiment runner. Uses `--benchmark` flag to switch between fast local and TabArena eval.
- `tabarena_eval.py` wraps the model for the official TabArena Autogluon benchmark pipeline.
- `run_all.sh` runs all seed × activation combos.
- Prior data dump: `300k_150x5_2.h5` (990MB, downloaded from figshare, not committed to git).

## Hardware

- **Local:** Ryzen 5950X + Radeon RX 9700 XT (16GB VRAM), ROCm 7.2, PyTorch 2.11.0+rocm7.2
- **Cluster:** BwUniClust3.0 (for full TabArena evaluation — needs 48GB VRAM recommended)

## Environment

- Python 3.11.15 via mise, dependencies managed with uv
- PyTorch ROCm 7.2 wheel from `https://download.pytorch.org/whl/rocm7.2`
- Dev tools: pytest, ruff (line-length=120)

## Evaluation Strategy

- **Fast local eval:** `sklearn.datasets.load_breast_cancer` — quick sanity check during development (`--benchmark breast_cancer`).
- **Full eval (on cluster):** TabArena benchmark (`--benchmark tabarena`). Currently runs the `lite` subset for testing, can be switched to `full` (51 datasets) in `tabarena_eval.py` when running on the cluster.

## Key Design Decisions

- **Gated activations (SwiGLU, GeGLU, ReGLU)** will use identical parameter counts to non-gated variants (LLaMA-style `E → 2*(2H/3)` trick) for fair comparison.
- **Seed determinism** is critical: `torch.cuda.manual_seed_all`, `cudnn.deterministic=True`, `cudnn.benchmark=False`. Verified by pytest.
- Default training: 2500 steps, batch_size=32, lr=4e-3, eval every 25 steps, checkpoint every 250 steps.
- Model: 356K params (embedding=96, heads=4, mlp_hidden=192, layers=3, outputs=2).

## Research Plan

1. End-to-end loop ASAP
2. Try ~15 activation functions, ~10 seeds each, everything else frozen
3. Aggregate over seeds
4. Take two best functions
5. Investigate how these behave when changing architecture (depth, etc.)
6. Evaluate on TabArena for final results + statistical significance

## Desired Final Plots

1. Pre-training steps vs. TabArena Score (baseline + 2 best variants, show std across seeds)
2. ROC-AUC per activation function (box-and-whisker)
3. Model depth vs. TabArena Score for best variants
