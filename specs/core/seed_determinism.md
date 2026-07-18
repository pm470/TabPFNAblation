---
id: CORE-003
title: "Seed Determinism"
status: implemented
module: train.py
last_synced: 2026-07-18
---

# Seed Determinism

## User Story

As a researcher, I want full control over random seeds so that experiments are exactly reproducible and I can attribute performance differences to activation function changes rather than randomness.

## Acceptance Criteria

### set_randomness_seed()

- [x] AC-1: Sets `random.seed(seed)` for Python's built-in random module.
- [x] AC-2: Sets `np.random.seed(seed)` for NumPy's random module.
- [x] AC-3: Sets `torch.manual_seed(seed)` for PyTorch CPU operations.
- [x] AC-4: Sets `torch.cuda.manual_seed_all(seed)` for all CUDA devices.
- [x] AC-5: Sets `torch.backends.cudnn.deterministic = True`.
- [x] AC-6: Sets `torch.backends.cudnn.benchmark = False`.

### Determinism Guarantees

- [x] AC-7: Two runs with the same seed produce identical loss sequences at every eval step.
- [x] AC-8: Two runs with different seeds produce at least one different loss value across eval steps.
- [x] AC-9: Two model instantiations with the same seed produce identical initial weights (`state_dict` tensors).
- [x] AC-10: Two model instantiations with different seeds produce at least one differing initial weight tensor.
- [x] AC-11: Two `NanopriorDataset` batches generated with the same seed are identical (`x`, `y`, and `train_test_split_index`).
- [x] AC-12: Two `NanopriorDataset` batches generated with different seeds differ.
- [x] AC-13: Two short `train()` runs with the same seed (same model init, same data, same optimizer steps) produce identical weights afterward.
- [x] AC-14: Two short `train()` runs with different seeds produce different weights afterward.
- [x] AC-15: Two forward passes on identical input, with the model constructed under the same seed, produce identical output tensors.
- [x] AC-16: Two forward passes on identical input, with the model constructed under different seeds, produce different output tensors.

### PriorDumpDataLoader Data Diversity

- [x] AC-15: `PriorDumpDataLoader` with different seeds reads different data slices from the HDF5 file, ensuring that the only shared variance source is not limited to weight initialization.

## Notes

- The determinism tests (`test_seed_determinism.py`) validate AC-7 and AC-8 by running short experiments (10 steps, eval every 5) and comparing the recorded loss values from `metrics.jsonl`.
- The weight determinism tests (`test_weight_determinism.py`) validate AC-9 and AC-10 directly against `NanoTabPFNModel.state_dict()`, without running a full training loop.
- The data generation determinism tests (`test_data_generation_determinism.py`) validate AC-11 and AC-12 by seeding and drawing a single batch from `NanopriorDataset`, without running a `DataLoader` or writing to HDF5.
- The training pipeline determinism tests (`test_training_pipeline_determinism.py`) validate AC-13 and AC-14 by seeding, then running a few steps of `train()` on a small in-memory `NanopriorDataset` and comparing the resulting `state_dict()`, isolating weight-trajectory determinism from the loss-only check in `test_seed_determinism.py`.
- The forward pass determinism tests (`test_forward_pass_determinism.py`) validate AC-15 and AC-16. The input tensor is generated from a separate fixed seed (`torch.manual_seed(12345)`) applied *after* model construction, so only the model's weights vary between calls, isolating weight-init determinism from input-data randomness — a check not covered by AC-9/AC-10 (which only inspect `state_dict()`, never run a forward pass) or AC-7/AC-8 (which run a full training loop, not an isolated inference-mode forward pass).
- `cudnn.benchmark = False` disables cuDNN's auto-tuning, which can introduce non-determinism when selecting algorithms.
- `cudnn.deterministic = True` forces cuDNN to use deterministic algorithms even if they are slower.
- The seed is set before model initialization and data loading to ensure both weight initialization and data ordering are deterministic.
