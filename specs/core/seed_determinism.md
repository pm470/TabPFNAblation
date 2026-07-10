---
id: CORE-003
title: "Seed Determinism"
status: implemented
module: train.py
last_synced: 2026-07-10
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

## Notes

- The determinism tests (`test_seed_determinism.py`) validate AC-7 and AC-8 by running short experiments (2 steps, eval every 1) and comparing the recorded loss values from `metrics.jsonl`.
- `cudnn.benchmark = False` disables cuDNN's auto-tuning, which can introduce non-determinism when selecting algorithms.
- `cudnn.deterministic = True` forces cuDNN to use deterministic algorithms even if they are slower.
- The seed is set before model initialization and data loading to ensure both weight initialization and data ordering are deterministic.
