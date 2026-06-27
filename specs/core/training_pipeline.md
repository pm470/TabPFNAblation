---
id: CORE-002
title: "Training Pipeline"
status: implemented
module: train.py
last_synced: 2026-06-26
---

# Training Pipeline

## User Story

As a researcher, I want a training loop that pre-trains NanoTabPFN on synthetic prior data from an HDF5 dump with periodic evaluation, structured metric logging, and checkpoint saving, so that I can track training progress and resume from saved states.

## Acceptance Criteria

### train() Function

- [x] AC-1: Uses `nn.CrossEntropyLoss()` as the loss criterion.
- [x] AC-2: Uses `schedulefree.AdamWScheduleFree` optimizer with `weight_decay=0.0` and the provided `lr`.
- [x] AC-3: Gradient clipping is applied via `torch.nn.utils.clip_grad_norm_` with `max_norm=1.0`.
- [x] AC-4: The optimizer step order is: `loss.backward()` → `clip_grad_norm_` → `optimizer.step()` → `optimizer.zero_grad()`.
- [x] AC-5: Evaluation runs every `steps_per_eval` steps (at step indices `steps_per_eval - 1`, `2*steps_per_eval - 1`, etc.).
- [x] AC-6: During evaluation, model and optimizer are switched to `.eval()` mode, then back to `.train()` mode after.
- [x] AC-7: Evaluation creates a `NanoTabPFNClassifier` and passes it to the `eval_func` callback.
- [x] AC-8: Each eval history entry contains keys `step`, `wall_time`, and `loss`; when `eval_func` is provided, it also includes the scores dict keys.
- [x] AC-9: When `eval_func` is None, eval history entries only contain `step`, `wall_time`, and `loss`.
- [x] AC-10: `step` in eval entries is 1-indexed (i.e., `step + 1`).
- [x] AC-11: `wall_time` tracks cumulative training time only (excludes evaluation time).
- [x] AC-12: Checkpoints are saved every `checkpoint_every` steps (at 1-indexed step) when both `checkpoint_dir` and `checkpoint_every` are truthy.
- [x] AC-13: Checkpoint filenames follow the pattern `step_{step:05d}.pt` (e.g., `step_00250.pt`).
- [x] AC-14: A `final.pt` checkpoint is always saved when `checkpoint_dir` is provided, regardless of `checkpoint_every`.
- [x] AC-15: Checkpoints are saved via `torch.save(model.state_dict(), path)`.
- [x] AC-16: `checkpoint_dir` is created with `os.makedirs(exist_ok=True)` if it doesn't exist.
- [x] AC-17: `KeyboardInterrupt` is caught and training terminates gracefully (final checkpoint still saved).
- [x] AC-18: Returns a tuple of `(model, eval_history)`.
- [x] AC-19: Training targets are sliced to test rows only: `targets[:, train_test_split_index:]`.
- [x] AC-20: Targets are reshaped to `(-1,)` and cast to `torch.long`; output is reshaped to `(-1, num_classes)`.
- [x] AC-21: If no device is provided, `get_default_device()` is used (prefers CUDA > MPS > CPU).
- [x] AC-21.1: At the end of training (the `train` function), if the device is CUDA, the peak GPU memory allocated is printed exactly once.

### PriorDumpDataLoader

- [x] AC-22: Inherits from `torch.utils.data.DataLoader`.
- [x] AC-23: Loads data from an HDF5 file specified by `filename`.
- [x] AC-24: Reads `max_num_classes` from the HDF5 file at construction time.
- [x] AC-25: Iterates for exactly `num_steps` batches per epoch.
- [x] AC-26: Each yielded batch is a dict with keys `x`, `y`, and `train_test_split_index`.
- [x] AC-27: `train_test_split_index` is a scalar value extracted via `[0].item()` from the first sample in the batch.
- [x] AC-28: X and y tensors are dynamically sized per batch: `num_features` and `max_seq_in_batch` are computed from the current batch.
- [x] AC-29: When the pointer reaches the end of the HDF5 dataset, it wraps around to 0.
- [x] AC-30: `__len__` returns `num_steps`.

## Notes

- The ScheduleFree optimizer eliminates the need for a learning rate scheduler while maintaining competitive performance.
- The `weight_decay=0.0` is explicit — no weight decay is applied.
- Training data (`y`) is sliced to only training rows (`y[:, :train_test_split_index]`) before being passed to the model, while full `y` is kept for computing loss on test rows.
