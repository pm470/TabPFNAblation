---
id: CORE-002
title: "Training Pipeline"
status: implemented
module: train.py
last_synced: 2026-07-17
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
- [x] AC-8: Each eval history entry contains keys `step`, `wall_time`, `loss`, and `param_count`; when `eval_func` is provided, it also includes the scores dict keys.
- [x] AC-9: When `eval_func` is None, eval history entries only contain `step`, `wall_time`, `loss`, and `param_count`.
- [x] AC-10: `step` in eval entries is 1-indexed (i.e., `step + 1`).
- [x] AC-11: `wall_time` tracks cumulative training time only (excludes evaluation time).
- [x] AC-12: Checkpoints are saved every `checkpoint_every` steps (at 1-indexed step) when both `checkpoint_dir` and `checkpoint_every` are truthy.
- [x] AC-13: Checkpoint filenames follow the pattern `step_{step:05d}.pt` (e.g., `step_00250.pt`).
- [x] AC-14: A `final.pt` checkpoint is always saved when `checkpoint_dir` is provided, regardless of `checkpoint_every`. It follows the eval→save→train pattern from AC-15.
- [x] AC-15: Before saving any checkpoint, `optimizer.eval()` must be called to swap in the averaged weights. After saving, `optimizer.train()` must be called to resume training. Checkpoints are saved via `torch.save(model.state_dict(), path)` and the saved `state_dict` must contain the eval-mode (averaged) weights.
- [x] AC-15.1: A helper function `_save_checkpoint(model, optimizer, checkpoint_dir, filename)` encapsulates the eval→save→train pattern to prevent future omissions.
- [x] AC-15.2: When `eval_func` is provided, the total cumulative time spent on inline evaluation is tracked and printed once at the end of training (e.g., `[NanoTabPFN] Total inline eval time: 123.4s`).
- [x] AC-16: `checkpoint_dir` is created with `os.makedirs(exist_ok=True)` if it doesn't exist.
- [x] AC-17: `KeyboardInterrupt` is caught and training terminates gracefully (final checkpoint still saved).
- [x] AC-18: Returns a tuple of `(model, eval_history)`.
- [x] AC-19: Training targets are sliced to test rows only: `targets[:, train_test_split_index:]`.
- [x] AC-20: Targets are reshaped to `(-1,)` and cast to `torch.long`; output is reshaped to `(-1, num_classes)`.
- [x] AC-21: If no device is provided, `get_default_device()` is used (prefers CUDA > MPS > CPU).
- [x] AC-21.1: At the end of training (the `train` function), if the device is CUDA, the peak GPU memory allocated is printed exactly once.
- [x] AC-21.2: If the batch loss is `NaN`, a warning is printed and the batch is skipped without updating gradients or weights.
- [x] AC-21.3: `train` accepts a `start_step` parameter to offset the logged step counter when resuming.

### Auto-Resumption / Idempotency

- [x] AC-21.4: `run_experiment.py` checks for existing checkpoints in the `checkpoint_dir` before starting.
- [x] AC-21.5: If checkpoints exist, it loads the model weights from the highest `step_XXXXX.pt` file.
- [x] AC-21.6: The dataloader's `num_steps` is automatically reduced to only process the remaining steps.
- [x] AC-21.7: New evaluation metrics are appended to the existing `metrics.jsonl` file instead of overwriting it.
- [x] AC-21.8: `slurm/generate_data.sbatch` and `slurm/train.sbatch` implement file existence checks (on the final HDF5 file and final checkpoint, respectively) to exit cleanly without re-running completed work.

### NanopriorDataLoader

- [x] AC-22: Inherits from `torch.utils.data.IterableDataset` and is wrapped by `torch.utils.data.DataLoader` for batching/multiprocessing.
- [x] AC-23: Generates synthetic datasets on the fly using `prior.py`'s `rand_dataset_filtered`.
- [x] AC-24: Batches are generated directly within the dataset so all datasets in a batch share the exact same `n_samples`, `n_features`, and `x_cat_sizes`.
- [x] AC-25: Iterates for exactly `num_steps` batches per epoch.
- [x] AC-26: Each yielded batch is a dict with keys `x`, `y`, and `train_test_split_index`.
- [x] AC-27: `train_test_split_index` is a randomly chosen index between 50% and 90% of the `n_samples`.
- [x] AC-28: `num_workers` is supported to accelerate generation.
- [x] AC-29: The `num_classes` parameter is randomized per batch but bounded by `max_classes`.
- [x] AC-30: Yields tensors matching the `x` and `y` shapes expected by `NanoTabPFNModel`.

### Mixed-Precision Training

- [x] AC-31: The forward pass and loss computation are wrapped in `torch.autocast(device_type=..., dtype=torch.bfloat16)` when the device supports it (CUDA or ROCm). On CPU/MPS, autocast is not used.
- [x] AC-32: Model parameters remain in float32 (master weights). Autocast handles the temporary bf16 downcast for compute-heavy ops (Linear, attention) while keeping precision-sensitive ops (LayerNorm, softmax, loss) in float32.
- [x] AC-33: No `GradScaler` is used — bfloat16 has the same dynamic range as float32, so loss scaling is unnecessary.
- [x] AC-34: The `train()` function accepts an `autocast_dtype` parameter (default: `torch.bfloat16`) that controls the autocast dtype. Passing `None` disables autocast entirely.
- [x] AC-35: The autocast dtype is logged in the experiment config JSON.

## Notes

- The ScheduleFree optimizer eliminates the need for a learning rate scheduler while maintaining competitive performance.
- The `weight_decay=0.0` is explicit — no weight decay is applied.
- Training data (`y`) is sliced to only training rows (`y[:, :train_test_split_index]`) before being passed to the model, while full `y` is kept for computing loss on test rows.
- Mixed-precision with bfloat16 autocast enables FlashAttention (which requires bf16/fp16 inputs) while maintaining float32 numerical stability in normalization layers and the optimizer. This follows the same approach used by Google's TabFM.
