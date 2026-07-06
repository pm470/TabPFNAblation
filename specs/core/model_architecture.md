---
id: CORE-001
title: "NanoTabPFN Model Architecture"
status: implemented
module: model.py
last_synced: 2026-07-03
---

# NanoTabPFN Model Architecture

## User Story

As a researcher, I want a compact tabular foundation model that encodes features and targets into embeddings, applies cross-attention between rows and columns, and decodes class logits, so that I can pre-train on synthetic priors and predict on downstream classification tasks.

## Acceptance Criteria

### NanoTabPFNModel

- [x] AC-1: The model constructor accepts `embedding_size`, `num_attention_heads`, `mlp_hidden_size`, `num_layers`, `num_outputs`, and `activation` (default `"gelu"`) as parameters.
- [x] AC-2: The model contains a `FeatureEncoder`, a `TargetEncoder`, a `nn.ModuleList` of `TransformerEncoderLayer` blocks (length `num_layers`), and a `Decoder`.
- [x] AC-3: Forward pass accepts a tuple `(x_src, y_src)` and an integer `train_test_split_index`.
- [x] AC-4: If `y_src` has fewer dimensions than `x_src`, an extra trailing dimension is added via `unsqueeze(-1)`.
- [x] AC-5: Feature and target embeddings are concatenated along dimension 2 (`torch.cat([x_src, y_src], 2)`).
- [x] AC-6: All transformer blocks are applied sequentially, each receiving the `train_test_split_index`.
- [x] AC-7: After the transformer stack, only test-row target embeddings are selected: `src_tensor[:, train_test_split_index:, -1, :]`.
- [x] AC-8: Output shape is `(batch_size, num_test_rows, num_outputs)` where `num_test_rows = num_rows - train_test_split_index`.
- [x] AC-9: With default config (embedding_size=96, heads=4, mlp_hidden=192, layers=3, outputs=2) the model has exactly 356,066 parameters.
- [x] AC-9.1: Gated activations dynamically adjust their inner dimension size to maintain a parameter count as close as possible to the baseline.

### FeatureEncoder

- [x] AC-10: Uses a single `nn.Linear(1, embedding_size)` to embed scalar features.
- [x] AC-11: Normalization is computed from training rows only: mean and std over `x[:, :train_test_split_index]` (cast to float) along dim=1 with `keepdim=True`.
- [x] AC-12: Std uses `unbiased=False` and an epsilon of `1e-20` to avoid division by zero and NaNs for single-element batches.
- [x] AC-13: Normalized features are clipped to `[-100, 100]` via `torch.clip`.
- [x] AC-14: Input `(B, R, C)` produces output `(B, R, C, E)` where E is `embedding_size`.

### TargetEncoder

- [x] AC-15: Uses a single `nn.Linear(1, embedding_size)` to embed scalar targets.
- [x] AC-16: Test-row targets are padded with the per-batch mean of `y_train` (cast to float, computed over dim=1, keepdim=True).
- [x] AC-17: Padding is created via `mean.repeat(1, num_rows - y_train.shape[1], 1)` and concatenated with `y_train`.
- [x] AC-18: Input `(B, N_train, 1)` produces output `(B, num_rows, 1, E)`.

### TransformerEncoderLayer

- [x] AC-19: Contains two separate `MultiheadAttention` modules: one for attention between features (columns) and one for attention between datapoints (rows).
- [x] AC-20: Contains a dynamically sized MLP (created via `create_mlp` returning `StandardMLP` or `GatedMLP`) with residual connection.
- [x] AC-21: Contains three `LayerNorm` layers (one after each of: feature attention, datapoint attention, MLP).
- [x] AC-22: Feature attention reshapes `(B, R, C, E)` → `(B*R, C, E)` so attention operates across columns per row.
- [x] AC-23: Datapoint attention transposes dims 1 and 2, then reshapes to `(B*C, R, E)` so attention operates across rows per column.
- [x] AC-24: Training rows attend only to themselves: `self_attention(src[:, :split], src[:, :split], src[:, :split])`.
- [x] AC-25: Test rows attend only to training rows: `self_attention(src[:, split:], src[:, :split], src[:, :split])`.
- [x] AC-26: Both attention outputs are concatenated and a residual connection is added from the pre-attention tensor.
- [x] AC-27: The MLP supports standard activations (GELU, ReLU, SiLU/Swish, Mish) and gated variants (SwiGLU, GeGLU, ReGLU).

### Decoder

- [x] AC-28: Uses the dynamically sized MLP (created via `create_mlp`) configured for `num_outputs`.
- [x] AC-29: No residual connection in the decoder.
- [x] AC-30: Input `(B, R, E)` produces output `(B, R, num_outputs)`.

### NanoTabPFNClassifier

- [x] AC-31: Implements a scikit-learn-like interface with `fit`, `predict_proba`, and `predict` methods, allowing optional `max_train_samples`, `max_total_samples`, and `n_ensemble` in constructor.
- [x] AC-32: `fit` stores `X_train`, `y_train`, and computes `num_classes = max(set(y_train)) + 1`.
- [x] AC-33: `predict_proba` concatenates `X_train_sub` and `X_test_chunk` into a single array, runs inference with `torch.no_grad()`.
- [x] AC-33.1: If constructor limits are not provided, they are calculated dynamically based on feature count and attention heads to target a max attention weights memory (e.g. 8.0 GiB) and clipped to a safe range (e.g. `[1000, 10000]`).
- [x] AC-33.2: Does not print GPU memory logs from within `predict_proba` to avoid verbose log clutter.
- [x] AC-33.3: Supports `n_ensemble` forward passes, using stratified random sampling (with different seeds) to subsample the training data while strictly preserving class balance. Falls back to uniform random sampling if a class is too small to stratify. Averages the resulting probabilities.
- [x] AC-33.4: Uses `torch.autocast(dtype=torch.bfloat16)` during the forward pass to halve memory usage. Bfloat16 is used instead of float16 because it shares float32's exponent range (max ~3.4e38), avoiding overflow→NaN issues.
- [x] AC-34: Model output is sliced to `[:, :num_classes]` outside the autocast context to remove unused output columns.
- [x] AC-35: Softmax is applied over dim=1 in float32, outside the autocast context, to convert logits to probabilities.
- [x] AC-36: Output is moved to CPU and converted to numpy before returning.
- [x] AC-37: `predict` returns `argmax(axis=1)` of `predict_proba` output.
- [x] AC-38: Input tensors are unsqueezed to add a batch dimension of 1 and cast to `torch.float`.

## Notes

- The architecture is from [nanoTabPFN](https://github.com/automl/nanoTabPFN), a simplified reimplementation of TabPFNv2.
- The causal attention mask (train attends to train, test attends to train) prevents label leakage during pre-training.
- `MultiheadAttention` is used with `batch_first=True`.
