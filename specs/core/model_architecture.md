---
id: CORE-001
title: "NanoTabPFN Model Architecture"
status: implemented
module: model.py
last_synced: 2026-06-25
---

# NanoTabPFN Model Architecture

## User Story

As a researcher, I want a compact tabular foundation model that encodes features and targets into embeddings, applies cross-attention between rows and columns, and decodes class logits, so that I can pre-train on synthetic priors and predict on downstream classification tasks.

## Acceptance Criteria

### NanoTabPFNModel

- [ ] AC-1: The model constructor accepts `embedding_size`, `num_attention_heads`, `mlp_hidden_size`, `num_layers`, and `num_outputs` as parameters.
- [ ] AC-2: The model contains a `FeatureEncoder`, a `TargetEncoder`, a `nn.ModuleList` of `TransformerEncoderLayer` blocks (length `num_layers`), and a `Decoder`.
- [ ] AC-3: Forward pass accepts a tuple `(x_src, y_src)` and an integer `train_test_split_index`.
- [ ] AC-4: If `y_src` has fewer dimensions than `x_src`, an extra trailing dimension is added via `unsqueeze(-1)`.
- [ ] AC-5: Feature and target embeddings are concatenated along dimension 2 (`torch.cat([x_src, y_src], 2)`).
- [ ] AC-6: All transformer blocks are applied sequentially, each receiving the `train_test_split_index`.
- [ ] AC-7: After the transformer stack, only test-row target embeddings are selected: `src_tensor[:, train_test_split_index:, -1, :]`.
- [ ] AC-8: Output shape is `(batch_size, num_test_rows, num_outputs)` where `num_test_rows = num_rows - train_test_split_index`.
- [ ] AC-9: With default config (embedding_size=96, heads=4, mlp_hidden=192, layers=3, outputs=2) the model has exactly 356,066 parameters.

### FeatureEncoder

- [ ] AC-10: Uses a single `nn.Linear(1, embedding_size)` to embed scalar features.
- [ ] AC-11: Normalization is computed from training rows only: mean and std over `x[:, :train_test_split_index]` along dim=1 with `keepdim=True`.
- [ ] AC-12: Std uses an epsilon of `1e-20` to avoid division by zero.
- [ ] AC-13: Normalized features are clipped to `[-100, 100]` via `torch.clip`.
- [ ] AC-14: Input `(B, R, C)` produces output `(B, R, C, E)` where E is `embedding_size`.

### TargetEncoder

- [ ] AC-15: Uses a single `nn.Linear(1, embedding_size)` to embed scalar targets.
- [ ] AC-16: Test-row targets are padded with the per-batch mean of `y_train` (computed over dim=1, keepdim=True).
- [ ] AC-17: Padding is created via `mean.repeat(1, num_rows - y_train.shape[1], 1)` and concatenated with `y_train`.
- [ ] AC-18: Input `(B, N_train, 1)` produces output `(B, num_rows, 1, E)`.

### TransformerEncoderLayer

- [ ] AC-19: Contains two separate `MultiheadAttention` modules: one for attention between features (columns) and one for attention between datapoints (rows).
- [ ] AC-20: Contains a 2-layer MLP (`Linear(E, H)` → GELU → `Linear(H, E)`) with residual connection.
- [ ] AC-21: Contains three `LayerNorm` layers (one after each of: feature attention, datapoint attention, MLP).
- [ ] AC-22: Feature attention reshapes `(B, R, C, E)` → `(B*R, C, E)` so attention operates across columns per row.
- [ ] AC-23: Datapoint attention transposes dims 1 and 2, then reshapes to `(B*C, R, E)` so attention operates across rows per column.
- [ ] AC-24: Training rows attend only to themselves: `self_attention(src[:, :split], src[:, :split], src[:, :split])`.
- [ ] AC-25: Test rows attend only to training rows: `self_attention(src[:, split:], src[:, :split], src[:, :split])`.
- [ ] AC-26: Both attention outputs are concatenated and a residual connection is added from the pre-attention tensor.
- [ ] AC-27: The MLP activation function is GELU (`F.gelu`).

### Decoder

- [ ] AC-28: 2-layer MLP: `Linear(embedding_size, mlp_hidden_size)` → GELU → `Linear(mlp_hidden_size, num_outputs)`.
- [ ] AC-29: No residual connection in the decoder.
- [ ] AC-30: Input `(B, R, E)` produces output `(B, R, num_outputs)`.

### NanoTabPFNClassifier

- [ ] AC-31: Implements a scikit-learn-like interface with `fit`, `predict_proba`, and `predict` methods.
- [ ] AC-32: `fit` stores `X_train`, `y_train`, and computes `num_classes = max(set(y_train)) + 1`.
- [ ] AC-33: `predict_proba` concatenates `X_train` and `X_test` into a single array, runs inference with `torch.no_grad()`.
- [ ] AC-34: Model output is sliced to `[:, :num_classes]` to remove unused output columns.
- [ ] AC-35: Softmax is applied over dim=1 to convert logits to probabilities.
- [ ] AC-36: Output is moved to CPU and converted to numpy before returning.
- [ ] AC-37: `predict` returns `argmax(axis=1)` of `predict_proba` output.
- [ ] AC-38: Input tensors are unsqueezed to add a batch dimension of 1 and cast to `torch.float`.

## Notes

- The architecture is from [nanoTabPFN](https://github.com/automl/nanoTabPFN), a simplified reimplementation of TabPFNv2.
- The causal attention mask (train attends to train, test attends to train) prevents label leakage during pre-training.
- `MultiheadAttention` is used with `batch_first=True`.
