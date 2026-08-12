---
id: EVAL-001
title: "Local Evaluation (Covertype)"
status: implemented
module: train.py
last_synced: 2026-07-25
---

# Local Evaluation (Covertype)

## User Story

As a researcher, I want a fast local evaluation on a known dataset so that I can quickly verify model training progress and sanity-check changes during development without waiting for the full TabArena benchmark.

## Acceptance Criteria

### get_eval_datasets()

- [x] AC-1: Returns a list of `(name, X_train, X_test, y_train, y_test)` tuples.
- [x] AC-2: Uses `sklearn.datasets.fetch_openml` (or `fetch_covtype`) as the datasets.
- [x] AC-3: Stratified sub-samples datasets to specified size.
- [x] AC-4: Splits rows with `test_size=0.5` and `random_state=42` via `train_test_split`.
- [x] AC-4.1: If all dataset fetches fail, raises a `RuntimeError` explaining dataset fetching failed.

### eval()

- [x] AC-5: Accepts a `classifier` object and an optional `datasets` parameter.
- [x] AC-6: If `datasets` is None, calls `get_eval_datasets()` to obtain the default datasets.
- [x] AC-7: Calls `classifier.fit(X_train, y_train)` for each dataset.
- [x] AC-8: Calls `classifier.predict_proba(X_test)` and derives predictions via `prob.argmax(axis=1)` (avoids a second forward pass).
- [x] AC-9: For binary classification (`prob.shape[1] == 2`), passes `prob[:, 1]` to `roc_auc_score`.
- [x] AC-10: Computes three metrics: `roc_auc` (via `roc_auc_score` with `multi_class="ovr"`), `acc` (via `accuracy_score`), and `balanced_acc` (via `balanced_accuracy_score`).
- [x] AC-11: Returns a dict with keys `"roc_auc"`, `"acc"`, and `"balanced_acc"`.
- [x] AC-12: Metric values are averaged over all datasets by dividing the sum by `len(datasets)`. Raises `ValueError` if `datasets` is empty.
- [x] AC-13: All returned values are Python floats (cast via `float()`).
- [x] AC-14: If the model predicts `NaN` probabilities, a warning is printed and the `NaN`s are replaced with uniform probabilities before calculating metrics to prevent failures.

## Notes

- The Covertype dataset is a standard classification benchmark with 7 classes and 54 features.
- We sub-sample it to 2000 rows to ensure `predict_proba` completes quickly without bottlenecking the training loop.
- The 50/50 split with `random_state=0` ensures deterministic train/test partitions across runs.
- The `multi_class="ovr"` parameter is passed to `roc_auc_score` to support multiclass extension if more datasets are added later.
