---
id: EVAL-001
title: "Local Evaluation (Breast Cancer)"
status: implemented
module: train.py
last_synced: 2026-06-25
---

# Local Evaluation (Breast Cancer)

## User Story

As a researcher, I want a fast local evaluation on a known dataset so that I can quickly verify model training progress and sanity-check changes during development without waiting for the full TabArena benchmark.

## Acceptance Criteria

### get_eval_datasets()

- [ ] AC-1: Returns a list of `(X_train, X_test, y_train, y_test)` tuples.
- [ ] AC-2: Uses `sklearn.datasets.load_breast_cancer(return_X_y=True)` as the dataset.
- [ ] AC-3: Splits with `test_size=0.5` and `random_state=0` via `sklearn.model_selection.train_test_split`.
- [ ] AC-4: Currently contains exactly one dataset (breast cancer only).

### eval()

- [ ] AC-5: Accepts a `classifier` object and an optional `datasets` parameter.
- [ ] AC-6: If `datasets` is None, calls `get_eval_datasets()` to obtain the default datasets.
- [ ] AC-7: Calls `classifier.fit(X_train, y_train)` for each dataset.
- [ ] AC-8: Calls `classifier.predict_proba(X_test)` and derives predictions via `prob.argmax(axis=1)` (avoids a second forward pass).
- [ ] AC-9: For binary classification (`prob.shape[1] == 2`), passes `prob[:, 1]` to `roc_auc_score`.
- [ ] AC-10: Computes three metrics: `roc_auc` (via `roc_auc_score` with `multi_class="ovr"`), `acc` (via `accuracy_score`), and `balanced_acc` (via `balanced_accuracy_score`).
- [ ] AC-11: Returns a dict with keys `"roc_auc"`, `"acc"`, and `"balanced_acc"`.
- [ ] AC-12: Metric values are averaged over all datasets by dividing the sum by `len(datasets)`.
- [ ] AC-13: All returned values are Python floats (cast via `float()`).

## Notes

- The breast cancer dataset is a standard sklearn binary classification benchmark with 569 samples and 30 features.
- The 50/50 split with `random_state=0` ensures deterministic train/test partitions across runs.
- The `multi_class="ovr"` parameter is passed to `roc_auc_score` to support multiclass extension if more datasets are added later.
