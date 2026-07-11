---
id: EVAL-002
title: "TabArena Evaluation"
status: implemented
module: tabarena_eval.py
last_synced: 2026-07-11
---

# TabArena Evaluation

## User Story

As a researcher, I want to evaluate my trained model on the TabArena benchmark so that I can compare activation function variants on a standardized, community-recognized tabular ML benchmark.

## Acceptance Criteria

### TabArenaNanoTabPFNModel

- [x] AC-1: Extends `autogluon.core.models.AbstractModel`.
- [x] AC-2: Sets `ag_key = "NanoTabPFN"` and `ag_name = "NanoTabPFN"` as class attributes.
- [x] AC-3: `_preprocess` uses `LabelEncoderFeatureGenerator` to label-encode categorical features (fitted only during `is_train=True`).
- [x] AC-4: `_preprocess` fills NaN values with 0 via `X.fillna(0)`.
- [x] AC-5: `_preprocess` calls `super()._preprocess(X, **kwargs)` before custom preprocessing.
- [x] AC-6: `_fit` converts the DataFrame to `np.float32` and the Series to `np.int64` before passing to `NanoTabPFNClassifier`.
- [x] AC-7: `_fit` retrieves the shared model and device from module-level globals `_CURRENT_PYTORCH_MODEL` and `_CURRENT_DEVICE`.
- [x] AC-8: `_fit` asserts that both `_CURRENT_PYTORCH_MODEL` and `_CURRENT_DEVICE` are not None, and instantiates `NanoTabPFNClassifier` with `n_ensemble=8`.
- [x] AC-9: `_get_default_auxiliary_params` sets `valid_raw_types` to `["int", "float", "category"]`.
- [x] AC-10: `supported_problem_types` returns `["binary", "multiclass"]`.
- [x] AC-11: `config_generator` returns a `ConfigGenerator` with a single empty manual config and no search space.
- [x] AC-11.1: `_predict_proba` calls `self.preprocess(X, **kwargs)` before converting to numpy — this is required because AutoGluon does NOT automatically preprocess X before calling `_predict_proba`. The `**kwargs` carries `preprocess_nonadaptive=False` from the BAG model so only stateful preprocessing runs (including `_preprocess_align_features`, which aligns features to the training fold).
- [x] AC-11.2: Overrides `score_with_y_pred_proba` to catch `ValueError` ("Only one class present in y_true") and return a dummy score (0.5). This prevents AutoGluon from crashing when computing ROC AUC on validation folds that only contain a single class due to high class imbalance in small datasets.
- [x] AC-11.3: `_predict_proba` returns `self._convert_proba_to_unified_form(probs)` to ensure the shape matches AutoGluon's expected unified format (e.g., extracting the positive class probability `(N,)` for binary classification tasks) rather than passing the raw `(N, C)` shape up to the BAG model.

### run_tabarena_eval()

- [x] AC-12: Sets module-level globals `_CURRENT_PYTORCH_MODEL` and `_CURRENT_DEVICE` from function arguments.
- [x] AC-13: Creates a `TabArenaV0pt1ExperimentBundle` with a single model entry (the `TabArenaNanoTabPFNModel` config generator at index 0) and sets `verbosity=0` and `model_verbosity=0` to reduce AutoGluon's fold-level logging noise.
- [x] AC-14: Uses `TabArenaContext` to build and run evaluation jobs.
- [x] AC-15: Results directory is `run_dir / "tabarena_exp"`.
- [x] AC-16: Uses the `subset` parameter in `build_and_run_jobs` (defaults to `"nanotabpfn"`).
- [x] AC-17: Runs with `debug_mode=True` (required for the global variable sharing pattern).
- [x] AC-18: Sets `new_result_prefix="[New] "` for result labeling.
- [x] AC-18.1: Defines a `SubsetPredicate` for "nanotabpfn" that combines TabArena's official "tabpfn" subset with a classification-only filter (`n_classes > 0`).
- [x] AC-18.2: Bypasses TabArena's `context.compare()` and instead manually parses the `list[dict]` returned by `build_and_run_jobs`.
- [x] AC-18.3: Computes metrics (ROC-AUC and Log Loss) natively via `sklearn.metrics` directly from the raw test probabilities (`pred_proba_dict_test`) and labels (`y_test`) nested inside `simulation_artifacts`. Handles `problem_type == "binary"` arrays correctly and dynamically suppresses `roc_auc` dimension mismatch or missing class errors using `labels=...`.
- [x] AC-18.4: Saves a clean CSV (`nanotabpfn_summary.csv`) containing average metric scores grouped by `task_id`.

### Global Model Sharing

- [x] AC-19: Module-level variables `_CURRENT_PYTORCH_MODEL` and `_CURRENT_DEVICE` are initialized to `None`.
- [x] AC-20: `run_tabarena_eval` sets these globals before building experiments so that `_fit` can access the pre-trained model.
- [x] AC-21: `TabArenaNanoTabPFNModel` overrides `predict_proba` to print peak GPU memory allocated once per dataset, extracting the dataset identifier from the model path.

## Notes

- The global variable pattern (`_CURRENT_PYTORCH_MODEL` / `_CURRENT_DEVICE`) is a deliberate hack to pass the pre-trained PyTorch model into the AutoGluon `AbstractModel._fit` method, which doesn't support custom constructor arguments. This only works with `debug_mode=True` (in-process execution).
- `_preprocess` makes a copy of `X` before label encoding to avoid mutating the input DataFrame.
- `_predict_proba` must explicitly call `self.preprocess(X, **kwargs)` because AutoGluon's `AbstractModel._predict_proba_internal` does NOT preprocess X before calling `_predict_proba`. Without this, the BAG model's cross-validation can fail when `drop_unique` removes a single-value feature from a training fold, causing a dimension mismatch between X_train (stored during `_fit`) and X_test (passed to `_predict_proba`). The `TabDPT` model in AutoGluon follows the same pattern.
