---
id: EVAL-002
title: "TabArena Evaluation"
status: implemented
module: tabarena_eval.py
last_synced: 2026-06-25
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
- [x] AC-8: `_fit` asserts that both `_CURRENT_PYTORCH_MODEL` and `_CURRENT_DEVICE` are not None.
- [x] AC-9: `_get_default_auxiliary_params` sets `valid_raw_types` to `["int", "float", "category"]`.
- [x] AC-10: `supported_problem_types` returns `["binary", "multiclass"]`.
- [x] AC-11: `config_generator` returns a `ConfigGenerator` with a single empty manual config and no search space.

### run_tabarena_eval()

- [x] AC-12: Sets module-level globals `_CURRENT_PYTORCH_MODEL` and `_CURRENT_DEVICE` from function arguments.
- [x] AC-13: Creates a `TabArenaV0pt1ExperimentBundle` with a single model entry (the `TabArenaNanoTabPFNModel` config generator at index 0).
- [x] AC-14: Uses `TabArenaContext` to build and run evaluation jobs.
- [x] AC-15: Results directory is `run_dir / "tabarena_exp"`.
- [x] AC-16: Uses `subset="full"` in `build_and_run_jobs`.
- [x] AC-17: Runs with `debug_mode=True` (required for the global variable sharing pattern).
- [x] AC-18: Sets `new_result_prefix="[New] "` for result labeling.

### Global Model Sharing

- [x] AC-19: Module-level variables `_CURRENT_PYTORCH_MODEL` and `_CURRENT_DEVICE` are initialized to `None`.
- [x] AC-20: `run_tabarena_eval` sets these globals before building experiments so that `_fit` can access the pre-trained model.

## Notes

- The global variable pattern (`_CURRENT_PYTORCH_MODEL` / `_CURRENT_DEVICE`) is a deliberate hack to pass the pre-trained PyTorch model into the AutoGluon `AbstractModel._fit` method, which doesn't support custom constructor arguments. This only works with `debug_mode=True` (in-process execution).
- `_preprocess` makes a copy of `X` before label encoding to avoid mutating the input DataFrame.
