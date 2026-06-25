---
id: EVAL-003
title: "Results Plotting"
status: implemented
module: plot_results.py
last_synced: 2026-06-25
---

# Results Plotting

## User Story

As a researcher, I want to visualize training curves and final evaluation metrics across activation functions and seeds so that I can compare performance and identify the best-performing variants.

## Acceptance Criteria

### load_all_results()

- [ ] AC-1: Reads from the `results/` directory (configurable via `results_dir` argument).
- [ ] AC-2: Raises `FileNotFoundError` if the results directory does not exist.
- [ ] AC-3: Iterates over sorted subdirectories of `results/`, treating each as an activation function name.
- [ ] AC-4: Within each activation directory, iterates over sorted seed subdirectories.
- [ ] AC-5: Reads `metrics.jsonl` from each seed directory, parsing each line as a JSON object.
- [ ] AC-6: Skips empty lines in `metrics.jsonl`.
- [ ] AC-7: Returns a dict mapping activation name → list of seed runs, where each seed run is a list of metric dicts.
- [ ] AC-8: Skips activation directories that have no valid seed runs with metrics.

### plot_training_curves()

- [ ] AC-9: Creates a single figure with size `(10, 6)`.
- [ ] AC-10: Plots training step vs. ROC-AUC for each activation function.
- [ ] AC-11: Assumes all seeds within an activation have the same step sequence (takes steps from the first seed run).
- [ ] AC-12: Computes mean and standard deviation of `roc_auc` across seeds using `np.mean` and `np.std` along axis=0.
- [ ] AC-13: Plots the mean line with a label of format `"{activation_name} (n={num_seeds})"`.
- [ ] AC-14: Renders ±1 std shaded bands using `ax.fill_between` with `alpha=0.2`.
- [ ] AC-15: Uses `float("nan")` as fallback if `roc_auc` key is missing from an entry.
- [ ] AC-16: Axes labels are "Training Step" (x) and "ROC-AUC" (y).
- [ ] AC-17: Title is "Training Curves: ROC-AUC vs. Step".
- [ ] AC-18: Saves to the specified `output_path` (default `"training_curves.png"`) at 150 DPI.
- [ ] AC-19: Calls `plt.close(fig)` after saving to free memory.

### plot_final_boxplot()

- [ ] AC-20: Creates a box-and-whisker plot of final ROC-AUC values per activation function.
- [ ] AC-21: Final ROC-AUC is taken from the last entry (`run[-1]`) of each seed run.
- [ ] AC-22: Figure width scales with the number of activations: `max(6, len(labels) * 1.5)`, height is 6.
- [ ] AC-23: Uses `ax.boxplot` with `patch_artist=True` for colored boxes.
- [ ] AC-24: Box colors are from `seaborn.color_palette("husl", len(labels))` with `alpha=0.7`.
- [ ] AC-25: Axes labels are "Activation Function" (x) and "Final ROC-AUC" (y).
- [ ] AC-26: Title is "Final ROC-AUC by Activation Function".
- [ ] AC-27: Saves to the specified `output_path` (default `"final_boxplot.png"`) at 150 DPI.
- [ ] AC-28: Calls `plt.close(fig)` after saving.

### Main Script

- [ ] AC-29: When run as `__main__`, calls `load_all_results()` with default arguments.
- [ ] AC-30: Prints discovered activations and seed counts before plotting.
- [ ] AC-31: Calls both `plot_training_curves` and `plot_final_boxplot` sequentially.

## Notes

- Depends on `matplotlib` and `seaborn` for plotting.
- The script expects the `results/` directory structure produced by `run_experiment.py`: `results/{activation}/seed_{N}/metrics.jsonl`.
- Grid is enabled on plots with `alpha=0.3`; the box plot grid is y-axis only.
