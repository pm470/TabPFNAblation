---
id: EVAL-003
title: "Results Plotting"
status: implemented
module: plot_results.py, generate_mock_data.py
last_synced: 2026-07-15
---

# Results Plotting

## User Story

As a researcher, I want to visualize training curves and final evaluation metrics across activation functions and seeds so that I can compare performance and identify the best-performing variants. I also want to generate the issue #9 evaluation plots (bar chart, learning curves, depth/width scaling) with mock data support and poster-ready output formats.

## Acceptance Criteria

### load_all_results()

- [x] AC-1: Reads from the `results/` directory (configurable via `results_dir` argument).
- [x] AC-2: Raises `FileNotFoundError` if the results directory does not exist.
- [x] AC-3: Iterates over sorted subdirectories of `results/`, treating each as an activation function name.
- [x] AC-4: Within each activation directory, iterates over sorted seed subdirectories.
- [x] AC-5: Reads `metrics.jsonl` from each seed directory, parsing each line as a JSON object.
- [x] AC-6: Skips empty lines in `metrics.jsonl`.
- [x] AC-7: Returns a dict mapping activation name → list of seed runs, where each seed run is a list of metric dicts.
- [x] AC-8: Skips activation directories that have no valid seed runs with metrics.



### Main Script

- [x] AC-29: When run as `__main__`, calls `load_all_results()` with default arguments.
- [x] AC-30: Prints discovered activations and seed counts before plotting.

### generate_mock_data.py

- [ ] AC-32: Generates mock results into `results_mock/` directory with 7 activations × 10 seeds.
- [ ] AC-33: Each seed directory contains `metrics.jsonl` and `config.json`.
- [ ] AC-34: Learning curves follow exponential saturation with per-seed offsets and per-step noise.
- [ ] AC-35: Generates `architecture_ablation.json` with depth and width scaling data for GELU, SwiGLU, GeGLU.
- [ ] AC-36: Uses fixed numpy random seed (42) for reproducibility.

### plot_bar_chart_with_error_bars()

- [ ] AC-37: Bar chart of Normalized ROC-AUC per activation with mean ± std error bars.
- [ ] AC-38: Bars sorted by descending mean performance.
- [ ] AC-39: GELU baseline bar is visually highlighted (distinct color or hatching).
- [ ] AC-40: Saves to `{output_dir}/bar_chart_roc_auc.{png,svg}` at 300 DPI.

### plot_learning_curves_best()

- [ ] AC-41: Learning curve of pre-training steps vs. Normalized ROC-AUC.
- [ ] AC-42: Shows GELU baseline + top-2 best-performing activations (by final mean ROC-AUC).
- [ ] AC-43: Renders ±1 std shaded uncertainty bands across seeds.
- [ ] AC-44: Saves to `{output_dir}/learning_curves_best.{png,svg}` at 300 DPI.

### plot_depth_scaling()

- [ ] AC-45: Line plot of number of transformer layers (x-axis, integer) vs. TabArena Score (y-axis).
- [ ] AC-46: Shows GELU + top-2 variants with uncertainty bands.
- [ ] AC-47: Saves to `{output_dir}/depth_scaling.{png,svg}` at 300 DPI.

### plot_width_scaling()

- [ ] AC-48: Line plot of FFN hidden dim (x-axis, log scale) vs. Normalized ROC-AUC (y-axis).
- [ ] AC-49: Shows GELU + top-2 variants with uncertainty bands.
- [ ] AC-50: Saves to `{output_dir}/width_scaling.{png,svg}` at 300 DPI.

### Mock Data Watermark

- [ ] AC-51: All plot functions accept an `is_mock: bool` parameter (default `False`).
- [ ] AC-52: When `is_mock=True`, a diagonal watermark text "MOCK DATA — NOT FROM REAL EXPERIMENTS" is rendered across the plot.
- [ ] AC-53: Watermark uses semi-transparent red text that doesn't obscure the data and is oriented negatively.

### CLI Interface

- [ ] AC-54: `--mock` flag switches data source to `results_mock/`.
- [ ] AC-55: `--output-dir` flag controls plot output directory (default `plots`).
- [ ] AC-56: When `--mock` is passed, all 6 plots are generated with watermarks.
- [ ] AC-57: Plots are saved as both PNG (300 DPI) and SVG for poster presentations.

## Notes

- Depends on `matplotlib` and `seaborn` for plotting.
- The script expects the `results/` directory structure produced by `run_experiment.py`: `results/{activation}/seed_{N}/metrics.jsonl`.
- Grid is enabled on plots with `alpha=0.3`; the box plot grid is y-axis only.
- Issue #9 plots use mock data when `--mock` is passed; all mock plots include a watermark.
- Architecture ablation data is loaded from `results_mock/architecture_ablation.json`.
- Output formats: PNG (300 DPI) + SVG for poster presentations.
