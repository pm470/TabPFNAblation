"""Plot results from ablation experiments.

Reads metrics.jsonl files from results/ and produces:
1. Training step vs. ROC-AUC per activation (mean ± std across seeds)
2. Final ROC-AUC box plot across activations

Issue #9 plots (with ``--mock`` flag for mock data):
3. Bar chart with error bars: Normalized ROC-AUC per activation
4. Learning curves for GELU baseline + top-2 best variants
5. Architecture scaling (depth): layers vs. TabArena Score
6. Architecture scaling (width): FFN hidden dim (log) vs. Normalized ROC-AUC
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

from nanotabpfn.analysis import (
    METRICS,
    aggregate_seed_scores,
    compute_relative_improvement,
    format_activation_display_name,
    load_tabarena_scores,
)

# Publication-quality defaults
PLOT_DPI = 600
PLOT_STYLE = "whitegrid"
WATERMARK_TEXT = "MOCK DATA — NOT FROM REAL EXPERIMENTS"
BASELINE_ACTIVATION = "gelu"

# Clean, publication-friendly Matplotlib rcParams (apply globally)
mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial"],
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 14,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 12,
        "figure.dpi": PLOT_DPI,
        "savefig.dpi": PLOT_DPI,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def _wrap_label(label: str, width: int = 14) -> str:
    """Wrap a label across lines to reduce x-axis overlap."""
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


def _add_watermark(ax: plt.Axes) -> None:
    """Add a diagonal watermark to indicate mock data.

    Renders semi-transparent gray text diagonally across the plot area
    so it is visible but does not obscure the underlying data.
    """
    ax.text(
        0.5,
        0.5,
        WATERMARK_TEXT,
        transform=ax.transAxes,
        fontsize=22,
        color="red",
        alpha=0.16,
        ha="center",
        va="center",
        rotation=-30,
        fontweight="normal",
        zorder=999,
    )


def _save_plot(fig: plt.Figure, output_dir: str, basename: str) -> None:
    """Save a figure as both PNG and SVG for poster presentations.

    Args:
        fig: The matplotlib figure to save.
        output_dir: Directory to write output files into.
        basename: Filename without extension (e.g. ``"bar_chart_roc_auc"``).
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    png_path = out_path / f"{basename}.png"
    svg_path = out_path / f"{basename}.svg"
    pdf_path = out_path / f"{basename}.pdf"

    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    # Vector PDF for poster printing (fonts embedded via rcParams)
    fig.savefig(pdf_path, bbox_inches="tight")
    print(f"  Saved: {png_path}, {svg_path}")


def _get_top_activations(all_results: dict[str, list[list[dict]]], n: int = 2) -> list[str]:
    """Return the top-N activation names by final mean ROC-AUC.

    Args:
        all_results: Dict mapping activation name → list of seed runs.
        n: Number of top activations to return.

    Returns:
        List of top-N activation names (excluding the baseline) sorted
        by descending final mean ROC-AUC.
    """
    final_means: dict[str, float] = {}
    for name, seed_runs in all_results.items():
        tabarena_aucs = [run[-1].get("tabarena_roc_auc") for run in seed_runs]
        if any(v is not None for v in tabarena_aucs):
            final_aucs = [v if v is not None else float("nan") for v in tabarena_aucs]
        else:
            final_aucs = [run[-1].get("roc_auc", float("nan")) for run in seed_runs]
        final_means[name] = float(np.nanmean(final_aucs))

    # Sort by mean, exclude baseline, take top-N
    sorted_names = sorted(final_means, key=lambda k: final_means[k], reverse=True)
    top = [name for name in sorted_names if name != BASELINE_ACTIVATION][:n]
    return top


# ---------------------------------------------------------------------------
# Existing plots (unchanged behavior, minor additions for output_dir support)
# ---------------------------------------------------------------------------


def load_all_results(results_dir: str = "results") -> dict[str, list[list[dict]]]:
    """Load all metrics from results directory.

    Returns:
        Dict mapping activation name -> list of seed runs,
        where each seed run is a list of metric dicts.
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    all_results = {}
    for activation_dir in sorted(results_path.iterdir()):
        if not activation_dir.is_dir():
            continue
        activation_name = activation_dir.name
        seed_runs = []
        for seed_dir in sorted(activation_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            metrics_file = seed_dir / "metrics.jsonl"
            if not metrics_file.exists():
                continue
            metrics = []
            with open(metrics_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        metrics.append(json.loads(line))
            if metrics:
                tabarena_csv = seed_dir / "benchmark_final" / "tabarena_exp" / "nanotabpfn_summary.csv"
                if tabarena_csv.exists():
                    try:
                        import pandas as pd

                        df = pd.read_csv(tabarena_csv)
                        if "roc_auc" in df.columns:
                            metrics[-1]["tabarena_roc_auc"] = float(df["roc_auc"].mean())
                    except Exception:
                        pass
                seed_runs.append(metrics)
        if seed_runs:
            all_results[activation_name] = seed_runs

    return all_results


# ---------------------------------------------------------------------------
# Issue #9 plots
# ---------------------------------------------------------------------------


def plot_bar_chart_with_error_bars(
    all_results: dict[str, list[list[dict]]],
    output_dir: str = "plots",
    is_mock: bool = False,
) -> None:
    """Bar chart of ROC-AUC per activation with error bars.

    Bars are sorted by descending mean performance. The GELU baseline
    is plotted as a horizontal reference line with a shaded uncertainty band.

    Args:
        all_results: Dict mapping activation name → list of seed runs.
        output_dir: Directory for output files.
        is_mock: If True, add a watermark indicating mock data.
    """
    sns.set_theme(style=PLOT_STYLE)

    # Compute final mean and std per activation
    names: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    baseline_mean: float = 0.0

    for activation_name, seed_runs in all_results.items():
        tabarena_aucs = [run[-1].get("tabarena_roc_auc") for run in seed_runs]
        if any(v is not None for v in tabarena_aucs):
            final_aucs = [v if v is not None else float("nan") for v in tabarena_aucs]
        else:
            final_aucs = [run[-1].get("roc_auc", float("nan")) for run in seed_runs]

        act_mean = float(np.nanmean(final_aucs))
        act_std = float(np.nanstd(final_aucs))

        display_name = format_activation_display_name(activation_name)
        if activation_name == BASELINE_ACTIVATION:
            baseline_mean = act_mean
            display_name = f"{display_name} (Baseline)"

        names.append(display_name)
        means.append(act_mean)
        stds.append(act_std)

    # Sort by descending mean
    order = np.argsort(means)[::-1]
    names = [names[i] for i in order]
    means = [means[i] for i in order]
    stds = [stds[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("deep", len(names))
    x_positions = np.arange(len(names))
    bars = ax.bar(x_positions, means, yerr=stds, capsize=5, color=palette, edgecolor="black", linewidth=0.8)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([_wrap_label(name) for name in names], ha="center")

    # Find GELU baseline color to match the line
    baseline_color = "black"
    for bar, name in zip(bars, names, strict=True):
        if "(Baseline)" in name:
            baseline_color = bar.get_facecolor()

    # Highlight GELU baseline as a horizontal reference (in front of bars)
    ax.axhline(
        baseline_mean,
        color=baseline_color,
        linestyle="--",
        linewidth=2,
        zorder=2,
        label=f"{format_activation_display_name(BASELINE_ACTIVATION)} (Baseline)",
    )
    ax.legend(fontsize=12, loc="upper right")

    ax.set_xlabel("Activation Function", fontsize=14)
    ax.set_ylabel("ROC-AUC", fontsize=14)
    ax.set_title("ROC-AUC by Activation Function", fontsize=18)
    y_min = min(m - s for m, s in zip(means, stds, strict=False))
    y_max = max(m + s for m, s in zip(means, stds, strict=False))
    y_range = y_max - y_min
    # Add 50% padding at the bottom, 80% at the top (to fit text labels)
    ax.set_ylim(bottom=max(0.5, y_min - y_range * 0.5), top=min(1.0, y_max + y_range * 0.8))
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            mean + std + 0.002,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="semibold",
            zorder=4,
        )

    if is_mock:
        _add_watermark(ax)

    fig.tight_layout()
    _save_plot(fig, output_dir, "bar_chart_roc_auc")
    plt.close(fig)


def plot_learning_curves_best(
    all_results: dict[str, list[list[dict]]],
    output_dir: str = "plots",
    is_mock: bool = False,
) -> None:
    """Learning curves for GELU baseline + top-2 best-performing activations.

    Shows pre-training steps vs. ROC-AUC with ±1 std shaded
    uncertainty bands across seeds.

    Args:
        all_results: Dict mapping activation name → list of seed runs.
        output_dir: Directory for output files.
        is_mock: If True, add a watermark indicating mock data.
    """
    sns.set_theme(style=PLOT_STYLE)
    top2 = _get_top_activations(all_results, n=2)
    show_activations = [BASELINE_ACTIVATION, *top2]

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("colorblind", len(show_activations))

    for idx, activation_name in enumerate(show_activations):
        if activation_name not in all_results:
            continue
        seed_runs = all_results[activation_name]
        steps = [entry["step"] for entry in seed_runs[0]]
        roc_aucs = []
        for run in seed_runs:
            run_aucs = [entry.get("roc_auc", float("nan")) for entry in run]
            roc_aucs.append(run_aucs)

        roc_aucs_arr = np.array(roc_aucs)
        mean_auc = np.nanmean(roc_aucs_arr, axis=0)
        std_auc = np.nanstd(roc_aucs_arr, axis=0)

        label = format_activation_display_name(activation_name)
        if activation_name == BASELINE_ACTIVATION:
            label = f"{label} (Baseline)"
        label = f"{label}"

        linestyle = "--" if activation_name == BASELINE_ACTIVATION else "-"
        ax.plot(steps, mean_auc, label=label, color=palette[idx], linewidth=2, linestyle=linestyle)
        ax.fill_between(steps, mean_auc - std_auc, mean_auc + std_auc, alpha=0.15, color=palette[idx])

    ax.set_xlabel("Pre-training Step", fontsize=14)
    ax.set_ylabel("Average ROC-AUC", fontsize=14)
    ax.set_title("Learning Curves: Baseline vs. Best Variants", fontsize=18)
    ax.legend(fontsize=11, loc="lower right")
    ax.tick_params(axis="both", which="major", labelsize=11, labelcolor="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("semibold")
    ax.grid(True, alpha=0.3)

    # Force the main plot x-axis to the requested range (500–5000), but
    # clamp to the actual data range to avoid empty plots when data is smaller.
    try:
        steps_arr_all = np.array(steps, dtype=float)
        global_min = float(np.nanmin(steps_arr_all))
        global_max = float(np.nanmax(steps_arr_all))
        xmin_main = max(global_min, 500.0)
        xmax_main = min(global_max, 5000.0)
        if xmin_main < xmax_main:
            ax.set_xlim(xmin_main, xmax_main)
    except Exception:
        pass

    # Highlight the dynamic (early) training interval with a shaded
    # rectangle on the main axes and a separate zoom panel to the right.
    # Use a fixed zoom window from 500 to 1500 (explicit indices), as requested.
    # Convert `steps` to a numeric array and locate start/end indices robustly.
    start_step = 750
    end_step = 2000
    try:
        steps_arr = np.array(steps, dtype=float)
        start_idx = int(np.searchsorted(steps_arr, start_step, side="left"))
        end_idx = int(np.searchsorted(steps_arr, end_step, side="right"))
        # Clamp indices to valid range and ensure at least one element
        start_idx = max(0, min(start_idx, len(steps_arr) - 1))
        end_idx = max(start_idx + 1, min(end_idx, len(steps_arr)))
        zoom_steps = steps[start_idx:end_idx]
    except Exception:
        # Fallback: use the original heuristic if something goes wrong
        start_idx = 0
        end_idx = max(3, int(len(steps) * 0.2))
        zoom_steps = steps[start_idx:end_idx]

    # shaded vertical band on the main axes
    xmin = zoom_steps[0]
    xmax = zoom_steps[-1]
    ymin, ymax = ax.get_ylim()
    ax.axvspan(xmin, xmax, ymin=0, ymax=1.0, color="#dddddd", alpha=0.35, zorder=0)

    # separate zoom panel (small axes on the lower-right)
    # Create a larger inset and center it inside the main axes. Move it
    # slightly upward to avoid overlapping the x-axis tick labels and legend
    # while keeping it visually central.
    # Make the inset larger and move it upward (centered horizontally,
    # higher vertically) so it no longer overlaps the x-axis tick labels
    # and legend. Reduce tick label size inside the inset for clarity.
    ax_zoom = inset_axes(
        ax,
        width="50%",
        height="50%",
        bbox_to_anchor=(0.25, 0.05,0.8,0.8),
        bbox_transform=ax.transAxes,
        loc="center",
    )
    ax_zoom.set_zorder(10)
    ax_zoom.patch.set_alpha(0.98)
    ax_zoom.set_facecolor("white")
    for spine in ax_zoom.spines.values():
        spine.set_edgecolor("#444444")
        spine.set_linewidth(0.6)
    # Reduce inset tick label size and set readable x tick labels
    ax_zoom.tick_params(axis="both", which="major", labelsize=7)
    # show three x-ticks: start, middle, end of the zoom window, formatted as integers
    try:
        # Place tick marks every 250 steps within the zoom window (user data
        # is at 250-step resolution). Align ticks to multiples of the
        # interval so labels read like 500, 750, 1000, ...
        tick_interval = 250
        start_val = float(xmin)
        end_val = float(xmax)
        first_tick = int(np.ceil(start_val / tick_interval) * tick_interval)
        candidate_ticks = np.arange(first_tick, end_val + 1, tick_interval)
        # Ensure ticks lie within the exact zoom bounds (inclusive)
        ticks = [t for t in candidate_ticks if t >= start_val - 1e-8 and t <= end_val + 1e-8]
        if len(ticks) == 0:
            # fallback to start/mid/end
            zs = list(zoom_steps)
            if len(zs) == 0:
                raise ValueError("zoom_steps empty")
            if len(zs) <= 3:
                ticks = zs
            else:
                idxs = np.linspace(0, len(zs) - 1, 3, dtype=int)
                ticks = list(np.array(zs)[idxs])

        ax_zoom.set_xticks(ticks)
        ax_zoom.set_xticklabels([f"{int(round(v))}" for v in ticks], fontsize=9, fontweight="semibold")
    except Exception:
        pass
    for idx, activation_name in enumerate(show_activations):
        if activation_name not in all_results:
            continue
        seed_runs = all_results[activation_name]
        roc_aucs = [[entry.get("roc_auc", float("nan")) for entry in run] for run in seed_runs]
        roc_aucs_arr = np.array(roc_aucs)
        # Slice per-run arrays to the chosen start/end indices for the zoom window
        mean_auc_zoom = np.nanmean(roc_aucs_arr[:, start_idx:end_idx], axis=0)
        std_auc_zoom = np.nanstd(roc_aucs_arr[:, start_idx:end_idx], axis=0)

        # interpolate to higher resolution for a smoother, denser zoom view
        orig_x = np.array(zoom_steps, dtype=float)
        if orig_x.size >= 2:
            dense_x = np.linspace(orig_x[0], orig_x[-1], max(120, orig_x.size * 10))
            mean_interp = np.interp(dense_x, orig_x, mean_auc_zoom)
            std_interp = np.interp(dense_x, orig_x, std_auc_zoom)
        else:
            dense_x = orig_x
            mean_interp = mean_auc_zoom
            std_interp = std_auc_zoom

        linestyle = "--" if activation_name == BASELINE_ACTIVATION else "-"
        ax_zoom.plot(dense_x, mean_interp, color=palette[idx], linewidth=1.6, linestyle=linestyle)
        ax_zoom.fill_between(dense_x, mean_interp - std_interp, mean_interp + std_interp, alpha=0.12, color=palette[idx])

    ax_zoom.set_xlim(xmin, xmax)
    # tighten y-limits to the data in the zoom window for better visibility
    all_zoom_vals = []
    for activation_name in show_activations:
        if activation_name not in all_results:
            continue
        seed_runs = all_results[activation_name]
        roc_aucs = [ [entry.get("roc_auc", float("nan")) for entry in run][start_idx:end_idx] for run in seed_runs ]
        all_zoom_vals.extend(np.concatenate(roc_aucs).tolist())
    # Force the inset y-limits to the requested range for consistent
    # comparison and to avoid overlap with main plot tick labels.
    ax_zoom.set_ylim(0.64, 0.70)

    ax_zoom.set_title("Early dynamic phase", fontsize=11, fontweight="semibold")
    ax_zoom.tick_params(axis="both", which="major", labelsize=9)
    for label in ax_zoom.get_xticklabels() + ax_zoom.get_yticklabels():
        label.set_fontweight("semibold")

    # Draw a rectangle on the main axes outlining the zoomed region and
    # connect it to the inset with dashed connector lines. `mark_inset`
    # uses the inset axes limits to compute the rectangle corners.
    try:
        mark_inset(ax, ax_zoom, loc1=1, loc2=3, fc="none", ec="#444444", linestyle=(0, (5, 3)), linewidth=0.4)
    except Exception:
        pass

    if is_mock:
        _add_watermark(ax)

    fig.tight_layout()
    _save_plot(fig, output_dir, "learning_curves_best")
    plt.close(fig)


def plot_depth_scaling(
    ablation_data: dict,
    output_dir: str = "plots",
    is_mock: bool = False,
) -> None:
    """Line plot of model depth vs. TabArena Score for best variants.

    Args:
        ablation_data: Architecture ablation data dict with ``"depth"`` key
            containing ``"layers"`` and ``"results"`` sub-keys.
        output_dir: Directory for output files.
        is_mock: If True, add a watermark indicating mock data.
    """
    sns.set_theme(style=PLOT_STYLE)
    depth_data = ablation_data["depth"]
    layers = depth_data["layers"]

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("colorblind", len(depth_data["results"]))
    markers = ["o", "s", "D", "^", "v"]

    for idx, (name, values) in enumerate(depth_data["results"].items()):
        mean = np.array(values["mean"])
        std = np.array(values["std"])
        label = format_activation_display_name(name)
        if name == BASELINE_ACTIVATION:
            label = f"{label} (Baseline)"

        linestyle = "--" if name == BASELINE_ACTIVATION else "-"
        marker = markers[idx % len(markers)]

        ax.plot(
            layers,
            mean,
            label=label,
            color=palette[idx],
            linewidth=2,
            linestyle=linestyle,
            marker=marker,
            markersize=8,
        )
        ax.fill_between(
            layers,
            mean - std,
            mean + std,
            color=palette[idx],
            alpha=0.2,
        )

    ax.set_xlabel("Number of Transformer Layers", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Architecture Scaling: Depth", fontsize=14)
    ax.set_xticks(layers)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if is_mock:
        _add_watermark(ax)

    fig.tight_layout()
    _save_plot(fig, output_dir, "depth_scaling")
    plt.close(fig)


def plot_width_scaling(
    ablation_data: dict,
    output_dir: str = "plots",
    is_mock: bool = False,
) -> None:
    """Line plot of FFN hidden dim (log scale) vs. ROC-AUC.

    Args:
        ablation_data: Architecture ablation data dict with ``"width"`` key
            containing ``"hidden_dims"`` and ``"results"`` sub-keys.
        output_dir: Directory for output files.
        is_mock: If True, add a watermark indicating mock data.
    """
    sns.set_theme(style=PLOT_STYLE)
    width_data = ablation_data["width"]
    hidden_dims = width_data["hidden_dims"]

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("colorblind", len(width_data["results"]))

    for idx, (name, values) in enumerate(width_data["results"].items()):
        mean = np.array(values["mean"])
        std = np.array(values["std"])
        label = format_activation_display_name(name)
        if name == BASELINE_ACTIVATION:
            label = f"{label} (Baseline)"

        linestyle = "--" if name == BASELINE_ACTIVATION else "-"

        ax.plot(hidden_dims, mean, label=label, color=palette[idx], linewidth=2, linestyle=linestyle, marker="o")
        ax.fill_between(hidden_dims, mean - std, mean + std, alpha=0.15, color=palette[idx])

    ax.set_xscale("log", base=2)
    ax.set_xticks(hidden_dims)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("FFN Hidden Dimension", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Architecture Scaling: Width", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if is_mock:
        _add_watermark(ax)

    fig.tight_layout()
    _save_plot(fig, output_dir, "width_scaling")
    plt.close(fig)


def _normalize_activation_name(name: str) -> str:
    """Normalize activation names by stripping unrestricted markers."""
    return name.rstrip("*").strip()


def collect_relative_improvements(
    results_dir: Path | str,
    baseline_activation: str,
    variant_activations: list[str],
    metric: str,
    split: str = "all",
) -> dict[str, list[float]]:
    """Compute per-dataset relative improvement (%) for the baseline and each variant.

    Args:
        results_dir: Base results directory (e.g. `results/`).
        baseline_activation: Baseline activation name (e.g. `"gelu"`).
        variant_activations: Activation names to compare against the baseline.
        metric: `"roc_auc"` or `"log_loss"`.
        split: Which datasets to include (`"all"`, `"id"`, or `"ood"`).

    Returns:
        Mapping of activation name to a list of per-dataset relative
        improvement percentages. The baseline is included as an all-zero
        reference series, matching the baseline-vs-itself convention used
        in relative-improvement plots.
    """
    higher_is_better = METRICS[metric]
    baseline_scores = aggregate_seed_scores(load_tabarena_scores(results_dir, baseline_activation, metric, split))

    series = {baseline_activation: [0.0] * len(baseline_scores)}
    for variant in variant_activations:
        variant_scores = aggregate_seed_scores(load_tabarena_scores(results_dir, variant, metric, split))
        improvements = compute_relative_improvement(baseline_scores, variant_scores, higher_is_better)
        series[variant] = [100 * v for v in improvements.values()]
    return series


def plot_relative_improvement(
    series: dict[str, list[float]],
    baseline_activation: str,
    metric: str,
    output_dir: str = "plots",
) -> None:
    """Box-and-whisker plot of per-dataset relative improvement over the baseline.

    Each box shows the distribution across datasets; individual dots are
    jittered per-dataset scores; the diamond marks the mean; the label
    below each box shows mean ± std.  Every activation gets a unique color
    so they are easy to distinguish on a poster.

    Args:
        series: Mapping of activation name to per-dataset relative improvement
            percentages, as returned by `collect_relative_improvements`.
        baseline_activation: Name of the baseline activation, used to
            highlight it differently from the variants.
        metric: `"roc_auc"` or `"log_loss"`, used for axis/title labeling.
        output_dir: Directory to save the figure (PNG + SVG).
    """
    sns.set_theme(style="whitegrid")

    raw_labels = list(series.keys())
    data = [series[label] for label in raw_labels]

    group_palette = {
        "Gated": "#55A868",
        "Smooth": "#4C72B0",
        "ReLU-family": "#DD8452",
        "Other": "#7E6148",
    }

    normalized_labels = [_normalize_activation_name(name) for name in raw_labels]

    def _group_for_activation(name: str) -> str:
        if "swiglu" in name or "bilinear" in name:
            return "Gated"
        if name in {"gelu", "swish"}:
            return "Smooth"
        if "relu" in name:
            return "ReLU-family"
        return "Other"

    group_labels = [_group_for_activation(name) for name in normalized_labels]
    means_for_sort = [float(np.mean(v)) if v else float("nan") for v in data]

    group_means: dict[str, float] = {}
    for group in sorted(set(group_labels)):
        idxs = [i for i, g in enumerate(group_labels) if g == group]
        group_means[group] = float(np.nanmean([means_for_sort[i] for i in idxs]))

    group_order = sorted(group_means, key=lambda g: group_means[g])
    order: list[int] = []
    for group in group_order:
        group_indices = [i for i, g in enumerate(group_labels) if g == group]
        order.extend(sorted(group_indices, key=lambda i: means_for_sort[i]))

    raw_labels = [raw_labels[i] for i in order]
    data = [data[i] for i in order]
    group_labels = [group_labels[i] for i in order]

    display_labels = [format_activation_display_name(name) for name in raw_labels]
    palette = [group_palette[group] for group in group_labels]

    fig, ax = plt.subplots(figsize=(max(8, len(raw_labels) * 1.3), 6))

    bp = ax.boxplot(
        data,
        tick_labels=[_wrap_label(dl) for dl in display_labels],
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )
    ax.set_xticklabels([_wrap_label(dl) for dl in display_labels], rotation=0, ha="center")
    ax.tick_params(axis="x", labelsize=9)
    for label in ax.get_xticklabels():
        label.set_fontweight("semibold")
    for patch, color in zip(bp["boxes"], palette, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    rng = np.random.default_rng(0)
    for i, (values, color) in enumerate(zip(data, palette, strict=False), start=1):
        jitter = rng.normal(0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            color=color,
            alpha=0.6,
            s=22,
            zorder=2,
            edgecolors="none",
        )

    means = np.array([np.mean(v) if v else float("nan") for v in data])
    stds = np.array([np.std(v) if v else float("nan") for v in data])
    ax.scatter(range(1, len(raw_labels) + 1), means, marker="D", color="black", s=60, zorder=3, label="Mean")

    for i, (mean, std) in enumerate(zip(means, stds, strict=False), start=1):
        ax.annotate(
            f"{mean:+.2f} ± {std:.1f}%",
            xy=(i, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -28),
            textcoords="offset points",
            ha="center",
            va="top",
            rotation=0,
            fontsize=9,
            fontweight="semibold",
        )

    baseline_display = format_activation_display_name(baseline_activation)
    metric_label = " Average ROC-AUC" if metric == "roc_auc" else "Log Loss"
    baseline_line_color = "#000000"
    ax.axhline(0, color=baseline_line_color, linewidth=1.5, linestyle="--", label=f"{baseline_display} baseline")
    ax.set_ylabel(f"Relative improvement over {baseline_display} ({metric_label})(%)", fontsize=11.5)
    ax.set_title(f"Per-dataset Relative Improvement", fontsize=18)
    ax.grid(False)
    ax.tick_params(axis="both", which="major", labelsize=9, labelcolor="black")
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("semibold")

    group_ranges: list[tuple[int, int, str]] = []
    start_idx = 1
    current_group = group_labels[0] if group_labels else ""
    for idx, group in enumerate(group_labels[1:], start=2):
        if group != current_group:
            group_ranges.append((start_idx, idx - 1, current_group))
            start_idx = idx
            current_group = group
    if group_labels:
        group_ranges.append((start_idx, len(group_labels), current_group))

    y_min, y_max = ax.get_ylim()
    y_padding = (y_max - y_min) * 0.02
    for start, end, group in group_ranges:
        rect = mpatches.Rectangle(
            (start - 0.5, y_min - y_padding),
            width=end - start + 1,
            height=(y_max - y_min) + 2 * y_padding,
            fill=False,
            edgecolor="#444444",
            linestyle="--",
            linewidth=1.0,
            alpha=0.8,
            zorder=1,
        )
        ax.add_patch(rect)

    group_handles = [mpatches.Patch(color=group_palette[group], label=group) for group in group_order if group in group_labels]
    ax.legend(handles=group_handles, title="Activation group", fontsize=9, title_fontsize=10, loc="upper left")

    has_unrestricted = any(name.endswith("*") for name in display_labels)
    if has_unrestricted:
        fig.subplots_adjust(bottom=0.12)
        fig.text(
            0.92, 0.5, "* = full hidden width (no reduction for parameter parity)",
            fontsize=10, fontstyle="italic", color="#555555",
            ha="right", va="center", rotation=90,
        )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    basename = f"relative_improvement_{metric}"
    png_path = out / f"{basename}.png"
    svg_path = out / f"{basename}.svg"
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"  Saved: {png_path}, {svg_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate evaluation plots for the ablation study")
    parser.add_argument("--mock", action="store_true", help="Use mock data from results_mock/")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory for plot output (default: plots)")
    return parser.parse_args()


def main() -> None:
    """Entry point: load data and generate all plots."""
    args = parse_args()

    import os

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if args.mock:
        results_dir = "results_mock"
    else:
        workspace_dir = os.environ.get("WORKSPACE_DIR", "/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data")
        results_dir = os.path.join(workspace_dir, "results")

    output_dir = args.output_dir

    results = load_all_results(results_dir)
    print(f"Found activations: {list(results.keys())}")
    for name, runs in results.items():
        print(f"  {name}: {len(runs)} seeds")

    print(f"\nGenerating plots → {output_dir}/")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Issue #9 plots
    print("\n--- Issue #9: Bar chart with error bars ---")
    plot_bar_chart_with_error_bars(results, output_dir=output_dir, is_mock=args.mock)

    print("--- Issue #9: Learning curves (best variants) ---")
    plot_learning_curves_best(results, output_dir=output_dir, is_mock=args.mock)

    # Architecture ablation plots need separate data
    ablation_path = Path(results_dir) / "architecture_ablation.json"
    if ablation_path.exists():
        with open(ablation_path) as f:
            ablation_data = json.load(f)

        print("--- Issue #9: Depth scaling ---")
        plot_depth_scaling(ablation_data, output_dir=output_dir, is_mock=args.mock)

        print("--- Issue #9: Width scaling ---")
        plot_width_scaling(ablation_data, output_dir=output_dir, is_mock=args.mock)
    else:
        print(f"\nWarning: {ablation_path} not found. Skipping depth/width plots.")
        print("Run 'python scripts/data/generate_mock_data.py' first to generate mock data.")

    # Variant comparison (relative improvement) plot — always run as part of the full report
    print("--- Variant comparison: Relative improvement over baseline ---")
    results_path = Path(results_dir)
    variant_activations = sorted(d.name for d in results_path.iterdir() if d.is_dir() and d.name != BASELINE_ACTIVATION)
    series = collect_relative_improvements(results_dir, BASELINE_ACTIVATION, variant_activations, "roc_auc")
    plot_relative_improvement(series, BASELINE_ACTIVATION, "roc_auc", output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
