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
import importlib.util
import json
import sys
import textwrap
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nanotabpfn.analysis import format_activation_display_name

# Publication-quality defaults
PLOT_DPI = 300
PLOT_STYLE = "whitegrid"
WATERMARK_TEXT = "MOCK DATA — NOT FROM REAL EXPERIMENTS"
BASELINE_ACTIVATION = "gelu"


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
        fontsize=16,
        color="red",
        alpha=0.35,
        ha="center",
        va="center",
        rotation=-30,
        fontweight="bold",
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

    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
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
    ax.legend(fontsize=11, loc="upper right")

    ax.set_xlabel("Activation Function", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("ROC-AUC by Activation Function", fontsize=14)
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
            fontsize=10,
            fontweight="bold",
            zorder=4,
            path_effects=[path_effects.withStroke(linewidth=3, foreground="white")],
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
    uncertainty bands across seeds. Includes an inset axis to zoom in
    on the early dynamic phase.

    Args:
        all_results: Dict mapping activation name → list of seed runs.
        output_dir: Directory for output files.
        is_mock: If True, add a watermark indicating mock data.
    """
    sns.set_theme(style=PLOT_STYLE)
    top2 = _get_top_activations(all_results, n=2)
    show_activations = [BASELINE_ACTIVATION, *top2]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create inset axes for the zoom-in (x, y, width, height) relative to main axes
    # Placed in the lower-right to avoid the lower-left legend and the top-right curves
    axins = ax.inset_axes((0.40, 0.08, 0.58, 0.60))

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

        linestyle = "--" if activation_name == BASELINE_ACTIVATION else "-"

        # Plot on main axis
        ax.plot(steps, mean_auc, label=label, color=palette[idx], linewidth=2, linestyle=linestyle)
        ax.fill_between(steps, mean_auc - std_auc, mean_auc + std_auc, alpha=0.15, color=palette[idx])

        # Plot the exact same lines on the inset axis
        axins.plot(steps, mean_auc, color=palette[idx], linewidth=2, linestyle=linestyle)
        axins.fill_between(steps, mean_auc - std_auc, mean_auc + std_auc, alpha=0.15, color=palette[idx])

    # Configure main axis
    ax.set_xlim(min(steps), max(steps))
    ax.set_xlabel("Pre-training Step", fontsize=12)
    ax.set_ylabel("Average ROC-AUC", fontsize=12)
    ax.set_title("Learning Curves: Baseline vs. Two Best Variants (proxy datasets)", fontsize=14)
    ax.legend(fontsize=11, loc="lower left")  # Moved to lower left
    ax.grid(True, alpha=0.3)

    # Configure inset axis (Early dynamic phase zoom)
    axins.set_xlim(750, 2000)
    axins.set_ylim(0.64, 0.70)
    axins.set_title("Early dynamic phase", fontsize=11, fontweight="bold")
    axins.grid(True, alpha=0.3)

    # Add connecting lines between the zoomed box and the inset plot
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="gray", alpha=0.8, linestyle="--")

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


def run_variant_comparison_plot(
    results_dir: str, output_dir: str, baseline: str = "gelu", metric: str = "roc_auc"
) -> None:
    """Dynamically import and run the variant-comparison plotting script.

    This avoids a hard import and keeps the original script as the source
    of truth for the relative-improvement plot.
    """
    mod_path = Path(__file__).parents[1] / "plot_variant_comparison.py"
    if not mod_path.exists():
        print(f"Variant comparison script not found: {mod_path}")
        return

    spec = importlib.util.spec_from_file_location("plot_variant_comparison", str(mod_path))
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    # Discover variants (exclude baseline)
    results_path = Path(results_dir)
    variant_activations = sorted(d.name for d in results_path.iterdir() if d.is_dir() and d.name != baseline)

    series = module.collect_relative_improvements(results_dir, baseline, variant_activations, metric)
    module.plot_relative_improvement(series, baseline, metric, output_dir)


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
    run_variant_comparison_plot(results_dir, output_dir, baseline=BASELINE_ACTIVATION, metric="roc_auc")

    print("\nDone.")


if __name__ == "__main__":
    main()
