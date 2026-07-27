"""Plot results from ablation experiments.

Reads metrics.jsonl files from results/ and produces:
1. Training step vs. ROC-AUC per activation (mean ± std across seeds)
2. Final ROC-AUC box plot across activations

Issue #9 plots (with ``--mock`` flag for mock data):
3. Bar chart with error bars: Normalized ROC-AUC per activation
4. Learning curves for GELU baseline + top-2 best variants
5. Architecture scaling (depth): layers vs. TabArena Score
6. Architecture scaling (width): FFN hidden dim (log) vs. Normalized ROC-AUC

Issue #26 architecture ablation plots (with ``--arch`` flag):
7. Layers scaling: transformer depth vs. ROC-AUC per activation
8. Hidden scaling: MLP hidden dim (log2) vs. ROC-AUC per activation
9. Embedding scaling: embedding size vs. ROC-AUC per activation
"""

import argparse
import json
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Publication-quality defaults
PLOT_DPI = 300
PLOT_STYLE = "whitegrid"
WATERMARK_TEXT = "MOCK DATA — NOT FROM REAL EXPERIMENTS"
BASELINE_ACTIVATION = "gelu"

ACTIVATION_DISPLAY_NAMES = {
    "gelu": "GELU",
    "relu": "ReLU",
    "swish": "Swish",
    "leaky_relu": "Leaky ReLU",
    "prelu": "PReLU",
    "swiglu": "SwiGLU",
    "bilinear": "Bilinear",
}


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


def load_arch_results(
    results_dir: str = "results_arch",
    sweep_axis: str = "layers",
) -> dict[str, dict[str, list[float]]]:
    """Load architecture ablation results for a single sweep axis.

    Reads the ``results_arch/{sweep_axis}/{activation}/e{E}_h{H}_l{L}/seed_{N}/``
    directory structure and extracts the final ROC-AUC from each run's ``metrics.jsonl``.

    Args:
        results_dir: Root results_arch directory.
        sweep_axis: One of ``'layers'``, ``'hidden'``, or ``'embedding'``.

    Returns:
        Dict mapping activation name → dict with keys:
        - ``'values'``: sorted list of unique parameter values (int)
        - ``'means'``: list of mean ROC-AUC per value
        - ``'stds'``: list of std ROC-AUC per value
    """
    axis_dir = Path(results_dir) / sweep_axis
    if not axis_dir.exists():
        raise FileNotFoundError(f"Architecture results not found: {axis_dir}")

    # Parse: {activation}/{arch_tag}/seed_{N}/metrics.jsonl
    # arch_tag = e{E}_h{H}_l{L}
    raw: dict[str, dict[int, list[float]]] = {}  # activation -> {param_value -> [final_aucs]}

    for activation_dir in sorted(axis_dir.iterdir()):
        if not activation_dir.is_dir():
            continue
        act_name = activation_dir.name
        if act_name not in raw:
            raw[act_name] = {}

        for arch_dir in sorted(activation_dir.iterdir()):
            if not arch_dir.is_dir():
                continue
            # Parse arch tag: e128_h256_l3
            tag = arch_dir.name
            parts = tag.split("_")
            tag_dict = {}
            for part in parts:
                if part.startswith("e"):
                    tag_dict["e"] = int(part[1:])
                elif part.startswith("h"):
                    tag_dict["h"] = int(part[1:])
                elif part.startswith("l"):
                    tag_dict["l"] = int(part[1:])

            # Determine the swept parameter value
            param_map = {"layers": "l", "hidden": "h", "embedding": "e"}
            param_val = tag_dict.get(param_map.get(sweep_axis, ""), 0)

            for seed_dir in sorted(arch_dir.iterdir()):
                if not seed_dir.is_dir():
                    continue
                metrics_file = seed_dir / "metrics.jsonl"
                if not metrics_file.exists():
                    continue
                # Read final metric entry
                last_line = None
                with open(metrics_file) as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            last_line = line
                if last_line:
                    entry = json.loads(last_line)
                    auc = entry.get("roc_auc", float("nan"))
                    if param_val not in raw[act_name]:
                        raw[act_name][param_val] = []
                    raw[act_name][param_val].append(auc)

    # Aggregate into sorted arrays
    result: dict[str, dict[str, list[float]]] = {}
    for act_name, value_map in raw.items():
        sorted_values = sorted(value_map.keys())
        means = [float(np.nanmean(value_map[v])) for v in sorted_values]
        stds = [float(np.nanstd(value_map[v])) for v in sorted_values]
        result[act_name] = {
            "values": [float(v) for v in sorted_values],
            "means": means,
            "stds": stds,
        }

    return result


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

        display_name = ACTIVATION_DISPLAY_NAMES.get(activation_name, activation_name.upper())
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
    bars = ax.bar(names, means, yerr=stds, capsize=5, color=palette, edgecolor="black", linewidth=0.8)

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
        label=f"{ACTIVATION_DISPLAY_NAMES.get(BASELINE_ACTIVATION, BASELINE_ACTIVATION.upper())} (Baseline)",
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

        label = ACTIVATION_DISPLAY_NAMES.get(activation_name, activation_name.upper())
        if activation_name == BASELINE_ACTIVATION:
            label = f"{label} (Baseline)"
        label = f"{label} (n={len(seed_runs)})"

        linestyle = "--" if activation_name == BASELINE_ACTIVATION else "-"
        ax.plot(steps, mean_auc, label=label, color=palette[idx], linewidth=2, linestyle=linestyle)
        ax.fill_between(steps, mean_auc - std_auc, mean_auc + std_auc, alpha=0.15, color=palette[idx])

    ax.set_xlabel("Pre-training Step", fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title("Learning Curves: Baseline vs. Best Variants", fontsize=14)
    ax.legend(fontsize=11, loc="lower right")
    ax.grid(True, alpha=0.3)

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
        label = ACTIVATION_DISPLAY_NAMES.get(name, name.upper())
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
        label = ACTIVATION_DISPLAY_NAMES.get(name, name.upper())
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


# ---------------------------------------------------------------------------
# Issue #26 architecture ablation plots (read from results_arch/)
# ---------------------------------------------------------------------------


def _plot_arch_scaling(
    arch_data: dict[str, dict[str, list[float]]],
    sweep_axis: str,
    xlabel: str,
    title: str,
    output_basename: str,
    output_dir: str = "plots",
    log_x: bool = False,
    is_mock: bool = False,
) -> None:
    """Generic architecture scaling plot for one sweep axis.

    Args:
        arch_data: Output of :func:`load_arch_results` for the given axis.
        sweep_axis: The sweep axis name (for labelling).
        xlabel: X-axis label.
        title: Plot title.
        output_basename: Filename base (without extension).
        output_dir: Directory for output files.
        log_x: If True, use log2 x-axis scaling.
        is_mock: If True, add a watermark indicating mock data.
    """
    sns.set_theme(style=PLOT_STYLE)
    fig, ax = plt.subplots(figsize=(8, 6))

    activations = sorted(arch_data.keys(), key=lambda k: (k != BASELINE_ACTIVATION, k))
    palette = sns.color_palette("colorblind", len(activations))
    markers = ["o", "s", "D", "^", "v"]

    for idx, act_name in enumerate(activations):
        data = arch_data[act_name]
        x_vals = np.array(data["values"])
        mean = np.array(data["means"])
        std = np.array(data["stds"])

        label = ACTIVATION_DISPLAY_NAMES.get(act_name, act_name.upper())
        if act_name == BASELINE_ACTIVATION:
            label = f"{label} (Baseline)"

        linestyle = "--" if act_name == BASELINE_ACTIVATION else "-"
        marker = markers[idx % len(markers)]

        ax.plot(
            x_vals,
            mean,
            label=label,
            color=palette[idx],
            linewidth=2,
            linestyle=linestyle,
            marker=marker,
            markersize=8,
        )
        ax.fill_between(x_vals, mean - std, mean + std, color=palette[idx], alpha=0.2)

    if log_x:
        ax.set_xscale("log", base=2)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
        # Set ticks at the actual data points
        all_values = set()
        for data in arch_data.values():
            all_values.update(int(v) for v in data["values"])
        ax.set_xticks(sorted(all_values))
    else:
        all_values = set()
        for data in arch_data.values():
            all_values.update(int(v) for v in data["values"])
        ax.set_xticks(sorted(all_values))

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("ROC-AUC", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    if is_mock:
        _add_watermark(ax)

    fig.tight_layout()
    _save_plot(fig, output_dir, output_basename)
    plt.close(fig)


def plot_arch_layers_scaling(
    arch_data: dict[str, dict[str, list[float]]],
    output_dir: str = "plots",
    is_mock: bool = False,
) -> None:
    """Plot transformer depth vs. ROC-AUC for the architecture ablation.

    Args:
        arch_data: Output of ``load_arch_results(sweep_axis='layers')``.
        output_dir: Directory for output files.
        is_mock: If True, add a watermark indicating mock data.
    """
    _plot_arch_scaling(
        arch_data,
        sweep_axis="layers",
        xlabel="Number of Transformer Layers",
        title="Architecture Ablation: Depth Scaling",
        output_basename="arch_layers_scaling",
        output_dir=output_dir,
        log_x=False,
        is_mock=is_mock,
    )


def plot_arch_hidden_scaling(
    arch_data: dict[str, dict[str, list[float]]],
    output_dir: str = "plots",
    is_mock: bool = False,
) -> None:
    """Plot MLP hidden dim vs. ROC-AUC for the architecture ablation.

    Args:
        arch_data: Output of ``load_arch_results(sweep_axis='hidden')``.
        output_dir: Directory for output files.
        is_mock: If True, add a watermark indicating mock data.
    """
    _plot_arch_scaling(
        arch_data,
        sweep_axis="hidden",
        xlabel="MLP Hidden Dimension",
        title="Architecture Ablation: Hidden Width Scaling",
        output_basename="arch_hidden_scaling",
        output_dir=output_dir,
        log_x=True,
        is_mock=is_mock,
    )


def plot_arch_embedding_scaling(
    arch_data: dict[str, dict[str, list[float]]],
    output_dir: str = "plots",
    is_mock: bool = False,
) -> None:
    """Plot embedding size vs. ROC-AUC for the architecture ablation.

    Args:
        arch_data: Output of ``load_arch_results(sweep_axis='embedding')``.
        output_dir: Directory for output files.
        is_mock: If True, add a watermark indicating mock data.
    """
    _plot_arch_scaling(
        arch_data,
        sweep_axis="embedding",
        xlabel="Embedding Size",
        title="Architecture Ablation: Embedding Scaling",
        output_basename="arch_embedding_scaling",
        output_dir=output_dir,
        log_x=False,
        is_mock=is_mock,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate evaluation plots for the ablation study")
    parser.add_argument("--mock", action="store_true", help="Use mock data from results_mock/")
    parser.add_argument("--output-dir", type=str, default="plots", help="Directory for plot output (default: plots)")
    parser.add_argument("--arch", action="store_true", help="Generate Issue #26 architecture ablation plots")
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

    # ─── Issue #26: Architecture ablation plots ───
    if args.arch:
        arch_results_dir = (
            os.path.join(
                os.environ.get("WORKSPACE_DIR", "."),
                "results_arch",
            )
            if not args.mock
            else "results_arch_mock"
        )

        for axis, plot_fn in [
            ("layers", plot_arch_layers_scaling),
            ("hidden", plot_arch_hidden_scaling),
            ("embedding", plot_arch_embedding_scaling),
        ]:
            try:
                arch_data = load_arch_results(arch_results_dir, sweep_axis=axis)
                print(f"\n--- Issue #26: {axis} scaling ---")
                plot_fn(arch_data, output_dir=output_dir, is_mock=args.mock)
            except FileNotFoundError:
                print(f"\nWarning: No {axis} sweep results in {arch_results_dir}. Skipping.")

    print("\nDone.")


if __name__ == "__main__":
    main()
