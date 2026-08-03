"""Plot per-dataset relative improvement over baseline as a box-and-whisker chart.

For each activation variant, shows the distribution of per-dataset relative
improvement over the baseline (one point per TabArena dataset, averaged
across seeds), with the mean marked and annotated below each box.
"""

import argparse
import os
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from nanotabpfn.analysis import (
    METRICS,
    aggregate_seed_scores,
    compute_relative_improvement,
    format_activation_display_name,
    load_tabarena_scores,
)

PLOT_DPI = 300
BASELINE_HIGHLIGHT_COLOR = "#4C72B0"


def _wrap_label(label: str, width: int = 14) -> str:
    """Wrap a label across lines to reduce x-axis overlap."""
    return "\n".join(textwrap.wrap(label, width=width, break_long_words=False))


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

    def get_family(name: str) -> str:
        n = name.lower().replace("*", "")
        if n.endswith("glu") or n == "bilinear":
            return "Gated"
        if n in ["relu", "leaky_relu", "prelu", "re"]:
            return "ReLU-family"
        return "Smooth"

    # Sort by family and ascending mean improvement
    family_order = {"Smooth": 0, "ReLU-family": 1, "Gated": 2}
    means_for_sort = [float(np.mean(v)) if v else float("nan") for v in data]
    order = sorted(range(len(raw_labels)), key=lambda i: (family_order[get_family(raw_labels[i])], means_for_sort[i]))
    raw_labels = [raw_labels[i] for i in order]
    data = [data[i] for i in order]

    display_labels = [format_activation_display_name(name) for name in raw_labels]

    family_colors = {
        "Smooth": sns.color_palette("deep")[0],
        "ReLU-family": sns.color_palette("deep")[1],
        "Gated": sns.color_palette("deep")[2],
    }
    palette = [family_colors[get_family(label)] for label in raw_labels]

    fig, ax = plt.subplots(figsize=(max(8, len(raw_labels) * 1.3), 6))

    bp = ax.boxplot(
        data,
        tick_labels=[_wrap_label(dl) for dl in display_labels],
        patch_artist=True,
        showfliers=False,
        widths=0.5,
    )
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
    ax.scatter(range(1, len(raw_labels) + 1), means, marker="D", color="black", s=60, zorder=3)

    for i, (mean, std) in enumerate(zip(means, stds, strict=False), start=1):
        ax.annotate(
            f"{mean:+.2f} ± {std:.1f}%",
            xy=(i, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -28),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    ax.annotate(
        "Mean:",
        xy=(0.6, 0),
        xycoords=("data", "axes fraction"),
        xytext=(0, -28),
        textcoords="offset points",
        ha="right",
        fontsize=9,
        fontweight="bold",
    )

    baseline_display = format_activation_display_name(baseline_activation)
    metric_label = "ROC-AUC" if metric == "roc_auc" else "Log Loss"
    ax.axhline(0, color=BASELINE_HIGHLIGHT_COLOR, linewidth=1.5, linestyle="--")
    ax.set_ylabel(f"Relative improvement ({metric_label}, %)")
    ax.set_title(f"Per-dataset relative improvement over {baseline_display} Baseline ({metric_label}, TabArena)")
    ax.grid(True, alpha=0.3, axis="y")

    import matplotlib.patches as mpatches

    legend_elements = [
        mpatches.Patch(facecolor=family_colors["Smooth"], edgecolor="black", label="Smooth", alpha=0.6),
        mpatches.Patch(facecolor=family_colors["ReLU-family"], edgecolor="black", label="ReLU-family", alpha=0.6),
        mpatches.Patch(facecolor=family_colors["Gated"], edgecolor="black", label="Gated", alpha=0.6),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Activation Family", fontsize=9)

    # Add footnote if any unrestricted variants are present
    has_unrestricted = any(name.endswith("*") for name in display_labels)
    fig.subplots_adjust(bottom=0.3)
    if has_unrestricted:
        fig.subplots_adjust(right=0.9)
        fig.text(
            0.92,
            0.5,
            "* = full hidden width (no reduction for parameter parity)",
            fontsize=10,
            fontstyle="italic",
            color="#555555",
            ha="center",
            va="center",
            rotation=-90,
        )

    # SHAPES
    def get_activation_shape(name: str, x: np.ndarray) -> np.ndarray:
        n = name.lower().replace("*", "")
        import scipy.special

        if n in ["relu", "re", "reglu"]:
            return np.maximum(0, x)
        if n in ["gelu", "ge", "geglu"]:
            return x * 0.5 * (1.0 + scipy.special.erf(x / np.sqrt(2.0)))
        if n in ["swish", "swi", "silu", "swiglu"]:
            return x * scipy.special.expit(x)
        if n in ["mish", "mi", "miglu"]:
            return x * np.tanh(np.log1p(np.exp(x)))
        if n in ["leaky_relu"]:
            return np.maximum(0.01 * x, x)
        if n in ["prelu"]:
            return np.maximum(0.25 * x, x)
        if n in ["identity", "linear"]:
            return x
        if n == "bilinear":
            return x * x
        return np.zeros_like(x)

    import matplotlib.transforms as transforms

    trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for i, (label, color) in enumerate(zip(raw_labels, palette, strict=False), start=1):
        ax_inset = ax.inset_axes((i - 0.35, -0.28, 0.7, 0.14), transform=trans)
        x_vals = np.linspace(-3, 3, 100)
        y_vals = get_activation_shape(label, x_vals)
        ax_inset.plot(x_vals, y_vals, color=color, linewidth=2)
        ax_inset.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_inset.axvline(0, color="gray", linewidth=0.5, linestyle="--")
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        for spine in ax_inset.spines.values():
            spine.set_edgecolor("gray")
            spine.set_alpha(0.3)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    basename = f"relative_improvement_{metric}"
    png_path = out / f"{basename}.png"
    svg_path = out / f"{basename}.svg"
    fig.savefig(png_path, dpi=PLOT_DPI, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"  Saved: {png_path}, {svg_path}")
    plt.close(fig)


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot per-dataset relative improvement over baseline")

    default_results = "/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data/results"
    if "WORKSPACE_DIR" in os.environ:
        default_results = str(Path(os.environ["WORKSPACE_DIR"]) / "results")

    parser.add_argument("--results_dir", type=str, default=default_results, help="Base results directory")
    parser.add_argument("--baseline", type=str, default="gelu", help="Baseline activation name")
    parser.add_argument("--metric", type=str, default="roc_auc", choices=list(METRICS), help="Metric to plot")
    parser.add_argument("--split", type=str, default="all", choices=["all", "id", "ood"], help="Dataset split")
    parser.add_argument("--output_dir", type=str, default="plots", help="Directory for plot output (default: plots)")
    return parser.parse_args(argv)


def main(argv=None):
    """Compute and plot per-dataset relative improvement for every discovered variant."""
    args = parse_args(argv)
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return

    variant_activations = sorted(d.name for d in results_dir.iterdir() if d.is_dir() and d.name != args.baseline)

    series = collect_relative_improvements(results_dir, args.baseline, variant_activations, args.metric, args.split)
    plot_relative_improvement(series, args.baseline, args.metric, args.output_dir)


if __name__ == "__main__":
    main()
