"""Plot per-dataset relative improvement over baseline as a box-and-whisker chart.

For each activation variant, shows the distribution of per-dataset relative
improvement over the baseline (one point per TabArena dataset, averaged
across seeds), with the mean marked and annotated below each box.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from nanotabpfn.analysis import (
    METRICS,
    aggregate_seed_scores,
    compute_relative_improvement,
    load_tabarena_scores,
)

BASELINE_COLOR = "#4C72B0"
VARIANT_COLOR = "#55A868"


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
    output_path: str = "relative_improvement.png",
):
    """Box-and-whisker plot of per-dataset relative improvement over the baseline.

    Each box shows the distribution across datasets; individual dots are
    jittered per-dataset scores; the diamond marks the mean; the label
    below each box shows mean ± std.

    Args:
        series: Mapping of activation name to per-dataset relative improvement
            percentages, as returned by `collect_relative_improvements`.
        baseline_activation: Name of the baseline activation, used to color
            it differently from the variants.
        metric: `"roc_auc"` or `"log_loss"`, used for axis/title labeling.
        output_path: Where to save the figure.
    """
    labels = list(series.keys())
    data = [series[label] for label in labels]
    colors = [BASELINE_COLOR if label == baseline_activation else VARIANT_COLOR for label in labels]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.3), 6))

    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False, widths=0.5)
    for patch, color in zip(bp["boxes"], colors, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)

    rng = np.random.default_rng(0)
    for i, (values, color) in enumerate(zip(data, colors, strict=False), start=1):
        jitter = rng.normal(0, 0.04, size=len(values))
        ax.scatter(
            np.full(len(values), i) + jitter,
            values,
            color=color,
            alpha=0.5,
            s=18,
            zorder=2,
            edgecolors="none",
        )

    means = [np.mean(v) if v else float("nan") for v in data]
    stds = [np.std(v) if v else float("nan") for v in data]
    ax.scatter(range(1, len(labels) + 1), means, marker="D", color="black", s=60, zorder=3, label="Mean")

    for i, (mean, std) in enumerate(zip(means, stds, strict=False), start=1):
        ax.annotate(
            f"{mean:.2f} ± {std:.1f}%",
            xy=(i, 0),
            xycoords=("data", "axes fraction"),
            xytext=(0, -28),
            textcoords="offset points",
            ha="center",
            fontsize=8,
        )

    metric_label = "ROC-AUC" if metric == "roc_auc" else "Log Loss"
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_ylabel(f"Relative improvement over {baseline_activation} ({metric_label}, %)")
    ax.set_title(f"Per-dataset relative improvement over {baseline_activation} ({metric_label})")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(loc="upper left")
    fig.subplots_adjust(bottom=0.2)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Plot per-dataset relative improvement over baseline")
    parser.add_argument("--results_dir", type=str, default="results", help="Base results directory")
    parser.add_argument("--baseline", type=str, default="gelu", help="Baseline activation name")
    parser.add_argument("--metric", type=str, default="roc_auc", choices=list(METRICS), help="Metric to plot")
    parser.add_argument("--split", type=str, default="all", choices=["all", "id", "ood"], help="Dataset split")
    parser.add_argument("--output_path", type=str, default=None, help="Where to save the figure")
    return parser.parse_args(argv)


def main(argv=None):
    """Compute and plot per-dataset relative improvement for every discovered variant."""
    args = parse_args(argv)
    results_dir = Path(args.results_dir)

    variant_activations = sorted(d.name for d in results_dir.iterdir() if d.is_dir() and d.name != args.baseline)

    series = collect_relative_improvements(results_dir, args.baseline, variant_activations, args.metric, args.split)
    output_path = args.output_path or f"relative_improvement_{args.metric}_{args.split}.png"
    plot_relative_improvement(series, args.baseline, args.metric, output_path)


if __name__ == "__main__":
    main()
