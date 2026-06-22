"""Plot results from ablation experiments.

Reads metrics.jsonl files from results/ and produces:
1. Training step vs. ROC-AUC per activation (mean ± std across seeds)
2. Final ROC-AUC box plot across activations
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
                seed_runs.append(metrics)
        if seed_runs:
            all_results[activation_name] = seed_runs

    return all_results


def plot_training_curves(all_results: dict[str, list[list[dict]]], output_path: str = "training_curves.png"):
    """Plot training step vs. ROC-AUC per activation with std bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for activation_name, seed_runs in all_results.items():
        # Collect steps and roc_auc values across seeds
        # Assume all seeds have the same steps
        steps = [entry["step"] for entry in seed_runs[0]]
        roc_aucs = []
        for run in seed_runs:
            run_aucs = [entry.get("roc_auc", float("nan")) for entry in run]
            roc_aucs.append(run_aucs)

        roc_aucs = np.array(roc_aucs)
        mean_auc = np.mean(roc_aucs, axis=0)
        std_auc = np.std(roc_aucs, axis=0)

        ax.plot(steps, mean_auc, label=f"{activation_name} (n={len(seed_runs)})")
        ax.fill_between(steps, mean_auc - std_auc, mean_auc + std_auc, alpha=0.2)

    ax.set_xlabel("Training Step")
    ax.set_ylabel("ROC-AUC")
    ax.set_title("Training Curves: ROC-AUC vs. Step")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Training curves saved to {output_path}")
    plt.close(fig)


def plot_final_boxplot(all_results: dict[str, list[list[dict]]], output_path: str = "final_boxplot.png"):
    """Plot box-and-whisker of final ROC-AUC per activation."""
    data = []
    labels = []
    for activation_name, seed_runs in all_results.items():
        final_aucs = []
        for run in seed_runs:
            # Take the last entry's roc_auc
            final_auc = run[-1].get("roc_auc", float("nan"))
            final_aucs.append(final_auc)
        data.append(final_aucs)
        labels.append(activation_name)

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 6))
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True)

    # Color the boxes
    colors = sns.color_palette("husl", len(labels))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xlabel("Activation Function")
    ax.set_ylabel("Final ROC-AUC")
    ax.set_title("Final ROC-AUC by Activation Function")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Box plot saved to {output_path}")
    plt.close(fig)


if __name__ == "__main__":
    results = load_all_results()
    print(f"Found activations: {list(results.keys())}")
    for name, runs in results.items():
        print(f"  {name}: {len(runs)} seeds")
    plot_training_curves(results)
    plot_final_boxplot(results)
