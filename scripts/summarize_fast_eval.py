"""Script to summarize and plot the fast evaluation metrics."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "matplotlib",
#     "pandas",
#     "seaborn",
# ]
# ///

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main():
    """Run the summarization and plotting script."""
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from nanotabpfn.utils import load_env
    load_env()

    parser = argparse.ArgumentParser(description="Summarize and plot fast evaluation metrics.")
    
    default_results = "/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data/results"
    if "WORKSPACE_DIR" in os.environ:
        default_results = str(Path(os.environ["WORKSPACE_DIR"]) / "results")

    parser.add_argument(
        "--results_dir",
        type=str,
        default=default_results,
        help="Path to the results directory",
    )
    parser.add_argument(
        "--output_plot", type=str, default="learning_curves.png", help="Path to save the learning curves plot"
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} not found.")
        return

    print("Gathering metrics...")
    data = []

    # Traverse results/<activation>/seed_<seed>/metrics.jsonl
    for act_dir in sorted(results_dir.iterdir()):
        if not act_dir.is_dir():
            continue
        activation = act_dir.name

        for seed_dir in act_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = seed_dir.name.split("_")[1]

            metrics_file = seed_dir / "metrics.jsonl"
            if not metrics_file.exists():
                continue

            with open(metrics_file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if "step" in record and "roc_auc" in record:
                            data.append(
                                {
                                    "activation": activation,
                                    "seed": seed,
                                    "step": record["step"],
                                    "roc_auc": record["roc_auc"],
                                }
                            )
                    except json.JSONDecodeError:
                        pass

    if not data:
        print("No metrics found. Ensure the paths are correct and metrics.jsonl files exist.")
        return

    df = pd.DataFrame(data)

    # 1. Print Summary Table for the final step
    max_step = df["step"].max()
    print(f"\n=== Summary of Fast Evaluation (Step {max_step}) ===")
    final_df = df[df["step"] == max_step]

    summary = final_df.groupby("activation")["roc_auc"].agg(["mean", "std", "count"]).reset_index()
    summary = summary.sort_values(by="mean", ascending=False)

    # Format the output nicely
    print(f"{'Activation':<15} | {'Mean ROC-AUC':<15} | {'Std Dev':<10} | {'Completed Seeds'}")
    print("-" * 65)
    for _, row in summary.iterrows():
        print(f"{row['activation']:<15} | {row['mean']:<15.4f} | {row['std']:<10.4f} | {int(row['count'])}/10")

    # 2. Plot Learning Curves
    print("\nGenerating learning curves plot...")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")

    # lineplot automatically aggregates over seeds and plots the mean + 95% CI (or std depending on settings)
    sns.lineplot(
        data=df,
        x="step",
        y="roc_auc",
        hue="activation",
        errorbar="sd",  # plots standard deviation across seeds
        linewidth=2,
    )

    plt.title("Fast Evaluation Proxy (3 Datasets) - Learning Curves")
    plt.xlabel("Pretraining Steps")
    plt.ylabel("ROC AUC")
    plt.legend(title="Activation Function")
    plt.tight_layout()

    plt.savefig(args.output_plot, dpi=300)
    print(f"Plot saved successfully to {args.output_plot}!")


if __name__ == "__main__":
    main()
