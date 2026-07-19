"""Script to compare GELU and SwiGLU performance across evaluation steps."""

# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "pandas",
# ]
# ///

import argparse
import json
import os
from pathlib import Path

import pandas as pd


def main():
    """Run the comparison script."""
    import sys

    sys.path.append(str(Path(__file__).parent.parent / "src"))
    from nanotabpfn.utils import load_env

    load_env()

    parser = argparse.ArgumentParser(description="Compare GELU and SwiGLU across all steps.")

    default_results = "/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data/results"
    if "WORKSPACE_DIR" in os.environ:
        default_results = str(Path(os.environ["WORKSPACE_DIR"]) / "results")

    parser.add_argument("--results_dir", type=str, default=default_results)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Error: {results_dir} not found.")
        return

    data = []
    for activation in ["gelu", "swiglu"]:
        act_dir = results_dir / activation
        if not act_dir.exists():
            continue
        for seed_dir in act_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
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
                                {"activation": activation, "step": record["step"], "roc_auc": record["roc_auc"]}
                            )
                    except json.JSONDecodeError:
                        pass

    if not data:
        print("No data found for gelu or swiglu.")
        return

    df = pd.DataFrame(data)

    # Calculate mean and std for each activation and step
    stats = df.groupby(["step", "activation"])["roc_auc"].agg(["mean", "std"]).reset_index()

    # Pivot so we can compare gelu and swiglu easily
    pivot_mean = stats.pivot(index="step", columns="activation", values="mean")
    pivot_std = stats.pivot(index="step", columns="activation", values="std")

    if "gelu" not in pivot_mean.columns or "swiglu" not in pivot_mean.columns:
        print("Missing data for either gelu or swiglu.")
        return

    print(
        f"{'Step':<6} | {'GELU ROC-AUC (±std)':<22} | "
        f"{'SwiGLU ROC-AUC (±std)':<22} | {'Abs Diff':<10} | {'Rel Improv.'}"
    )
    print("-" * 85)

    for step in sorted(pivot_mean.index):
        gelu_mean = pivot_mean.at[step, "gelu"]
        gelu_std = pivot_std.at[step, "gelu"]

        swiglu_mean = pivot_mean.at[step, "swiglu"]
        swiglu_std = pivot_std.at[step, "swiglu"]

        # Skip if missing data
        if pd.isna(gelu_mean) or pd.isna(swiglu_mean):
            continue

        abs_diff = swiglu_mean - gelu_mean
        rel_diff = (abs_diff / gelu_mean) * 100

        gelu_str = f"{gelu_mean:.4f} (±{gelu_std:.4f})"
        swiglu_str = f"{swiglu_mean:.4f} (±{swiglu_std:.4f})"
        abs_str = f"{abs_diff:+.4f}"
        rel_str = f"{rel_diff:+.2f}%"

        print(f"{step:<6} | {gelu_str:<22} | {swiglu_str:<22} | {abs_str:<10} | {rel_str}")


if __name__ == "__main__":
    main()
