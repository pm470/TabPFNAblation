"""Compare all activation variants against a baseline on TabArena results.

Reads `results/<activation>/seed_*/tabarena_exp/nanotabpfn_summary.csv` for
every activation, computes each variant's mean relative improvement over the
baseline (per dataset, averaged across seeds) for both ROC-AUC and log loss,
and runs a paired t-test across datasets to flag statistically significant
differences.
"""

import argparse
import os
from pathlib import Path

import pandas as pd

from nanotabpfn.analysis import METRICS, compare_variant_to_baseline


def parse_args(argv=None):
    """Parse command line arguments."""
    from nanotabpfn.utils import load_env
    load_env()

    parser = argparse.ArgumentParser(description="Compare activation variants against a baseline on TabArena results")
    
    default_results = "/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data/results"
    if "WORKSPACE_DIR" in os.environ:
        default_results = str(Path(os.environ["WORKSPACE_DIR"]) / "results")

    parser.add_argument(
        "--results_dir",
        type=str,
        default=default_results,
        help="Base results directory",
    )
    parser.add_argument("--baseline", type=str, default="gelu", help="Baseline activation name")
    parser.add_argument("--output_csv", type=str, default=None, help="Where to save the comparison table")
    return parser.parse_args(argv)


def main(argv=None):
    """Run the variant comparison and print/save the summary table."""
    args = parse_args(argv)
    results_dir = Path(args.results_dir)

    variant_activations = sorted(d.name for d in results_dir.iterdir() if d.is_dir() and d.name != args.baseline)

    rows = []
    for variant in variant_activations:
        for metric in METRICS:
            for split in ["all", "id", "ood"]:
                rows.append(compare_variant_to_baseline(results_dir, args.baseline, variant, metric, split))

    df = pd.DataFrame(rows)
    
    for split in ["all", "id", "ood"]:
        split_df = df[df["split"] == split].copy()
        if split_df.empty:
            continue
            
        print(f"\n=== Split: {split.upper()} ===")
        roc_df = split_df[split_df["metric"] == "roc_auc"]
        log_df = split_df[split_df["metric"] == "log_loss"]
        
        baseline_roc = roc_df["baseline_mean"].iloc[0] if not roc_df.empty else float("nan")
        baseline_log = log_df["baseline_mean"].iloc[0] if not log_df.empty else float("nan")
        
        print(f"Baseline ({args.baseline}) Mean ROC-AUC:  {baseline_roc:.6f}")
        print(f"Baseline ({args.baseline}) Mean Log-Loss: {baseline_log:.6f}\n")
        
        display_df = split_df.drop(columns=["split", "baseline_mean"])
        print(display_df.to_string(index=False))

    output_csv = args.output_csv or str(results_dir / "variant_comparison.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved full comparison table (with all columns) to: {output_csv}")


if __name__ == "__main__":
    main()
