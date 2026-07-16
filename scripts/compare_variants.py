"""Compare all activation variants against a baseline on TabArena results.

Reads `results/<activation>/seed_*/tabarena_exp/nanotabpfn_summary.csv` for
every activation, computes each variant's mean relative improvement over the
baseline (per dataset, averaged across seeds) for both ROC-AUC and log loss,
and runs a paired t-test across datasets to flag statistically significant
differences.
"""

import argparse
from pathlib import Path

import pandas as pd

from nanotabpfn.analysis import METRICS, compare_variant_to_baseline


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare activation variants against a baseline on TabArena results")
    parser.add_argument("--results_dir", type=str, default="results", help="Base results directory")
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
            rows.append(compare_variant_to_baseline(results_dir, args.baseline, variant, metric))

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))

    output_csv = args.output_csv or str(results_dir / "variant_comparison.csv")
    df.to_csv(output_csv, index=False)
    print(f"\nSaved comparison table to: {output_csv}")


if __name__ == "__main__":
    main()
