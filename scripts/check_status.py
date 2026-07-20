#!/usr/bin/env python3
"""Check the completion status of TabPFN Ablation experiments.

Identifies runs that have not completed their training (reached 5000 steps)
or have not completed the TabArena evaluation.
"""

import argparse
import json
import os
from pathlib import Path

# Default expected activations
EXPECTED_ACTIVATIONS = [
    "bilinear",
    "gelu",
    "leaky_relu",
    "prelu",
    "relu",
    "swiglu",
    "swish",
]
EXPECTED_SEEDS = list(range(10))
EXPECTED_STEPS = 5000


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Check the completion status of experiments.")
    parser.add_argument("--results-dir", type=str, default=None, help="Base results directory")
    return parser.parse_args()


def main():
    """Run the status check logic."""
    args = parse_args()

    # Try to load workspace dir from .env
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass

    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        workspace_dir = os.environ.get("WORKSPACE_DIR", "/pfs/work9/workspace/scratch/fr_lf453-nanotabpfn_data")
        results_dir = Path(workspace_dir) / "results"

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return

    # Find all activations (expected + any new ones)
    activations = set(EXPECTED_ACTIVATIONS)
    for d in results_dir.iterdir():
        if d.is_dir() and d.name != "results_mock":
            activations.add(d.name)
    activations = sorted(list(activations))

    print(f"Checking results in: {results_dir}")
    print(f"Expected seeds per activation: {EXPECTED_SEEDS}")
    print(f"Expected steps: {EXPECTED_STEPS}\n")

    missing_training = []
    missing_tabarena = []

    for act in activations:
        act_dir = results_dir / act
        for seed in EXPECTED_SEEDS:
            seed_dir = act_dir / f"seed_{seed}"
            seed_name = f"{act}/seed_{seed}"

            # Check training (metrics.jsonl)
            metrics_file = seed_dir / "metrics.jsonl"
            training_complete = False
            if metrics_file.exists():
                try:
                    # check last line
                    with open(metrics_file) as f:
                        lines = [line.strip() for line in f if line.strip()]
                        if lines:
                            last_metric = json.loads(lines[-1])
                            if last_metric.get("step", 0) >= EXPECTED_STEPS:
                                training_complete = True
                except Exception:
                    pass

            if not training_complete:
                missing_training.append(seed_name)

            # Check TabArena
            tabarena_file = seed_dir / "benchmark_final" / "tabarena_exp" / "nanotabpfn_summary.csv"
            if not tabarena_file.exists():
                missing_tabarena.append(seed_name)

    # Print Report
    print("=== Training Status (Missing or < 5000 steps) ===")
    if not missing_training:
        print("  ✅ All training completed!")
    else:
        for m in missing_training:
            print(f"  ❌ {m}")

    print("\n=== TabArena Status (Missing nanotabpfn_summary.csv) ===")
    if not missing_tabarena:
        print("  ✅ All TabArena evaluations completed!")
    else:
        for m in missing_tabarena:
            print(f"  ❌ {m}")

    print("\nDone.")


if __name__ == "__main__":
    main()
