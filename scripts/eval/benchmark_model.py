"""Standalone benchmarking script for trained models."""

import argparse
import re
from pathlib import Path

import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.tabarena_eval import run_tabarena_eval
from nanotabpfn.train import get_default_device


def discover_checkpoints(checkpoint_dir: Path) -> list[Path]:
    """Find all checkpoint files in a directory, sorted by training step.

    Discovers ``step_*.pt`` files (sorted numerically by step number) and
    appends ``final.pt`` if it exists.

    Args:
        checkpoint_dir: Directory containing checkpoint ``.pt`` files.

    Returns:
        Sorted list of checkpoint paths.
    """
    step_pattern = re.compile(r"^step_(\d+)\.pt$")
    step_files: list[tuple[int, Path]] = []
    for f in checkpoint_dir.iterdir():
        m = step_pattern.match(f.name)
        if m:
            step_files.append((int(m.group(1)), f))
    step_files.sort(key=lambda t: t[0])

    checkpoints = [p for _, p in step_files]

    final = checkpoint_dir / "final.pt"
    if final.exists():
        checkpoints.append(final)

    return checkpoints


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Benchmark a trained model")

    ckpt_group = parser.add_mutually_exclusive_group(required=True)
    ckpt_group.add_argument("--checkpoint", type=str, help="Path to a single trained model .pt file")
    ckpt_group.add_argument("--checkpoint-dir", type=str, help="Directory containing checkpoint .pt files")

    parser.add_argument(
        "--subset", type=str, default="classification", help="TabArena subset (e.g., 'lite', 'classification', 'tiny')"
    )
    parser.add_argument("--n_ensemble", type=int, default=8, help="Number of ensemble members for NanoTabPFN")

    parser.add_argument("--embedding_size", type=int, default=128)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--mlp_hidden_size", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_outputs", type=int, default=10)
    parser.add_argument("--activation", type=str, default="gelu")
    parser.add_argument(
        "--gated-unrestricted",
        action="store_true",
        help="Run gated activations with unrestricted hidden width without scaling down for parameter parity",
    )

    return parser.parse_args()


def main() -> None:
    """Run the benchmark on one or more checkpoints."""
    args = parse_args()
    device = get_default_device()

    # Resolve checkpoints and run directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoints = discover_checkpoints(checkpoint_dir)
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            return
        run_dir = checkpoint_dir.parent
    else:
        checkpoint_path = Path(args.checkpoint)
        checkpoints = [checkpoint_path]
        run_dir = checkpoint_path.parent.parent  # Assumes .../checkpoints/model.pt

    for i, ckpt_path in enumerate(checkpoints):
        if i > 0:
            print("\n" + "=" * 80)
            print(f"  Checkpoint {i + 1}/{len(checkpoints)}")
            print("=" * 80 + "\n")

        print(f"Loading checkpoint: {ckpt_path}")

        # Fresh model for each checkpoint
        model = NanoTabPFNModel(
            embedding_size=args.embedding_size,
            num_attention_heads=args.num_attention_heads,
            mlp_hidden_size=args.mlp_hidden_size,
            num_layers=args.num_layers,
            num_outputs=args.num_outputs,
            activation=args.activation,
            gated_unrestricted=args.gated_unrestricted,
        )

        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()

        ckpt_run_dir = run_dir / f"benchmark_{ckpt_path.stem}"
        print(f"Running TabArena evaluation on {args.subset} → {ckpt_run_dir}")
        run_tabarena_eval(model, device, ckpt_run_dir, subset=args.subset, n_ensemble=args.n_ensemble)

    print(f"\nAll {len(checkpoints)} checkpoint(s) benchmarked.")


if __name__ == "__main__":
    main()
