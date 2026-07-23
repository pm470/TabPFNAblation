"""CLI training script for the activation function ablation study."""

import argparse
import json

from nanotabpfn import config
from nanotabpfn.experiment_utils import setup_experiment
from nanotabpfn.train import (
    PriorDumpDataLoader,
    train,
)


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a single ablation model")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function name")
    parser.add_argument("--data_file", type=str, default="50k_10000x120_10.h5", help="Path to HDF5 prior data dump")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=2500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--eval_every", type=int, default=25, help="Evaluate every N steps")
    parser.add_argument("--checkpoint_every", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument("--checkpoint_every_minutes", type=float, default=10.0, help="Save checkpoint every N minutes")
    parser.add_argument("--output_dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--embedding_size", type=int, default=128, help="Embedding size")
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--mlp_hidden_size", type=int, default=192, help="MLP hidden size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num_outputs", type=int, default=config.MAX_CLASSES, help="Number of output classes")
    parser.add_argument("--no-autocast", action="store_true", help="Disable bfloat16 autocast (use pure float32)")
    return parser.parse_args(argv)


def run_training(args):
    """Run training with the given configuration."""
    run_dir, checkpoint_dir, device, model, autocast_dtype = setup_experiment(args)

    # Create dataloader using the dumped HDF5
    prior = PriorDumpDataLoader(
        filename=args.data_file,
        num_steps=args.num_steps,
        batch_size=args.batch_size,
        device=device,
        seed=args.seed,
    )

    # Train
    model, eval_history = train(
        model,
        prior,
        lr=args.lr,
        device=device,
        steps_per_eval=args.eval_every,
        eval_func=None,  # Disabled inline eval for pure pretraining speed
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=args.checkpoint_every,
        checkpoint_every_minutes=args.checkpoint_every_minutes,
        autocast_dtype=autocast_dtype,
    )

    metrics_path = run_dir / "metrics.jsonl"
    with open(metrics_path, "w") as f:
        for entry in eval_history:
            f.write(json.dumps(entry) + "\n")

    print(f"\nMetrics saved to {metrics_path}")
    print(f"Checkpoints saved to {checkpoint_dir}")

    return eval_history


if __name__ == "__main__":
    args = parse_args()
    run_training(args)
