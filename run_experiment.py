"""CLI experiment runner for the activation function ablation study.

Wraps the existing nanoTabPFN train.py with structured logging,
seed control, and checkpointing. For now, always uses GELU —
the --activation flag is a placeholder for the next phase.
"""

import argparse
import json
from pathlib import Path

import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import (
    NanopriorDataset,
    PriorDumpDataLoader,
    eval,
    get_default_device,
    set_randomness_seed,
    train,
)
from nanotabpfn.utils import save_memory_stat


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a single ablation experiment")
    parser.add_argument("--activation", type=str, default="gelu", help="Activation function name (placeholder for now)")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["quick", "tabarena"],
        default="quick",
        help="Benchmark to use for final evaluation",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_steps", type=int, default=2500, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=4e-3, help="Learning rate")
    parser.add_argument("--eval_every", type=int, default=25, help="Evaluate every N steps")
    parser.add_argument("--checkpoint_every", type=int, default=250, help="Save checkpoint every N steps")
    parser.add_argument("--data_file", type=str, default=None, help="Path to HDF5 prior data file")
    parser.add_argument("--output_dir", type=str, default="results", help="Base output directory")
    parser.add_argument("--max_seq_len", type=int, default=3000, help="Maximum number of rows per dataset")
    parser.add_argument("--max_features", type=int, default=45, help="Maximum number of features per dataset")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes per dataset")
    parser.add_argument("--embedding_size", type=int, default=128, help="Embedding size")
    parser.add_argument("--num_attention_heads", type=int, default=4, help="Number of attention heads")
    parser.add_argument("--mlp_hidden_size", type=int, default=192, help="MLP hidden size")
    parser.add_argument("--num_layers", type=int, default=3, help="Number of transformer layers")
    parser.add_argument("--num_outputs", type=int, default=10, help="Number of output classes")
    parser.add_argument("--no-autocast", action="store_true", help="Disable bfloat16 autocast (use pure float32)")
    return parser.parse_args(argv)


def run_experiment(args):
    """Run a single experiment with the given configuration."""
    # Set up output directory
    run_dir = Path(args.output_dir) / args.activation / f"seed_{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"

    # Set seed before anything else
    set_randomness_seed(args.seed)

    # Create model
    device = get_default_device()
    model = NanoTabPFNModel(
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        num_layers=args.num_layers,
        num_outputs=args.num_outputs,
        activation=args.activation,
    )

    # Save config
    param_count = sum(p.numel() for p in model.parameters())
    # Resolve autocast dtype
    autocast_dtype = None if args.no_autocast else torch.bfloat16

    config = {
        "activation": args.activation,
        "benchmark": args.benchmark,
        "seed": args.seed,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "eval_every": args.eval_every,
        "checkpoint_every": args.checkpoint_every,
        "embedding_size": args.embedding_size,
        "num_attention_heads": args.num_attention_heads,
        "mlp_hidden_size": args.mlp_hidden_size,
        "num_layers": args.num_layers,
        "num_outputs": args.num_outputs,
        "max_seq_len": args.max_seq_len,
        "max_features": args.max_features,
        "max_classes": args.max_classes,
        "data_file": args.data_file,
        "param_count": param_count,
        "device": str(device),
        "autocast_dtype": str(autocast_dtype) if autocast_dtype else None,
    }
    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    print(f"Model parameters: {param_count:,}")

    # Auto-resume logic
    start_step = 0
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("step_*.pt"))
        if checkpoints:
            # Sort by step number extracted from step_XXXXX.pt
            latest_ckpt = sorted(checkpoints, key=lambda x: int(x.stem.split("_")[1]))[-1]
            start_step = int(latest_ckpt.stem.split("_")[1])
            print(f"Resuming from checkpoint {latest_ckpt} at step {start_step}")
            model.load_state_dict(torch.load(latest_ckpt, map_location=device))

    if start_step >= args.num_steps:
        print(f"Experiment already completed up to {args.num_steps} steps. Exiting.")
        return []

    remaining_steps = args.num_steps - start_step

    # Create dataloader
    if args.data_file:
        print(f"Loading prior data from {args.data_file}")
        prior = PriorDumpDataLoader(
            filename=args.data_file,
            num_steps=remaining_steps,
            batch_size=args.batch_size,
            device=device,
        )
    else:
        from torch.utils.data import DataLoader

        print("Generating prior data on the fly")
        dataset = NanopriorDataset(
            num_steps=remaining_steps,
            batch_size=args.batch_size,
            max_seq_len=args.max_seq_len,
            max_features=args.max_features,
            max_classes=args.max_classes,
            device=device,
        )
        # Using 0 workers for safety/compatibility; increase if CPU allows.
        prior = DataLoader(dataset, batch_size=None, num_workers=0)

    # Setup metrics file
    metrics_path = run_dir / "metrics.jsonl"

    # Train
    model, eval_history = train(
        model,
        prior,
        lr=args.lr,
        device=device,
        steps_per_eval=args.eval_every,
        eval_func=eval,
        checkpoint_dir=str(checkpoint_dir),
        checkpoint_every=args.checkpoint_every,
        start_step=start_step,
        autocast_dtype=autocast_dtype,
        metrics_file=metrics_path,
    )

    print(f"\nTraining completed. Checkpoints saved to {checkpoint_dir}")

    # Final evaluation
    model.eval()

    # Reset peak-memory tracking so eval VRAM is measured separately from pretraining VRAM.
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    if args.benchmark == "tabarena":
        from nanotabpfn.tabarena_eval import run_tabarena_eval

        print("Running TabArena final evaluation...")
        run_tabarena_eval(model, device, run_dir)
    else:
        from nanotabpfn.model import NanoTabPFNClassifier
        from nanotabpfn.train import eval as eval_fn

        classifier = NanoTabPFNClassifier(model, device)
        final_scores = eval_fn(classifier)
        print(f"Final scores: {final_scores}")
        if device.type == "cuda":
            peak_mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
            print(f"[NanoTabPFN] Peak GPU memory allocated for benchmark {args.benchmark}: {peak_mem_gb:.2f} GB")
            save_memory_stat(run_dir, "peak_vram_eval_gb", peak_mem_gb)

        # Save final local scores to a separate file so we don't mix them with step metrics
        final_scores_path = run_dir / "final_scores.json"
        with open(final_scores_path, "w") as f:
            json.dump(final_scores, f, indent=2)
        print(f"Final local scores saved to {final_scores_path}")

    return eval_history


if __name__ == "__main__":
    args = parse_args()
    run_experiment(args)
