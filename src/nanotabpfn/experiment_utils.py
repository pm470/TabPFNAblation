"""Experiment and tracking utilities."""

import json
from pathlib import Path

import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.utils import get_default_device, set_randomness_seed


def _build_run_dir(args) -> Path:
    """Build the run directory path based on the experiment mode.

    For standard activation ablation: ``{output_dir}/{activation}/seed_{seed}/``.
    For architecture sweep: ``{output_dir}/{sweep_axis}/{activation}/e{E}_h{H}_l{L}/seed_{seed}/``.
    """
    base = Path(args.output_dir)
    arch_sweep = getattr(args, "arch_sweep", None)
    if arch_sweep:
        arch_tag = f"e{args.embedding_size}_h{args.mlp_hidden_size}_l{args.num_layers}"
        return base / arch_sweep / args.activation / arch_tag / f"seed_{args.seed}"
    return base / args.activation / f"seed_{args.seed}"


def setup_experiment(args) -> tuple[Path, Path, torch.device, NanoTabPFNModel, torch.dtype | None]:
    """Sets up the output directory, sets seeds, creates the model, and saves config."""
    run_dir = _build_run_dir(args)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_dir / "checkpoints"

    set_randomness_seed(args.seed)
    device = get_default_device()

    model = NanoTabPFNModel(
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        num_layers=args.num_layers,
        num_outputs=args.num_outputs,
        activation=args.activation,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    param_count = sum(p.numel() for p in model.parameters())
    autocast_dtype = None if args.no_autocast else torch.bfloat16

    config = vars(args).copy()
    config.update(
        {
            "param_count": param_count,
            "device": str(device),
            "autocast_dtype": str(autocast_dtype) if autocast_dtype else None,
        }
    )

    config_path = run_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to {config_path}")
    print(f"Model parameters: {param_count:,}")

    return run_dir, checkpoint_dir, device, model, autocast_dtype


def save_memory_stat(run_dir: Path, key: str, value: float | dict) -> None:
    """Merge a single peak-VRAM measurement into run_dir/memory_stats.json.

    Read-modify-write so pretrain and eval stats (written at different points
    in the pipeline) accumulate into the same file instead of overwriting each other.
    """
    stats_path = Path(run_dir) / "memory_stats.json"
    stats = {}
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
    stats[key] = value
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
