"""Standalone benchmarking script for trained models."""

import argparse
import torch
from pathlib import Path

from model import NanoTabPFNModel
from tabarena_eval import run_tabarena_eval
from train import get_default_device


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a trained model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the trained model.pt")
    parser.add_argument("--subset", type=str, default="classification", help="TabArena subset (e.g., 'lite', 'classification', 'tiny')")
    parser.add_argument("--n_ensemble", type=int, default=8, help="Number of ensemble members for NanoTabPFN")
    
    # Model architecture args (must match training)
    parser.add_argument("--embedding_size", type=int, default=96)
    parser.add_argument("--num_attention_heads", type=int, default=4)
    parser.add_argument("--mlp_hidden_size", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--num_outputs", type=int, default=10)
    parser.add_argument("--activation", type=str, default="gelu")
    
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_default_device()
    
    # Instantiate model
    model = NanoTabPFNModel(
        embedding_size=args.embedding_size,
        num_attention_heads=args.num_attention_heads,
        mlp_hidden_size=args.mlp_hidden_size,
        num_layers=args.num_layers,
        num_outputs=args.num_outputs,
        activation=args.activation,
    )
    
    # Load checkpoint
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Benchmark
    checkpoint_path = Path(args.checkpoint)
    run_dir = checkpoint_path.parent.parent  # Assumes results/{act}/seed_{seed}/checkpoints/model.pt
    
    print(f"Running TabArena final evaluation on {args.subset}...")
    run_tabarena_eval(model, device, run_dir, subset=args.subset, n_ensemble=args.n_ensemble)

if __name__ == "__main__":
    main()
