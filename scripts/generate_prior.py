"""Generate synthetic prior data to an HDF5 file.

Iterates over NanopriorDataset and dumps it sequentially to HDF5.
"""

import argparse
import os
import sys

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train import NanopriorDataset, set_randomness_seed


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate synthetic TabPFN prior datasets to HDF5.")
    parser.add_argument("--output_file", type=str, default="prior_dump.h5", help="Path to output HDF5 file")
    parser.add_argument("--num_datasets", type=int, default=100000, help="Total number of datasets to generate")
    parser.add_argument("--batch_size", type=int, default=16, help="Generation batch size")
    parser.add_argument("--max_seq_len", type=int, default=2000, help="Maximum number of samples per dataset")
    parser.add_argument("--max_features", type=int, default=100, help="Maximum number of features per dataset")
    parser.add_argument("--max_classes", type=int, default=10, help="Maximum number of classes")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    """Generate datasets based on parsed arguments."""
    args = parse_args()
    set_randomness_seed(args.seed)

    print(f"Generating {args.num_datasets} datasets into {args.output_file}...")
    print(
        f"Parameters: max_seq_len={args.max_seq_len}, max_features={args.max_features}, max_classes={args.max_classes}"
    )

    # Create dataset and loader
    # num_steps is the total number of batches
    num_steps = (args.num_datasets + args.batch_size - 1) // args.batch_size
    dataset = NanopriorDataset(
        num_steps=num_steps,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        max_features=args.max_features,
        max_classes=args.max_classes,
        device=torch.device("cpu"),  # Generate on CPU, save to disk
    )

    loader = DataLoader(dataset, batch_size=None, num_workers=args.num_workers)

    # Initialize HDF5 file
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)), exist_ok=True)
    with h5py.File(args.output_file, "w") as f:
        f.create_dataset("max_num_classes", data=np.array([args.max_classes]))

        # We need chunking for X and y because they are large
        chunk_size = min(100, args.num_datasets)
        X_ds = f.create_dataset(
            "X",
            shape=(args.num_datasets, args.max_seq_len, args.max_features),
            dtype=np.float32,
            chunks=(chunk_size, args.max_seq_len, args.max_features),
            fillvalue=0.0,
        )
        y_ds = f.create_dataset(
            "y",
            shape=(args.num_datasets, args.max_seq_len),
            dtype=np.float32,
            chunks=(chunk_size, args.max_seq_len),
            fillvalue=0.0,
        )
        num_features_ds = f.create_dataset("num_features", shape=(args.num_datasets,), dtype=np.int64)
        num_datapoints_ds = f.create_dataset("num_datapoints", shape=(args.num_datasets,), dtype=np.int64)
        single_eval_pos_ds = f.create_dataset("single_eval_pos", shape=(args.num_datasets,), dtype=np.int64)

        pointer = 0
        for batch_idx, batch in enumerate(loader):
            x_batch = batch["x"].numpy()  # (batch_size, n_samples, n_features)
            y_batch = batch["y"].numpy()  # (batch_size, n_samples)
            train_test_split_index = batch["train_test_split_index"]

            current_batch_size = x_batch.shape[0]
            n_samples = x_batch.shape[1]
            n_features = x_batch.shape[2]

            # In case the last batch is smaller due to rounding
            if pointer + current_batch_size > args.num_datasets:
                current_batch_size = args.num_datasets - pointer
                x_batch = x_batch[:current_batch_size]
                y_batch = y_batch[:current_batch_size]

            end = pointer + current_batch_size

            # Write to HDF5
            X_ds[pointer:end, :n_samples, :n_features] = x_batch
            y_ds[pointer:end, :n_samples] = y_batch

            num_features_ds[pointer:end] = n_features
            num_datapoints_ds[pointer:end] = n_samples
            single_eval_pos_ds[pointer:end] = train_test_split_index

            pointer += current_batch_size

            if (batch_idx + 1) % max(1, num_steps // 20) == 0:
                print(f"Generated {pointer}/{args.num_datasets} datasets ({(pointer / args.num_datasets) * 100:.1f}%)")

    print(f"Successfully generated {args.num_datasets} datasets and saved to {args.output_file}")


if __name__ == "__main__":
    main()
