"""Tests for training pipeline and data loading."""

import h5py
import numpy as np
import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import PriorDumpDataLoader, train


def test_train_without_eval_func():
    """train() correctly handles eval_func=None, only logging step, wall_time, loss."""
    model = NanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=2,
        mlp_hidden_size=64,
        num_layers=1,
        num_outputs=2,
    )

    # Mock prior with a single batch
    mock_prior = [
        {
            "x": torch.randn(2, 20, 5, dtype=torch.float32),
            "y": torch.randint(0, 2, (2, 20), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]

    # Train for 1 step, eval every 1 step
    _, eval_history = train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=1,
        eval_func=None,
    )
    assert len(eval_history) == 1
    entry = eval_history[0]
    expected_keys = {"step", "wall_time", "loss"}
    assert set(entry.keys()) == expected_keys


def test_prior_dump_dataloader_wraparound(tmp_path):
    """PriorDumpDataLoader correctly wraps around when reaching the end of the HDF5 dataset."""
    # Create a tiny HDF5 file
    h5_path = tmp_path / "tiny_prior.h5"
    num_samples = 3

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("max_num_classes", data=[2])
        # Need X, y, num_features, num_datapoints, single_eval_pos
        f.create_dataset("X", data=np.random.randn(num_samples, 10, 5).astype(np.float32))
        f.create_dataset("y", data=np.random.randint(0, 2, (num_samples, 10)).astype(np.float32))
        f.create_dataset("num_features", data=np.array([5, 5, 5], dtype=np.int32))
        f.create_dataset("num_datapoints", data=np.array([10, 10, 10], dtype=np.int32))
        f.create_dataset("single_eval_pos", data=np.array([5, 5, 5], dtype=np.int32))

    loader = PriorDumpDataLoader(filename=str(h5_path), num_steps=3, batch_size=2, device=torch.device("cpu"))

    # Batch 1: consumes 2 samples (index 0, 1) -> pointer = 2
    # Batch 2: consumes 2 samples (but dataset only has 3, so pointer = 4, then wrapping triggers pointer = 0)
    # Batch 3: consumes 2 samples (index 0, 1) -> pointer = 2

    batches = list(loader)

    assert len(batches) == 3
    for batch in batches:
        assert "x" in batch
        assert "y" in batch
        assert "train_test_split_index" in batch

    # Verify wraparound reset pointer to 0
    assert loader.pointer == 2
