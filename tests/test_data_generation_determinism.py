"""Test that seeding produces reproducible (and seed-dependent) synthetic prior batches."""

import torch

from nanotabpfn.train import NanopriorDataset, set_randomness_seed


def get_first_batch(seed):
    set_randomness_seed(seed)
    dataset = NanopriorDataset(
        num_steps=1,
        batch_size=2,
        max_seq_len=150,
        max_features=5,
        max_classes=3,
        device=torch.device("cpu"),
    )
    return next(iter(dataset))


def test_same_seed_produces_identical_batch():
    """Generating a batch with the same seed twice must yield identical data."""
    batch_a = get_first_batch(42)
    batch_b = get_first_batch(42)

    x_a, x_b = batch_a["x"], batch_b["x"]
    y_a, y_b = batch_a["y"], batch_b["y"]
    assert isinstance(x_a, torch.Tensor) and isinstance(x_b, torch.Tensor)
    assert isinstance(y_a, torch.Tensor) and isinstance(y_b, torch.Tensor)
    assert torch.equal(x_a, x_b)
    assert torch.equal(y_a, y_b)
    assert batch_a["train_test_split_index"] == batch_b["train_test_split_index"]


def test_different_seeds_produce_different_batch():
    """Generating a batch with different seeds must yield different data."""
    batch_a = get_first_batch(0)
    batch_b = get_first_batch(1)

    x_a, x_b = batch_a["x"], batch_b["x"]
    y_a, y_b = batch_a["y"], batch_b["y"]
    assert isinstance(x_a, torch.Tensor) and isinstance(x_b, torch.Tensor)
    assert isinstance(y_a, torch.Tensor) and isinstance(y_b, torch.Tensor)

    same_x = x_a.shape == x_b.shape and torch.equal(x_a, x_b)
    same_y = y_a.shape == y_b.shape and torch.equal(y_a, y_b)
    assert not (same_x and same_y), "Different seeds produced an identical batch"
