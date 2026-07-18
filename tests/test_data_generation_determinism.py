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

    assert isinstance(batch_a["x"], torch.Tensor) and isinstance(batch_b["x"], torch.Tensor)
    assert isinstance(batch_a["y"], torch.Tensor) and isinstance(batch_b["y"], torch.Tensor)
    assert torch.equal(batch_a["x"], batch_b["x"])
    assert torch.equal(batch_a["y"], batch_b["y"])
    assert batch_a["train_test_split_index"] == batch_b["train_test_split_index"]


def test_different_seeds_produce_different_batch():
    """Generating a batch with different seeds must yield different data."""
    batch_a = get_first_batch(0)
    batch_b = get_first_batch(1)

    same_x = (
        isinstance(batch_a["x"], torch.Tensor)
        and isinstance(batch_b["x"], torch.Tensor)
        and batch_a["x"].shape == batch_b["x"].shape
        and torch.equal(batch_a["x"], batch_b["x"])
    )
    same_y = (
        isinstance(batch_a["y"], torch.Tensor)
        and isinstance(batch_b["y"], torch.Tensor)
        and batch_a["y"].shape == batch_b["y"].shape
        and torch.equal(batch_a["y"], batch_b["y"])
    )
    assert not (same_x and same_y), "Different seeds produced an identical batch"
