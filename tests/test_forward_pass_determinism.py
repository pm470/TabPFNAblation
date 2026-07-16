"""Test that seeding produces reproducible (and seed-dependent) forward pass outputs."""

import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import set_randomness_seed


def make_model():
    return NanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=2,
        mlp_hidden_size=64,
        num_layers=2,
        num_outputs=2,
    )


def make_fixed_input():
    """Fixed input, decoupled from the seed under test, so only model weights vary between calls."""
    torch.manual_seed(12345)
    x = torch.randn(2, 20, 5)
    y = torch.randn(2, 10)
    return x, y


def run_forward(seed):
    set_randomness_seed(seed)
    model = make_model()
    model.eval()

    x, y = make_fixed_input()
    with torch.no_grad():
        return model((x, y), train_test_split_index=10)


def test_same_seed_produces_identical_forward_pass():
    """Seeding with the same seed twice must yield an identical forward pass output on the same input."""
    output_a = run_forward(42)
    output_b = run_forward(42)
    assert torch.equal(output_a, output_b)


def test_different_seeds_produce_different_forward_pass():
    """Seeding with different seeds must yield a different forward pass output on the same input."""
    output_a = run_forward(0)
    output_b = run_forward(1)
    assert not torch.equal(output_a, output_b)
