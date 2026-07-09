"""Test that seeding produces reproducible (and seed-dependent) model weights."""

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


def test_same_seed_produces_identical_weights():
    """Initializing with the same seed twice must yield identical weights."""
    set_randomness_seed(42)
    model_a = make_model()

    set_randomness_seed(42)
    model_b = make_model()

    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"Weight mismatch at {key}"


def test_different_seeds_produce_different_weights():
    """Initializing with different seeds must yield different weights."""
    set_randomness_seed(0)
    model_a = make_model()

    set_randomness_seed(1)
    model_b = make_model()

    state_a, state_b = model_a.state_dict(), model_b.state_dict()
    assert any(not torch.equal(state_a[key], state_b[key]) for key in state_a), (
        "Different seeds produced identical weights"
    )
