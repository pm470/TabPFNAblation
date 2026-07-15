"""Test that seeding produces reproducible (and seed-dependent) weights after training steps."""

import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import NanopriorDataset, set_randomness_seed, train


def run_training(seed):
    set_randomness_seed(seed)
    model = NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=2,
        mlp_hidden_size=32,
        num_layers=1,
        num_outputs=2,
    )
    prior = NanopriorDataset(
        num_steps=3,
        batch_size=2,
        max_seq_len=150,
        max_features=4,
        max_classes=2,
        device=torch.device("cpu"),
    )
    trained_model, _ = train(
        model,
        prior,
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=100,
        eval_func=None,
    )
    return trained_model.state_dict()


def test_same_seed_produces_identical_trained_weights():
    """Training with the same seed twice must yield identical weights after training."""
    state_a = run_training(42)
    state_b = run_training(42)

    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key]), f"Weight mismatch at {key}"


def test_different_seeds_produce_different_trained_weights():
    """Training with different seeds must yield different weights after training."""
    state_a = run_training(0)
    state_b = run_training(1)

    assert any(not torch.equal(state_a[key], state_b[key]) for key in state_a), (
        "Different seeds produced identical trained weights"
    )
