from unittest.mock import patch

import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import train


class DummyPrior:
    def __init__(self, num_steps):
        self.num_steps = num_steps

    def __iter__(self):
        for _ in range(self.num_steps):
            x = torch.randn(2, 5, 3)
            y = torch.randint(0, 2, (2, 5))
            yield {"x": x, "y": y, "train_test_split_index": 3}

    def __len__(self):
        return self.num_steps


def test_gradient_accumulation_steps():
    """Verify that optimizer.step() is called the correct number of times."""
    device = torch.device("cpu")
    model = NanoTabPFNModel(
        embedding_size=16,
        num_attention_heads=1,
        mlp_hidden_size=32,
        num_layers=1,
        num_outputs=2,
    )
    prior = DummyPrior(num_steps=4)

    with (
        patch("schedulefree.AdamWScheduleFree.step") as mock_step,
        patch("schedulefree.AdamWScheduleFree.zero_grad") as mock_zero,
    ):
        train(
            model,
            prior,  # type: ignore
            lr=1e-3,
            device=device,
            steps_per_eval=10,
            accumulation_steps=2,
            autocast_dtype=None,
        )
        # For 4 steps with accumulation=2, step() should be called 2 times.
        assert mock_step.call_count == 2
        assert mock_zero.call_count == 2


def test_gradient_checkpointing_equivalence():
    """Verify that gradients are identical with and without checkpointing."""

    def create_model(checkpointing):
        torch.manual_seed(42)
        return NanoTabPFNModel(
            embedding_size=16,
            num_attention_heads=1,
            mlp_hidden_size=32,
            num_layers=1,
            num_outputs=2,
            gradient_checkpointing=checkpointing,
        )

    model_normal = create_model(False)
    model_ckpt = create_model(True)

    torch.manual_seed(42)
    x = torch.randn(2, 5, 3)
    y = torch.randint(0, 2, (2, 5)).float()

    # Normal forward/backward
    model_normal.train()
    out_normal = model_normal((x, y), train_test_split_index=3)
    loss_normal = out_normal.sum()
    loss_normal.backward()

    # Checkpoint forward/backward
    model_ckpt.train()
    out_ckpt = model_ckpt((x, y), train_test_split_index=3)
    loss_ckpt = out_ckpt.sum()
    loss_ckpt.backward()

    # Check outputs
    assert torch.allclose(out_normal, out_ckpt)

    # Check gradients
    for p_norm, p_ckpt in zip(model_normal.parameters(), model_ckpt.parameters(), strict=True):
        if p_norm.grad is not None and p_ckpt.grad is not None:
            assert torch.allclose(p_norm.grad, p_ckpt.grad)


def test_gradient_accumulation_equivalence():
    """Verify that accumulation is mathematically identical to a larger batch size."""

    def create_model():
        torch.manual_seed(42)
        return NanoTabPFNModel(
            embedding_size=16,
            num_attention_heads=1,
            mlp_hidden_size=32,
            num_layers=1,
            num_outputs=2,
        )

    model_accum = create_model()
    model_large = create_model()

    torch.manual_seed(42)
    x = torch.randn(4, 5, 3)
    y = torch.randint(0, 2, (4, 5)).float()
    split = 3

    # Large single batch (size 4)
    model_large.train()
    out_large = model_large((x, y), train_test_split_index=split)
    loss_large = out_large.mean()
    loss_large.backward()

    # Accumulated batches (size 2 x 2)
    model_accum.train()
    out_1 = model_accum((x[:2], y[:2]), train_test_split_index=split)
    loss_1 = out_1.mean() / 2.0
    loss_1.backward()

    out_2 = model_accum((x[2:], y[2:]), train_test_split_index=split)
    loss_2 = out_2.mean() / 2.0
    loss_2.backward()

    # Verify outputs match
    assert torch.allclose(torch.cat([out_1, out_2], dim=0), out_large)

    # Verify gradients match exactly
    for p_large, p_accum in zip(model_large.parameters(), model_accum.parameters(), strict=True):
        if p_large.grad is not None and p_accum.grad is not None:
            assert torch.allclose(p_large.grad, p_accum.grad, atol=1e-5)
