"""Tests for mixed-precision training with torch.autocast."""

import torch

from nanotabpfn.model import NanoTabPFNModel
from nanotabpfn.train import train


def _make_mock_prior(batch_size=2, rows=20, features=5, num_classes=2):
    """Create a single-batch mock prior for testing."""
    return [
        {
            "x": torch.randn(batch_size, rows, features, dtype=torch.float32),
            "y": torch.randint(0, num_classes, (batch_size, rows), dtype=torch.float32),
            "train_test_split_index": 10,
        }
    ]


def _make_small_model():
    """Create a small model for testing."""
    return NanoTabPFNModel(
        embedding_size=32,
        num_attention_heads=2,
        mlp_hidden_size=64,
        num_layers=1,
        num_outputs=2,
    )


def test_train_autocast_none_disables(capsys):
    """Passing autocast_dtype=None disables autocast — no mixed-precision message printed."""
    model = _make_small_model()
    mock_prior = _make_mock_prior()

    _, eval_history = train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=1,
        eval_func=None,
        autocast_dtype=None,
    )
    assert len(eval_history) == 1

    captured = capsys.readouterr()
    assert "Mixed-precision training enabled" not in captured.out


def test_train_autocast_on_cpu_prints_warning(capsys):
    """On CPU, autocast is not applied and a warning is printed."""
    model = _make_small_model()
    mock_prior = _make_mock_prior()

    _, eval_history = train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=1,
        eval_func=None,
        autocast_dtype=torch.bfloat16,
    )
    assert len(eval_history) == 1

    captured = capsys.readouterr()
    assert "not supported" in captured.out


def test_train_default_autocast_dtype_is_bfloat16():
    """The default autocast_dtype parameter is torch.bfloat16."""
    import inspect

    sig = inspect.signature(train)
    default = sig.parameters["autocast_dtype"].default
    assert default is torch.bfloat16


def test_train_autocast_none_produces_finite_loss():
    """Training with autocast_dtype=None still produces finite loss on CPU."""
    model = _make_small_model()
    mock_prior = _make_mock_prior()

    _, eval_history = train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=1,
        eval_func=None,
        autocast_dtype=None,
    )
    assert len(eval_history) == 1
    assert eval_history[0]["loss"] > 0  # Loss should be positive and finite
    assert not torch.isnan(torch.tensor(eval_history[0]["loss"]))


def test_train_model_weights_remain_float32():
    """Model parameters should remain in float32 regardless of autocast setting."""
    model = _make_small_model()
    mock_prior = _make_mock_prior()

    train(
        model,
        mock_prior,  # type: ignore
        lr=1e-3,
        device=torch.device("cpu"),
        steps_per_eval=1,
        eval_func=None,
        autocast_dtype=torch.bfloat16,
    )

    for name, param in model.named_parameters():
        assert param.dtype == torch.float32, f"Parameter {name} has dtype {param.dtype}, expected float32"
