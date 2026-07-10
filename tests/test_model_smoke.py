"""Smoke tests for the NanoTabPFN model."""

import torch

from nanotabpfn.model import NanoTabPFNModel


def make_model():
    return NanoTabPFNModel(
        embedding_size=128,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
    )


def test_model_output_shape():
    """Model forward pass produces expected output shape."""
    model = make_model()
    model.eval()

    batch_size = 2
    num_rows = 20
    num_features = 5
    train_test_split_index = 10

    x = torch.randn(batch_size, num_rows, num_features)
    y = torch.randn(batch_size, train_test_split_index)

    with torch.no_grad():
        output = model((x, y), train_test_split_index=train_test_split_index)

    expected_test_rows = num_rows - train_test_split_index
    assert output.shape == (batch_size, expected_test_rows, 2), (
        f"Expected shape ({batch_size}, {expected_test_rows}, 2), got {output.shape}"
    )


def test_model_output_finite():
    """Model produces finite outputs (no NaN or Inf)."""
    model = make_model()
    model.eval()

    x = torch.randn(2, 20, 5)
    y = torch.randn(2, 10)

    with torch.no_grad():
        output = model((x, y), train_test_split_index=10)

    assert torch.isfinite(output).all(), "Model output contains NaN or Inf"


def test_model_param_count():
    """Parameter count matches expected value for default config."""
    model = make_model()
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 572_674, f"Expected 572,674 parameters, got {param_count:,}"


def test_classifier_predict():
    """NanoTabPFNClassifier.predict returns class labels via argmax."""
    import numpy as np

    from nanotabpfn.model import NanoTabPFNClassifier

    model = make_model()
    classifier = NanoTabPFNClassifier(model, torch.device("cpu"))

    # Dummy data
    X_train = np.random.randn(10, 5)
    y_train = np.random.randint(0, 2, (10,))
    X_test = np.random.randn(5, 5)

    classifier.fit(X_train, y_train)
    predictions = classifier.predict(X_test)

    assert isinstance(predictions, np.ndarray)
    assert predictions.shape == (5,)
    assert predictions.dtype == np.int64 or predictions.dtype == np.int32
    assert set(predictions).issubset({0, 1})
