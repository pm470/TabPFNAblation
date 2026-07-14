import pytest

from nanotabpfn.model import NanoTabPFNModel


def test_param_count_gelu():
    model = NanoTabPFNModel(
        embedding_size=128,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
    )
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 572_674, f"GELU param count is {param_count}"


@pytest.mark.parametrize("activation", ["swiglu", "bilinear"])
def test_param_count_gated(activation):
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
        activation=activation,
    )
    param_count = sum(p.numel() for p in model.parameters())
    # The gated parameter logic targets the same param count as GELU/Standard MLP
    # Due to integer rounding of hidden dimensions, it might be slightly off.
    assert abs(param_count - 356_066) < 500, f"{activation} param count is {param_count} (expected ~356066)"
