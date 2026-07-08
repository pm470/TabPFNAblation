import pytest

from nanotabpfn.model import NanoTabPFNModel


def test_param_count_gelu():
    model = NanoTabPFNModel(
        embedding_size=96,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
    )
    param_count = sum(p.numel() for p in model.parameters())
    assert param_count == 356_066, f"GELU param count is {param_count}"


@pytest.mark.parametrize("activation", ["swiglu", "geglu", "reglu"])
def test_param_count_gated(activation):
    # This will fail right now because NanoTabPFNModel doesn't accept `activation` yet
    pass
