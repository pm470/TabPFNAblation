import pytest

from nanotabpfn.model import GatedMLP, NanoTabPFNModel, TransformerEncoderLayer


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
        gated_unrestricted=False,
    )
    param_count = sum(p.numel() for p in model.parameters())
    # The gated parameter logic targets the same param count as GELU/Standard MLP
    # Due to integer rounding of hidden dimensions, it might be slightly off.
    assert abs(param_count - 356_066) < 500, f"{activation} param count is {param_count} (expected ~356066)"


@pytest.mark.parametrize("activation", ["swiglu", "bilinear"])
def test_gated_unrestricted_param_count(activation):
    model_restricted = NanoTabPFNModel(
        embedding_size=128,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
        activation=activation,
        gated_unrestricted=False,
    )
    model_unrestricted = NanoTabPFNModel(
        embedding_size=128,
        num_attention_heads=4,
        mlp_hidden_size=192,
        num_layers=3,
        num_outputs=2,
        activation=activation,
        gated_unrestricted=True,
    )

    params_restricted = sum(p.numel() for p in model_restricted.parameters())
    params_unrestricted = sum(p.numel() for p in model_unrestricted.parameters())

    assert params_unrestricted > params_restricted, (
        f"Unrestricted ({params_unrestricted}) should have more parameters than restricted ({params_restricted})"
    )

    # Check hidden features on the first transformer block's MLP
    block = model_unrestricted.transformer_blocks[0]
    assert isinstance(block, TransformerEncoderLayer)
    gated_mlp_unrestricted = block.mlp
    assert isinstance(gated_mlp_unrestricted, GatedMLP)
    assert gated_mlp_unrestricted.hidden_features == 192, (
        f"Expected hidden_features=192, got {gated_mlp_unrestricted.hidden_features}"
    )
