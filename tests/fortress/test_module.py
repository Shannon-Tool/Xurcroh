from __future__ import annotations

import torch

from xurcroh.fortress.compiler import compile_operation
from xurcroh.fortress.module import FortressLinear, replace_linear_layers


def test_linear_replacement_preserves_fixed_point_forward_and_bias() -> None:
    torch.manual_seed(812)
    original = torch.nn.Linear(4, 4, bias=True)
    operation = compile_operation(original.weight.detach().clone(), threshold=2, slice_count=3)
    replacement = FortressLinear("projection", operation, bias=original.bias.detach().clone())
    model = torch.nn.Sequential(original)
    replaced = replace_linear_layers(model, {"0.weight": operation})
    assert isinstance(replaced[0], FortressLinear)
    hidden = torch.randn(2, 4, dtype=torch.float64)
    expected = replacement.reference(hidden)
    actual = replaced(hidden)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    assert not hasattr(replaced[0], "weight")
    assert replaced[0].bias is not None


def test_reference_mode_supports_metadata_only_operations() -> None:
    from xurcroh.fortress.module import set_reference_mode

    original = torch.nn.Linear(4, 4, bias=False, dtype=torch.float64)
    operation = compile_operation(
        original.weight.detach().clone(),
        threshold=2,
        slice_count=3,
        materialize_shares=False,
    )
    model = replace_linear_layers(torch.nn.Sequential(original), {"0.weight": operation})
    set_reference_mode(model)
    hidden = torch.randn(3, 4, dtype=torch.float64)
    torch.testing.assert_close(model(hidden), model[0].reference(hidden))
