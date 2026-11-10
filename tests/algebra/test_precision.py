from __future__ import annotations

import torch

from xurcroh.algebra.attention import sample_orthogonal
from xurcroh.algebra.orthogonal import conjugate_linear


def test_linear_conjugation_fp32_and_bf16_error_is_bounded() -> None:
    torch.manual_seed(127)
    weight = torch.randn(12, 8)
    bias = torch.randn(12)
    x = torch.randn(5, 8)
    basis_in = sample_orthogonal(8, seed=131, dtype=torch.float32)
    basis_out = sample_orthogonal(12, seed=137, dtype=torch.float32)
    t_weight, t_bias = conjugate_linear(weight, bias, basis_in, basis_out)
    reference = x @ weight.T + bias
    transformed = (x @ basis_in.T) @ t_weight.T + t_bias
    torch.testing.assert_close(transformed @ basis_out, reference, atol=2e-5, rtol=2e-5)

    bf_reference = (reference @ basis_out.T).bfloat16().float() @ basis_out
    bf_transformed = transformed.bfloat16().float() @ basis_out
    max_error = (bf_reference - bf_transformed).abs().max().item()
    assert max_error < 0.05
