from __future__ import annotations

import torch

from xurcroh.algebra.attention import sample_orthogonal
from xurcroh.algebra.modulation import SignedPermutation, adaln, block_scalar_adaln


def test_signed_permutation_adaln_is_exact() -> None:
    torch.manual_seed(103)
    x = torch.randn(6, 8, dtype=torch.float64)
    z = torch.randn(6, 8, dtype=torch.float64)
    gamma, beta, alpha = (torch.randn(8, dtype=torch.float64) for _ in range(3))
    basis = SignedPermutation.generate(8, seed=107)
    original = adaln(x, gamma, beta, alpha, z)
    transformed = adaln(
        basis.apply(x),
        gamma[..., basis.permutation],
        basis.apply(beta),
        alpha[..., basis.permutation],
        basis.apply(z),
    )
    torch.testing.assert_close(basis.inverse(transformed), original, atol=1e-12, rtol=1e-12)


def test_block_scalar_adaln_supports_dense_rotation() -> None:
    torch.manual_seed(109)
    x, z = torch.randn(6, 8, dtype=torch.float64), torch.randn(6, 8, dtype=torch.float64)
    basis = sample_orthogonal(8, seed=113)
    beta = torch.randn(8, dtype=torch.float64)
    original = block_scalar_adaln(x, 0.7, beta, -0.4, z)
    rotated = block_scalar_adaln(x @ basis.T, 0.7, beta @ basis.T, -0.4, z @ basis.T)
    torch.testing.assert_close(rotated @ basis, original, atol=1e-12, rtol=1e-12)
