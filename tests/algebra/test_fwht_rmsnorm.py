from __future__ import annotations

import pytest
import torch

from xurcroh.algebra.fwht import fwht, hadamard_matrix
from xurcroh.algebra.rmsnorm import absorb_rms_gain, unit_rms_norm


def test_keyed_fwht_matches_dense_matrix_and_inverts() -> None:
    torch.manual_seed(7)
    x = torch.randn(3, 8, dtype=torch.float64)
    signs = torch.tensor([1, -1, 1, 1, -1, 1, -1, 1], dtype=torch.float64)
    actual = fwht(x, signs)
    expected = (x * signs) @ hadamard_matrix(8, x.dtype).T / 8**0.5
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(fwht(actual, signs, inverse=True), x, atol=1e-12, rtol=1e-12)


def test_fwht_rejects_non_power_of_two_without_silent_padding() -> None:
    with pytest.raises(ValueError, match="power of two"):
        fwht(torch.ones(2, 6))


def test_rms_gain_absorption_preserves_projection() -> None:
    torch.manual_seed(11)
    x = torch.randn(4, 16, dtype=torch.float64)
    gain = torch.randn(16, dtype=torch.float64)
    weight = torch.randn(12, 16, dtype=torch.float64)
    original = (unit_rms_norm(x, 1e-6) * gain) @ weight.T
    absorbed = unit_rms_norm(x, 1e-6) @ absorb_rms_gain(weight, gain).T
    torch.testing.assert_close(original, absorbed, atol=1e-12, rtol=1e-12)


def test_unit_rms_norm_is_orthogonally_equivariant() -> None:
    torch.manual_seed(13)
    x = torch.randn(4, 8, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(8, 8, dtype=torch.float64))
    torch.testing.assert_close(
        unit_rms_norm(x @ q.T, 1e-6),
        unit_rms_norm(x, 1e-6) @ q.T,
        atol=1e-12,
        rtol=1e-12,
    )
