from __future__ import annotations

import torch

from xurcroh.attacks import (
    gram_spectrum,
    nearest_embedding_by_norm,
    orthogonal_procrustes,
)


def test_procrustes_recovers_public_basis_and_contragredient_changes_spectrum() -> None:
    torch.manual_seed(337)
    public_weight = torch.randn(6, 4, dtype=torch.float64)
    basis = torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q
    transformed = public_weight @ basis.T
    recovered = orthogonal_procrustes(public_weight, transformed)
    assert (recovered - basis).abs().max().item() < 1e-8

    scaling = torch.diag(torch.linspace(0.5, 1.5, 6, dtype=torch.float64))
    contragredient = scaling @ transformed
    public_spectrum = gram_spectrum(public_weight)
    transformed_spectrum = gram_spectrum(contragredient)
    assert not torch.allclose(public_spectrum, transformed_spectrum)


def test_distinct_embedding_norms_leak_token_identity() -> None:
    embeddings = torch.eye(4, dtype=torch.float64) * torch.tensor([1.0, 2.0, 3.0, 4.0])
    rotated = embeddings @ torch.linalg.qr(torch.randn(4, 4, dtype=torch.float64)).Q
    matches = nearest_embedding_by_norm(rotated, embeddings)
    assert matches.tolist() == [0, 1, 2, 3]
