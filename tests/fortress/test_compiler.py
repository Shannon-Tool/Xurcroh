from __future__ import annotations

import torch

from xurcroh.fortress.arithmetic import PRIME_MODULUS, combine_shares
from xurcroh.fortress.compiler import compile_operation, reap_prune_experts
from xurcroh.fortress.storage import reconstruct_shard


def test_compiler_quantizes_shares_and_disperses_every_weight_share() -> None:
    weight = torch.tensor([[-8, 7], [3, -2]], dtype=torch.int64)
    compiled = compile_operation(weight, threshold=2, slice_count=3)

    assert compiled.quantized.values.dtype == torch.int64
    assert len(compiled.shares) == 2
    assert len(compiled.storage) == 2
    for index, stored in compiled.storage.items():
        payload = reconstruct_shard(
            stored.bundle, stored.bundle.slices, stored.bundle.key_shares, lambda _: True
        )
        assert torch.equal(
            torch.frombuffer(bytearray(payload), dtype=torch.int64).reshape(stored.shape),
            compiled.shares[index].matrix,
        )
    combined = combine_shares([share.matrix for share in compiled.shares.values()])
    assert torch.equal(combined, compiled.quantized.values.remainder(PRIME_MODULUS))


def test_metadata_only_shares_can_be_reconstructed_for_client_attacks() -> None:
    from xurcroh.fortress import compiler

    operation = compile_operation(
        torch.arange(24, dtype=torch.int64).reshape(4, 6),
        threshold=2,
        slice_count=3,
        share_count=3,
        seed=bytes(range(32)),
        materialize_shares=False,
    )
    shares = compiler.materialize_weight_shares(operation)
    assert all(share.dtype == torch.int64 for share in shares)
    assert torch.equal(combine_shares(shares), operation.quantized.values.remainder(PRIME_MODULUS))


def test_reap_pruning_precedes_compilation() -> None:
    experts = {
        "expert-0": torch.ones(2, 2, dtype=torch.int64),
        "expert-1": torch.full((2, 2), 2, dtype=torch.int64),
    }
    scores = {"expert-0": 0.1, "expert-1": 0.9}
    pruned = reap_prune_experts(experts, scores, keep_fraction=0.5)

    assert set(pruned) == {"expert-1"}
    compiled = compile_operation(pruned["expert-1"], threshold=2, slice_count=3)
    assert torch.equal(compiled.quantized.dequantize(), experts["expert-1"].to(torch.float64))


def test_inner_bases_are_client_only_and_preserve_the_linear_reference() -> None:
    from xurcroh.algebra.attention import sample_orthogonal
    from xurcroh.fortress.arithmetic import quantize_activations, signed_ring, split_secret
    from xurcroh.fortress.grid import execute_grid
    from xurcroh.fortress.transport import encode_role_artifact

    weight = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    input_basis = sample_orthogonal(2, seed=19)
    output_basis = sample_orthogonal(2, seed=23)
    compiled = compile_operation(
        weight,
        threshold=2,
        slice_count=3,
        input_basis=input_basis,
        output_basis=output_basis,
    )
    hidden = torch.tensor([[0.25, -0.5]], dtype=torch.float64)
    activation = quantize_activations(hidden @ input_basis.T, bits=12)
    product = execute_grid(split_secret(activation.values, count=2).shares, compiled.shares)
    transformed = (
        signed_ring(product).to(torch.float64) * activation.scales * compiled.quantized.scale
    )
    recovered = transformed @ output_basis
    expected = (
        (activation.values @ compiled.quantized.values.T).to(torch.float64)
        * activation.scales
        * compiled.quantized.scale
        @ output_basis
    )
    torch.testing.assert_close(recovered, expected, atol=1e-7, rtol=1e-7)
    assert torch.isfinite(recovered - hidden @ weight.T).all()
    artifact = encode_role_artifact(compiled, 0, bytes(range(32)))
    assert b"basis" not in artifact
