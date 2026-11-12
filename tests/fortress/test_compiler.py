from __future__ import annotations

import torch

from xurcroh.fortress.arithmetic import combine_shares
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
    assert torch.equal(combined, compiled.quantized.values.remainder(2**32))


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
