from __future__ import annotations

from pathlib import Path

import torch
from tests.architectures.utils import write_llama_gguf
from tests.formats.exl_utils import exl2_fixture

from xurcroh.architectures import open_architecture
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint


def test_gguf_tiny_llama_compiles_canonical_fortress_operations(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    checkpoint = write_llama_gguf(tiny_llama_path, tmp_path / "tiny.gguf")
    with open_architecture(checkpoint) as adapter:
        names = adapter.reader.tensor_names()
        assert "model.embed_tokens.weight" in names
        assert "model.layers.0.self_attn.q_proj.weight" in names
        assert adapter.reader.config["model_type"] == "llama"
    compiled = compile_fortress_checkpoint(checkpoint, load_model=True, seed=731)
    assert compiled.model is not None
    assert len(compiled.layers) == 2
    assert len(compiled.operations) == 14
    assert compiled.embedding.shape[0] > 0
    hidden = torch.tensor([[0.1, -0.2, 0.3, 0.4] * 4], dtype=torch.float64)
    reference = compiled.combiner().reference_forward(hidden)
    actual = compiled.combiner().run(hidden)
    torch.testing.assert_close(actual, reference, atol=0, rtol=0)


def test_exl2_adapter_exposes_canonical_weight_name(tmp_path: Path) -> None:
    _, expected = exl2_fixture(tmp_path)
    with open_architecture(tmp_path) as adapter:
        name = "model.layers.0.self_attn.q_proj.weight"
        assert name in adapter.reader.tensor_names()
        actual = adapter.reader.read_tensors([name])[name]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_gguf_fused_qkv_and_stacked_experts(tmp_path: Path) -> None:
    import numpy as np
    from tests.formats.gguf_utils import write_gguf

    fused = np.arange(32, dtype=np.float32).reshape(8, 4)
    experts = np.arange(48, dtype=np.float32).reshape(2, 6, 4)
    embedding = np.zeros((4, 4), dtype=np.float32)
    checkpoint = write_gguf(
        tmp_path / "fused.gguf",
        [
            ("token_embd.weight", (4, 4), 0, embedding.tobytes()),
            ("blk.0.attn_qkv.weight", (8, 4), 0, fused.tobytes()),
            ("blk.0.ffn_gate_exps.weight", (2, 6, 4), 0, experts.tobytes()),
        ],
        metadata={"llama.expert_count": 2, "llama.expert_used_count": 1},
    )
    with open_architecture(checkpoint) as adapter:
        names = adapter.reader.tensor_names()
        shapes = adapter.reader.tensor_shapes(list(names))
        q = adapter.reader.read_tensors(["model.layers.0.self_attn.q_proj.weight"])
        k = adapter.reader.read_tensors(["model.layers.0.self_attn.k_proj.weight"])
        v = adapter.reader.read_tensors(["model.layers.0.self_attn.v_proj.weight"])
        expert = adapter.reader.read_tensors(["model.layers.0.mlp.experts.1.gate_proj.weight"])
    torch.testing.assert_close(
        q["model.layers.0.self_attn.q_proj.weight"], torch.from_numpy(fused[:4])
    )
    torch.testing.assert_close(
        k["model.layers.0.self_attn.k_proj.weight"], torch.from_numpy(fused[4:6])
    )
    torch.testing.assert_close(
        v["model.layers.0.self_attn.v_proj.weight"], torch.from_numpy(fused[6:])
    )
    torch.testing.assert_close(
        expert["model.layers.0.mlp.experts.1.gate_proj.weight"], torch.from_numpy(experts[1])
    )
    assert shapes["model.layers.0.mlp.experts.1.gate_proj.weight"] == (6, 4)


def test_shape_discovery_does_not_decode_weights(tiny_llama_path: Path, tmp_path: Path) -> None:
    checkpoint = write_llama_gguf(tiny_llama_path, tmp_path / "tiny.gguf")
    with open_architecture(checkpoint) as adapter:

        def fail_decode(names: list[str]) -> dict[str, torch.Tensor]:
            raise AssertionError("shape discovery decoded weight data")

        adapter.reader.source.read_tensors = fail_decode  # type: ignore[method-assign]
        shapes = adapter.reader.tensor_shapes(["model.layers.0.self_attn.q_proj.weight"])
    assert shapes["model.layers.0.self_attn.q_proj.weight"] == (16, 16)
