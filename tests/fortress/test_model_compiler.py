from __future__ import annotations

from pathlib import Path

import torch

from xurcroh.fortress.combiner import FortressLayerMetadata
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
from xurcroh.fortress.transport import encode_role_artifact


def test_real_checkpoint_compiles_without_retaining_full_client_weights(
    tiny_llama_path: Path,
) -> None:
    compiled = compile_fortress_checkpoint(tiny_llama_path)
    assert len(compiled.layers) == 2
    assert len(compiled.operations) == 14
    assert all(
        operation.quantized.values.dtype == torch.int8 for operation in compiled.operations.values()
    )
    assert all(isinstance(layer, FortressLayerMetadata) for layer in compiled.layers)
    combiner = compiled.combiner()
    hidden = torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.4] * 4,
            [0.5, 0.2, -0.1, -0.3] * 4,
        ],
        dtype=torch.float64,
    )
    expected = combiner.reference_forward(hidden)
    prefill = combiner.run_cached("compiled-request", hidden[:1])
    decoded = combiner.run_cached("compiled-request", hidden[1:])
    actual = torch.cat([prefill, decoded])
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    assert combiner.sample(actual[-1:]).item() == combiner.reference_sample(expected[-1:]).item()
    assert all(
        b"basis" not in encode_role_artifact(operation, 0, bytes(range(32)))
        for operation in compiled.operations.values()
    )
    combiner.drop_request("compiled-request")


def test_remote_client_compilation_discards_worker_weight_matrices(
    tiny_llama_path: Path,
) -> None:
    full = compile_fortress_checkpoint(tiny_llama_path, seed=731)
    client = compile_fortress_checkpoint(tiny_llama_path, seed=731, materialize_shares=False)
    assert client.operations.keys() == full.operations.keys()
    for name, operation in client.operations.items():
        reference = full.operations[name]
        assert set(operation.shares) == set(reference.shares)
        assert operation.storage == {}
        assert all(share.matrix is None for share in operation.shares.values())
        for index, share in operation.shares.items():
            assert share.seed == reference.shares[index].seed


def test_integrity_challenges_are_not_derived_from_deployment_seed(
    tiny_llama_path: Path,
) -> None:
    first = compile_fortress_checkpoint(tiny_llama_path, seed=731, load_model=False)
    second = compile_fortress_checkpoint(tiny_llama_path, seed=731, load_model=False)
    for name, operation in first.operations.items():
        other = second.operations[name]
        assert not torch.equal(operation.shares[0].challenge.r, other.shares[0].challenge.r)
        assert torch.equal(operation.input_basis.permutation, other.input_basis.permutation)
        assert operation.shares[0].seed == other.shares[0].seed


def test_remote_client_can_discard_quantized_weight_values(tiny_llama_path: Path) -> None:
    compiled = compile_fortress_checkpoint(
        tiny_llama_path,
        seed=731,
        materialize_shares=False,
        retain_quantized=False,
        load_model=False,
    )
    for operation in compiled.operations.values():
        assert operation.quantized.values is None
        assert len(operation.quantized.dimensions) == 2
        assert operation.quantized.scale > 0


def test_remote_model_uses_fp32_and_shares_boundary_storage(tiny_llama_path: Path) -> None:
    compiled = compile_fortress_checkpoint(
        tiny_llama_path,
        seed=731,
        materialize_shares=False,
        model_dtype=torch.float32,
    )
    assert compiled.model is not None
    assert compiled.embedding.dtype == torch.float32
    assert compiled.model.get_input_embeddings().weight.data_ptr() == compiled.embedding.data_ptr()
    assert compiled.model.get_output_embeddings().weight.data_ptr() == compiled.lm_head.data_ptr()


def test_reference_compilation_can_omit_the_official_model(tiny_llama_path: Path) -> None:
    compiled = compile_fortress_checkpoint(
        tiny_llama_path, seed=731, materialize_shares=False, load_model=False
    )
    assert compiled.model is None
    hidden = torch.tensor([[0.1, -0.2, 0.3, 0.4] * 4], dtype=torch.float64)
    reference = compiled.combiner().reference_forward(hidden)
    assert reference.shape == hidden.shape


def test_checkpoint_compiles_and_executes_three_way_shares(tiny_llama_path: Path) -> None:
    compiled = compile_fortress_checkpoint(tiny_llama_path, seed=731, share_count=3)
    assert all(len(operation.shares) == 3 for operation in compiled.operations.values())
    hidden = torch.tensor([[0.1, -0.2, 0.3, 0.4] * 4], dtype=torch.float64)
    combiner = compiled.combiner()
    expected = combiner.reference_forward(hidden)
    actual = combiner.run(hidden)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)


def test_moe_checkpoint_keeps_router_local_and_compiles_each_expert(tmp_path: Path) -> None:
    import json

    from safetensors.torch import save_file

    config = {
        "model_type": "test_moe",
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 2,
        "num_hidden_layers": 1,
        "num_experts_per_tok": 1,
        "rope_theta": 10000,
    }
    (tmp_path / "config.json").write_text(json.dumps(config))
    generator = torch.Generator().manual_seed(912)

    def matrix(*shape: int) -> torch.Tensor:
        return torch.randn(*shape, generator=generator, dtype=torch.float64) / 8

    tensors = {
        "model.embed_tokens.weight": matrix(12, 4),
        "model.norm.weight": torch.ones(4, dtype=torch.float64),
        "lm_head.weight": matrix(12, 4),
        "model.layers.0.input_layernorm.weight": torch.ones(4, dtype=torch.float64),
        "model.layers.0.post_attention_layernorm.weight": torch.ones(4, dtype=torch.float64),
        "model.layers.0.mlp.gate.weight": matrix(2, 4),
    }
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        tensors[f"model.layers.0.self_attn.{projection}.weight"] = matrix(4, 4)
    for expert in range(2):
        tensors[f"model.layers.0.mlp.experts.{expert}.gate_proj.weight"] = matrix(6, 4)
        tensors[f"model.layers.0.mlp.experts.{expert}.up_proj.weight"] = matrix(6, 4)
        tensors[f"model.layers.0.mlp.experts.{expert}.down_proj.weight"] = matrix(4, 6)
    save_file(tensors, str(tmp_path / "model.safetensors"))

    compiled = compile_fortress_checkpoint(tmp_path)
    assert len(compiled.operations) == 10
    assert compiled.layers[0].router is not None
    combiner = compiled.combiner()
    hidden = torch.tensor([[0.2, -0.1, 0.3, 0.4], [0.1, 0.5, -0.2, 0.2]], dtype=torch.float64)
    expected = combiner.reference_forward(hidden)
    actual = combiner.run_cached("moe-request", hidden)
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-7)
    combiner.drop_request("moe-request")


def test_language_model_prefixed_checkpoint_compiles(tiny_llama_path: Path, tmp_path: Path) -> None:
    from safetensors import safe_open
    from safetensors.torch import save_file

    with safe_open(tiny_llama_path / "model.safetensors", framework="pt") as handle:
        tensors = {
            (
                name.replace("model.", "model.language_model.", 1)
                if name.startswith("model.")
                else name
            ): handle.get_tensor(name)
            for name in handle.keys()  # noqa: SIM118
        }
    config = __import__("json").loads((tiny_llama_path / "config.json").read_text())
    config["model_type"] = "test_prefixed"
    (tmp_path / "config.json").write_text(__import__("json").dumps(config))
    save_file(tensors, str(tmp_path / "model.safetensors"))
    compiled = compile_fortress_checkpoint(tmp_path)
    assert compiled.embedding.shape[0] > 0
    assert compiled.operations
    assert all(name.startswith("model.language_model.layers.") for name in compiled.operations)


def test_gemma4_text_loader_excludes_multimodal_towers_and_executes(tmp_path: Path) -> None:
    import json

    from safetensors.torch import save_file
    from transformers.models.gemma4 import Gemma4ForCausalLM, Gemma4TextConfig

    text_config = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "global_head_dim": 4,
        "vocab_size": 12,
        "vocab_size_per_layer_input": 12,
        "hidden_size_per_layer_input": 4,
        "layer_types": ["full_attention", "full_attention"],
        "num_kv_shared_layers": 1,
        "use_double_wide_mlp": False,
        "max_position_embeddings": 128,
    }
    config = Gemma4TextConfig(**text_config)
    with torch.device("meta"):
        reference = Gemma4ForCausalLM(config)
    ignored = {f"model.{name}" for name in reference.model._keys_to_ignore_on_load_unexpected}
    generator = torch.Generator().manual_seed(812)
    tensors = {}
    for name, parameter in reference.state_dict().items():
        if name in ignored:
            continue
        checkpoint_name = (
            name.replace("model.", "model.language_model.", 1)
            if name.startswith("model.")
            else name
        )
        tensors[checkpoint_name] = (
            torch.randn(tuple(parameter.shape), generator=generator, dtype=torch.float64) / 8
        )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "gemma4",
                "architectures": ["Gemma4ForConditionalGeneration"],
                "text_config": text_config,
            }
        )
    )
    save_file(tensors, str(tmp_path / "model.safetensors"))
    compiled = compile_fortress_checkpoint(tmp_path, seed=731)
    assert type(compiled.model).__name__ == "Gemma4ForCausalLM"
    assert not hasattr(compiled.model.model, "vision_tower")
    with torch.inference_mode():
        output = compiled.model(
            input_ids=torch.tensor([[1, 2]]),
            return_shared_kv_states=True,
            use_cache=True,
        )
    assert output.logits.shape == (1, 2, 12)
    assert torch.isfinite(output.logits).all()


def test_qwen3_5_text_loader_excludes_multimodal_towers_and_executes(tmp_path: Path) -> None:
    import json

    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    text_config = {
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "vocab_size": 12,
        "layer_types": ["full_attention"],
        "max_position_embeddings": 128,
    }
    config = AutoConfig.for_model("qwen3_5_text", **text_config)
    with torch.device("meta"):
        reference = AutoModelForCausalLM.from_config(config)
    generator = torch.Generator().manual_seed(813)
    tensors = {}
    for name, parameter in reference.state_dict().items():
        checkpoint_name = (
            name.replace("model.", "model.language_model.", 1)
            if name.startswith("model.")
            else name
        )
        tensors[checkpoint_name] = (
            torch.randn(tuple(parameter.shape), generator=generator, dtype=torch.float64) / 8
        )
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "qwen3_5",
                "architectures": ["Qwen3_5ForConditionalGeneration"],
                "text_config": {"model_type": "qwen3_5_text", **text_config},
            }
        )
    )
    save_file(tensors, str(tmp_path / "model.safetensors"))
    compiled = compile_fortress_checkpoint(tmp_path, seed=731)
    assert type(compiled.model).__name__ == "Qwen3_5ForCausalLM"
    assert not hasattr(compiled.model.model, "visual")
    with torch.inference_mode():
        output = compiled.model(
            input_ids=torch.tensor([[1, 2]]),
            return_shared_kv_states=True,
            use_cache=True,
        )
    assert output.logits.shape == (1, 2, 12)
    assert torch.isfinite(output.logits).all()
