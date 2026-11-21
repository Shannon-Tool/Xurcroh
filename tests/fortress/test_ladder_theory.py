"""Local theory-mode execution of every text ladder rung.

Each rung of ``configs/ladders/text-ladder.yaml`` runs here with a miniature
checkpoint of the rung's own model family through four loopback role workers
and the real ``verify_ladder`` path. The evidence is local: it shows the rung's
graph, placement, verification and SDK path execute, and says nothing about the
official checkpoint or physical hardware.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import torch
import yaml
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)
from tests.fortress.test_ladder import manifest_mapping

from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.ladder_deployment import sdk_configuration
from xurcroh.fortress.ladder_verifier import verify_ladder
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint, load_reap_saliency
from xurcroh.fortress.transport import encode_role_artifact
from xurcroh.fortress.worker import FortressRoleWorker

pytestmark = pytest.mark.slow

LADDER = Path(__file__).resolve().parents[2] / "configs" / "ladders" / "text-ladder.yaml"
RUNGS: list[dict[str, Any]] = yaml.safe_load(LADDER.read_text())["rungs"]
VOCABULARY = 32


def _tokenizer(root: Path) -> None:
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    words = ["<pad>", "<eos>", "user", "reply", "with", "a", "short", "technical", "sentence"]
    backend = Tokenizer(
        WordLevel({word: index for index, word in enumerate(words)}, unk_token="<pad>")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", unk_token="<pad>"
    ).save_pretrained(root)


def _save(model: Any, root: Path) -> None:
    model.save_pretrained(root, safe_serialization=True)


def _wrapped_text_checkpoint(root: Path, outer: str, text_type: str, text: dict[str, Any]) -> None:
    """Write a text decoder the way official multimodal checkpoints nest it."""
    from safetensors.torch import save_file
    from transformers import AutoConfig, AutoModelForCausalLM

    with torch.random.fork_rng():
        torch.manual_seed(812)
        model = AutoModelForCausalLM.from_config(AutoConfig.for_model(text_type, **text))
    ignored = {
        f"model.{name}" for name in getattr(model.model, "_keys_to_ignore_on_load_unexpected", [])
    }
    tensors = {
        (
            name.replace("model.", "model.language_model.", 1)
            if name.startswith("model.")
            else name
        ): (tensor.detach().clone())
        for name, tensor in model.state_dict().items()
        if name not in ignored
    }
    root.mkdir(parents=True, exist_ok=True)
    architecture = {
        "gemma4": "Gemma4ForConditionalGeneration",
        "qwen3_5": "Qwen3_5ForConditionalGeneration",
    }[outer]
    (root / "config.json").write_text(
        json.dumps(
            {
                "model_type": outer,
                "architectures": [architecture],
                "text_config": {"model_type": text_type, **text},
            }
        )
    )
    save_file(tensors, str(root / "model.safetensors"))


def _gemma4(root: Path, **extra: Any) -> None:
    _wrapped_text_checkpoint(
        root,
        "gemma4",
        "gemma4_text",
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "global_head_dim": 8,
            "vocab_size": VOCABULARY,
            "vocab_size_per_layer_input": VOCABULARY,
            "hidden_size_per_layer_input": 4,
            # As in the official checkpoints, the shared-KV tail reuses states
            # produced by an earlier layer of each attention type.
            "layer_types": [
                "sliding_attention",
                "full_attention",
                "sliding_attention",
                "full_attention",
            ],
            "num_kv_shared_layers": 2,
            "use_double_wide_mlp": True,
            "sliding_window": 8,
            "max_position_embeddings": 128,
            **extra,
        },
    )


def _gemma4_moe(root: Path) -> None:
    _gemma4(
        root,
        enable_moe_block=True,
        # Four experts put more than 64 operations on each role worker.
        num_experts=4,
        top_k_experts=2,
        moe_intermediate_size=8,
    )


def _qwen3_5(root: Path) -> None:
    _wrapped_text_checkpoint(
        root,
        "qwen3_5",
        "qwen3_5_text",
        {
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "vocab_size": VOCABULARY,
            "layer_types": ["full_attention", "full_attention"],
            "max_position_embeddings": 128,
        },
    )


def _qwen2(root: Path) -> None:
    from transformers import Qwen2Config, Qwen2ForCausalLM

    _save(
        Qwen2ForCausalLM(
            Qwen2Config(
                vocab_size=VOCABULARY,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=128,
            )
        ),
        root,
    )


def _qwen3_moe(root: Path) -> None:
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    _save(
        Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                vocab_size=VOCABULARY,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=1,
                num_experts=4,
                num_experts_per_tok=2,
                max_position_embeddings=128,
            )
        ),
        root,
    )


def _deepseek(root: Path) -> None:
    from transformers.models.deepseek_v3 import DeepseekV3Config, DeepseekV3ForCausalLM

    _save(
        DeepseekV3ForCausalLM(
            DeepseekV3Config(
                vocab_size=VOCABULARY,
                hidden_size=16,
                intermediate_size=32,
                moe_intermediate_size=8,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=4,
                n_shared_experts=1,
                n_routed_experts=4,
                num_experts_per_tok=1,
                first_k_dense_replace=0,
                n_group=1,
                topk_group=1,
                q_lora_rank=8,
                kv_lora_rank=8,
                qk_rope_head_dim=4,
                qk_nope_head_dim=4,
                v_head_dim=4,
                num_mtp_layers=0,
                max_position_embeddings=128,
            )
        ),
        root,
    )


# The miniature for each rung uses the model family named by its manifest entry.
MINIATURES: dict[str, Callable[[Path], None]] = {
    "gemma-4-e2b": _gemma4,
    "gemma-4-e4b": _gemma4,
    "gemma-4-12b": _gemma4,
    "gemma-4-26b-a4b": _gemma4_moe,
    "gemma-4-31b": _gemma4,
    "qwen-3.8-27b": _qwen3_5,
    "dense-100-300b-class": _qwen2,
    "qwen3-235b-a22b": _qwen3_moe,
    "deepseek-v3-reap": _deepseek,
}


def test_every_text_rung_has_a_miniature_family() -> None:
    assert {entry["model"] for entry in RUNGS} == set(MINIATURES)
    assert [entry["rung"] for entry in RUNGS] == sorted(entry["rung"] for entry in RUNGS)
    assert {entry["rung"] for entry in RUNGS} == {f"L{index}" for index in range(7)}


@pytest.mark.parametrize("entry", RUNGS, ids=[f"{e['rung']}-{e['model']}" for e in RUNGS])
def test_text_rung_executes_in_theory_with_its_model_family(
    entry: dict[str, Any],
    tmp_path: Path,
    mtls_identities: dict[str, bytes],
) -> None:
    checkpoint = tmp_path / "checkpoint"
    with torch.random.fork_rng():
        torch.manual_seed(731)
        MINIATURES[entry["model"]](checkpoint)
    _tokenizer(checkpoint)
    document = {key: value for key, value in entry.items() if key not in {"resolution_required"}}
    document["workers"] = manifest_mapping()["workers"]
    document["checkpoint_fingerprint"] = "b" * 64
    verification = dict(document.get("verification", {}))
    reap: dict[str, Any] = {}
    if "reap_keep_fraction" in verification:
        # Operator-measured saliency is a deployment input; the miniature uses
        # complete synthetic scores for its own four routed experts.
        saliency = tmp_path / "saliency.json"
        saliency.write_text(
            json.dumps(
                {
                    str(layer): {str(expert): 0.1 + 0.2 * expert for expert in range(4)}
                    for layer in range(2)
                }
            )
        )
        verification["reap_saliency_path"] = str(saliency)
        document["verification"] = verification
        reap = {
            "reap_keep_fraction": verification["reap_keep_fraction"],
            "reap_saliency": load_reap_saliency(saliency),
        }
    rung = LadderRung.from_mapping(document)
    compiled = compile_fortress_checkpoint(checkpoint, seed=731, **reap)
    workers: list[FortressRoleWorker] = []
    try:
        for index, placement in enumerate(rung.workers):
            private = X25519PrivateKey.generate()
            public = private.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
            worker = FortressRoleWorker(
                {
                    name: encode_role_artifact(operation, placement.weight_index, public)
                    for name, operation in compiled.operations.items()
                },
                private.private_bytes(Encoding.Raw, PrivateFormat.Raw, NoEncryption()),
                certificates=(mtls_identities["server_certificate"], mtls_identities["ca"]),
                private_key=mtls_identities["server_key"],
                client_ca=mtls_identities["ca"],
                allow_unattested=True,
            )
            worker.start()
            workers.append(worker)
            document["workers"][index]["endpoint"] = worker.address
            document["workers"][index]["recipient_public"] = public.hex()
        del compiled
        placed = LadderRung.from_mapping(document)
        for filename, key in (
            ("client.crt", "client_certificate"),
            ("client.key", "client_key"),
            ("ca.crt", "ca"),
        ):
            (tmp_path / filename).write_bytes(mtls_identities[key])
        sdk = sdk_configuration(
            placed,
            checkpoint_path=checkpoint,
            fortress_seed=731,
            certificate_path=tmp_path / "client.crt",
            key_path=tmp_path / "client.key",
            ca_path=tmp_path / "ca.crt",
        )
        report = verify_ladder({"rung": document, "sdk": sdk})
    finally:
        for worker in workers:
            worker.close()
    assert report["hardware_validated"] is False
    assert report["production_accepted"] is False
    assert report["quantized_reference_max_abs_error"] == 0
    assert report["original_model_max_abs_error"] >= 0
    assert report["sdk"]["response"]["execution"]["engine"] == "fortress-grid"
    assert report["bytes_sent"] > 0
    assert report["bytes_received"] > 0
    assert report["attack_results"]["full_weight_collusion_exact"] is True
    assert report["canary_tripped"] is True
    assert report["freivalds_failures"] == 0
    assert report["lifecycle"]["cancelled"] is True
    assert report["lifecycle"]["concurrent_requests"] == 2
