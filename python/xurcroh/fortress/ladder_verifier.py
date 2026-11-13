from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch

from xurcroh.errors import CapabilityError
from xurcroh.fortress.combiner import FortressCombiner
from xurcroh.fortress.ladder import LadderRung
from xurcroh.fortress.model_compiler import compile_fortress_checkpoint
from xurcroh.fortress.transport import FortressRoleClient, MutualTLSIdentity, RoleEndpoint


def _identity(cluster: list[dict[str, Any]]) -> MutualTLSIdentity:
    labels = cluster[0]["labels"]
    return MutualTLSIdentity(
        certificates=(
            Path(labels["client_certificate_path"]).read_bytes(),
            Path(labels["ca_path"]).read_bytes(),
        ),
        private_key=Path(labels["client_key_path"]).read_bytes(),
        pinned_ca=Path(labels["ca_path"]).read_bytes(),
        server_name=labels.get("server_name", "localhost"),
    )


def _roles(cluster: list[dict[str, Any]]) -> list[RoleEndpoint]:
    return [
        RoleEndpoint(
            activation_index=int(worker["labels"]["activation_index"]),
            weight_index=int(worker["labels"]["weight_index"]),
            address=worker["endpoint"].removeprefix("quic://"),
            recipient_public=bytes.fromhex(worker["labels"]["recipient_public"]),
            provider_id=worker["name"],
        )
        for worker in cluster
    ]


def verify_ladder(deployment: dict[str, Any]) -> dict[str, Any]:
    rung = LadderRung.from_mapping(deployment["rung"])
    if rung.modality not in {"text", "vlm"}:
        raise CapabilityError("this verifier currently accepts text and VLM text-boundary rungs")
    sdk = deployment["sdk"]
    definition = sdk["model_path"][rung.model]
    started = time.perf_counter()
    compiled = compile_fortress_checkpoint(
        definition["path"],
        seed=definition["fortress_seed"],
        share_count=rung.share_count,
    )
    cluster = sdk["cluster"]
    identity = _identity(cluster)
    roles = _roles(cluster)
    clients = {
        name: FortressRoleClient(operation, roles, identity, operation_name=name)
        for name, operation in compiled.operations.items()
    }
    prompt = "user: Reply with a short technical sentence."
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            definition["path"],
            revision=rung.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        hidden = compiled.embedding[input_ids[0]]
        reference = compiled.combiner()
        expected = reference.reference_forward(hidden)
        remote = FortressCombiner(
            layers=list(compiled.layers),
            operations=compiled.operations,
            query_heads=compiled.query_heads,
            kv_heads=compiled.kv_heads,
            final_norm=compiled.final_norm,
            lm_head=compiled.lm_head,
            rope_theta=compiled.rope_theta,
            activation_bits=compiled.activation_bits,
            grid_clients=clients,
        )
        actual = remote.run(hidden)
        quantized_error = float((actual - expected).abs().max())
        quantized_logits = reference.logits(expected[-1:])[0]
        original = AutoModelForCausalLM.from_pretrained(
            definition["path"],
            revision=rung.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        original.eval()  # type: ignore[no-untyped-call]
        with torch.inference_mode():
            original_logits = original(input_ids=input_ids).logits[0, -1].to(torch.float64)
        original_error = float((quantized_logits - original_logits).abs().max())
    finally:
        for role_client in clients.values():
            role_client.close()
    direct_sent = sum(client.bytes_sent for client in clients.values())
    direct_received = sum(client.bytes_received for client in clients.values())
    from Xurcroh import Xurcroh

    with Xurcroh(**sdk) as client:
        response = client.chat.completions.create(
            model=rung.model,
            messages=[{"role": "user", "content": "Reply with a short technical sentence."}],
            temperature=0,
            max_tokens=1,
        )
        metrics = client.metrics()
    return {
        "hardware_validated": deployment.get("hardware_validated", False),
        "production_accepted": False,
        "quantized_reference_max_abs_error": quantized_error,
        "original_model_max_abs_error": original_error,
        "bytes_sent": direct_sent + int(metrics.get("fortress_bytes_sent", 0)),
        "bytes_received": direct_received + int(metrics.get("fortress_bytes_received", 0)),
        "inference_seconds": time.perf_counter() - started,
        "sdk": {
            "response": response.model_dump(mode="json"),
            "metrics": metrics,
        },
    }
