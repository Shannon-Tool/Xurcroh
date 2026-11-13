from __future__ import annotations

import logging
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
    if rung.modality in {"image", "video"}:
        return _verify_media(deployment, rung)
    if rung.modality not in {"text", "vlm"}:
        raise CapabilityError("this verifier currently accepts text and VLM text-boundary rungs")
    sdk = deployment["sdk"]
    definition = sdk["model_path"][rung.model]
    started = time.perf_counter()
    compiled = compile_fortress_checkpoint(
        definition["path"],
        seed=definition["fortress_seed"],
        share_count=rung.share_count,
        materialize_shares=False,
        load_model=False,
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
        del original
        from xurcroh.fortress.attacks import exercise_adversarial_grid

        adversarial = exercise_adversarial_grid(clients, roles, compiled.operations)
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
        **adversarial,
        "sdk": {
            "response": response.model_dump(mode="json"),
            "metrics": metrics,
        },
    }


def _verify_media(deployment: dict[str, Any], rung: LadderRung) -> dict[str, Any]:
    import inspect

    from xurcroh.config import ModelDefinition
    from xurcroh.fortress.diffusers_loader import compile_diffusers_component
    from xurcroh.fortress.module import replace_linear_layers
    from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
    from xurcroh.runtime.media_boundary import DiffusionMediaBoundary

    sdk = deployment["sdk"]
    definition = ModelDefinition.model_validate(sdk["model_path"][rung.model])
    component = str(
        deployment["rung"].get(
            "diffusion_component",
            "transformer" if (Path(definition.path) / "transformer").is_dir() else "unet",
        )
    )
    started = time.perf_counter()
    operations = compile_diffusers_component(
        definition.path,
        component=component,
        seed=definition.fortress_seed,
        share_count=rung.share_count,
    )
    cluster = sdk["cluster"]
    identity = _identity(cluster)
    roles = _roles(cluster)
    clients = {
        name: FortressRoleClient(operation, roles, identity, operation_name=name)
        for name, operation in operations.items()
    }
    phase_times: dict[str, float] = {}
    phase_started = time.perf_counter()

    def phase(name: str) -> None:
        nonlocal phase_started
        now = time.perf_counter()
        phase_times[name] = now - phase_started
        logging.getLogger(__name__).info(
            "Fortress media phase %s took %.3fs", name, now - phase_started
        )
        phase_started = now

    phase("compile_and_connect")

    def boundary(grid: dict[str, FortressRoleClient] | None) -> Any:
        result = DiffusionMediaBoundary.from_definition(definition)
        module = getattr(result.pipeline, component)
        replace_linear_layers(module, operations, grid_clients=grid)
        return result

    try:
        local = boundary(None)
        remote = boundary(clients)
        original = DiffusionMediaBoundary.from_definition(definition)
        phase("load_boundaries")
        verification = deployment["rung"].get("verification", {})
        common: dict[str, Any] = {
            "model": rung.model,
            "prompt": "a small red square",
            "steps": int(verification.get("steps", 1)),
            "seed": 11,
            "size": str(verification.get("size", "64x64")),
            "timeout": float(verification.get("timeout", 3600)),
        }
        frames = int(verification.get("frames", 4))
        fps = int(verification.get("fps", 4))
        width, height = (int(value) for value in str(common["size"]).split("x"))
        request: ImageGenerationRequest | VideoGenerationRequest
        if rung.modality == "image":
            request = ImageGenerationRequest(**common)
            parameters = {
                "prompt": request.prompt,
                "width": width,
                "height": height,
                "num_inference_steps": request.steps,
                "output_type": "latent",
            }
        else:
            request = VideoGenerationRequest(**common, frames=frames, fps=fps)
            parameters = {
                "prompt": request.prompt,
                "width": width,
                "height": height,
                "num_frames": frames,
                "fps": fps,
                "num_inference_steps": request.steps,
                "output_type": "latent",
            }

        supports_latent_decode = hasattr(original.pipeline, "decode_latents") or hasattr(
            original.pipeline, "vae"
        )
        if not supports_latent_decode:
            parameters["output_type"] = "pt"

        def generate(pipeline: Any) -> torch.Tensor:
            signature = inspect.signature(pipeline.__call__).parameters
            selected = {key: value for key, value in parameters.items() if key in signature}
            selected["generator"] = torch.Generator().manual_seed(11)
            result = pipeline(**selected)
            values = (
                getattr(result, "images", None)
                if rung.modality == "image"
                else getattr(result, "frames", None)
            )
            if not isinstance(values, torch.Tensor):
                raise CapabilityError("media pipeline did not return a latent tensor")
            return values

        expected = generate(local.pipeline)
        phase("quantized_reference")
        actual = generate(remote.pipeline)
        phase("remote_grid")
        original_values = generate(original.pipeline)
        phase("original_model")
        quantized_error = float((actual - expected).abs().max())
        original_error = float((expected - original_values).abs().max())
        from xurcroh.fortress.attacks import exercise_adversarial_grid

        adversarial = exercise_adversarial_grid(clients, roles, operations)
        phase("adversarial")
    finally:
        for role_client in clients.values():
            role_client.close()
    direct_sent = sum(client.bytes_sent for client in clients.values())
    direct_received = sum(client.bytes_received for client in clients.values())
    from Xurcroh import Xurcroh

    response: Any
    with Xurcroh(**sdk) as client:
        if rung.modality == "image":
            response = client.images.generate(**common)
        else:
            response = client.videos.generate(**common, frames=frames, fps=fps)
        metrics = client.metrics()
    phase("public_sdk")
    return {
        "hardware_validated": deployment.get("hardware_validated", False),
        "phase_seconds": phase_times,
        "production_accepted": False,
        "quantized_reference_max_abs_error": quantized_error,
        "original_model_max_abs_error": original_error,
        "bytes_sent": direct_sent + int(metrics.get("fortress_bytes_sent", 0)),
        "bytes_received": direct_received + int(metrics.get("fortress_bytes_received", 0)),
        "inference_seconds": time.perf_counter() - started,
        **adversarial,
        "sdk": {"response": response.model_dump(mode="json"), "metrics": metrics},
    }
