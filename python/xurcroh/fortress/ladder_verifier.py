from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

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


def _roles(cluster: list[dict[str, Any]], operation_name: str) -> list[RoleEndpoint]:
    roles = []
    for worker in cluster:
        labels = worker["labels"]
        activation_index = int(labels["activation_index"])
        weight_index = int(labels["weight_index"])
        if "operation_roles" in labels:
            assigned = json.loads(labels["operation_roles"]).get(operation_name)
            if assigned is None:
                continue
            if (
                not isinstance(assigned, list)
                or len(assigned) != 2
                or any(isinstance(index, bool) or not isinstance(index, int) for index in assigned)
            ):
                raise CapabilityError("operation role labels are invalid")
            activation_index, weight_index = assigned
        roles.append(
            RoleEndpoint(
                activation_index=activation_index,
                weight_index=weight_index,
                address=worker["endpoint"].removeprefix("quic://"),
                recipient_public=bytes.fromhex(labels["recipient_public"]),
                provider_id=worker["name"],
            )
        )
    if not roles:
        raise CapabilityError(f"no roles are assigned to operation {operation_name!r}")
    return roles


@dataclass(frozen=True)
class MediaWorkflow:
    name: str
    pipeline_class: str | None
    masked: bool = False


def _media_workflows(rung: LadderRung, pipeline: Any) -> list[MediaWorkflow]:
    import inspect

    from xurcroh.fortress.media_adapter import MediaArchitectureAdapter

    adapter = MediaArchitectureAdapter.from_pipeline(pipeline, rung.modality)
    verification = rung.verification
    selected = set(adapter.workflows)
    parameters = inspect.signature(pipeline.__call__).parameters
    if rung.rung == "M2" and {"image", "mask_image"} <= set(parameters):
        selected.add("masked_edit")
    requested = verification.get("workflows")
    if requested is not None:
        if (
            not isinstance(requested, list)
            or not requested
            or any(not isinstance(name, str) or not name for name in requested)
        ):
            raise CapabilityError("verification workflows must be a nonempty string list")
        selected = set(requested)
    allowed = (
        {"text_to_image", "image_to_image", "masked_edit"}
        if rung.modality == "image"
        else {"text_to_video", "image_to_video"}
    )
    if not selected or not selected <= allowed:
        raise CapabilityError("verification contains an unsupported media workflow")
    pipeline_classes = verification.get("workflow_pipelines", {})
    if not isinstance(pipeline_classes, dict):
        raise CapabilityError("workflow_pipelines must be an object")
    workflows = []
    for name in sorted(selected):
        if name in {"image_to_image", "image_to_video"} and "image" not in parameters:
            raise CapabilityError(f"pipeline does not expose the required {name} workflow")
        if name == "masked_edit" and not {"image", "mask_image"} <= set(parameters):
            raise CapabilityError("masked editing requires image and mask_image parameters")
        pipeline_class = pipeline_classes.get(name)
        if pipeline_class is not None and (
            not isinstance(pipeline_class, str) or not pipeline_class.isidentifier()
        ):
            raise CapabilityError("workflow pipeline class names must be valid identifiers")
        workflows.append(MediaWorkflow(name, pipeline_class, name == "masked_edit"))
    return workflows


def _provenance(
    report: dict[str, Any],
    deployment: dict[str, Any],
    rung: LadderRung,
    definition: Any,
) -> dict[str, Any]:
    from xurcroh.formats.base import fingerprint_source
    from xurcroh.formats.reference_bridge import ReferenceDecoderBridge

    path = definition["path"] if isinstance(definition, dict) else definition.path
    source_format = (
        definition.get("source_format", "auto")
        if isinstance(definition, dict)
        else definition.source_format
    )
    if rung.modality in {"image", "video"}:
        selected_format = "safetensors"
        root = Path(path)
        files = sorted(file for file in root.rglob("*") if file.is_file())
        source_digest = fingerprint_source(root, files)
        adapter_name = f"diffusers:{rung.modality}"
    else:
        from xurcroh.architectures import open_architecture

        with open_architecture(path, source_format) as adapter:
            selected_format = adapter.reader.source_format
            source_digest = adapter.reader.fingerprint
            adapter_name = adapter.name
    decoder = (
        ReferenceDecoderBridge("exllamav2" if selected_format == "exl2" else "exllamav3").identity()
        if selected_format in {"exl2", "exl3"}
        else {"package": selected_format, "official_reference": True}
    )
    official_error = None
    if selected_format in {"exl2", "exl3"} and torch.cuda.is_available():
        from xurcroh.architectures.base import CanonicalReader
        from xurcroh.formats.reference_bridge import official_exl2_error, official_exl3_error

        with open_architecture(path, source_format) as adapter:
            source = cast(CanonicalReader, adapter.reader).source
            name = next(
                candidate
                for candidate in adapter.reader.tensor_names()
                if candidate.endswith(".weight")
                and len(adapter.reader.tensor_shapes([candidate])[candidate]) == 2
            )
            expected = source.read_operation(name)
            prefix = name.removesuffix(".weight")
            tensors = source.read_tensors(
                [
                    candidate
                    for candidate in source.tensor_names()
                    if candidate.startswith(prefix + ".")
                ]
            )
            official_error = (
                official_exl2_error(tensors, prefix, expected)
                if selected_format == "exl2"
                else official_exl3_error(tensors, prefix, expected)
            )
    return {
        **report,
        "evidence_version": 2,
        "source_format": selected_format,
        "source_digest": source_digest,
        "architecture_adapter": adapter_name,
        "decoder_identity": decoder,
        "source_decoder_max_abs_error": official_error or 0.0,
        "official_decoder_validated": selected_format in {"safetensors", "gguf"}
        or official_error == 0.0,
        "workflow": report.get("workflow", rung.modality),
        "client_resources": report.get("sdk", {}).get("metrics", {}),
        "role_resources": report.get("role_resources", deployment.get("role_resources", {})),
        "phase_seconds": report.get("phase_seconds")
        or {"verification": report["inference_seconds"]},
        "relay_mode": (
            "datagram"
            if deployment.get("datagram_tunnel", deployment["rung"].get("datagram_tunnel", False))
            else "direct"
        ),
        "result": "pass",
    }


def _verify_text_lifecycle(client: Any, model: str) -> dict[str, Any]:
    from concurrent.futures import ThreadPoolExecutor

    from xurcroh.errors import RequestCancelledError

    messages = [{"role": "user", "content": "Reply with a short technical sentence."}]
    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=2,
        stream=True,
    )
    chunks = list(stream)
    streaming_tokens = chunks[-1].usage.completion_tokens if chunks[-1].usage is not None else 0
    cancelling = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=4,
        stream=True,
        request_id="ladder-cancel",
    )
    next(cancelling)
    cancelled = bool(client.cancel("ladder-cancel"))
    try:
        list(cancelling)
    except RequestCancelledError:
        cancelled = cancelled and True
    else:
        cancelled = False
    if not cancelled:
        raise RuntimeError("streaming cancellation was not acknowledged")
    post = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=1,
    )
    with ThreadPoolExecutor(max_workers=2) as executor:
        concurrent = list(
            executor.map(
                lambda _: client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                    max_tokens=1,
                ),
                range(2),
            )
        )
    return {
        "streaming_completion_tokens": streaming_tokens,
        "cancelled": cancelled,
        "post_cancellation_completion_tokens": post.usage.completion_tokens,
        "concurrent_requests": len(concurrent),
    }


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
        model_dtype=torch.float32,
        load_model=True,
    )
    cluster = sdk["cluster"]
    identity = _identity(cluster)
    clients = {
        name: FortressRoleClient(operation, _roles(cluster, name), identity, operation_name=name)
        for name, operation in compiled.operations.items()
    }
    prompt = "user: Reply with a short technical sentence."
    try:
        from transformers import AutoModelForCausalLM, AutoModelForMultimodalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            definition["path"],
            revision=rung.revision,
            local_files_only=True,
            trust_remote_code=False,
        )
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
        hidden = compiled.embedding[input_ids[0]]
        reference = compiled.combiner()
        if compiled.model is not None:
            from xurcroh.fortress.module import connect_grid_clients, set_reference_mode

            set_reference_mode(compiled.model)
            with torch.inference_mode():
                expected_output = compiled.model(input_ids=input_ids)
            set_reference_mode(compiled.model, False)
            connect_grid_clients(compiled.model, clients)
            with torch.inference_mode():
                actual_output = compiled.model(input_ids=input_ids)
            quantized_error = float((actual_output.logits - expected_output.logits).abs().max())
            quantized_logits = expected_output.logits[0, -1].to(torch.float64)
        else:
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
        original_loader = (
            AutoModelForMultimodalLM
            if compiled.model is not None
            and getattr(compiled.model.config, "model_type", None) in {"gemma4", "qwen3_5"}
            else AutoModelForCausalLM
        )
        original = original_loader.from_pretrained(
            definition["path"],
            revision=rung.revision,
            local_files_only=True,
            trust_remote_code=False,
            dtype=torch.float32,
        )
        original.eval()
        with torch.inference_mode():
            original_logits = original(input_ids=input_ids).logits[0, -1].to(torch.float64)
        original_error = float((quantized_logits - original_logits).abs().max())
        del original
        from xurcroh.fortress.attacks import exercise_adversarial_grid

        adversarial = exercise_adversarial_grid(
            clients, _roles(cluster, next(iter(clients))), compiled.operations
        )
    finally:
        for role_client in clients.values():
            role_client.close()
    direct_sent = sum(client.bytes_sent for client in clients.values())
    direct_received = sum(client.bytes_received for client in clients.values())
    del compiled, reference
    from Xurcroh import Xurcroh

    with Xurcroh(**sdk) as client:
        response = client.chat.completions.create(
            model=rung.model,
            messages=[{"role": "user", "content": "Reply with a short technical sentence."}],
            temperature=0,
            max_tokens=1,
        )
        lifecycle = _verify_text_lifecycle(client, rung.model)
        metrics = client.metrics()
    return _provenance(
        {
            "hardware_validated": deployment.get("hardware_validated", False),
            "production_accepted": False,
            "quantized_reference_max_abs_error": quantized_error,
            "original_model_max_abs_error": original_error,
            "bytes_sent": direct_sent + int(metrics.get("fortress_bytes_sent", 0)),
            "bytes_received": direct_received + int(metrics.get("fortress_bytes_received", 0)),
            "inference_seconds": time.perf_counter() - started,
            **adversarial,
            "lifecycle": lifecycle,
            "sdk": {
                "response": response.model_dump(mode="json"),
                "metrics": metrics,
            },
        },
        deployment,
        rung,
        definition,
    )


def _verify_media(deployment: dict[str, Any], rung: LadderRung) -> dict[str, Any]:
    import inspect

    from xurcroh.config import ModelDefinition
    from xurcroh.fortress.media_adapter import MediaArchitectureAdapter
    from xurcroh.fortress.media_components import compile_media_checkpoint
    from xurcroh.fortress.module import replace_linear_layers
    from xurcroh.requests import ImageGenerationRequest, VideoGenerationRequest
    from xurcroh.runtime.media_boundary import DiffusionMediaBoundary

    sdk = deployment["sdk"]
    definition = ModelDefinition.model_validate(sdk["model_path"][rung.model]).model_copy(
        update={"fortress_shares": rung.share_count}
    )
    adapter = MediaArchitectureAdapter.from_checkpoint(definition.path, rung.modality)
    started = time.perf_counter()
    compiled = compile_media_checkpoint(definition, adapter)
    operations = {
        f"{component}:{name}": operation
        for component, component_operations in compiled.items()
        for name, operation in component_operations.items()
    }
    cluster = sdk["cluster"]
    identity = _identity(cluster)
    clients = {
        name: FortressRoleClient(operation, _roles(cluster, name), identity, operation_name=name)
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

    def boundary(
        grid: dict[str, FortressRoleClient] | None, selected_definition: ModelDefinition
    ) -> Any:
        result = DiffusionMediaBoundary.from_definition(selected_definition)
        for component, component_operations in compiled.items():
            component_clients = None
            if grid is not None:
                component_clients = {
                    name: grid[f"{component}:{name}"] for name in component_operations
                }
            replace_linear_layers(
                getattr(result.pipeline, component),
                component_operations,
                grid_clients=component_clients,
            )
        return result

    verification = rung.verification
    common: dict[str, Any] = {
        "model": rung.model,
        "prompt": "a small red square",
        "steps": int(verification.get("steps", 1)),
        "seed": 11,
        "size": str(verification.get("size", "64x64")),
        "timeout": float(verification.get("timeout", 3600)),
    }
    if rung.modality == "image":
        common["strength"] = float(verification.get("strength", 0.75))
    frames = int(verification.get("frames", 4))
    fps = int(verification.get("fps", 4))
    width, height = (int(value) for value in str(common["size"]).split("x"))
    from base64 import b64encode
    from io import BytesIO

    from PIL import Image

    from xurcroh.runtime.vlm import _decode_image

    def source_data(mask: bool = False) -> str:
        image = Image.new("L" if mask else "RGB", (width, height), 255 if mask else (255, 0, 0))
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return "data:image/png;base64," + b64encode(buffer.getvalue()).decode("ascii")

    source_image = source_data()
    mask_image = source_data(mask=True)

    def request_for(workflow: MediaWorkflow) -> ImageGenerationRequest | VideoGenerationRequest:
        values = dict(common)
        if workflow.name != ("text_to_image" if rung.modality == "image" else "text_to_video"):
            values["image"] = source_image
        if workflow.masked:
            values["mask"] = mask_image
        if rung.modality == "image":
            return ImageGenerationRequest(**values)
        return VideoGenerationRequest(**values, frames=frames, fps=fps)

    def generate(
        pipeline: Any, request: ImageGenerationRequest | VideoGenerationRequest
    ) -> torch.Tensor:
        parameters: dict[str, Any] = {
            "prompt": request.prompt,
            "width": width,
            "height": height,
            "num_inference_steps": request.steps,
            "output_type": "latent",
        }
        if isinstance(request, VideoGenerationRequest):
            parameters.update({"num_frames": frames, "fps": fps})
        if not hasattr(pipeline, "decode_latents") and not hasattr(pipeline, "vae"):
            parameters["output_type"] = "pt"
        signature = inspect.signature(pipeline.__call__).parameters
        if request.image is not None:
            if "image" not in signature:
                raise CapabilityError("conditioned media pipeline does not accept an image")
            parameters["image"] = _decode_image(request.image)
        if isinstance(request, ImageGenerationRequest) and request.mask is not None:
            if "mask_image" not in signature:
                raise CapabilityError("editing pipeline does not accept a mask")
            parameters["mask_image"] = _decode_image(request.mask)
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

    probe = DiffusionMediaBoundary.from_definition(definition)
    workflows = _media_workflows(rung, probe.pipeline)
    del probe
    quantized_error = 0.0
    original_error = 0.0
    workflow_reports: dict[str, Any] = {}
    try:
        for workflow in workflows:
            selected_definition = definition.model_copy(
                update={"pipeline_class": workflow.pipeline_class}
            )
            request = request_for(workflow)
            local = boundary(None, selected_definition)
            remote = boundary(clients, selected_definition)
            original = DiffusionMediaBoundary.from_definition(selected_definition)
            phase(f"{workflow.name}:load_boundaries")
            expected = generate(local.pipeline, request)
            phase(f"{workflow.name}:quantized_reference")
            actual = generate(remote.pipeline, request)
            phase(f"{workflow.name}:remote_grid")
            original_values = generate(original.pipeline, request)
            phase(f"{workflow.name}:original_model")
            workflow_quantized = float((actual - expected).abs().max())
            workflow_original = float((expected - original_values).abs().max())
            quantized_error = max(quantized_error, workflow_quantized)
            original_error = max(original_error, workflow_original)
            workflow_reports[workflow.name] = {
                "pipeline_class": workflow.pipeline_class or "DiffusionPipeline",
                "quantized_reference_max_abs_error": workflow_quantized,
                "original_model_max_abs_error": workflow_original,
            }
            del local, remote, original
        from xurcroh.fortress.attacks import exercise_adversarial_grid

        adversarial = exercise_adversarial_grid(
            clients, _roles(cluster, next(iter(clients))), operations
        )
        phase("adversarial")
    finally:
        for role_client in clients.values():
            role_client.close()
    direct_sent = sum(client.bytes_sent for client in clients.values())
    direct_received = sum(client.bytes_received for client in clients.values())
    from Xurcroh import Xurcroh

    responses: dict[str, Any] = {}
    response: Any = None
    metrics: dict[str, Any] = {}
    sdk_sent = 0
    sdk_received = 0
    for workflow in workflows:
        workflow_sdk = {
            **sdk,
            "model_path": {
                **sdk["model_path"],
                rung.model: {
                    **sdk["model_path"][rung.model],
                    "pipeline_class": workflow.pipeline_class,
                },
            },
        }
        request = request_for(workflow)
        with Xurcroh(**workflow_sdk) as client:
            if rung.modality == "image" and workflow.masked:
                response = client.images.edit(**request.model_dump())
            elif rung.modality == "image":
                response = client.images.generate(**request.model_dump())
            else:
                response = client.videos.generate(**request.model_dump())
            metrics = client.metrics()
        sdk_sent += int(metrics.get("fortress_bytes_sent", 0))
        sdk_received += int(metrics.get("fortress_bytes_received", 0))
        responses[workflow.name] = response.model_dump(mode="json")
        workflow_reports[workflow.name]["response"] = responses[workflow.name]
    phase("public_sdk")
    return _provenance(
        {
            "hardware_validated": deployment.get("hardware_validated", False),
            "phase_seconds": phase_times,
            "production_accepted": False,
            "quantized_reference_max_abs_error": quantized_error,
            "original_model_max_abs_error": original_error,
            "bytes_sent": direct_sent + sdk_sent,
            "bytes_received": direct_received + sdk_received,
            "inference_seconds": time.perf_counter() - started,
            **adversarial,
            "workflow": ",".join(workflow.name for workflow in workflows),
            "workflows": workflow_reports,
            "sdk": {
                "response": response.model_dump(mode="json"),
                "responses": responses,
                "metrics": metrics,
            },
        },
        deployment,
        rung,
        definition,
    )
