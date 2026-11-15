from __future__ import annotations

import sys
from pathlib import Path

import pytest
from tests.integration.test_ssh_executor import ssh_testbed

from Xurcroh import Xurcroh
from xurcroh.config import WorkerConfig

pytestmark = pytest.mark.slow


def _cpu_workers(base: WorkerConfig) -> list[WorkerConfig]:
    return [
        WorkerConfig(
            name=f"cpu-{index}",
            host=base.host,
            ssh=base.ssh,
            labels={
                "device": "cpu",
                "bootstrap": "false",
                "python_path": sys.executable,
            },
            provider={
                "country": "us" if index % 2 else "de",
                "cpu_vendor": "amd" if index % 2 else "intel",
            },
            resources={"max_host_memory_gb": 16},
        )
        for index in range(12)
    ]


def test_sdk_managed_fortress_inference_over_real_ssh(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    with ssh_testbed(tmp_path) as base:
        workers = _cpu_workers(base)
        with Xurcroh(
            cluster=workers,
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                }
            },
        ) as client:
            response = client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=1,
                timeout=120,
            )
            assert response.usage.completion_tokens == 1
            assert response.execution.engine == "fortress-grid"
            assert len(response.execution.worker_ids) == 12
            endpoints = client._runtime._workers_for_model("tiny")
            assert [worker.endpoint for worker in endpoints] == [
                f"quic://127.0.0.1:{9444 + index}" for index in range(12)
            ]


def test_sdk_managed_image_generation_over_real_ssh(tmp_path: Path) -> None:
    import pytest

    diffusers = pytest.importorskip("diffusers")
    model = diffusers.UNet2DModel(
        sample_size=8,
        in_channels=3,
        out_channels=3,
        layers_per_block=1,
        block_out_channels=(8, 16),
        down_block_types=("DownBlock2D", "AttnDownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D"),
        norm_num_groups=4,
    )
    checkpoint = tmp_path / "image-model"
    diffusers.DDPMPipeline(model, scheduler=diffusers.DDPMScheduler()).save_pretrained(checkpoint)
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny-image": {
                    "path": str(checkpoint),
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                }
            },
        ) as client,
    ):
        response = client.images.generate(
            model="tiny-image",
            prompt="a small red square",
            size="64x64",
            steps=1,
            seed=11,
            timeout=120,
        )
        assert response.data[0].b64_json
        assert response.execution.engine == "fortress-diffusion-grid"
        assert len(response.execution.worker_ids) == 12


def test_sdk_managed_moe_inference_over_real_ssh(tiny_llama_path: Path, tmp_path: Path) -> None:
    from tests.fortress.test_official_models import copy_tokenizer
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=128,
    )
    checkpoint = tmp_path / "moe-model"
    Qwen3MoeForCausalLM(config).save_pretrained(checkpoint, safe_serialization=True)
    copy_tokenizer(tiny_llama_path, checkpoint)
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny-moe": {
                    "path": str(checkpoint),
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                }
            },
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny-moe",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=1,
            timeout=120,
        )
        assert response.usage.completion_tokens == 1
        assert response.execution.engine == "fortress-grid"
        assert len(response.execution.worker_ids) == 12


def test_managed_role_crash_restarts_and_resumes_inference(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                }
            },
        ) as client,
    ):
        request = {
            "model": "tiny",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0,
            "max_tokens": 1,
            "timeout": 120,
        }
        first = client.chat.completions.create(**request)
        assert first.usage.completion_tokens == 1
        deployment = client._runtime._deployment
        assert deployment is not None
        role = deployment._roles[0]
        pid = Path(deployment._remote_root) / "model-0" / role.worker.name / "role.pid"
        deployment._executor.run(role.worker, f"kill -KILL $(cat {pid})")
        second = client.chat.completions.create(**request)
        assert second.usage.completion_tokens == 1
        assert client.metrics()["fortress_reconnections"] > 0


def test_multiple_models_share_managed_cluster_with_isolated_roles(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    from tests.fortress.test_official_models import copy_tokenizer
    from transformers.models.qwen3_moe import Qwen3MoeConfig, Qwen3MoeForCausalLM

    moe_checkpoint = tmp_path / "moe-model"
    config = Qwen3MoeConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        num_experts=2,
        num_experts_per_tok=1,
        max_position_embeddings=128,
    )
    Qwen3MoeForCausalLM(config).save_pretrained(moe_checkpoint, safe_serialization=True)
    copy_tokenizer(tiny_llama_path, moe_checkpoint)
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                },
                "moe": {
                    "path": str(moe_checkpoint),
                    "engine": "fortress",
                    "fortress_seed": 732,
                },
            },
        ) as client,
    ):
        for model in ("tiny", "moe"):
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "hello"}],
                temperature=0,
                max_tokens=1,
                timeout=120,
            )
            assert response.usage.completion_tokens == 1
            assert len(response.execution.worker_ids) == 12
        moe_endpoints = client._runtime._workers_for_model("moe")
        assert [worker.endpoint for worker in moe_endpoints] == [
            f"quic://127.0.0.1:{10444 + index}" for index in range(12)
        ]


def test_managed_datagram_tunnel_executes_real_fortress_inference(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster={"workers": _cpu_workers(base), "datagram_tunnel": True},
            model_path={
                "tiny": {
                    "path": str(tiny_llama_path),
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                }
            },
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "hello"}],
            temperature=0,
            max_tokens=1,
            timeout=120,
        )
        assert response.usage.completion_tokens == 1
        endpoints = client._runtime._workers_for_model("tiny")
        assert all(
            worker.endpoint is not None
            and worker.endpoint.startswith("quic://127.0.0.1:")
            and not worker.endpoint.endswith(":9444")
            for worker in endpoints
        )


def test_managed_multimodal_boundary_inference_over_real_ssh(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    import base64
    import io
    import shutil
    import wave

    import numpy as np
    import torch
    from PIL import Image
    from safetensors import safe_open
    from safetensors.torch import save_file

    checkpoint = tmp_path / "vlm-model"
    shutil.copytree(tiny_llama_path, checkpoint)
    tensors = {}
    with safe_open(checkpoint / "model.safetensors", framework="pt") as handle:
        for name in handle.keys():  # noqa: SIM118
            tensors[name] = handle.get_tensor(name)
    tensors["vision_projection.weight"] = torch.randn(16, 6912, dtype=torch.float64) / 100
    tensors["audio_projection.weight"] = torch.randn(16, 640, dtype=torch.float64) / 100
    save_file(tensors, str(checkpoint / "model.safetensors"))
    image_path = tmp_path / "red.png"
    Image.new("RGB", (48, 48), color=(255, 0, 0)).save(image_path)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16_000)
        output.writeframes(np.arange(800, dtype="<i2").tobytes())
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny-vlm": {
                    "path": str(checkpoint),
                    "engine": "fortress",
                    "modality": "vlm",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                }
            },
        ) as client,
    ):
        response = client.chat.completions.create(
            model="tiny-vlm",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": image_path.as_uri()}},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(buffer.getvalue()).decode(),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=1,
            timeout=120,
        )
        assert response.usage.completion_tokens == 1
        assert response.execution.engine == "fortress-grid"
        assert len(response.execution.worker_ids) == 12


def test_managed_official_gemma_vlm_image_inference_over_real_ssh(tmp_path: Path) -> None:
    import base64
    import io

    from PIL import Image
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast
    from transformers.models.gemma4 import (
        Gemma4AudioFeatureExtractor,
        Gemma4ImageProcessor,
        Gemma4Processor,
        Gemma4VideoProcessor,
    )
    from transformers.models.gemma4.configuration_gemma4 import (
        Gemma4AudioConfig,
        Gemma4Config,
        Gemma4TextConfig,
        Gemma4VisionConfig,
    )
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    checkpoint = tmp_path / "gemma-vlm"
    text = Gemma4TextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=4,
        global_head_dim=4,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=4,
        layer_types=["full_attention", "full_attention"],
        num_kv_shared_layers=0,
        use_double_wide_mlp=False,
        max_position_embeddings=512,
        eos_token_id=1,
        pad_token_id=0,
    )
    vision = Gemma4VisionConfig(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        position_embedding_size=48,
        patch_size=16,
        pooling_kernel_size=3,
        default_output_length=4,
    )
    audio = Gemma4AudioConfig(
        hidden_size=16, num_hidden_layers=1, num_attention_heads=2, output_proj_dims=16
    )
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(
            text_config=text,
            vision_config=vision,
            audio_config=audio,
            image_token_id=8,
            boi_token_id=7,
            eoi_token_id=9,
            audio_token_id=12,
            boa_token_id=10,
            eoa_token_index=11,
        )
    )
    model.save_pretrained(checkpoint, safe_serialization=True)
    vocabulary = {
        "<pad>": 0,
        "<eos>": 1,
        "hello": 2,
        "world": 3,
        "system": 4,
        "user": 5,
        "assistant": 6,
        "<boi>": 7,
        "<image>": 8,
        "<eoi>": 9,
        "<boa>": 10,
        "<eoa>": 11,
        "<audio>": 12,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<pad>",
        additional_special_tokens=["<boi>", "<image>", "<eoi>", "<boa>", "<eoa>", "<audio>"],
    )
    tokenizer.boi_token = "<boi>"
    tokenizer.eoi_token = "<eoi>"
    tokenizer.image_token = "<image>"
    tokenizer.boa_token = "<boa>"
    tokenizer.eoa_token = "<eoa>"
    tokenizer.audio_token = "<audio>"
    tokenizer.boi_token_id = 7
    tokenizer.eoi_token_id = 9
    tokenizer.image_token_id = 8
    tokenizer.boa_token_id = 10
    tokenizer.eoa_token_id = 11
    tokenizer.audio_token_id = 12
    template = (
        "{% for message in messages %}{{ message.role + ':' }}"
        "{% for part in message.content %}"
        "{% if part.type == 'text' %}{{ part.text }}"
        "{% elif part.type in ['input_audio', 'audio'] %}{{ '<audio>' }}"
        "{% else %}{{ '<image>' }}{% endif %}"
        "{% endfor %}{{ '\\n' }}{% endfor %}assistant:"
    )
    processor = Gemma4Processor(
        Gemma4AudioFeatureExtractor(),
        Gemma4ImageProcessor(),
        tokenizer,
        video_processor=Gemma4VideoProcessor(),
        chat_template=template,
        image_seq_length=4,
        audio_seq_length=8,
    )
    processor.save_pretrained(checkpoint)
    import json

    tokenizer_config = checkpoint / "tokenizer_config.json"
    tokenizer_values = json.loads(tokenizer_config.read_text())
    tokenizer_values.update(
        {
            "boi_token_id": 7,
            "eoi_token_id": 9,
            "image_token_id": 8,
            "boi_token": "<boi>",
            "eoi_token": "<eoi>",
            "image_token": "<image>",
            "boa_token_id": 10,
            "eoa_token_id": 11,
            "audio_token_id": 12,
            "boa_token": "<boa>",
            "eoa_token": "<eoa>",
            "audio_token": "<audio>",
        }
    )
    tokenizer_config.write_text(json.dumps(tokenizer_values))
    image = Image.new("RGB", (32, 32), (20, 40, 60))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "gemma-vlm": {
                    "path": str(checkpoint),
                    "engine": "fortress",
                    "modality": "vlm",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                }
            },
        ) as client,
    ):
        response = client.chat.completions.create(
            model="gemma-vlm",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {"type": "image_url", "image_url": {"url": image_url}},
                    ],
                }
            ],
            temperature=0,
            max_tokens=1,
            timeout=120,
        )
        assert response.usage.completion_tokens == 1
        assert response.execution.engine == "fortress-grid"
        import wave

        import numpy as np

        audio_buffer = io.BytesIO()
        with wave.open(audio_buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(16_000)
            output.writeframes(np.arange(5120, dtype="<i2").tobytes())
        audio_response = client.chat.completions.create(
            model="gemma-vlm",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hello"},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": base64.b64encode(audio_buffer.getvalue()).decode(),
                                "format": "wav",
                            },
                        },
                    ],
                }
            ],
            temperature=0,
            max_tokens=1,
            timeout=120,
        )
        assert audio_response.usage.completion_tokens == 1


def test_managed_video_generation_over_real_ssh(tiny_llama_path: Path, tmp_path: Path) -> None:
    import pytest

    diffusers = pytest.importorskip("diffusers")
    transformers = pytest.importorskip("transformers")
    unet = diffusers.UNet3DConditionModel(
        sample_size=2,
        in_channels=4,
        out_channels=4,
        down_block_types=("DownBlock3D",),
        up_block_types=("UpBlock3D",),
        block_out_channels=(4,),
        layers_per_block=1,
        norm_num_groups=4,
        cross_attention_dim=16,
        attention_head_dim=4,
    )
    vae = diffusers.AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D",),
        up_block_types=("UpDecoderBlock2D",),
        block_out_channels=(4,),
        layers_per_block=1,
        latent_channels=4,
        norm_num_groups=4,
        sample_size=4,
    )
    text_config = transformers.CLIPTextConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        projection_dim=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=8,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(tiny_llama_path)
    tokenizer.model_max_length = 8
    pipeline = diffusers.TextToVideoSDPipeline(
        unet=unet,
        vae=vae,
        text_encoder=transformers.CLIPTextModel(text_config),
        tokenizer=tokenizer,
        scheduler=diffusers.DDIMScheduler(),
    )
    checkpoint = tmp_path / "video-model"
    pipeline.save_pretrained(checkpoint)
    with (
        ssh_testbed(tmp_path) as base,
        Xurcroh(
            cluster=_cpu_workers(base),
            model_path={
                "tiny-video": {
                    "path": str(checkpoint),
                    "engine": "fortress",
                    "fortress_seed": 731,
                    "fortress_max_provider_fraction": 1.0,
                }
            },
        ) as client,
    ):
        response = client.videos.generate(
            model="tiny-video",
            prompt="ocean waves",
            size="32x32",
            frames=2,
            fps=2,
            steps=1,
            seed=12,
            timeout=900,
        )
        assert response.data[0].b64_json
        assert response.execution.engine == "fortress-diffusion-grid"
        assert len(response.execution.worker_ids) == 12
