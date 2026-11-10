from __future__ import annotations

import base64
import io
from pathlib import Path

from PIL import Image
from tokenizers import Tokenizer, pre_tokenizers
from tokenizers.models import WordLevel
from transformers import (
    Gemma3Config,
    Gemma3ForConditionalGeneration,
    Gemma3ImageProcessor,
    Gemma3Processor,
    PreTrainedTokenizerFast,
)

from Xurcroh import Xurcroh


def tiny_gemma_checkpoint(root: Path) -> Path:
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
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<pad>",
        additional_special_tokens=["<boi>", "<image>", "<eoi>"],
    )
    tokenizer.boi_token = "<boi>"
    tokenizer.eoi_token = "<eoi>"
    tokenizer.image_token = "<image>"
    tokenizer.boi_token_id = 7
    config = Gemma3Config(
        text_config={
            "vocab_size": 512,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 128,
            "eos_token_id": 1,
            "pad_token_id": 0,
        },
        vision_config={
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "image_size": 32,
            "patch_size": 16,
            "num_channels": 3,
        },
        mm_tokens_per_image=4,
        boi_token_index=7,
        eoi_token_index=9,
        image_token_index=8,
    )
    model = Gemma3ForConditionalGeneration(config)
    model.save_pretrained(root, safe_serialization=True)
    image_processor = Gemma3ImageProcessor(
        size={"height": 32, "width": 32}, crop_size={"height": 32, "width": 32}
    )
    processor = Gemma3Processor(image_processor, tokenizer, image_seq_length=4)
    processor.save_pretrained(root)
    import json

    tokenizer_config = root / "tokenizer_config.json"
    values = json.loads(tokenizer_config.read_text())
    values["boi_token_id"] = 7
    tokenizer_config.write_text(json.dumps(values))
    return root


def png_data_url() -> str:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 32), (20, 40, 60)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_real_vlm_image_chat_and_stream(tmp_path: Path) -> None:
    checkpoint = tiny_gemma_checkpoint(tmp_path / "gemma")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the image."},
                {"type": "image_url", "image_url": {"url": png_data_url()}},
            ],
        }
    ]
    with Xurcroh(
        cluster={"endpoint": "local://cpu"},
        model_path={"gemma": {"path": str(checkpoint), "modality": "vlm"}},
    ) as client:
        response = client.chat.completions.create(
            model="gemma", messages=messages, max_tokens=1, temperature=0
        )
        assert response.execution.engine == "transformers-vlm"
        assert response.usage.completion_tokens == 1
        chunks = list(
            client.chat.completions.create(
                model="gemma", messages=messages, max_tokens=1, temperature=0, stream=True
            )
        )
    assert chunks[-1].usage is not None
    assert chunks[-1].usage.completion_tokens == 1


def test_remote_vlm_worker_executes_image_content(tmp_path: Path) -> None:
    import threading
    import time

    import uvicorn

    from xurcroh.config import ModelRegistry
    from xurcroh.worker import create_worker_app

    checkpoint = tiny_gemma_checkpoint(tmp_path / "gemma-remote")
    app = create_worker_app(
        ModelRegistry.normalize({"gemma": {"path": str(checkpoint), "modality": "vlm"}}),
        worker_id="vlm-test-worker",
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    started = time.monotonic()
    while not server.started:
        if time.monotonic() - started > 10:
            raise RuntimeError("VLM worker did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    try:
        with Xurcroh(
            cluster={
                "endpoint": f"http+dev://127.0.0.1:{port}",
                "security": {"mode": "development"},
            },
            model_path={"gemma": "remote-gemma"},
        ) as client:
            response = client.chat.completions.create(
                model="gemma",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Describe the image."},
                            {"type": "image_url", "image_url": {"url": png_data_url()}},
                        ],
                    }
                ],
                max_tokens=1,
                temperature=0,
            )
        assert response.execution.engine == "remote-openai"
        assert response.usage.completion_tokens == 1
    finally:
        server.should_exit = True
        thread.join(timeout=10)
