from __future__ import annotations

from pathlib import Path

import pytest
from tests.architectures.utils import write_llama_gguf

from Xurcroh import Xurcroh
from xurcroh.errors import ConfigurationError


def test_public_sdk_gguf_auto_detection_and_chat(tiny_llama_path: Path, tmp_path: Path) -> None:
    checkpoint = write_llama_gguf(tiny_llama_path, tmp_path / "tiny.gguf")
    with Xurcroh(model_path={"tiny": str(checkpoint)}) as client:
        report = client.preflight("tiny")["tiny"]
        assert report.supported is True
        assert report.source_format == "gguf"
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
            max_tokens=1,
        )
        assert response.execution is not None
        assert response.execution.engine == "fortress-local"
        assert isinstance(response.choices[0].message.content, str)


def test_quantized_checkpoint_rejects_plaintext_engine_override(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    checkpoint = write_llama_gguf(tiny_llama_path, tmp_path / "tiny.gguf")
    with (
        Xurcroh(model_path={"tiny": {"path": str(checkpoint), "engine": "transformers"}}) as client,
        pytest.raises(ConfigurationError, match="Fortress"),
    ):
        client.preflight("tiny")


def test_public_sdk_exl2_full_model_chat(tiny_llama_path: Path, tmp_path: Path) -> None:
    from tests.architectures.utils import write_llama_exl2

    checkpoint = write_llama_exl2(tiny_llama_path, tmp_path / "tiny-exl2")
    with Xurcroh(model_path={"tiny": str(checkpoint)}) as client:
        report = client.preflight("tiny")["tiny"]
        assert report.supported is True
        assert report.source_format == "exl2"
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
            max_tokens=1,
        )
        assert response.execution is not None
        assert response.execution.engine == "fortress-local"


def test_public_sdk_exl3_full_model_chat(tiny_llama_path: Path, tmp_path: Path) -> None:
    from tests.architectures.utils import write_llama_exl3

    checkpoint = write_llama_exl3(tiny_llama_path, tmp_path / "tiny-exl3")
    with Xurcroh(model_path={"tiny": str(checkpoint)}) as client:
        report = client.preflight("tiny")["tiny"]
        assert report.supported is True
        assert report.source_format == "exl3"
        response = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
            max_tokens=1,
        )
        assert response.execution is not None
        assert response.execution.engine == "fortress-local"


def test_public_sdk_gguf_sync_stream_and_async_client(
    tiny_llama_path: Path, tmp_path: Path
) -> None:
    from Xurcroh import AsyncXurcroh

    checkpoint = write_llama_gguf(tiny_llama_path, tmp_path / "tiny.gguf")
    with Xurcroh(model_path={"tiny": str(checkpoint)}) as client:
        stream = client.chat.completions.create(
            model="tiny",
            messages=[{"role": "user", "content": "Hello"}],
            temperature=0,
            max_tokens=1,
            stream=True,
        )
        chunks = list(stream)
        assert chunks
        assert chunks[-1].choices[0].finish_reason is not None

    async def run() -> None:
        async with AsyncXurcroh(model_path={"tiny": str(checkpoint)}) as client:
            response = await client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "Hello"}],
                temperature=0,
                max_tokens=1,
            )
            assert response.execution is not None
            assert response.execution.engine == "fortress-local"

    import asyncio

    asyncio.run(run())
