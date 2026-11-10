from __future__ import annotations

from pathlib import Path

import pytest
from tokenizers import Tokenizer, pre_tokenizers, processors
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("tiny-neural-model")
    vocab = {
        "<pad>": 0,
        "<eos>": 1,
        "hello": 2,
        "world": 3,
        "transformers": 4,
        "explain": 5,
        "you": 6,
        "are": 7,
        "helpful": 8,
        "test": 9,
        "a": 10,
        "b": 11,
        "c": 12,
        "d": 13,
        "e": 14,
        "f": 15,
        "g": 16,
        "h": 17,
        "i": 18,
        "j": 19,
        "k": 20,
        "l": 21,
        "m": 22,
        "n": 23,
        "o": 24,
        "p": 25,
        "q": 26,
        "r": 27,
        "s": 28,
        "t": 29,
        "u": 30,
        "v": 31,
        "w": 32,
        "x": 33,
        "y": 34,
        "z": 35,
        ".": 36,
        "!": 37,
        "?": 38,
        ",": 39,
        "system": 42,
        "user": 43,
        "assistant": 44,
        "0": 45,
        "1": 46,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="<eos> $A",
        special_tokens=[("<eos>", 1)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        eos_token="<eos>",
        pad_token="<pad>",
        unk_token="<pad>",
    )
    tokenizer.save_pretrained(root)
    config = GPT2Config(
        vocab_size=max(vocab.values()) + 1,
        n_positions=128,
        n_embd=32,
        n_layer=2,
        n_head=2,
        bos_token_id=0,
        eos_token_id=1,
        pad_token_id=0,
    )
    model = GPT2LMHeadModel(config)
    model.save_pretrained(root)
    return root


@pytest.fixture(scope="session")
def tiny_llama_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    from transformers import LlamaConfig, LlamaForCausalLM

    root = tmp_path_factory.mktemp("tiny-llama-model")
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        attention_bias=False,
        mlp_bias=False,
    )
    model = LlamaForCausalLM(config)
    model.save_pretrained(root, safe_serialization=True)
    vocabulary = {token: index for index, token in enumerate(["<pad>", "<eos>", "hello", "world"])}
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<pad>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, eos_token="<eos>", pad_token="<pad>", unk_token="<pad>"
    )
    tokenizer.save_pretrained(root)
    return root


@pytest.fixture
def running_worker(tiny_model_path: Path):
    import threading
    import time

    import uvicorn

    from xurcroh.config import ModelRegistry
    from xurcroh.worker import create_worker_app

    app = create_worker_app(
        ModelRegistry.normalize({"tiny": tiny_model_path}), worker_id="local-test-worker"
    )
    config = uvicorn.Config(app, host="127.0.0.1", port=0, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    start = time.monotonic()
    while not server.started:
        if time.monotonic() - start > 10:
            raise RuntimeError("worker did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http+dev://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=10)
