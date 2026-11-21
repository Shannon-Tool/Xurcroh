from __future__ import annotations

from pathlib import Path

import pytest

from xurcroh.fortress.prompt_armour import PromptArmour, RecognizedEntity


def test_prompt_armour_replaces_and_restores_names_and_explicit_secrets() -> None:
    armour = PromptArmour(
        recognizer=lambda text: [
            RecognizedEntity(0, 5, "PER"),
            RecognizedEntity(text.rindex("Alice"), text.rindex("Alice") + 5, "PER"),
        ]
    )
    protected = armour.armour(
        "Alice user@example.com uses api_key=super-secret-token and Alice again."
    )
    assert "Alice" not in protected.text
    assert "user@example.com" not in protected.text
    assert "super-secret-token" not in protected.text
    assert "Alice" not in repr(protected)
    restored = protected.restore(
        f"Response for {protected.placeholders[0]} and {protected.placeholders[0]}"
    )
    assert restored == "Response for Alice and Alice"
    protected.close()
    assert all(not any(value) for value in protected._originals.values())
    with pytest.raises(ValueError, match="closed"):
        protected.restore("anything")


def test_prompt_armour_uses_real_local_transformers_model(tmp_path: Path) -> None:
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocabulary = {"<pad>": 0, "hello": 1, "alice": 2, "<unk>": 3}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<unk>", pad_token="<pad>"
    )
    tokenizer.save_pretrained(tmp_path)
    config = transformers.BertConfig(
        vocab_size=len(vocabulary),
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_labels=3,
        id2label={0: "O", 1: "B-PER", 2: "I-PER"},
        label2id={"O": 0, "B-PER": 1, "I-PER": 2},
    )
    model = transformers.BertForTokenClassification(config)
    model.save_pretrained(tmp_path)

    armour = PromptArmour(model_path=tmp_path)
    protected = armour.armour("hello alice")
    assert protected.restore(protected.text) == "hello alice"
    protected.close()
    armour.close()


def test_stream_restorer_handles_placeholders_split_across_deltas() -> None:
    from xurcroh.fortress.prompt_armour import MessageArmour
    from xurcroh.requests import InputMessage

    session = MessageArmour(PromptArmour()).protect(
        [
            InputMessage(role="system", content="Reply to user@example.com."),
            InputMessage(
                role="user",
                content=[{"type": "text", "text": "My password: hunter2-secret ok"}],
            ),
        ]
    )
    texts = [message.plain_text() for message in session.messages]
    assert "user@example.com" not in texts[0]
    assert "hunter2-secret" not in texts[1]
    assert session.entities == 2
    email, password = session.placeholders

    generated = f"Sent to {email}, never {password}__ and _under_score__X"
    restorer = session.stream()
    restored = "".join(restorer.feed(character) for character in generated) + restorer.finish()

    assert restored == "Sent to user@example.com, never hunter2-secret__ and _under_score__X"
    assert session.restore(generated) == restored
    session.close()
    with pytest.raises(ValueError, match="closed"):
        session.restore("x")


def test_stream_restorer_does_not_hold_back_ordinary_text() -> None:
    from xurcroh.fortress.prompt_armour import MessageArmour
    from xurcroh.requests import InputMessage

    session = MessageArmour(PromptArmour()).protect(
        [InputMessage(role="user", content="mail a@b.io")]
    )
    restorer = session.stream()

    assert restorer.feed("hello world") == "hello world"
    assert restorer.feed(" tail_") == " tail"
    assert restorer.finish() == "_"


def test_sdk_prompt_armour_is_opt_in_and_reported(
    tiny_model_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from Xurcroh import Xurcroh
    from xurcroh.runtime.local import LocalTransformersEngine

    seen: list[str] = []
    original = LocalTransformersEngine.generate

    def spy(self, request, **arguments):  # type: ignore[no-untyped-def]
        seen.append(" ".join(message.plain_text() for message in request.messages))
        return original(self, request, **arguments)

    monkeypatch.setattr(LocalTransformersEngine, "generate", spy)
    messages = [{"role": "user", "content": "hello user@example.com"}]
    models = {"tiny": tiny_model_path}
    with Xurcroh(cluster={"endpoint": "local://cpu"}, model_path=models) as client:
        plain = client.chat.completions.create(model="tiny", messages=messages, max_tokens=2)
    with Xurcroh(
        cluster={"endpoint": "local://cpu"},
        model_path=models,
        security={"prompt_armour": True},
    ) as client:
        armoured = client.chat.completions.create(model="tiny", messages=messages, max_tokens=2)
        chunks = list(
            client.chat.completions.create(
                model="tiny", messages=messages, max_tokens=2, stream=True
            )
        )

    assert "user@example.com" in seen[0]
    assert "user@example.com" not in seen[1] and "__XURCROH_" in seen[1]
    assert "user@example.com" not in seen[2]
    assert plain.execution.armoured_entities is None
    assert armoured.execution.armoured_entities == 1
    assert chunks[-1].execution is not None
    assert chunks[-1].execution.armoured_entities == 1
