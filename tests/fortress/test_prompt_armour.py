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
