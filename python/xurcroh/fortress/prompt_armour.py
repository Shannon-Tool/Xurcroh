from __future__ import annotations

import re
import secrets
import threading
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RecognizedEntity:
    start: int
    end: int
    label: str


@dataclass(repr=False, eq=False)
class ArmouredText:
    text: str
    _originals: dict[str, bytearray] = field(repr=False)
    _closed: bool = False

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    @property
    def placeholders(self) -> tuple[str, ...]:
        return tuple(self._originals)

    def restore(self, generated: str) -> str:
        with self._lock:
            if self._closed:
                raise ValueError("armoured text is closed")
            restored = generated
            for placeholder, original in self._originals.items():
                restored = restored.replace(placeholder, original.decode())
            return restored

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for original in self._originals.values():
                for index in range(len(original)):
                    original[index] = 0


class PromptArmour:
    def __init__(
        self,
        *,
        model_path: str | Path | None = None,
        recognizer: Callable[[str], list[RecognizedEntity]] | None = None,
    ) -> None:
        if model_path is not None and recognizer is not None:
            raise ValueError("model_path and recognizer are mutually exclusive")
        self._recognizer = recognizer
        self._pipeline: Any = None
        self._protected: weakref.WeakSet[ArmouredText] = weakref.WeakSet()
        self._closed = False
        if model_path is not None:
            from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

            tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            model = AutoModelForTokenClassification.from_pretrained(
                model_path, local_files_only=True
            )
            self._pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="simple",
            )

    def _model_entities(self, text: str) -> list[RecognizedEntity]:
        if self._pipeline is None:
            return []
        entities = []
        for prediction in self._pipeline(text):
            if prediction.get("entity_group") == "PER":
                entities.append(
                    RecognizedEntity(int(prediction["start"]), int(prediction["end"]), "PER")
                )
        return entities

    def _secret_entities(self, text: str) -> list[RecognizedEntity]:
        entities = [
            RecognizedEntity(match.start(), match.end(), "EMAIL")
            for match in re.finditer(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", text)
        ]
        credential = re.compile(
            r"\b(?:api[_-]?key|token|password|secret)\s*[:=]\s*([^\s,;]+)",
            flags=re.IGNORECASE,
        )
        for match in credential.finditer(text):
            entities.append(RecognizedEntity(match.start(1), match.end(1), "SECRET"))
        return entities

    def armour(self, text: str) -> ArmouredText:
        if self._closed:
            raise ValueError("prompt armour is closed")
        recognizer = self._recognizer or self._model_entities
        entities = [*self._secret_entities(text), *recognizer(text)]
        spans: list[tuple[int, int, str]] = []
        for entity in sorted(entities, key=lambda item: (item.start, item.end)):
            if not 0 <= entity.start < entity.end <= len(text):
                raise ValueError("recognizer returned an invalid entity span")
            if spans and entity.start < spans[-1][1]:
                continue
            spans.append((entity.start, entity.end, text[entity.start : entity.end]))
        originals: dict[str, bytearray] = {}
        prefix = secrets.token_hex(4).upper()
        protected = text
        for index, (start, end, original) in reversed(list(enumerate(spans, start=1))):
            placeholder = f"__XURCROH_{prefix}_{index:04d}__"
            originals[placeholder] = bytearray(original.encode())
            protected = protected[:start] + placeholder + protected[end:]
        result = ArmouredText(protected, originals)
        self._protected.add(result)
        return result

    def close(self) -> None:
        self._closed = True
        for protected in list(self._protected):
            protected.close()
        self._pipeline = None
