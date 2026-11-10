from __future__ import annotations

from dataclasses import dataclass

FORMAT_VERSION = 1
PROTOCOL_LABEL = "xurcroh-v1"


@dataclass(frozen=True, slots=True)
class ProtocolVersion:
    value: int
    label: str

    @classmethod
    def parse(cls, label: str) -> ProtocolVersion:
        if label != PROTOCOL_LABEL:
            raise ValueError(f"unsupported protocol version: {label!r}")
        return cls(value=FORMAT_VERSION, label=label)

    def __str__(self) -> str:
        return self.label
