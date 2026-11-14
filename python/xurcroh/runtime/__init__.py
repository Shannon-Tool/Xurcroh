"""Execution lifecycle shared by public SDK and distributed workers."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from xurcroh.runtime.service import RuntimeService

__all__ = ["RuntimeService"]


def __getattr__(name: str) -> object:
    if name == "RuntimeService":
        from xurcroh.runtime.service import RuntimeService

        return RuntimeService
    raise AttributeError(name)
