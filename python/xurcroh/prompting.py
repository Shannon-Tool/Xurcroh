from __future__ import annotations

from typing import Any

from xurcroh.requests import InputMessage


def render_chat_prompt(tokenizer: Any, messages: list[InputMessage]) -> str:
    template = getattr(tokenizer, "apply_chat_template", None)
    if callable(template) and getattr(tokenizer, "chat_template", None) is not None:
        rendered = template(
            [{"role": message.role, "content": message.plain_text()} for message in messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(rendered, str):
            raise TypeError("tokenizer chat template must return text when tokenize=False")
        return rendered
    prompt = "\n".join(f"{message.role}: {message.plain_text()}" for message in messages)
    return f"{prompt}\nassistant:"
