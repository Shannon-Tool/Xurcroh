from __future__ import annotations

from pathlib import Path

import pytest

from Xurcroh import Xurcroh
from xurcroh.errors import ConfigurationError


def test_invalid_sampling_and_messages_fail_before_model_execution(tiny_model_path: Path) -> None:
    with Xurcroh(
        cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}
    ) as client:
        with pytest.raises(ConfigurationError, match="temperature"):
            client.chat.completions.create(
                model="tiny",
                messages=[{"role": "user", "content": "hello"}],
                temperature=-1,
            )
        with pytest.raises(ConfigurationError, match="messages"):
            client.chat.completions.create(model="tiny", messages=[])
