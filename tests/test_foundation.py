from __future__ import annotations

import pytest

from xurcroh.protocol import FORMAT_VERSION, PROTOCOL_LABEL, ProtocolVersion


def test_shared_protocol_version_is_explicitly_versioned() -> None:
    assert FORMAT_VERSION == 1
    assert PROTOCOL_LABEL == "xurcroh-v1"
    assert ProtocolVersion.parse("xurcroh-v1").value == FORMAT_VERSION


def test_unknown_protocol_version_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported protocol"):
        ProtocolVersion.parse("xurcroh-v2")
