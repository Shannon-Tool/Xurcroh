from __future__ import annotations

from pathlib import Path

import pytest

from xurcroh.errors import ShardIntegrityError
from xurcroh.shards.keys import read_key_envelope, write_key_envelope


def test_stage_key_envelope_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "stage-0.key.json"
    stage_key = bytes(range(32))
    wrapping_key = bytes(reversed(range(32)))
    write_key_envelope(path, stage_key, wrapping_key, key_id="stage-0")
    assert stage_key.hex() not in path.read_text()
    assert read_key_envelope(path, wrapping_key, expected_key_id="stage-0") == stage_key
    with pytest.raises(ShardIntegrityError):
        read_key_envelope(path, bytes(32), expected_key_id="stage-0")
