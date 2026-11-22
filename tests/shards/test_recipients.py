from __future__ import annotations

import base64

import pytest

from xurcroh.errors import ShardIntegrityError
from xurcroh.kem import HybridKemKey
from xurcroh.shards.recipients import StageRecipient, open_stage_key, seal_stage_key


def _recipient(index: int, private: HybridKemKey) -> StageRecipient:
    public = private.public_key
    return StageRecipient(stage_index=index, public_key=public, key_id=f"stage-{index}")


def test_stage_key_is_opened_only_by_its_recipient() -> None:
    first = HybridKemKey.generate()
    second = HybridKemKey.generate()
    stage_key = bytes(range(32))
    envelope = seal_stage_key(
        stage_key,
        _recipient(0, first),
        manifest_digest="ab" * 32,
        checkpoint_fingerprint="checkpoint",
    )

    assert stage_key.hex() not in str(envelope)
    assert (
        open_stage_key(
            envelope,
            first,
            manifest_digest="ab" * 32,
            checkpoint_fingerprint="checkpoint",
        )
        == stage_key
    )
    with pytest.raises(ShardIntegrityError):
        open_stage_key(
            envelope,
            second,
            manifest_digest="ab" * 32,
            checkpoint_fingerprint="checkpoint",
        )


def test_stage_key_envelope_rejects_tampering_and_wrong_manifest() -> None:
    private = HybridKemKey.generate()
    envelope = seal_stage_key(
        bytes(32),
        _recipient(0, private),
        manifest_digest="cd" * 32,
        checkpoint_fingerprint="checkpoint",
    )
    with pytest.raises(ShardIntegrityError):
        open_stage_key(
            envelope,
            private,
            manifest_digest="ef" * 32,
            checkpoint_fingerprint="checkpoint",
        )

    tampered = dict(envelope)
    wrapped = bytearray(base64.b64decode(tampered["sealed_key"], validate=True))
    wrapped[-1] ^= 1
    tampered["sealed_key"] = base64.b64encode(wrapped).decode()
    with pytest.raises(ShardIntegrityError):
        open_stage_key(
            tampered,
            private,
            manifest_digest="cd" * 32,
            checkpoint_fingerprint="checkpoint",
        )
