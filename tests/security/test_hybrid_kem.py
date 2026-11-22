from __future__ import annotations

from pathlib import Path

import pytest

from xurcroh.kem import (
    ALGORITHM,
    PRIVATE_KEY_BYTES,
    PUBLIC_KEY_BYTES,
    SEAL_OVERHEAD_BYTES,
    HybridKemKey,
    KemError,
    open_sealed,
    seal,
)


def test_hybrid_kem_round_trips_with_fixed_sizes() -> None:
    key = HybridKemKey.generate()
    sealed = seal(key.public_key, b"info", b"secret")

    assert ALGORITHM == "X25519+ML-KEM-768"
    assert len(key.public_key) == PUBLIC_KEY_BYTES == 1184 + 32
    assert len(key.to_bytes()) == PRIVATE_KEY_BYTES
    assert len(sealed) == SEAL_OVERHEAD_BYTES + len(b"secret")
    assert open_sealed(key, b"info", sealed) == b"secret"


def test_key_is_restored_from_its_private_bytes() -> None:
    key = HybridKemKey.generate()
    restored = HybridKemKey.from_bytes(key.to_bytes())

    assert restored.public_key == key.public_key
    assert open_sealed(restored, b"i", seal(key.public_key, b"i", b"m")) == b"m"


def test_sealed_message_is_bound_to_recipient_and_info() -> None:
    key = HybridKemKey.generate()
    sealed = seal(key.public_key, b"info", b"secret")

    with pytest.raises(KemError):
        open_sealed(HybridKemKey.generate(), b"info", sealed)
    with pytest.raises(KemError):
        open_sealed(key, b"other", sealed)


@pytest.mark.parametrize("offset", [0, 1087, 1088, 1119, 1120, -1])
def test_damage_to_any_part_of_a_sealed_message_is_rejected(offset: int) -> None:
    """Offsets cover the ML-KEM ciphertext, the X25519 share and the AEAD body."""

    key = HybridKemKey.generate()
    sealed = bytearray(seal(key.public_key, b"info", b"secret"))
    sealed[offset] ^= 1

    with pytest.raises(KemError):
        open_sealed(key, b"info", bytes(sealed))


def test_each_seal_uses_fresh_randomness() -> None:
    key = HybridKemKey.generate()

    assert seal(key.public_key, b"i", b"m") != seal(key.public_key, b"i", b"m")


@pytest.mark.parametrize("length", [0, 32, PUBLIC_KEY_BYTES - 1, PUBLIC_KEY_BYTES + 1])
def test_malformed_keys_and_messages_are_rejected(length: int) -> None:
    key = HybridKemKey.generate()

    with pytest.raises(KemError):
        seal(bytes(length), b"i", b"m")
    with pytest.raises(KemError):
        HybridKemKey.from_bytes(bytes(length))
    with pytest.raises(KemError):
        open_sealed(key, b"i", bytes(length))


def test_key_file_is_private_and_never_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xurcroh.kem import main, write_kem_key

    path = tmp_path / "onion.key"
    main([str(path), "--generate"])
    public = capsys.readouterr().out.strip()

    assert path.stat().st_mode & 0o777 == 0o600
    assert HybridKemKey.from_bytes(path.read_bytes()).public_key.hex() == public
    main([str(path)])
    assert capsys.readouterr().out.strip() == public
    with pytest.raises(FileExistsError):
        write_kem_key(path)
