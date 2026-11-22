from __future__ import annotations

from pathlib import Path

import pytest

from xurcroh.signatures import (
    ALGORITHM,
    PUBLIC_KEY_BYTES,
    SIGNATURE_BYTES,
    HybridSigningKey,
    SignatureError,
    verify_hybrid,
)


def test_hybrid_signature_round_trips_and_has_fixed_sizes() -> None:
    key = HybridSigningKey.generate()
    signature = key.sign(b"context", b"message")

    assert ALGORITHM == "Ed25519+ML-DSA-65"
    assert len(key.public_key) == PUBLIC_KEY_BYTES == 32 + 1952
    assert len(signature) == SIGNATURE_BYTES == 64 + 3309
    verify_hybrid(key.public_key, b"context", b"message", signature)


def test_hybrid_key_is_restored_from_its_private_bytes() -> None:
    key = HybridSigningKey.generate()
    restored = HybridSigningKey.from_bytes(key.to_bytes())

    assert restored.public_key == key.public_key
    verify_hybrid(key.public_key, b"c", b"m", restored.sign(b"c", b"m"))


@pytest.mark.parametrize("offset", [0, 63, 64, SIGNATURE_BYTES - 1])
def test_damage_to_either_half_of_the_signature_is_rejected(offset: int) -> None:
    key = HybridSigningKey.generate()
    signature = bytearray(key.sign(b"c", b"m"))
    signature[offset] ^= 1

    with pytest.raises(SignatureError):
        verify_hybrid(key.public_key, b"c", b"m", bytes(signature))


def test_one_valid_half_is_not_enough() -> None:
    """A forger who breaks only one of the two schemes must still be rejected."""

    key = HybridSigningKey.generate()
    other = HybridSigningKey.generate()
    genuine = key.sign(b"c", b"m")
    forged = other.sign(b"c", b"m")

    with pytest.raises(SignatureError):
        verify_hybrid(key.public_key, b"c", b"m", genuine[:64] + forged[64:])
    with pytest.raises(SignatureError):
        verify_hybrid(key.public_key, b"c", b"m", forged[:64] + genuine[64:])


def test_signature_is_bound_to_its_context_and_message() -> None:
    key = HybridSigningKey.generate()
    signature = key.sign(b"trace", b"m")

    with pytest.raises(SignatureError):
        verify_hybrid(key.public_key, b"release", b"m", signature)
    with pytest.raises(SignatureError):
        verify_hybrid(key.public_key, b"trace", b"other", signature)
    # A context cannot be shifted into the message.
    with pytest.raises(SignatureError):
        verify_hybrid(key.public_key, b"trac", b"em", signature)


@pytest.mark.parametrize("length", [0, 32, PUBLIC_KEY_BYTES - 1, PUBLIC_KEY_BYTES + 1])
def test_malformed_public_keys_and_signatures_are_rejected(length: int) -> None:
    key = HybridSigningKey.generate()
    signature = key.sign(b"c", b"m")

    with pytest.raises(SignatureError):
        verify_hybrid(bytes(length), b"c", b"m", signature)
    with pytest.raises(SignatureError):
        verify_hybrid(key.public_key, b"c", b"m", signature[:length])
    with pytest.raises(SignatureError):
        HybridSigningKey.from_bytes(bytes(length))


def test_key_file_is_private_and_never_overwritten(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from xurcroh.signatures import load_signing_key, main, write_signing_key

    path = tmp_path / "signing.key"
    main([str(path), "--generate"])
    public = capsys.readouterr().out.strip()

    assert path.stat().st_mode & 0o777 == 0o600
    assert load_signing_key(path).public_key.hex() == public
    main([str(path), "--public-out", str(tmp_path / "signing.pub")])
    assert capsys.readouterr().out.strip() == public
    assert (tmp_path / "signing.pub").read_bytes().hex() == public
    with pytest.raises(FileExistsError):
        write_signing_key(path)
