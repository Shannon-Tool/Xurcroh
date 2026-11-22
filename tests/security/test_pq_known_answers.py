"""Known-answer vectors for the post-quantum primitives this project relies on.

Key generation and signature verification vectors come from the public NIST ACVP server
test files (the ``source`` field of the vector file); decapsulation and encapsulation-key
validation vectors come from Project Wycheproof (``wycheproof_source``). They check the
library that implements ML-KEM-768 and ML-DSA-65 here, so a build against a
non-conforming backend fails these tests.

Two operations have no known-answer test because the library draws their randomness
itself and offers no way to supply it: ML-KEM encapsulation and ML-DSA signing. Both
are covered by round-trip tests instead (``test_hybrid_kem.py``,
``test_hybrid_signatures.py``), and their inverse operations are covered here.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.mldsa import MLDSA65PrivateKey, MLDSA65PublicKey
from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PrivateKey, MLKEM768PublicKey

from xurcroh.kem import HybridKemKey
from xurcroh.signatures import HybridSigningKey

VECTORS = json.loads((Path(__file__).parent / "vectors" / "pq-kat.json").read_text())


def test_vector_file_has_every_kind_of_vector() -> None:
    assert len(VECTORS["ml_kem_768_keygen"]) >= 2
    assert len(VECTORS["ml_dsa_65_keygen"]) >= 2
    outcomes = {vector["testPassed"] for vector in VECTORS["ml_dsa_65_sigver"]}
    assert outcomes == {True, False}
    decaps = VECTORS["ml_kem_768_decaps"]
    assert {vector["result"] for vector in decaps} == {"valid", "invalid"}
    assert any("MalleableCiphertext" in vector["flags"] for vector in decaps)
    assert len(VECTORS["ml_kem_768_invalid_encapsulation_keys"]) >= 4


@pytest.mark.parametrize("vector", VECTORS["ml_kem_768_keygen"], ids=lambda v: f"tc{v['tcId']}")
def test_ml_kem_768_key_generation_matches_nist(vector: dict) -> None:
    seed = bytes.fromhex(vector["d"]) + bytes.fromhex(vector["z"])
    expected = bytes.fromhex(vector["ek"])

    assert MLKEM768PrivateKey.from_seed_bytes(seed).public_key().public_bytes_raw() == expected
    # The project's hybrid key stores the same seed and must expose the same ML-KEM key.
    hybrid = HybridKemKey.from_bytes(seed + bytes(32))
    assert hybrid.public_key[: len(expected)] == expected


@pytest.mark.parametrize("vector", VECTORS["ml_dsa_65_keygen"], ids=lambda v: f"tc{v['tcId']}")
def test_ml_dsa_65_key_generation_matches_nist(vector: dict) -> None:
    seed = bytes.fromhex(vector["seed"])
    expected = bytes.fromhex(vector["pk"])

    assert MLDSA65PrivateKey.from_seed_bytes(seed).public_key().public_bytes_raw() == expected
    hybrid = HybridSigningKey.from_bytes(bytes(32) + seed)
    assert hybrid.public_key[32:] == expected


@pytest.mark.parametrize("vector", VECTORS["ml_dsa_65_sigver"], ids=lambda v: f"tc{v['tcId']}")
def test_ml_dsa_65_verification_matches_nist(vector: dict) -> None:
    key = MLDSA65PublicKey.from_public_bytes(bytes.fromhex(vector["pk"]))
    arguments = (
        bytes.fromhex(vector["signature"]),
        bytes.fromhex(vector["message"]),
        bytes.fromhex(vector["context"]),
    )

    if vector["testPassed"]:
        key.verify(*arguments)
    else:
        with pytest.raises((InvalidSignature, ValueError)):
            key.verify(*arguments)


@pytest.mark.parametrize(
    "vector", VECTORS["ml_kem_768_decaps"], ids=lambda v: f"{v['result']}-tc{v['tcId']}"
)
def test_ml_kem_768_decapsulation_matches_known_answers(vector: dict) -> None:
    """Includes ciphertexts altered in transit, which must give the implicit-rejection key."""

    seed = bytes.fromhex(vector["seed"])
    ciphertext = bytes.fromhex(vector["c"])

    if vector["result"] == "invalid":
        # A malformed seed or ciphertext must be refused, not processed.
        with pytest.raises(ValueError):
            MLKEM768PrivateKey.from_seed_bytes(seed).decapsulate(ciphertext)
        return
    key = MLKEM768PrivateKey.from_seed_bytes(seed)
    assert key.public_key().public_bytes_raw() == bytes.fromhex(vector["ek"])
    assert key.decapsulate(ciphertext) == bytes.fromhex(vector["K"])
    # The hybrid key holds the same ML-KEM key.
    assert HybridKemKey.from_bytes(seed + bytes(32)).public_key[:1184] == bytes.fromhex(
        vector["ek"]
    )


@pytest.mark.parametrize(
    "vector",
    VECTORS["ml_kem_768_invalid_encapsulation_keys"],
    ids=lambda v: f"tc{v['tcId']}",
)
def test_ml_kem_768_rejects_invalid_encapsulation_keys(vector: dict) -> None:
    """A key with coefficients outside the modulus must not be encapsulated to."""

    from xurcroh.kem import KemError, seal

    encoded = bytes.fromhex(vector["ek"])
    with pytest.raises(ValueError):
        MLKEM768PublicKey.from_public_bytes(encoded).encapsulate()
    # The project's own sealing refuses the same key inside a hybrid public key.
    with pytest.raises((KemError, ValueError)):
        seal(encoded + bytes(32), b"info", b"payload")
