"""NIST ACVP known-answer vectors for the post-quantum primitives this project relies on.

The vectors were taken from the public ACVP server test files (the ``source`` field
of the vector file). They check the library that implements ML-KEM-768 and ML-DSA-65
here, so a build against a non-conforming backend fails these tests.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.mldsa import MLDSA65PrivateKey, MLDSA65PublicKey
from cryptography.hazmat.primitives.asymmetric.mlkem import MLKEM768PrivateKey

from xurcroh.kem import HybridKemKey
from xurcroh.signatures import HybridSigningKey

VECTORS = json.loads((Path(__file__).parent / "vectors" / "pq-kat.json").read_text())


def test_vector_file_has_every_kind_of_vector() -> None:
    assert len(VECTORS["ml_kem_768_keygen"]) >= 2
    assert len(VECTORS["ml_dsa_65_keygen"]) >= 2
    outcomes = {vector["testPassed"] for vector in VECTORS["ml_dsa_65_sigver"]}
    assert outcomes == {True, False}


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
