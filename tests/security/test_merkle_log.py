from __future__ import annotations

import hashlib

import pytest

from xurcroh.merkle_log import (
    MerkleLog,
    MerkleProofError,
    leaf_hash,
    verify_consistency,
    verify_inclusion,
    verify_tree_head,
)
from xurcroh.signatures import HybridSigningKey

RFC6962_LEAVES = [
    b"",
    b"\x00",
    b"\x10",
    b"\x20\x21",
    b"\x30\x31",
    b"\x40\x41\x42\x43",
    b"\x50\x51\x52\x53\x54\x55\x56\x57",
    bytes(range(0x60, 0x70)),
]


def filled(count: int) -> MerkleLog:
    log = MerkleLog(HybridSigningKey.generate())
    for index in range(count):
        assert log.append(f"leaf-{index}".encode()) == index
    return log


def test_known_answers_for_empty_single_and_eight_leaf_trees() -> None:
    log = MerkleLog(HybridSigningKey.generate())
    assert log.root().hex() == hashlib.sha256(b"").hexdigest()
    log.append(RFC6962_LEAVES[0])
    assert log.root().hex() == "6e340b9cffb37a989ca544e6bb780a2c78901d3fb33738768511a30617afa01d"
    for leaf in RFC6962_LEAVES[1:]:
        log.append(leaf)
    assert log.root().hex() == "5dc9da79a70659a9ad559cb701ded9a2ab9d823aad2f4960cfe370eff4604328"


def test_every_inclusion_proof_verifies_and_any_change_fails() -> None:
    log = filled(33)
    for size in range(1, 34):
        root = log.root(size)
        for index in range(size):
            path = log.inclusion_proof(index, size)
            leaf = leaf_hash(f"leaf-{index}".encode())
            verify_inclusion(leaf, index, size, path, root)
            with pytest.raises(MerkleProofError):
                verify_inclusion(leaf_hash(b"other"), index, size, path, root)
            if size > 1:
                with pytest.raises(MerkleProofError):
                    verify_inclusion(leaf, (index + 1) % size, size, path, root)
                with pytest.raises(MerkleProofError):
                    verify_inclusion(leaf, index, size, path[:-1], root)


def test_every_consistency_proof_verifies_and_a_rewritten_log_fails() -> None:
    log = filled(20)
    for second in range(1, 21):
        for first in range(1, second + 1):
            path = log.consistency_proof(first, second)
            verify_consistency(first, second, log.root(first), log.root(second), path)
    rewritten = MerkleLog(HybridSigningKey.generate())
    for index in range(20):
        rewritten.append(b"changed" if index == 3 else f"leaf-{index}".encode())
    with pytest.raises(MerkleProofError):
        verify_consistency(
            8, 20, log.root(8), rewritten.root(20), rewritten.consistency_proof(8, 20)
        )
    with pytest.raises(MerkleProofError):
        verify_consistency(8, 20, log.root(8), log.root(20), [])


def test_signed_tree_head_binds_size_root_and_log_key() -> None:
    key = HybridSigningKey.generate()
    log = MerkleLog(key)
    log.append(b"a")
    head = log.signed_tree_head()

    assert verify_tree_head(head, log.public_key) == (1, log.root())
    with pytest.raises(MerkleProofError, match="signature"):
        verify_tree_head({**head, "tree_size": 2}, log.public_key)
    with pytest.raises(MerkleProofError, match="signature"):
        verify_tree_head(head, bytes(32))
    with pytest.raises(MerkleProofError):
        log.inclusion_proof(1)
