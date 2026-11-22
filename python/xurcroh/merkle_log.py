"""An append-only Merkle log with inclusion and consistency proofs (RFC 6962 hashing).

An inclusion proof shows one entry is in the log at a given size. A consistency proof
shows a later log only appended to an earlier one. A signed tree head binds a size
and root to the log's key. Entries are opaque bytes; callers log digests, not content.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Mapping, Sequence
from typing import Any

from xurcroh.signatures import HybridSigningKey, SignatureError, verify_hybrid

TREE_HEAD_FORMAT = "xurcroh-tree-head-v2"
TREE_HEAD_CONTEXT = b"xurcroh-tree-head"


class MerkleProofError(ValueError):
    pass


def leaf_hash(leaf: bytes) -> bytes:
    return hashlib.sha256(b"\x00" + leaf).digest()


def _node(left: bytes, right: bytes) -> bytes:
    return hashlib.sha256(b"\x01" + left + right).digest()


def _split(count: int) -> int:
    """Largest power of two strictly less than ``count``."""

    return 1 << ((count - 1).bit_length() - 1)


def _canonical(document: Mapping[str, Any]) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":")).encode()


class MerkleLog:
    def __init__(self, signing_key: HybridSigningKey) -> None:
        self._key = signing_key
        self._leaves: list[bytes] = []
        self._cache: dict[tuple[int, int], bytes] = {}
        self._lock = threading.Lock()

    @property
    def public_key(self) -> bytes:
        return self._key.public_key

    @property
    def size(self) -> int:
        return len(self._leaves)

    def append(self, leaf: bytes) -> int:
        with self._lock:
            self._leaves.append(leaf_hash(leaf))
            return len(self._leaves) - 1

    def _hash(self, start: int, end: int) -> bytes:
        count = end - start
        if count == 0:
            return hashlib.sha256(b"").digest()
        if count == 1:
            return self._leaves[start]
        cached = self._cache.get((start, end))
        if cached is None:
            middle = start + _split(count)
            cached = _node(self._hash(start, middle), self._hash(middle, end))
            if count & (count - 1) == 0:
                self._cache[(start, end)] = cached
        return cached

    def _checked_size(self, size: int | None) -> int:
        size = len(self._leaves) if size is None else size
        if not 0 <= size <= len(self._leaves):
            raise MerkleProofError("tree size is outside the log")
        return size

    def root(self, size: int | None = None) -> bytes:
        return self._hash(0, self._checked_size(size))

    def inclusion_proof(self, index: int, size: int | None = None) -> list[bytes]:
        size = self._checked_size(size)
        if not 0 <= index < size:
            raise MerkleProofError("leaf index is outside the tree")

        def path(position: int, start: int, end: int) -> list[bytes]:
            if end - start == 1:
                return []
            middle = start + _split(end - start)
            if position < middle:
                return [*path(position, start, middle), self._hash(middle, end)]
            return [*path(position, middle, end), self._hash(start, middle)]

        return path(index, 0, size)

    def consistency_proof(self, first: int, second: int | None = None) -> list[bytes]:
        second = self._checked_size(second)
        if not 0 < first <= second:
            raise MerkleProofError("consistency proof needs 0 < first <= second")

        def proof(size: int, start: int, end: int, complete: bool) -> list[bytes]:
            if size == end - start:
                return [] if complete else [self._hash(start, end)]
            middle = start + _split(end - start)
            if size <= middle - start:
                return [*proof(size, start, middle, complete), self._hash(middle, end)]
            return [
                *proof(size - (middle - start), middle, end, False),
                self._hash(start, middle),
            ]

        return proof(first, 0, second, True)

    def signed_tree_head(self, size: int | None = None) -> dict[str, Any]:
        size = self._checked_size(size)
        head = {
            "format": TREE_HEAD_FORMAT,
            "tree_size": size,
            "root_hash": self.root(size).hex(),
            "timestamp": time.time(),
            "log_key": self.public_key.hex(),
        }
        signature = self._key.sign(TREE_HEAD_CONTEXT, _canonical(head))
        return {**head, "signature": signature.hex()}


def verify_tree_head(head: Mapping[str, Any], public_key: bytes) -> tuple[int, bytes]:
    """Return ``(tree_size, root)`` if the head is signed by ``public_key``."""

    body = {name: value for name, value in head.items() if name != "signature"}
    try:
        if head["format"] != TREE_HEAD_FORMAT or bytes.fromhex(head["log_key"]) != public_key:
            raise ValueError("unexpected format or log key")
        verify_hybrid(
            public_key, TREE_HEAD_CONTEXT, _canonical(body), bytes.fromhex(head["signature"])
        )
        return int(head["tree_size"]), bytes.fromhex(head["root_hash"])
    except (KeyError, TypeError, ValueError, SignatureError) as error:
        raise MerkleProofError("tree head signature is missing or does not verify") from error


def verify_inclusion(
    leaf: bytes, index: int, size: int, path: Sequence[bytes], root: bytes
) -> None:
    """Check that the hashed ``leaf`` is entry ``index`` of the tree with this root."""

    if not 0 <= index < size:
        raise MerkleProofError("leaf index is outside the tree")
    position, last, current = index, size - 1, leaf
    for sibling in path:
        if last == 0:
            raise MerkleProofError("inclusion proof is too long")
        if position & 1 or position == last:
            current = _node(sibling, current)
            while not position & 1 and position:
                position >>= 1
                last >>= 1
        else:
            current = _node(current, sibling)
        position >>= 1
        last >>= 1
    if last != 0 or current != root:
        raise MerkleProofError("inclusion proof does not lead to the tree root")


def verify_consistency(
    first: int, second: int, first_root: bytes, second_root: bytes, path: Sequence[bytes]
) -> None:
    """Check that the tree of size ``second`` extends the tree of size ``first``."""

    if not 0 < first <= second:
        raise MerkleProofError("consistency check needs 0 < first <= second")
    if first == second:
        if path or first_root != second_root:
            raise MerkleProofError("equal-sized trees must have equal roots")
        return
    nodes = list(path)
    if not nodes:
        raise MerkleProofError("consistency proof is empty")
    if first & (first - 1) == 0:
        nodes.insert(0, first_root)
    position, last = first - 1, second - 1
    while position & 1:
        position >>= 1
        last >>= 1
    old = new = nodes[0]
    for sibling in nodes[1:]:
        if last == 0:
            raise MerkleProofError("consistency proof is too long")
        if position & 1 or position == last:
            old = _node(sibling, old)
            new = _node(sibling, new)
            while not position & 1 and position:
                position >>= 1
                last >>= 1
        else:
            new = _node(new, sibling)
        position >>= 1
        last >>= 1
    if last != 0 or old != first_root or new != second_root:
        raise MerkleProofError("consistency proof does not connect the two roots")
