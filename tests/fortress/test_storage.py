from __future__ import annotations

import itertools

import pytest

from xurcroh.errors import AttestationError
from xurcroh.fortress.storage import StorageIntegrityError, disperse_shard, reconstruct_shard


def test_aont_rs_reconstructs_with_every_threshold_subset() -> None:
    plaintext = bytes(range(256)) * 4 + b"fortress"
    bundle = disperse_shard(plaintext, threshold=3, slice_count=5)

    for selected in itertools.combinations(range(5), 3):
        slices = {index: bundle.slices[index] for index in selected}
        key_shares = {index: bundle.key_shares[index] for index in selected}
        assert reconstruct_shard(bundle, slices, key_shares, lambda _: True) == plaintext


def test_aont_rs_rejects_inconsistent_slice_lengths() -> None:
    bundle = disperse_shard(b"secret model weights", threshold=3, slice_count=5)
    slices = dict(bundle.slices)
    slices[0] = slices[0][:-1]

    with pytest.raises(StorageIntegrityError):
        reconstruct_shard(bundle, slices, bundle.key_shares, lambda _: True)


def test_aont_rs_detects_tampering_and_requires_threshold_and_attestation() -> None:
    bundle = disperse_shard(b"secret model weights", threshold=3, slice_count=5)
    slices = dict(bundle.slices)
    slices[0] = bytes([slices[0][0] ^ 1]) + slices[0][1:]
    with pytest.raises(StorageIntegrityError):
        reconstruct_shard(bundle, slices, bundle.key_shares, lambda _: True)

    with pytest.raises(ValueError, match="threshold"):
        reconstruct_shard(bundle, {0: bundle.slices[0]}, bundle.key_shares, lambda _: True)
    with pytest.raises(AttestationError, match="attestation"):
        reconstruct_shard(bundle, bundle.slices, bundle.key_shares, lambda _: False)
