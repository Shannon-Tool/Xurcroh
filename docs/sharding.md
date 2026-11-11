# Encrypted Model Shards

## Compile

Generate a distinct X25519 identity for every worker, then supply only their public keys to the trusted compiler. The client boundary key is independent and remains on the trusted client.

```bash
export XURCROH_CLIENT_BOUNDARY_KEY="$(openssl rand -hex 32)"
xurcroh compile \
  --checkpoint /models/llama \
  --output /artifacts/llama-xurcroh \
  --stages 2 \
  --stage-recipient "0=${STAGE_0_PUBLIC_KEY}" \
  --stage-recipient "1=${STAGE_1_PUBLIC_KEY}"
```

The compiler requires `config.json` and `model.safetensors` or `model.safetensors.index.json`. It rejects plaintext-only or unsupported layouts rather than silently copying them. Llama-class dense and standard expert-indexed MoE layers use verified attention, RoPE/GQA, RMSNorm absorption, contragredient value/output, FFN, router, and expert transformations. Other architectures must use a dedicated verified adapter.

The output contains contiguous `stage-N.xurcroh` files, recipient-sealed stage-key envelopes, `client-boundary.xurcroh`, and manifest v2. Worker shards exclude token embeddings, the LM head, and explicit basis/transition tensors. Stage data keys are independent and are never written unencrypted. Each envelope binds its recipient, stage index, checkpoint fingerprint, and manifest digest. The client boundary is encrypted with the separate client-only key.

## Format and loading

Shard format v1 uses an 8-byte magic, explicit format version, bounded canonical JSON header, random 96-bit nonce, and ChaCha20-Poly1305 ciphertext. The prefix and header are authenticated associated data. Decryption also verifies the plaintext SHA-256 digest. Files are written atomically with owner-only permissions. Manifest v2 authenticates recipient public keys and the manifest digest.

`SecureShardLoader` requires a `.xurcroh` path, correct stage key, and optional exact stage metadata. It decrypts into page-locked mutable RAM, loads safetensors from bytes, wipes and unlocks the staging buffer, and zeroes CPU tensors after model loading. Wrong recipients or keys, modified headers/ciphertext, missing/truncated artifacts, unexpected stages, and plaintext model paths fail closed. The vLLM adapter registers `xurcroh_sharded` when vLLM is installed. SGLang uses its `private` loader with the same recipient checks.

Python's `bytes(plaintext)` conversion required by safetensors creates an immutable copy that cannot be explicitly wiped. Mutable staging buffers and returned CPU tensors are wiped, but interpreters, allocators, and framework copies cannot receive a universal physical-erasure guarantee. Host root/hypervisor access remains outside the non-TEE confidentiality guarantee.

## Verify artifacts

```python
from xurcroh.shards.manifest import verify_manifest

report = verify_manifest(
    "/artifacts/llama-xurcroh/manifest.json",
    client_boundary_key,
    stage_private_keys={0: stage_0_private_key, 1: stage_1_private_key},
)
```

Verification authenticates the manifest, boundary, recipient envelopes, and supplied stage artifacts; checks ciphertext and shard identities; validates sequential layer coverage and the common checkpoint fingerprint; and rejects unsafe paths. If stage private keys are omitted, only structural checks and boundary decryption are performed, which does not prove stage plaintext correctness.

## EROFS and dm-verity

`VerityPackager` invokes `mkfs.erofs` and `veritysetup` directly without a shell. It packages an encrypted shard into a read-only image, obtains the Merkle root hash, and supports explicit verification. Missing tools and command failures are reported; no root-hash success is assumed. Binding the verified root into a CVM/kernel launch configuration is deployment-specific and remains hardware-pending.
