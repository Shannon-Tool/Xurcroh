# Encrypted Model Shards

## Compile

```bash
export XURCROH_MASTER_KEY="$(openssl rand -hex 32)"
xurcroh compile \
  --checkpoint /models/llama \
  --output /artifacts/llama-xurcroh \
  --stages 4
```

The compiler requires `config.json` and `model.safetensors` or `model.safetensors.index.json`. It rejects plaintext-only or unsupported layouts rather than silently copying them. Llama-class dense and standard expert-indexed MoE layers use verified attention, RoPE/GQA, RMSNorm absorption, contragredient value/output, FFN, router, and expert transformations. Other architectures must use a dedicated verified adapter.

The output contains contiguous `stage-N.xurcroh` files, per-stage ChaCha20-Poly1305-wrapped key envelopes, `client-boundary.xurcroh`, and `manifest.json`. Worker shards exclude token embeddings and the LM head. Stage keys are independent and are never written unencrypted. Manifests bind stage/layer coverage, checkpoint fingerprint, shard identity, and ciphertext digests.

## Format and loading

Version 1 uses an 8-byte magic, explicit format version, bounded canonical JSON header, random 96-bit nonce, and ChaCha20-Poly1305 ciphertext. The prefix and header are authenticated associated data. Decryption also verifies the plaintext SHA-256 digest. Files are written atomically with owner-only permissions.

`SecureShardLoader` requires a `.xurcroh` path, correct stage key, and optional exact stage metadata. It decrypts into page-locked mutable RAM, loads safetensors from bytes, wipes and unlocks the staging buffer, and zeroes CPU tensors after model loading. Wrong keys, modified headers/ciphertext, missing/truncated artifacts, unexpected stages, and plaintext model paths fail closed. The vLLM adapter registers `xurcroh_sharded` when vLLM is installed and otherwise exposes the same strict loading implementation for local validation.

Python's `bytes(plaintext)` conversion required by safetensors creates an immutable copy that cannot be explicitly wiped. Mutable staging buffers and returned CPU tensors are wiped, but interpreters, allocators, and framework copies cannot receive a universal physical-erasure guarantee. Host root/hypervisor access remains outside the non-TEE confidentiality guarantee.

## Verify artifacts

```python
from xurcroh.shards.manifest import verify_manifest
report = verify_manifest("/artifacts/llama-xurcroh/manifest.json", master_key)
```

Verification decrypts every artifact, checks all ciphertext and shard identities, validates sequential stage coverage and the common checkpoint fingerprint, and rejects unsafe manifest paths.

## EROFS and dm-verity

`VerityPackager` invokes `mkfs.erofs` and `veritysetup` directly without a shell. It packages an encrypted shard into a read-only image, obtains the Merkle root hash, and supports explicit verification. Missing tools and command failures are reported; no root-hash success is assumed. Binding the verified root into a CVM/kernel launch configuration is deployment-specific and remains hardware-pending.
