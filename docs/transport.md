# Secure Transport Contracts

The Rust workspace separates three ownership boundaries:

- `xurcroh-protocol`: bounded binary tensor descriptors and frames. Every frame carries protocol version, request/stream identity, sequence, tensor type, shape, and exact payload length. Malformed lengths, overflow, unsupported versions, and trailing bytes are rejected without panics.
- `xurcroh-crypto`: X25519 agreement, HKDF-SHA256 directional keys, ChaCha20-Poly1305, monotonic nonces, replay windows, and onion-style hop metadata. Return secrets occur only in the innermost layer.
- `xurcroh-transport`: QUIC streams with pinned worker certificates and explicit buffer ownership. Python bindings use the same protocol implementation rather than a second serializer. Bidirectional `TensorClient`, `TensorServer`, and worker-connection APIs release the GIL during waits, support graceful peer shutdown, and explicitly abort endpoints on timeout.

Application AEAD and QUIC TLS are separate layers. PRG-based tensor masking is computational protection, not an information-theoretic one-time pad. Onion-style routing limits per-hop metadata exposure but does not hide endpoint existence, global timing, traffic volume, collusion, or malicious computation. NAT relays must be authenticated and are never given session private keys.

Nonce exhaustion and replayed/too-old packets fail closed. Key derivation separates send and receive directions. Epoch-based rotation replaces key material and starts a fresh nonce domain; recent reordering is accepted while duplicates and packets outside the 64-sequence replay window are rejected. `DenseStageEngine` executes compiled stages over `quic://` with per-stage AEAD keys, CA paths, response validation, per-worker locks, and timeout-triggered aborts. Stage workers expose this listener with `--quic-listen`, `--quic-cert`, and `--quic-key`; `labels.server_name` controls certificate naming and `labels.attestation_endpoint` selects the HTTPS evidence side channel. Physical multi-node NAT behavior, packet-loss tuning, and throughput remain hardware/network validation targets.
