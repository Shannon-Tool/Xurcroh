from __future__ import annotations

import base64
import binascii
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ModelLoadError, ProtocolError, ShardIntegrityError
from xurcroh.models.dense import DenseLayerWeights, DenseStageRunner, layer_weights_from_dict
from xurcroh.models.moe import MoELayerWeights, moe_layer_weights_from_dict
from xurcroh.runtime.stage_sessions import StageSessionStore
from xurcroh.runtime.tensor_codec import decode_stage_tensor, encode_stage_tensor
from xurcroh.shards.format import read_shard
from xurcroh.shards.recipients import open_stage_key, verify_manifest_digest


class StageExecutionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    session_id: str = Field(pattern=r"^[a-f0-9]{32}$")
    client_public: str = Field(min_length=44, max_length=44)
    stage_index: int = Field(ge=0)
    dtype: Literal["f64", "f32", "f16", "bf16"] = "f64"
    shape: list[int] = Field(min_length=1, max_length=2)
    packet: str = Field(min_length=1)


class DenseStage:
    def __init__(
        self,
        manifest_path: str | Path,
        stage_index: int,
        private_key: X25519PrivateKey,
    ) -> None:
        path = Path(manifest_path)
        try:
            document = json.loads(path.read_text())
            if document["format"] != "xurcroh-manifest" or document["version"] != 2:
                raise ShardIntegrityError("unsupported stage manifest")
            digest = verify_manifest_digest(document)
            stage = document["stages"][stage_index]
            if len(document["stages"]) != document["stage_count"]:
                raise ShardIntegrityError("stage count does not match manifest")
            if Path(stage["file"]).name != stage["file"]:
                raise ShardIntegrityError("unsafe stage artifact path")
            if Path(stage["key_envelope"]).name != stage["key_envelope"]:
                raise ShardIntegrityError("unsafe stage key path")
            envelope = json.loads((path.parent / stage["key_envelope"]).read_text())
        except (OSError, KeyError, IndexError, TypeError, json.JSONDecodeError) as error:
            raise ShardIntegrityError("invalid stage manifest") from error
        public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
        if public.hex() != stage["recipient_public"]:
            raise ShardIntegrityError("stage recipient identity mismatch")
        key = open_stage_key(
            envelope,
            private_key,
            manifest_digest=digest,
            checkpoint_fingerprint=document["checkpoint_fingerprint"],
        )
        self._tensors, metadata = read_shard(path.parent / stage["file"], key)
        if metadata["ciphertext_sha256"] != stage["ciphertext_sha256"]:
            raise ShardIntegrityError("stage manifest ciphertext digest mismatch")
        if metadata["shard_id"] != stage["shard_id"]:
            raise ShardIntegrityError("stage manifest shard identity mismatch")
        if metadata["checkpoint_fingerprint"] != document["checkpoint_fingerprint"]:
            raise ShardIntegrityError("stage checkpoint fingerprint mismatch")
        if metadata["stage_index"] != stage_index:
            raise ModelLoadError("stage index does not match encrypted shard metadata")
        if metadata["stage_count"] != document["stage_count"]:
            raise ModelLoadError("stage count does not match encrypted shard metadata")
        layers: list[DenseLayerWeights | MoELayerWeights] = []
        for layer in metadata["layer_indices"]:
            prefix = f"model.layers.{layer}"
            if f"{prefix}.mlp.gate.weight" in self._tensors:
                layers.append(
                    moe_layer_weights_from_dict(
                        self._tensors, prefix, top_k=int(metadata.get("moe_top_k", 1))
                    )
                )
            else:
                layers.append(layer_weights_from_dict(self._tensors, prefix))
        self.runner = DenseStageRunner(
            [layers],
            query_heads=int(metadata.get("query_heads", 4)),
            kv_heads=int(metadata.get("kv_heads", 2)),
            interleaved_rope=bool(metadata.get("interleaved_rope", False)),
        )
        self.sessions = StageSessionStore(private_key, stage_index)
        self.stage_index = stage_index
        self.closed = False

    def execute(self, hidden: torch.Tensor, request_id: str | None = None) -> torch.Tensor:
        if self.closed:
            raise ModelLoadError("stage worker is closed")
        if request_id is not None:
            return self.runner.run_cached(request_id, hidden)
        return self.runner.run(hidden)

    def close(self) -> None:
        if not self.closed:
            for tensor in self._tensors.values():
                tensor.zero_()
            self._tensors.clear()
            self.sessions.close()
            self.closed = True


def _decode_tensor(
    packet: bytes,
    receiver: PipelineTensorInterceptor,
    *,
    expected_dtype: str | None = None,
    expected_shape: list[int] | None = None,
) -> torch.Tensor:
    return decode_stage_tensor(
        packet,
        receiver,
        expected_dtype=expected_dtype,
        expected_shape=expected_shape,
    )


def _encode_tensor(
    hidden: torch.Tensor, request_id: str, sender: PipelineTensorInterceptor
) -> tuple[bytes, list[int]]:
    packet, _, shape = encode_stage_tensor(hidden, request_id, sender)
    return packet, shape


def create_stage_app(
    manifest_path: str | Path,
    *,
    stage_index: int,
    private_key: X25519PrivateKey,
    attestation_document: dict[str, Any] | None = None,
) -> FastAPI:
    stage = DenseStage(manifest_path, stage_index, private_key)
    cancelled: dict[str, threading.Event] = {}
    lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        with lock:
            events = list(cancelled.values())
            cancelled.clear()
        for event in events:
            event.set()
        stage.close()

    app = FastAPI(title="Xurcroh Dense Stage", version="0.1.0", lifespan=lifespan)

    @app.get("/v1/attestation/challenge")
    def attestation_challenge() -> dict[str, str]:
        import secrets

        return {"nonce": secrets.token_bytes(32).hex()}

    @app.get("/v1/attestation/evidence")
    def attestation_evidence(nonce: str) -> dict[str, Any]:
        import hashlib

        from xurcroh.errors import AttestationError

        if attestation_document is None:
            raise AttestationError("worker has no attestation evidence")
        try:
            nonce_bytes = bytes.fromhex(nonce)
        except ValueError as error:
            raise AttestationError("attestation nonce must be hexadecimal") from error
        if len(nonce_bytes) != 32:
            raise AttestationError("attestation nonce must contain 32 bytes")
        document = dict(attestation_document)
        document["nonce"] = nonce
        if "report_data" not in document:
            binding = (
                nonce_bytes
                + bytes.fromhex(document["workload_hash"])
                + bytes.fromhex(document["artifact_hash"])
                + bytes.fromhex(document["public_key_hash"])
            )
            document["report_data"] = hashlib.sha256(binding).hexdigest()
        return document

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "stage_index": stage_index}

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "runtime": "dense-stage",
            "stage_index": stage_index,
            "dtype": "f64",
            "cache_entries": stage.runner.cache_entries,
        }

    @app.post("/v1/stages/execute")
    def execute(values: dict[str, Any]) -> dict[str, Any]:
        request = StageExecutionRequest.model_validate(values)
        if request.stage_index != stage_index:
            raise ProtocolError("request was routed to the wrong stage")
        event = threading.Event()
        with lock:
            if request.request_id in cancelled:
                raise ProtocolError("stage request ID is already active")
            cancelled[request.request_id] = event
        try:
            receiver, sender = stage.sessions.get_or_create(
                request.session_id, request.client_public
            )
            packet = base64.b64decode(request.packet, validate=True)
            hidden = _decode_tensor(
                packet,
                receiver,
                expected_dtype=request.dtype,
                expected_shape=request.shape,
            )
            if event.is_set():
                raise ProtocolError("stage request was cancelled")
            output = stage.execute(hidden, request.request_id)
            if event.is_set():
                raise ProtocolError("stage request was cancelled")
            response_packet, shape = _encode_tensor(output, request.request_id, sender)
            hidden.zero_()
            output.zero_()
            return {
                "stage_index": stage_index,
                "dtype": request.dtype,
                "shape": shape,
                "packet": base64.b64encode(response_packet).decode(),
            }
        except (ValueError, binascii.Error) as error:
            raise ProtocolError("invalid stage tensor encoding") from error
        finally:
            with lock:
                cancelled.pop(request.request_id, None)

    @app.post("/v1/relay")
    async def relay(request: Request) -> Any:
        values = await request.json()
        return execute(values)

    @app.delete("/v1/stages/{request_id}", status_code=204)
    def cancel(request_id: str) -> None:
        with lock:
            event = cancelled.get(request_id)
            if event is not None:
                event.set()
        stage.runner.drop_request(request_id)

    @app.delete("/v1/stages/sessions/{session_id}", status_code=204)
    def close_session(session_id: str) -> None:
        stage.sessions.remove(session_id)

    return app


def main() -> None:
    import argparse

    import uvicorn

    parser = argparse.ArgumentParser(prog="xurcroh-stage")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--stage-index", type=int, required=True)
    parser.add_argument("--stage-private-key", type=Path, required=True)
    parser.add_argument("--attestation-evidence", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9444)
    parser.add_argument("--development-http", action="store_true")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    parser.add_argument("--quic-listen")
    parser.add_argument("--quic-cert", type=Path)
    parser.add_argument("--quic-key", type=Path)
    args = parser.parse_args()
    if not args.development_http and (not args.tls_cert or not args.tls_key):
        parser.error("stage workers require TLS certificate/key or explicit --development-http")
    identity_bytes = args.stage_private_key.read_bytes()
    if len(identity_bytes) != 32:
        parser.error("stage private identity must contain exactly 32 bytes")
    stage_identity = X25519PrivateKey.from_private_bytes(identity_bytes)
    attestation_document = (
        json.loads(args.attestation_evidence.read_text())
        if args.attestation_evidence is not None
        else None
    )
    app = create_stage_app(
        args.manifest,
        stage_index=args.stage_index,
        private_key=stage_identity,
        attestation_document=attestation_document,
    )
    if any((args.quic_listen, args.quic_cert, args.quic_key)) and not all(
        (args.quic_listen, args.quic_cert, args.quic_key)
    ):
        parser.error("QUIC requires --quic-listen, --quic-cert, and --quic-key")
    native_server = None
    if args.quic_listen:
        from xurcroh.native_stage import NativeStageServer

        native_server = NativeStageServer(
            args.manifest,
            stage_index=args.stage_index,
            stage_private_key=stage_identity,
            certificate=args.quic_cert.read_bytes(),
            private_key=args.quic_key.read_bytes(),
            address=args.quic_listen,
        )
        native_server.start()
    try:
        uvicorn.run(
            app,
            host=args.host,
            port=args.port,
            ssl_certfile=args.tls_cert,
            ssl_keyfile=args.tls_key,
            log_level="info",
        )
    finally:
        if native_server is not None:
            native_server.close()


if __name__ == "__main__":
    main()
