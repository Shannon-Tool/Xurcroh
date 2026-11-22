"""A public, append-only log of digests with proofs anyone can check.

The log holds only digests: task-trace hashes, model manifest digests and
release digests. It reveals nothing about a request. Each entry gets an
inclusion proof against a signed tree head, and any two tree heads can be
connected by a consistency proof, so the operator cannot remove or rewrite an
entry without that being detectable by someone who kept an earlier head.

A witness is an independent party that keeps the last head it saw, checks each
new head extends it, and countersigns. A head countersigned by a witness the
verifier trusts shows the operator did not present a different history to that
witness. The log operator alone can still show different histories to parties
who never compare heads; witnesses are what narrow that.
"""

from __future__ import annotations

import hmac
import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from xurcroh.errors import XurcrohError
from xurcroh.merkle_log import (
    MerkleLog,
    MerkleProofError,
    leaf_hash,
    verify_consistency,
    verify_inclusion,
    verify_tree_head,
)
from xurcroh.signatures import (
    ALGORITHM,
    HybridSigningKey,
    SignatureError,
    key_fingerprint,
    verify_hybrid,
)

KINDS = ("trace", "manifest", "release")
COSIGNATURE_CONTEXT = b"xurcroh-log-witness"


class PublicLogError(XurcrohError):
    """A public-log request or proof failed; the message says which part."""


class _Refused(PublicLogError):
    """The log answered and said no, as opposed to not answering."""


def entry_leaf(kind: str, digest: str) -> bytes:
    """The bytes logged for a digest; the kind keeps different uses apart."""

    if kind not in KINDS:
        raise PublicLogError(f"unknown log entry kind {kind!r}")
    try:
        raw = bytes.fromhex(digest)
    except ValueError as error:
        raise PublicLogError("log entry digest must be hexadecimal") from error
    if len(raw) != 32 or digest != raw.hex():
        raise PublicLogError("log entry digest must be 64 lowercase hexadecimal characters")
    return kind.encode() + b":" + raw


def _witnessed(head: Mapping[str, Any]) -> bytes:
    statement = {name: head[name] for name in ("log_key", "tree_size", "root_hash")}
    return json.dumps(statement, sort_keys=True, separators=(",", ":")).encode()


def verify_cosignature(head: Mapping[str, Any], cosignature: Mapping[str, Any]) -> str:
    """Check a witness signature over a tree head; return the witness key fingerprint."""

    try:
        witness = bytes.fromhex(cosignature["witness_key"])
        verify_hybrid(
            witness, COSIGNATURE_CONTEXT, _witnessed(head), bytes.fromhex(cosignature["signature"])
        )
    except (KeyError, TypeError, ValueError, SignatureError) as error:
        raise PublicLogError("witness signature does not verify") from error
    return key_fingerprint(witness)


class LogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: str = Field(pattern=r"^(trace|manifest|release)$")
    digest: str = Field(pattern=r"^[0-9a-f]{64}$")


class LogCosignature(BaseModel):
    model_config = ConfigDict(extra="forbid")
    tree_size: int = Field(gt=0)
    witness_key: str = Field(pattern=r"^[0-9a-f]+$")
    signature: str = Field(pattern=r"^[0-9a-f]+$")


class PublicLog:
    """The log itself: entries on disk, proofs and signed heads in memory."""

    def __init__(self, directory: str | Path, signing_key: HybridSigningKey) -> None:
        self._directory = Path(directory)
        self._directory.mkdir(parents=True, exist_ok=True)
        self._entries_path = self._directory / "entries.jsonl"
        self._witness_path = self._directory / "cosignatures.json"
        self._merkle = MerkleLog(signing_key)
        self._entries: list[tuple[str, str]] = []
        self._index: dict[tuple[str, str], int] = {}
        self._cosignatures: dict[int, list[dict[str, str]]] = {}
        self._lock = threading.Lock()
        if self._entries_path.is_file():
            for line in self._entries_path.read_text().splitlines():
                record = json.loads(line)
                self._remember(str(record["kind"]), str(record["digest"]))
        if self._witness_path.is_file():
            stored = json.loads(self._witness_path.read_text())
            self._cosignatures = {int(size): list(items) for size, items in stored.items()}

    def _remember(self, kind: str, digest: str) -> int:
        index = self._merkle.append(entry_leaf(kind, digest))
        self._entries.append((kind, digest))
        self._index.setdefault((kind, digest), index)
        return index

    @property
    def public_key(self) -> bytes:
        return self._merkle.public_key

    @property
    def size(self) -> int:
        return self._merkle.size

    def append(self, kind: str, digest: str) -> int:
        """Log a digest once; logging it again returns its existing position."""

        entry_leaf(kind, digest)
        with self._lock:
            existing = self._index.get((kind, digest))
            if existing is not None:
                return existing
            # Written and synced before it is counted, so an acknowledged entry
            # survives a restart.
            descriptor = os.open(self._entries_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            with os.fdopen(descriptor, "w") as stream:
                stream.write(json.dumps({"kind": kind, "digest": digest}) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            return self._remember(kind, digest)

    def lookup(self, kind: str, digest: str) -> int | None:
        with self._lock:
            return self._index.get((kind, digest))

    def tree_head(self, size: int | None = None) -> dict[str, Any]:
        head = self._merkle.signed_tree_head(size)
        with self._lock:
            head["cosignatures"] = list(self._cosignatures.get(head["tree_size"], []))
        return head

    def proof(self, index: int, size: int | None = None) -> dict[str, Any]:
        head = self.tree_head(size)
        path = self._merkle.inclusion_proof(index, head["tree_size"])
        kind, digest = self._entries[index]
        return {
            "kind": kind,
            "digest": digest,
            "leaf_index": index,
            "path": [node.hex() for node in path],
            "tree_head": head,
        }

    def consistency(self, first: int, second: int | None = None) -> list[str]:
        return [node.hex() for node in self._merkle.consistency_proof(first, second)]

    def entries(self, start: int, end: int) -> list[dict[str, str]]:
        return [{"kind": kind, "digest": digest} for kind, digest in self._entries[start:end]]

    def add_cosignature(self, tree_size: int, cosignature: Mapping[str, Any]) -> None:
        if not 0 < tree_size <= self.size:
            raise PublicLogError("cosignature is for a tree size outside the log")
        head = self._merkle.signed_tree_head(tree_size)
        verify_cosignature(head, cosignature)
        record = {
            "witness_key": str(cosignature["witness_key"]),
            "signature": str(cosignature["signature"]),
        }
        with self._lock:
            held = self._cosignatures.setdefault(tree_size, [])
            if all(item["witness_key"] != record["witness_key"] for item in held):
                held.append(record)
                temporary = self._witness_path.with_suffix(".tmp")
                temporary.write_text(json.dumps(self._cosignatures, sort_keys=True))
                temporary.replace(self._witness_path)


def create_app(log: PublicLog, *, write_token: str) -> Any:
    """HTTP interface: anyone may read and fetch proofs; appending needs the token."""

    from fastapi import FastAPI, Header, HTTPException

    if not write_token:
        raise ValueError("public log write token cannot be empty")

    app = FastAPI(title="Xurcroh public log")

    def fail(error: Exception) -> HTTPException:
        return HTTPException(status_code=400, detail=str(error))

    @app.post("/v1/entries", status_code=201)
    def append(entry: LogEntry, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        expected = f"Bearer {write_token}"
        if authorization is None or not hmac.compare_digest(authorization, expected):
            raise HTTPException(status_code=401, detail="write authentication required")
        return log.proof(log.append(entry.kind, entry.digest))

    @app.get("/v1/tree-head")
    def tree_head() -> dict[str, Any]:
        return log.tree_head()

    @app.get("/v1/entries/{kind}/{digest}")
    def find(kind: str, digest: str) -> dict[str, Any]:
        try:
            index = log.lookup(kind, digest)
        except PublicLogError as error:
            raise fail(error) from error
        if index is None:
            raise HTTPException(status_code=404, detail="digest is not in the log")
        return log.proof(index)

    @app.get("/v1/entries")
    def entries(start: int = 0, end: int | None = None) -> dict[str, Any]:
        stop = log.size if end is None else min(end, log.size)
        if start < 0 or stop - start > 10_000:
            raise HTTPException(status_code=400, detail="request at most 10000 entries")
        return {"start": start, "entries": log.entries(start, stop)}

    @app.get("/v1/consistency")
    def consistency(first: int, second: int | None = None) -> dict[str, Any]:
        try:
            return {"path": log.consistency(first, second), "tree_head": log.tree_head(second)}
        except MerkleProofError as error:
            raise fail(error) from error

    @app.post("/v1/cosignatures", status_code=201)
    def cosign(cosignature: LogCosignature) -> dict[str, str]:
        try:
            log.add_cosignature(cosignature.tree_size, cosignature.model_dump())
        except PublicLogError as error:
            raise fail(error) from error
        return {"status": "recorded"}

    @app.get("/v1/public-key")
    def public_key() -> dict[str, str]:
        return {"algorithm": ALGORITHM, "public_key": log.public_key.hex()}

    return app


def verify_entry_proof(
    proof: Mapping[str, Any], *, kind: str, digest: str, log_key: bytes
) -> dict[str, Any]:
    """Check that ``digest`` is in the log signed by ``log_key``; return the tree head."""

    try:
        head = proof["tree_head"]
        size, root = verify_tree_head(
            {name: value for name, value in head.items() if name != "cosignatures"}, log_key
        )
        verify_inclusion(
            leaf_hash(entry_leaf(kind, digest)),
            int(proof["leaf_index"]),
            size,
            [bytes.fromhex(node) for node in proof["path"]],
            root,
        )
    except (KeyError, TypeError, ValueError, MerkleProofError) as error:
        raise PublicLogError(f"digest is not in the log: {error}") from error
    return dict(head)


def witnesses_of(head: Mapping[str, Any], trusted: Sequence[bytes]) -> list[str]:
    """Fingerprints of trusted witnesses whose signature on this head verifies."""

    wanted = {key.hex() for key in trusted}
    found = []
    for cosignature in head.get("cosignatures", []):
        if cosignature.get("witness_key") in wanted:
            found.append(verify_cosignature(head, cosignature))
    return sorted(set(found))


class LogClient:
    """Talk to a public log and check everything it returns against a pinned key."""

    def __init__(
        self,
        endpoint: str,
        log_key: bytes,
        *,
        write_token: str | None = None,
        client: Any | None = None,
    ) -> None:
        from urllib.parse import urlsplit

        parsed = urlsplit(endpoint)
        if client is None and (parsed.scheme != "https" or not parsed.hostname):
            raise PublicLogError("a public log must be reached over HTTPS")
        self._endpoint = endpoint
        self._key = log_key
        self._token = write_token
        self._client = client

    def _request(self, method: str, path: str, **options: Any) -> Any:
        import httpx

        owned = self._client is None
        client = self._client or httpx.Client(base_url=self._endpoint, timeout=30)
        try:
            response = client.request(method, path, **options)
            if response.status_code == 404:
                return None
            if response.status_code >= 400:
                raise _Refused(f"public log refused the request with status {response.status_code}")
            return response.json()
        except _Refused:
            raise
        except (httpx.HTTPError, OSError, ValueError) as error:
            raise PublicLogError(f"public log request failed: {error}") from error
        finally:
            if owned:
                client.close()

    def publish(self, kind: str, digest: str) -> dict[str, Any]:
        """Log a digest and return its verified inclusion proof."""

        if self._token is None:
            raise PublicLogError("publishing to the public log requires a write token")
        proof = self._request(
            "POST",
            "/v1/entries",
            json={"kind": kind, "digest": digest},
            headers={"Authorization": f"Bearer {self._token}"},
        )
        if not isinstance(proof, dict):
            raise PublicLogError("public log returned no proof")
        verify_entry_proof(proof, kind=kind, digest=digest, log_key=self._key)
        return proof

    def find(self, kind: str, digest: str) -> dict[str, Any] | None:
        """Return a verified inclusion proof, or ``None`` if the digest is not logged."""

        proof = self._request("GET", f"/v1/entries/{kind}/{digest}")
        if proof is None:
            return None
        verify_entry_proof(proof, kind=kind, digest=digest, log_key=self._key)
        return dict(proof)

    def extends(self, earlier: Mapping[str, Any]) -> dict[str, Any]:
        """Check the log's current head only appended to ``earlier``; return the current head."""

        try:
            first, first_root = verify_tree_head(
                {name: value for name, value in earlier.items() if name != "cosignatures"},
                self._key,
            )
            answer = self._request("GET", "/v1/consistency", params={"first": first})
            if not isinstance(answer, dict):
                raise PublicLogError("public log returned no consistency proof")
            head = answer["tree_head"]
            second, second_root = verify_tree_head(
                {name: value for name, value in head.items() if name != "cosignatures"},
                self._key,
            )
            verify_consistency(
                first, second, first_root, second_root, [bytes.fromhex(n) for n in answer["path"]]
            )
        except (KeyError, TypeError, ValueError, MerkleProofError, _Refused) as error:
            # A log that cannot connect its current head to one it signed earlier has
            # dropped or rewritten entries. An unreachable log is a different failure.
            raise PublicLogError(
                f"the public log no longer contains the history it showed before: {error}"
            ) from error
        return dict(head)


class Witness:
    """Countersign tree heads, refusing any head that does not extend the last one seen."""

    def __init__(self, signing_key: HybridSigningKey, state: str | Path) -> None:
        self._key = signing_key
        self._state = Path(state)

    @property
    def public_key(self) -> bytes:
        return self._key.public_key

    def observe(self, log: LogClient) -> dict[str, Any]:
        """Fetch the current head, check it against the stored one, countersign and submit."""

        if self._state.is_file():
            head = log.extends(json.loads(self._state.read_text()))
        else:
            head = log._request("GET", "/v1/tree-head")
            if not isinstance(head, dict):
                raise PublicLogError("public log returned no tree head")
            verify_tree_head(
                {name: value for name, value in head.items() if name != "cosignatures"}, log._key
            )
        if head["tree_size"] == 0:
            raise PublicLogError("an empty log has nothing to witness")
        cosignature = {
            "tree_size": head["tree_size"],
            "witness_key": self.public_key.hex(),
            "signature": self._key.sign(COSIGNATURE_CONTEXT, _witnessed(head)).hex(),
        }
        temporary = self._state.with_suffix(".tmp")
        temporary.write_text(json.dumps(head, sort_keys=True))
        temporary.replace(self._state)
        log._request("POST", "/v1/cosignatures", json=cosignature)
        return cosignature


def main(argv: list[str] | None = None) -> None:
    import argparse

    from xurcroh.signatures import load_signing_key

    parser = argparse.ArgumentParser(description="Run or use a Xurcroh public digest log")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve", help="serve a log over HTTPS")
    serve.add_argument("--storage", required=True, type=Path)
    serve.add_argument("--signing-key", required=True, type=Path)
    serve.add_argument("--write-token-env", default="XURCROH_LOG_WRITE_TOKEN")
    serve.add_argument("--host", default="0.0.0.0")  # noqa: S104
    serve.add_argument("--port", type=int, default=9401)
    serve.add_argument("--ssl-certfile", required=True, type=Path)
    serve.add_argument("--ssl-keyfile", required=True, type=Path)
    for name, text in (
        ("publish", "log a digest and print its proof"),
        ("find", "print the proof for a logged digest"),
        ("witness", "check the log extends the last head seen and countersign it"),
    ):
        command = commands.add_parser(name, help=text)
        command.add_argument("--url", required=True)
        command.add_argument("--log-key", required=True, help="hex key of the log")
        if name == "witness":
            command.add_argument("--signing-key", required=True, type=Path)
            command.add_argument("--state", required=True, type=Path)
        else:
            command.add_argument("--kind", required=True, choices=KINDS)
            command.add_argument("--digest", required=True)
        if name == "publish":
            command.add_argument("--write-token-env", default="XURCROH_LOG_WRITE_TOKEN")
    args = parser.parse_args(argv)
    if args.command == "serve":
        import uvicorn

        token = os.environ.get(args.write_token_env)
        if not token:
            raise PublicLogError(f"write token variable {args.write_token_env} is unset")
        log = PublicLog(args.storage, load_signing_key(args.signing_key))
        uvicorn.run(
            create_app(log, write_token=token),
            host=args.host,
            port=args.port,
            access_log=False,
            ssl_certfile=str(args.ssl_certfile),
            ssl_keyfile=str(args.ssl_keyfile),
        )
        return
    token = os.environ.get(args.write_token_env) if args.command == "publish" else None
    client = LogClient(args.url, bytes.fromhex(args.log_key), write_token=token)
    if args.command == "witness":
        witness = Witness(load_signing_key(args.signing_key), args.state)
        print(json.dumps(witness.observe(client), sort_keys=True))
    elif args.command == "publish":
        print(json.dumps(client.publish(args.kind, args.digest), sort_keys=True))
    else:
        found = client.find(args.kind, args.digest)
        if found is None:
            raise SystemExit("digest is not in the log")
        print(json.dumps(found, sort_keys=True))


if __name__ == "__main__":
    main()
