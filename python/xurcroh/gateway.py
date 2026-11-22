"""A hosted gateway: the trusted client run by an operator on behalf of API users.

The gateway holds the client side of the protocol, so it sees every prompt and
output in the clear. Requesters authenticate with an API key, get a task ID for
each request, and only the requester who made a task can read its trace.

Every trace is marked ``hosted`` and ``operator-asserted``: the operator's
software produced and signed it, and the requester did not run it. Its signature
shows the trace was not changed after the gateway produced it and, with the
public log, that it was not replaced later. It does not show the gateway was
honest about the model or the settings. Only running the client yourself, or a
gateway in attested hardware (not built), gives that.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
import uuid
from collections import OrderedDict
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, HTTPException, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from xurcroh.errors import (
    ConfigurationError,
    ModelNotFoundError,
    RequestCancelledError,
    RequestTimeoutError,
    XurcrohError,
)
from xurcroh.requests import CompletionRequest, ImageGenerationRequest, VideoGenerationRequest
from xurcroh.signatures import ALGORITHM, HybridSigningKey

ASSURANCE = "operator-asserted"
TASK_HEADER = "X-Xurcroh-Task-Id"


def key_digest(api_key: str) -> str:
    """What the gateway stores for an API key; the key itself is never kept."""

    return hashlib.sha256(api_key.encode()).hexdigest()


def load_api_keys(text: str) -> dict[str, str]:
    """Parse ``owner:sha256-of-key`` lines into a digest-to-owner map."""

    keys: dict[str, str] = {}
    for number, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        owner, separator, digest = line.partition(":")
        if not separator or not owner or len(digest) != 64:
            raise ConfigurationError(f"API key line {number} must be owner:sha256-of-key")
        try:
            bytes.fromhex(digest)
        except ValueError as error:
            raise ConfigurationError(f"API key line {number} has a malformed digest") from error
        if digest in keys:
            raise ConfigurationError(f"API key line {number} repeats a key")
        keys[digest] = owner
    if not keys:
        raise ConfigurationError("a gateway needs at least one API key")
    return keys


class _Owners:
    """Which requester made which task, bounded like the trace history."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._owners: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def claim(self, owner: str) -> str:
        task_id = f"task-{uuid.uuid4().hex}"
        with self._lock:
            self._owners[task_id] = owner
            while len(self._owners) > self._limit:
                self._owners.popitem(last=False)
        return task_id

    def owns(self, owner: str, task_id: str) -> bool:
        with self._lock:
            return self._owners.get(task_id) == owner


def create_gateway_app(
    client: Any,
    *,
    api_keys: Mapping[str, str],
    signing_key: HybridSigningKey | None = None,
    gateway_id: str = "gateway",
) -> Any:
    """Serve an ``Xurcroh`` client over HTTP to authenticated requesters.

    ``api_keys`` maps the SHA-256 of each key to its owner. ``signing_key`` fixes
    the trace key so requesters can pin it across restarts.
    """

    if not api_keys:
        raise ConfigurationError("a gateway needs at least one API key")
    runtime = client._runtime
    runtime.configure_traces(mode="hosted", assurance=ASSURANCE, signing_key=signing_key)
    owners = _Owners(runtime.cluster.security.trace_history)
    keys = dict(api_keys)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        client.close()

    app = FastAPI(title="Xurcroh Gateway", lifespan=lifespan)

    @app.exception_handler(XurcrohError)
    def failed(_: Request, error: XurcrohError) -> JSONResponse:
        status = 500
        if isinstance(error, ModelNotFoundError):
            status = 404
        elif isinstance(error, ConfigurationError):
            status = 400
        elif isinstance(error, RequestTimeoutError):
            status = 504
        elif isinstance(error, RequestCancelledError):
            status = 499
        return JSONResponse(
            {"error": {"type": type(error).__name__, "message": str(error)}}, status
        )

    def requester(authorization: str | None) -> str:
        if authorization is None or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="an API key is required")
        presented = key_digest(authorization.removeprefix("Bearer "))
        owner = None
        # Compare against every key so timing does not show which one matched.
        for digest, name in keys.items():
            if hmac.compare_digest(digest, presented):
                owner = name
        if owner is None:
            raise HTTPException(status_code=401, detail="the API key is not recognised")
        return owner

    def owned(authorization: str | None, task_id: str) -> None:
        # The same answer for someone else's task and for no task at all, so task
        # identifiers cannot be probed.
        if not owners.owns(requester(authorization), task_id):
            raise HTTPException(status_code=404, detail="no such task for this API key")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "gateway_id": gateway_id}

    @app.get("/v1/trace-key")
    def trace_key() -> dict[str, str]:
        return {
            "algorithm": ALGORITHM,
            "public_key": runtime.trace_public_key.hex(),
            "log_key": runtime.trace_log_key.hex(),
            "mode": "hosted",
            "assurance": ASSURANCE,
        }

    @app.post("/v1/chat/completions")
    def complete(
        values: dict[str, Any], response: Response, authorization: str | None = Header(default=None)
    ) -> Any:
        task_id = owners.claim(requester(authorization))
        request = CompletionRequest.validated(**{**values, "request_id": task_id})
        if request.stream:

            def chunks() -> Iterator[bytes]:
                for chunk in runtime.stream(request):
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(
                chunks(), media_type="text/event-stream", headers={TASK_HEADER: task_id}
            )
        response.headers[TASK_HEADER] = task_id
        return runtime.complete(request)

    def media(
        kind: str, values: dict[str, Any], response: Response, authorization: str | None
    ) -> Any:
        task_id = owners.claim(requester(authorization))
        response.headers[TASK_HEADER] = task_id
        body = {**values, "request_id": task_id}
        if kind == "video":
            return runtime.generate_video(VideoGenerationRequest.model_validate(body))
        request = ImageGenerationRequest.model_validate(body)
        if kind == "edit" and request.image is None:
            raise ConfigurationError("image editing requires an input image")
        return runtime.generate_image(request)

    @app.post("/v1/images/generations")
    def generate_image(
        values: dict[str, Any], response: Response, authorization: str | None = Header(default=None)
    ) -> Any:
        return media("image", values, response, authorization)

    @app.post("/v1/images/edits")
    def edit_image(
        values: dict[str, Any], response: Response, authorization: str | None = Header(default=None)
    ) -> Any:
        return media("edit", values, response, authorization)

    @app.post("/v1/videos/generations")
    def generate_video(
        values: dict[str, Any], response: Response, authorization: str | None = Header(default=None)
    ) -> Any:
        return media("video", values, response, authorization)

    @app.get("/v1/tasks/{task_id}/trace")
    def trace(task_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        owned(authorization, task_id)
        return dict(runtime.trace(task_id))

    @app.get("/v1/tasks/{task_id}/proof")
    def proof(task_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
        owned(authorization, task_id)
        return dict(runtime.trace_proof(task_id))

    return app


def main(argv: list[str] | None = None) -> None:
    import argparse
    from pathlib import Path

    import uvicorn
    import yaml

    from Xurcroh import Xurcroh
    from xurcroh.signatures import load_signing_key

    parser = argparse.ArgumentParser(description="Serve a hosted Xurcroh gateway over HTTPS")
    parser.add_argument("--config", required=True, type=Path, help="YAML with cluster and models")
    parser.add_argument("--api-keys", required=True, type=Path, help="owner:sha256-of-key lines")
    parser.add_argument("--trace-key", required=True, type=Path, help="hybrid signing key file")
    parser.add_argument("--host", default="0.0.0.0")  # noqa: S104
    parser.add_argument("--port", type=int, default=8443)
    parser.add_argument("--ssl-certfile", required=True, type=Path)
    parser.add_argument("--ssl-keyfile", required=True, type=Path)
    args = parser.parse_args(argv)
    settings = yaml.safe_load(args.config.read_text())
    if not isinstance(settings, dict) or "models" not in settings:
        raise ConfigurationError("gateway configuration needs a models section")
    client = Xurcroh(
        cluster=settings.get("cluster"),
        model_path=settings["models"],
        security=settings.get("security"),
    )
    app = create_gateway_app(
        client,
        api_keys=load_api_keys(args.api_keys.read_text()),
        signing_key=load_signing_key(args.trace_key),
    )
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        access_log=False,
        ssl_certfile=str(args.ssl_certfile),
        ssl_keyfile=str(args.ssl_keyfile),
    )


if __name__ == "__main__":
    main()
