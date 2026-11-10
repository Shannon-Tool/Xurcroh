from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from xurcroh.config import ClusterConfig, ModelRegistry
from xurcroh.requests import CompletionRequest, ImageGenerationRequest, VideoGenerationRequest
from xurcroh.runtime import RuntimeService
from xurcroh.runtime.media_sessions import install_protected_media_routes


def create_worker_app(
    models: ModelRegistry, *, worker_id: str, runtime: RuntimeService | None = None
) -> FastAPI:
    if runtime is None:
        cluster = ClusterConfig.normalize({"name": worker_id, "endpoint": "local://cpu"})
        runtime = RuntimeService(cluster, models)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        runtime.close()
        if hasattr(app.state, "protected_media_sessions"):
            app.state.protected_media_sessions.close()

    app = FastAPI(title="Xurcroh Worker", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {"status": "ok", "worker_id": worker_id}

    @app.get("/metrics")
    def metrics() -> dict[str, int | float]:
        return runtime.metrics()

    @app.get("/v1/capabilities")
    def capabilities() -> dict[str, Any]:
        return {
            "worker_id": worker_id,
            "models": list(models.names()),
            "modalities": sorted(
                {
                    definition.modality if definition.modality != "auto" else "text"
                    for _, definition in models.items()
                }
            ),
            "engine": "xurcroh-runtime",
        }

    @app.post("/v1/chat/completions")
    def complete(values: dict[str, Any]) -> Any:
        request = CompletionRequest.validated(**values)
        if request.stream:

            def chunks() -> Iterator[bytes]:
                for chunk in runtime.stream(request):
                    yield f"data: {chunk.model_dump_json(exclude_none=True)}\n\n".encode()
                yield b"data: [DONE]\n\n"

            return StreamingResponse(chunks(), media_type="text/event-stream")
        return runtime.complete(request)

    @app.post("/v1/images/generations")
    def generate_image(values: dict[str, Any]) -> Any:
        request = ImageGenerationRequest.model_validate(values)
        return runtime.generate_image(request)

    @app.post("/v1/images/edits")
    def edit_image(values: dict[str, Any]) -> Any:
        request = ImageGenerationRequest.model_validate(values)
        if request.image is None:
            from xurcroh.errors import ConfigurationError

            raise ConfigurationError("image editing requires an input image")
        return runtime.generate_image(request)

    @app.post("/v1/videos/generations")
    def generate_video(values: dict[str, Any]) -> Any:
        request = VideoGenerationRequest.model_validate(values)
        return runtime.generate_video(request)

    install_protected_media_routes(app, runtime)
    return app


def main() -> None:
    import argparse
    import base64
    import threading
    import time

    import httpx
    import uvicorn
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    from xurcroh.controller.identity import Registration, sign_registration

    parser = argparse.ArgumentParser(prog="xurcroh-worker")
    parser.add_argument("--worker-id", required=True)
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--controller")
    parser.add_argument("--advertised-endpoint")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=9443)
    parser.add_argument("--development-http", action="store_true")
    parser.add_argument("--tls-cert")
    parser.add_argument("--tls-key")
    args = parser.parse_args()
    if not args.development_http and (not args.tls_cert or not args.tls_key):
        parser.error(
            "production workers require --tls-cert and --tls-key or explicit --development-http"
        )
    model_values = {}
    for item in args.model:
        name, separator, path = item.partition("=")
        if not separator or not name or not path:
            parser.error("--model requires name=/absolute/model/path")
        model_values[name] = path
    app = create_worker_app(ModelRegistry.normalize(model_values), worker_id=args.worker_id)
    scheme = "http+dev" if args.development_http else "https"
    endpoint = args.advertised_endpoint or f"{scheme}://{args.host}:{args.port}"
    if args.controller:
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes_raw()
        controller = args.controller.replace("http+dev://", "http://", 1).rstrip("/")

        def register() -> None:
            delay = 0.5
            while True:
                time.sleep(delay)
                registration = Registration(
                    worker_id=args.worker_id,
                    endpoint=endpoint,
                    models=tuple(model_values),
                    modalities=("text",),
                    timestamp=time.time(),
                )
                signature = sign_registration(private_key, registration)
                try:
                    response = httpx.post(
                        f"{controller}/v1/workers/register",
                        json={
                            "registration": registration.model_dump(mode="json"),
                            "signature": base64.b64encode(signature).decode(),
                            "public_key": base64.b64encode(public_key).decode(),
                        },
                        timeout=30,
                    )
                    response.raise_for_status()
                    delay = 20
                except httpx.HTTPError:
                    delay = 5

        threading.Thread(target=register, daemon=True, name="worker-registration").start()
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        ssl_certfile=args.tls_cert,
        ssl_keyfile=args.tls_key,
        log_level="info",
    )


if __name__ == "__main__":
    main()
