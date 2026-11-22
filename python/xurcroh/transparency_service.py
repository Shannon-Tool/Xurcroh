from __future__ import annotations

import hmac
import json
import os
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from xurcroh.signatures import ALGORITHM, HybridSigningKey
from xurcroh.transparency import (
    TRANSPARENCY_FORMAT,
    TransparencyError,
    TransparencyLog,
    verify_transparency_log,
)


class MeasurementEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    measurement: str = Field(pattern=r"^[0-9a-f]{64}$")


class TransparencyService:
    def __init__(
        self,
        storage: str | Path,
        *,
        signing_key: HybridSigningKey,
        write_token: str,
    ) -> None:
        if not write_token:
            raise ValueError("transparency write token cannot be empty")
        self.storage = Path(storage)
        self.storage.parent.mkdir(parents=True, exist_ok=True)
        self.log = TransparencyLog(signing_key)
        self._public_key = signing_key.public_key
        self._write_token = write_token
        self._lock = threading.Lock()
        self.app = FastAPI(title="Xurcroh transparency log")
        self._routes()

    def _document(self) -> dict[str, Any]:
        if not self.storage.is_file():
            return {"format": TRANSPARENCY_FORMAT, "entries": []}
        document = json.loads(self.storage.read_text())
        if not isinstance(document, dict):
            raise TransparencyError("stored transparency document must be an object")
        verify_transparency_log(
            document,
            public_key=self._public_key,
            expected_measurements={},
        )
        return document

    def _save(self) -> None:
        temporary = self.storage.with_suffix(".tmp")
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            json.dump(
                {"format": TRANSPARENCY_FORMAT, "entries": self.log.entries},
                stream,
                indent=2,
                sort_keys=True,
            )
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(self.storage)

    def _routes(self) -> None:
        @self.app.post("/v1/entries", status_code=201)
        def append(
            entry: MeasurementEntry, authorization: str | None = Header(default=None)
        ) -> dict[str, Any]:
            expected = f"Bearer {self._write_token}"
            if authorization is None or not hmac.compare_digest(authorization, expected):
                raise HTTPException(status_code=401, detail="write authentication required")
            with self._lock:
                document = self._document()
                self.log.entries = list(document["entries"])
                appended = self.log.append(entry.image_digest, entry.measurement)
                try:
                    self._save()
                except BaseException:
                    self.log.entries.pop()
                    raise
                return appended

        @self.app.get("/v1/log")
        def read() -> dict[str, Any]:
            with self._lock:
                return self._document()

        @self.app.get("/v1/measurements/{image_digest}")
        def latest(image_digest: str) -> dict[str, Any]:
            if len(image_digest) != 64:
                raise HTTPException(status_code=404, detail="measurement not found")
            with self._lock:
                entries = self._document()["entries"]
            for entry in reversed(entries):
                if entry["image_digest"] == image_digest:
                    return dict(entry)
            raise HTTPException(status_code=404, detail="measurement not found")

        @self.app.get("/v1/public-key")
        def public_key() -> dict[str, str]:
            return {
                "algorithm": ALGORITHM,
                "public_key": self._public_key.hex(),
            }
