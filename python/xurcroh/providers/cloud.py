from __future__ import annotations

import ipaddress
import os
import re
from collections.abc import Iterable
from typing import Any

import httpx

from xurcroh.errors import ProviderError


class CloudRESTClient:
    def __init__(self, client: httpx.Client) -> None:
        self._client = client

    def request(
        self,
        method: str,
        path: str,
        **kwargs: Any,
    ) -> Any:
        try:
            response = self._client.request(method, path, **kwargs)
            response.raise_for_status()
            if not response.content:
                return {}
            return response.json()
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                return None
            raise ProviderError(f"cloud API returned HTTP {error.response.status_code}") from error
        except (httpx.HTTPError, ValueError) as error:
            raise ProviderError(f"cloud API {method} {path} failed") from error


def instance_label(value: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,62}", value) is None:
        raise ValueError("instance label must contain 1-63 letters, digits, or hyphens")
    return value


def allowed_source_ranges(values: Any = None) -> list[str]:
    if values is None:
        values = os.environ.get("XURCROH_ALLOWED_SOURCE_RANGES")
    if isinstance(values, str):
        values = values.split(",")
    if not isinstance(values, Iterable):
        raise ProviderError(
            "allowed_source_ranges or XURCROH_ALLOWED_SOURCE_RANGES must restrict inbound access"
        )
    networks = []
    for value in values:
        network = ipaddress.ip_network(str(value).strip(), strict=False)
        networks.append(str(network))
    if not networks:
        raise ProviderError("at least one allowed source range is required")
    return networks
