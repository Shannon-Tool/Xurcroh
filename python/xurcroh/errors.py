from __future__ import annotations


class XurcrohError(Exception):
    """Base exception for stable public error handling."""


class ConfigurationError(XurcrohError, ValueError):
    pass


class ModelNotFoundError(XurcrohError, LookupError):
    pass


class ModelLoadError(XurcrohError):
    pass


class CapabilityError(XurcrohError):
    pass


class ClusterError(XurcrohError):
    pass


class ClusterUnavailableError(ClusterError):
    pass


class RequestError(XurcrohError):
    pass


class RequestTimeoutError(RequestError, TimeoutError):
    pass


class RequestCancelledError(RequestError):
    pass


class ClientClosedError(XurcrohError):
    pass


class ProtocolError(XurcrohError):
    pass


class AttestationError(XurcrohError):
    pass


class ShardError(XurcrohError):
    pass


class ShardIntegrityError(ShardError):
    pass


class CompatibilityError(XurcrohError):
    pass
