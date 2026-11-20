from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import numpy as np
import torch
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

RING_MODULUS = 2**32
PRIME_MODULUS = 2**32 - 5
FIELD_MODULUS = PRIME_MODULUS
MAX_SIGNED_RING_VALUE = 2**31 - 1
MAX_SIGNED_FIELD_VALUE = PRIME_MODULUS // 2
MAX_ACTIVATION_BITS = 12
WEIGHT_QUANTIZATION_BITS = 4


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor | None
    scale: float
    zero_point: int
    shape: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        if self.values is None:
            if self.shape is None or len(self.shape) != 2 or any(value < 1 for value in self.shape):
                raise ValueError("quantized descriptors require positive matrix dimensions")
        else:
            if self.values.dtype not in {torch.int8, torch.int16, torch.int32, torch.int64}:
                raise ValueError("quantized values must use a signed integer dtype")
            if self.values.ndim != 2:
                raise ValueError("quantized tensors must be matrices")
            if self.shape is not None and self.shape != tuple(self.values.shape):
                raise ValueError("quantized descriptor shape does not match its values")
        if self.scale <= 0:
            raise ValueError("quantization scale must be positive")

    @property
    def dimensions(self) -> tuple[int, int]:
        if self.values is not None:
            return (self.values.shape[0], self.values.shape[1])
        assert self.shape is not None
        return self.shape

    def require_values(self) -> torch.Tensor:
        if self.values is None:
            raise ValueError("reference arithmetic requires materialized quantized values")
        return self.values

    def dequantize(self) -> torch.Tensor:
        return (self.require_values() - self.zero_point).to(torch.float64) * self.scale


@dataclass(frozen=True)
class QuantizedProduct:
    values: torch.Tensor
    scale: float

    def dequantize(self) -> torch.Tensor:
        return self.values.to(torch.float64) * self.scale


@dataclass(frozen=True)
class ActivationQuantization:
    values: torch.Tensor
    scales: torch.Tensor
    zero_point: int


class ShareSecurity(Enum):
    INFORMATION_THEORETIC = "information-theoretic"
    COMPUTATIONAL = "computational"


@dataclass(frozen=True)
class ShareGrid:
    shares: list[torch.Tensor]
    security: ShareSecurity
    epoch: int = 0


@dataclass(frozen=True)
class ExactFreivaldsChallenge:
    r: torch.Tensor
    v: torch.Tensor
    prime: int


def symmetric_quantize(values: torch.Tensor, *, bits: int) -> QuantizedTensor:
    if not 2 <= bits <= 16:
        raise ValueError("quantization bits must be between 2 and 16")
    if values.ndim != 2:
        raise ValueError("quantization requires a matrix")
    maximum = 2 ** (bits - 1) - 1
    minimum = -(2 ** (bits - 1))
    largest = float(values.abs().max().item()) if values.numel() else 0.0
    scale = largest / maximum if largest else 1.0
    quantized = (
        torch.round(values.to(torch.float64) / scale).clamp(minimum, maximum).to(torch.int64)
    )
    return QuantizedTensor(quantized, scale, 0)


def quantize_activations(values: torch.Tensor, *, bits: int) -> ActivationQuantization:
    if not 8 <= bits <= 12:
        raise ValueError("activation bits must be between 8 and 12")
    if values.ndim not in {1, 2}:
        raise ValueError("activation quantization requires a token vector or matrix")
    maximum = 2 ** (bits - 1) - 1
    minimum = -(2 ** (bits - 1))
    largest = values.abs().amax(dim=-1)
    scales = torch.where(largest > 0, largest / maximum, torch.ones_like(largest))
    quantized = torch.round(values.to(torch.float64) / scales.unsqueeze(-1)).clamp(minimum, maximum)
    return ActivationQuantization(quantized.to(torch.int64), scales, 0)


def check_unreduced_product_bound(inner: int, left_bound: int, right_bound: int) -> None:
    if left_bound * right_bound * inner > MAX_SIGNED_FIELD_VALUE:
        raise OverflowError("unreduced product exceeds the signed field decoding bound")


def quantized_weight_bound(quantized: QuantizedTensor) -> int:
    if quantized.values is not None:
        return int((quantized.values - quantized.zero_point).abs().max().item())
    # compiler.compile_operation quantizes every weight with symmetric_quantize(bits=4)
    return 2 ** (WEIGHT_QUANTIZATION_BITS - 1)


def quantized_matmul(left: QuantizedTensor, right: QuantizedTensor) -> QuantizedProduct:
    if left.dimensions[1] != right.dimensions[0]:
        raise ValueError("matrix dimensions do not match")
    left_values = left.require_values().to(torch.int64) - left.zero_point
    right_values = right.require_values().to(torch.int64) - right.zero_point
    inner = left_values.shape[1]
    left_bound = int(left_values.abs().max().item()) if left_values.numel() else 0
    right_bound = int(right_values.abs().max().item()) if right_values.numel() else 0
    check_unreduced_product_bound(inner, left_bound, right_bound)
    return QuantizedProduct(left_values @ right_values, left.scale * right.scale)


def _secure_uniform(shape: tuple[int, ...], modulus: int) -> torch.Tensor:
    if modulus <= 1 or modulus > 2**63:
        raise ValueError("unsupported share modulus")
    count = 1
    for dimension in shape:
        count *= dimension
    if count < 1:
        raise ValueError("share shape must be nonempty")
    limit = (2**64 // modulus) * modulus
    accepted: list[torch.Tensor] = []
    remaining = count
    while remaining:
        candidate = np.frombuffer(os.urandom(remaining * 8), dtype=np.uint64)
        selected = candidate[candidate < limit].astype(np.int64)
        accepted.append(torch.from_numpy(selected).remainder(modulus))
        remaining = count - sum(tensor.numel() for tensor in accepted)
    return torch.cat(accepted).reshape(shape)


def signed_ring(values: torch.Tensor) -> torch.Tensor:
    return (values + 2**31).remainder(RING_MODULUS) - 2**31


def signed_field(values: torch.Tensor) -> torch.Tensor:
    canonical = values.remainder(FIELD_MODULUS)
    return torch.where(canonical > MAX_SIGNED_FIELD_VALUE, canonical - FIELD_MODULUS, canonical)


def split_secret(secret: torch.Tensor, *, count: int, modulus: int = FIELD_MODULUS) -> ShareGrid:
    if count < 2:
        raise ValueError("additive sharing requires at least two roles")
    if secret.dtype != torch.int64:
        raise ValueError("shared secrets must use int64")
    canonical = secret.remainder(modulus)
    shares = [_secure_uniform(secret.shape, modulus) for _ in range(count - 1)]
    shares.append((canonical - torch.sum(torch.stack(shares), dim=0)).remainder(modulus))
    return ShareGrid(shares, ShareSecurity.INFORMATION_THEORETIC)


def combine_shares(shares: list[torch.Tensor], *, modulus: int = FIELD_MODULUS) -> torch.Tensor:
    if not shares:
        raise ValueError("at least one share is required")
    return torch.sum(torch.stack(shares), dim=0).remainder(modulus)


def refresh_shares(
    shares: list[torch.Tensor], *, first: int, second: int, modulus: int = FIELD_MODULUS
) -> list[torch.Tensor]:
    if first == second or not 0 <= first < len(shares) or not 0 <= second < len(shares):
        raise ValueError("share refresh requires two distinct roles")
    delta = _secure_uniform(shares[first].shape, modulus)
    refreshed = [share.clone() for share in shares]
    refreshed[first] = (refreshed[first] + delta).remainder(modulus)
    refreshed[second] = (refreshed[second] - delta).remainder(modulus)
    return refreshed


def _seeded_uniform(seed: bytes, shape: tuple[int, ...]) -> torch.Tensor:
    if len(seed) != 32:
        raise ValueError("seed-expanded shares require a 32-byte seed")
    count = 1
    for dimension in shape:
        count *= dimension
    cipher = Cipher(algorithms.ChaCha20(seed, bytes(16)), mode=None)
    stream = cipher.encryptor().update(bytes(count * 8))
    return (
        torch.frombuffer(bytearray(stream), dtype=torch.uint64)
        .to(torch.int64)
        .remainder(FIELD_MODULUS)
        .reshape(shape)
    )


def seeded_share_seed(seed: bytes, index: int) -> bytes:
    if len(seed) != 32 or index < 0:
        raise ValueError("a 32-byte seed and nonnegative share index are required")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"xurcroh-fortress-v1",
        info=b"weight-share:" + index.to_bytes(4, "big"),
    ).derive(seed)


def activation_row_seed(seed: bytes, row: int) -> bytes:
    if len(seed) != 32 or row < 0:
        raise ValueError("activation row seeds require a 32-byte seed and nonnegative row")
    return HKDF(
        algorithm=hashes.SHA256(),
        length=32,
        salt=b"xurcroh-fortress-v1",
        info=b"activation-row:" + row.to_bytes(8, "big"),
    ).derive(seed)


def seeded_split_secret(secret: torch.Tensor, *, seed: bytes, count: int) -> ShareGrid:
    if count < 2:
        raise ValueError("additive sharing requires at least two roles")
    if secret.dtype != torch.int64:
        raise ValueError("shared secrets must use int64")
    canonical = secret.remainder(FIELD_MODULUS)
    shares = [
        _seeded_uniform(seeded_share_seed(seed, index), secret.shape) for index in range(count - 1)
    ]
    shares.append((canonical - torch.sum(torch.stack(shares), dim=0)).remainder(FIELD_MODULUS))
    return ShareGrid(shares, ShareSecurity.COMPUTATIONAL)


def _modular_matmul(left: torch.Tensor, right: torch.Tensor, modulus: int) -> torch.Tensor:
    if left.ndim not in {1, 2} or right.ndim not in {1, 2}:
        raise ValueError("modular multiplication requires vectors or matrices")
    left_matrix = left.reshape(1, -1) if left.ndim == 1 else left
    right_matrix = right.reshape(-1, 1) if right.ndim == 1 else right
    if left_matrix.shape[1] != right_matrix.shape[0]:
        raise ValueError("matrix and vector dimensions do not match")
    values = []
    for left_row in left_matrix:
        for right_column in right_matrix.T:
            values.append(
                sum(int(a) * int(b) for a, b in zip(left_row, right_column, strict=True)) % modulus
            )
    if left.ndim == 1 and right.ndim == 1:
        return torch.tensor(values, dtype=torch.int64)
    if left.ndim == 1:
        return torch.tensor(values, dtype=torch.int64).reshape(right_matrix.shape[1])
    return torch.tensor(values, dtype=torch.int64).reshape(
        left_matrix.shape[0], right_matrix.shape[1]
    )


def _modular_matvec(matrix: torch.Tensor, vector: torch.Tensor, modulus: int) -> torch.Tensor:
    if matrix.ndim != 2 or vector.ndim != 1 or matrix.shape[1] != vector.shape[0]:
        raise ValueError("matrix and vector dimensions do not match")
    values = [
        sum(int(left) * int(right) for left, right in zip(row, vector, strict=True)) % modulus
        for row in matrix
    ]
    return torch.tensor(values, dtype=torch.int64)


def _modular_dot(left: torch.Tensor, right: torch.Tensor, modulus: int) -> int:
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("dot-product dimensions do not match")
    return sum(int(a) * int(b) for a, b in zip(left, right, strict=True)) % modulus


def freivalds_challenge(
    weight: torch.Tensor, *, r: torch.Tensor, prime: int = PRIME_MODULUS
) -> ExactFreivaldsChallenge:
    if weight.ndim != 2 or weight.dtype != torch.int64 or r.dtype != torch.int64:
        raise ValueError("Freivalds inputs must be integer matrices")
    if r.shape != (weight.shape[0],):
        raise ValueError("challenge vector must match matrix output dimension")
    import xurcroh_quic

    values = xurcroh_quic.fortress_challenge_values(
        weight.shape[0],
        weight.shape[1],
        weight.detach().cpu().contiguous().numpy().tobytes(),
        r.detach().cpu().tolist(),
        prime,
    )
    return ExactFreivaldsChallenge(r=r, v=torch.tensor(values, dtype=torch.int64), prime=prime)


def verify_freivalds(
    activation: torch.Tensor, product: torch.Tensor, challenge: ExactFreivaldsChallenge
) -> bool:
    if activation.shape[-1] != challenge.v.shape[0] or product.shape[-1] != challenge.r.shape[0]:
        raise ValueError("input/output dimensions do not match challenge")
    left = _modular_matmul(activation, challenge.v, challenge.prime)
    right = _modular_matmul(product, challenge.r, challenge.prime)
    return bool(torch.equal(left, right))
