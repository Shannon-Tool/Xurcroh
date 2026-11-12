from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum

import torch
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms

RING_MODULUS = 2**32
PRIME_MODULUS = 2**32 - 5
MAX_SIGNED_RING_VALUE = 2**31 - 1


@dataclass(frozen=True)
class QuantizedTensor:
    values: torch.Tensor
    scale: float
    zero_point: int

    def __post_init__(self) -> None:
        if self.values.dtype != torch.int64:
            raise ValueError("quantized values must use int64")
        if self.values.ndim != 2:
            raise ValueError("quantized tensors must be matrices")
        if self.scale <= 0:
            raise ValueError("quantization scale must be positive")

    def dequantize(self) -> torch.Tensor:
        return (self.values - self.zero_point).to(torch.float64) * self.scale


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


def quantized_matmul(left: QuantizedTensor, right: QuantizedTensor) -> QuantizedProduct:
    if left.values.shape[1] != right.values.shape[0]:
        raise ValueError("matrix dimensions do not match")
    left_values = left.values - left.zero_point
    right_values = right.values - right.zero_point
    inner = left_values.shape[1]
    left_bound = int(left_values.abs().max().item()) if left_values.numel() else 0
    right_bound = int(right_values.abs().max().item()) if right_values.numel() else 0
    if left_bound * right_bound * inner > MAX_SIGNED_RING_VALUE:
        raise OverflowError("unreduced product exceeds the signed 2^32 ring bound")
    return QuantizedProduct(left_values @ right_values, left.scale * right.scale)


def _secure_uniform(shape: tuple[int, ...], modulus: int) -> torch.Tensor:
    if modulus <= 1 or modulus > 2**63:
        raise ValueError("unsupported share modulus")
    count = 1
    for dimension in shape:
        count *= dimension
    limit = (2**64 // modulus) * modulus
    accepted: list[torch.Tensor] = []
    remaining = count
    while remaining:
        candidate = torch.frombuffer(bytearray(os.urandom(remaining * 8)), dtype=torch.uint64)
        accepted.append(candidate if limit == 2**64 else candidate[candidate < limit])
        remaining = count - sum(tensor.numel() for tensor in accepted)
    return torch.cat(accepted).to(torch.int64).reshape(shape)


def signed_ring(values: torch.Tensor) -> torch.Tensor:
    return (values + 2**31).remainder(RING_MODULUS) - 2**31


def split_secret(secret: torch.Tensor, *, count: int, modulus: int = RING_MODULUS) -> ShareGrid:
    if count < 2:
        raise ValueError("additive sharing requires at least two roles")
    if secret.dtype != torch.int64:
        raise ValueError("shared secrets must use int64")
    canonical = secret.remainder(modulus)
    shares = [_secure_uniform(secret.shape, modulus) for _ in range(count - 1)]
    shares.append((canonical - torch.sum(torch.stack(shares), dim=0)).remainder(modulus))
    return ShareGrid(shares, ShareSecurity.INFORMATION_THEORETIC)


def combine_shares(shares: list[torch.Tensor], *, modulus: int = RING_MODULUS) -> torch.Tensor:
    if not shares:
        raise ValueError("at least one share is required")
    return torch.sum(torch.stack(shares), dim=0).remainder(modulus)


def refresh_shares(
    shares: list[torch.Tensor], *, first: int, second: int, modulus: int = RING_MODULUS
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
        .remainder(RING_MODULUS)
        .reshape(shape)
    )


def seeded_split_secret(secret: torch.Tensor, *, seed: bytes, count: int) -> ShareGrid:
    if count < 2:
        raise ValueError("additive sharing requires at least two roles")
    if secret.dtype != torch.int64:
        raise ValueError("shared secrets must use int64")
    canonical = secret.remainder(RING_MODULUS)
    shares = [_seeded_uniform(seed, secret.shape) for _ in range(count - 1)]
    shares.append((canonical - torch.sum(torch.stack(shares), dim=0)).remainder(RING_MODULUS))
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
    return ExactFreivaldsChallenge(r=r, v=_modular_matvec(weight.T, r, prime), prime=prime)


def verify_freivalds(
    activation: torch.Tensor, product: torch.Tensor, challenge: ExactFreivaldsChallenge
) -> bool:
    if activation.shape[-1] != challenge.v.shape[0] or product.shape[-1] != challenge.r.shape[0]:
        raise ValueError("input/output dimensions do not match challenge")
    left = _modular_matmul(activation, challenge.v, challenge.prime)
    right = _modular_matmul(product, challenge.r, challenge.prime)
    return bool(torch.equal(left, right))
