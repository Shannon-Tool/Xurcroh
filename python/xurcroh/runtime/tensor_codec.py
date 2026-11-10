from __future__ import annotations

import torch

from xurcroh.adapters.pipeline import PipelineTensorInterceptor
from xurcroh.errors import ProtocolError

DTYPES: dict[str, tuple[torch.dtype, int]] = {
    "f64": (torch.float64, 8),
    "f32": (torch.float32, 4),
    "f16": (torch.float16, 2),
    "bf16": (torch.bfloat16, 2),
}
DTYPE_NAMES = {dtype: name for name, (dtype, _) in DTYPES.items()}


def dtype_name(dtype: torch.dtype) -> str:
    try:
        return DTYPE_NAMES[dtype]
    except KeyError as error:
        raise ProtocolError(f"unsupported stage tensor dtype: {dtype}") from error


def encode_stage_tensor(
    hidden: torch.Tensor,
    request_id: str,
    sender: PipelineTensorInterceptor,
) -> tuple[bytes, str, list[int]]:
    contiguous = hidden.detach().cpu().contiguous()
    if contiguous.ndim != 2:
        raise ProtocolError("stage tensor must be a two-dimensional matrix")
    name = dtype_name(contiguous.dtype)
    shape = list(contiguous.shape)
    payload = contiguous.view(torch.uint8).numpy().tobytes()
    return sender.seal(request_id, name, shape, payload), name, shape


def decode_stage_tensor(
    packet: bytes,
    receiver: PipelineTensorInterceptor,
    *,
    expected_dtype: str | None = None,
    expected_shape: list[int] | None = None,
) -> torch.Tensor:
    name, shape, payload = receiver.open_tensor(packet)
    if name not in DTYPES or len(shape) != 2 or any(dimension < 1 for dimension in shape):
        raise ProtocolError("stage tensor descriptor is invalid")
    if expected_dtype is not None and name != expected_dtype:
        raise ProtocolError("stage tensor dtype does not match the request descriptor")
    if expected_shape is not None and shape != expected_shape:
        raise ProtocolError("stage tensor shape does not match the request descriptor")
    if shape[0] * shape[1] > 16_777_216:
        raise ProtocolError("stage tensor exceeds the element limit")
    dtype, item_size = DTYPES[name]
    buffer = bytearray(payload)
    try:
        if len(buffer) != shape[0] * shape[1] * item_size:
            raise ProtocolError("stage tensor byte count does not match its descriptor")
        return torch.frombuffer(buffer, dtype=dtype).reshape(shape).clone()
    finally:
        buffer[:] = bytes(len(buffer))
