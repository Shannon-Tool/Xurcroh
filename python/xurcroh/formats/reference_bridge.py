from __future__ import annotations

import hashlib
from importlib import metadata
from pathlib import Path
from typing import cast

import torch

from xurcroh.errors import CapabilityError


class ReferenceDecoderBridge:
    def __init__(self, package: str) -> None:
        if package not in {"exllamav2", "exllamav3"}:
            raise ValueError(f"unsupported reference decoder package: {package}")
        self.package = package

    def identity(self) -> dict[str, str]:
        try:
            version = metadata.version(self.package)
            distribution = metadata.distribution(self.package)
        except metadata.PackageNotFoundError as error:
            raise CapabilityError(
                f"pinned reference decoder {self.package} is not installed"
            ) from error
        digest = hashlib.sha256()
        decoder_names = (
            ("linear.py", "ext.py") if self.package == "exllamav2" else ("exl3.py", "quantize.py")
        )
        files = sorted(
            path
            for path in distribution.files or []
            if str(path).startswith(self.package) and path.name in decoder_names
        )
        if not files:
            raise CapabilityError(f"cannot fingerprint {self.package} decoder files")
        for file in files:
            digest.update(str(file).encode() + b"\0")
            digest.update(Path(file.locate()).read_bytes())
        return {"package": self.package, "version": version, "revision": digest.hexdigest()}


def reconstruct_exl2_official(
    tensors: dict[str, torch.Tensor], prefix: str, shape: tuple[int, int]
) -> torch.Tensor:
    from exllamav2 import ext

    extension = ext.exllamav2_ext  # type: ignore[attr-defined]

    if not torch.cuda.is_available():
        raise CapabilityError("official EXL2 differential validation requires CUDA")
    is_exl2 = f"{prefix}.q_weight" in tensors
    names = (
        ("q_weight", "q_invperm", "q_scale", "q_scale_max", "q_groups", "q_group_map")
        if is_exl2
        else ("qweight", "qzeros", "scales", "g_idx")
    )
    quantized = {
        name: tensors[f"{prefix}.{name}"].cuda() for name in names if f"{prefix}.{name}" in tensors
    }
    if is_exl2 and "q_invperm" in quantized:
        quantized["q_perm"] = torch.argsort(quantized.pop("q_invperm")).to(torch.int16)
    if is_exl2 and "q_group_map" not in quantized:
        quantized["q_group_map"] = ext.make_group_map(
            quantized["q_groups"], quantized["q_weight"].shape[0]
        )
    output_size, input_size = shape
    temporary = torch.empty(input_size * output_size, dtype=torch.float16, device="cuda")
    handle = ext.make_q_matrix(quantized, temporary)
    try:
        official = torch.empty((input_size, output_size), dtype=torch.float16, device="cuda")
        extension.reconstruct(handle, official)
        return official.T.contiguous().cpu()
    finally:
        extension.free_q_matrix(handle)
        temporary.zero_()
        for tensor in quantized.values():
            tensor.zero_()


def official_exl2_error(
    tensors: dict[str, torch.Tensor], prefix: str, expected: torch.Tensor
) -> float:
    official = reconstruct_exl2_official(tensors, prefix, (expected.shape[0], expected.shape[1]))
    return float((official - expected).abs().max())


def reconstruct_exl3_official(
    tensors: dict[str, torch.Tensor], prefix: str, shape: tuple[int, int]
) -> torch.Tensor:
    from exllamav3.modules.quant.exl3 import LinearEXL3

    if not torch.cuda.is_available():
        raise CapabilityError("official EXL3 differential validation requires CUDA")
    output_size, input_size = shape
    arguments = {
        name: tensors[f"{prefix}.{name}"].cuda()
        for name in ("su", "sv", "suh", "svh", "trellis", "mcg", "mul1", "bias")
        if f"{prefix}.{name}" in tensors
    }
    layer = LinearEXL3(
        config=None,
        in_features=input_size,
        out_features=output_size,
        su=arguments.get("su"),
        sv=arguments.get("sv"),
        suh=arguments.get("suh"),
        svh=arguments.get("svh"),
        trellis=arguments["trellis"],
        mcg=arguments.get("mcg"),
        mul1=arguments.get("mul1"),
        bias=arguments.get("bias"),
    )
    official = layer.get_weight_tensor().T.contiguous()  # type: ignore[no-untyped-call]
    layer.unload()  # type: ignore[no-untyped-call]
    for tensor in arguments.values():
        tensor.zero_()
    return cast(torch.Tensor, official.cpu())


def official_exl3_error(
    tensors: dict[str, torch.Tensor], prefix: str, expected: torch.Tensor
) -> float:
    official = reconstruct_exl3_official(tensors, prefix, (expected.shape[0], expected.shape[1]))
    return float((official - expected).abs().max())
