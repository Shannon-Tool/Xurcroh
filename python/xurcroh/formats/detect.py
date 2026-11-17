from __future__ import annotations

from pathlib import Path

from safetensors import safe_open

from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.formats.base import CheckpointFormat, TensorReader


def detect_checkpoint_format(path: str | Path) -> CheckpointFormat:
    checkpoint = Path(path).expanduser()
    if checkpoint.is_file():
        try:
            with checkpoint.open("rb") as stream:
                magic = stream.read(4)
        except OSError as error:
            raise ModelLoadError(f"cannot inspect checkpoint: {checkpoint}") from error
        if magic == b"GGUF":
            return "gguf"
        raise ModelLoadError(f"unrecognized checkpoint container: {checkpoint}")
    if not checkpoint.is_dir():
        raise ModelLoadError(f"checkpoint path does not exist: {checkpoint}")
    files = sorted(checkpoint.glob("*.safetensors"))
    if not files:
        raise ModelLoadError(f"checkpoint contains no supported weight files: {checkpoint}")
    for file in files:
        try:
            with safe_open(file, framework="pt") as handle:
                names = tuple(handle.keys())
        except Exception as error:
            raise ModelLoadError(f"cannot inspect safetensors metadata: {file}") from error
        if any(name.endswith(".trellis") for name in names):
            return "exl3"
        if any(name.endswith((".q_weight", ".qweight")) for name in names):
            return "exl2"
    return "safetensors"


def open_checkpoint(path: str | Path, source_format: str = "auto") -> TensorReader:
    selected = detect_checkpoint_format(path) if source_format == "auto" else source_format
    if selected == "safetensors":
        from xurcroh.formats.safetensors import SafetensorsReader

        return SafetensorsReader(Path(path).expanduser())
    if selected == "gguf":
        from xurcroh.formats.gguf import GGUFReader

        return GGUFReader(Path(path).expanduser())
    if selected in {"exl2", "exl3"}:
        raise CapabilityError(f"{selected.upper()} reader support is not installed yet")
    raise ModelLoadError(f"unknown checkpoint format: {selected}")
