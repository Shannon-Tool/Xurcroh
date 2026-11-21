from __future__ import annotations

import torch

from xurcroh.architectures.base import ArchitectureAdapter, CanonicalReader
from xurcroh.errors import ModelLoadError
from xurcroh.formats import TensorReader

_COMPANIONS = {
    "q_weight",
    "q_scale",
    "q_scale_max",
    "q_groups",
    "q_invperm",
    "qweight",
    "qzeros",
    "scales",
    "g_idx",
    "trellis",
    "su",
    "sv",
    "suh",
    "svh",
    "mcg",
    "mul1",
}


class QuantizedSafetensorsReader(CanonicalReader):
    def __init__(self, source: TensorReader) -> None:
        super().__init__(source)
        self._logical: dict[str, str] = {}
        # Checkpoints that store experts as one stacked tensor expose each
        # expert projection under the same per-expert name as unstacked ones.
        self._stacked: dict[str, tuple[str, int, str]] = {}
        names: list[str] = []
        for raw in source.tensor_names():
            if raw.endswith((".experts.gate_up_proj", ".experts.down_proj")):
                shape = source.tensor_shapes([raw])[raw]
                if len(shape) != 3:
                    raise ModelLoadError(f"stacked expert tensor is not three-dimensional: {raw}")
                prefix, kind = raw.rsplit(".", 1)
                projections = ("gate_proj", "up_proj") if kind == "gate_up_proj" else ("down_proj",)
                if kind == "gate_up_proj" and shape[1] % 2 != 0:
                    raise ModelLoadError(f"stacked gate/up expert tensor has an odd width: {raw}")
                for expert in range(shape[0]):
                    for projection in projections:
                        logical = f"{prefix}.{expert}.{projection}.weight"
                        self._stacked[logical] = (raw, expert, projection)
                        names.append(logical)
                continue
            parts = raw.rsplit(".", 1)
            if len(parts) == 2 and parts[0] and parts[1] in {"q_weight", "qweight", "trellis"}:
                logical = parts[0] + ".weight"
                self._logical[logical] = raw
                names.append(logical)
            elif len(parts) == 2 and parts[1] not in _COMPANIONS:
                names.append(raw)
        self._names = tuple(names)

    def tensor_names(self) -> tuple[str, ...]:
        self._ensure_open()
        return self._names

    def _output_size(self, name: str) -> int | None:
        config = {**self.config, **self.config.get("text_config", {})}
        if name.endswith("lm_head.weight"):
            return int(config["vocab_size"]) if "vocab_size" in config else None
        if name.endswith((".o_proj.weight", ".down_proj.weight")):
            return int(config["hidden_size"])
        if name.endswith(".q_proj.weight"):
            return int(config["num_attention_heads"]) * int(
                config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
            )
        if name.endswith((".k_proj.weight", ".v_proj.weight")):
            return int(config.get("num_key_value_heads", config["num_attention_heads"])) * int(
                config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
            )
        if name.endswith((".gate_proj.weight", ".up_proj.weight")):
            return int(config["intermediate_size"])
        return None

    def tensor_shapes(self, names: list[str]) -> dict[str, tuple[int, ...]]:
        self._ensure_open()
        shapes: dict[str, tuple[int, ...]] = {}
        for name in names:
            if name in self._stacked:
                stacked_raw, _, projection = self._stacked[name]
                _, rows, columns = self.source.tensor_shapes([stacked_raw])[stacked_raw]
                shapes[name] = (rows if projection == "down_proj" else rows // 2, columns)
                continue
            raw = self._logical.get(name)
            if raw is None:
                shapes[name] = self.source.tensor_shapes([name])[name]
                continue
            expected = self._output_size(name)
            if raw.endswith(".trellis"):
                trellis = self.source.tensor_shapes([raw])[raw]
                shapes[name] = (expected or trellis[1] * 16, trellis[0] * 16)
                continue
            raw_shape = self.source.tensor_shapes([raw])[raw]
            if raw.endswith(".qweight"):
                shapes[name] = (expected or raw_shape[1], raw_shape[0] * 8)
                continue
            inverse = raw.removesuffix(".q_weight") + ".q_invperm"
            if inverse in self.source.tensor_names():
                input_size = self.source.tensor_shapes([inverse])[inverse][0]
            else:
                group_name = raw.removesuffix(".q_weight") + ".q_groups"
                groups_flat = self.source.read_tensors([group_name])[group_name].flatten()
                groups = groups_flat.reshape(-1, 2).to(torch.int64)
                input_size = 0
                for index, (bits, packed_row) in enumerate(groups):
                    if index + 1 < len(groups):
                        input_size += int(groups[index + 1, 1] - packed_row) * 32 // int(bits)
                    else:
                        input_size += (raw_shape[0] - int(packed_row)) * 32 // int(bits)
            shapes[name] = (expected or raw_shape[1], input_size)
        return shapes

    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        output = {}
        cached_name: str | None = None
        cached: torch.Tensor | None = None
        for name in names:
            if name in self._stacked:
                stacked_raw, expert, projection = self._stacked[name]
                if cached_name != stacked_raw or cached is None:
                    cached_name = stacked_raw
                    cached = self.source.read_tensors([stacked_raw])[stacked_raw]
                matrix = cached[expert]
                if projection != "down_proj":
                    half = matrix.shape[0] // 2
                    matrix = matrix[:half] if projection == "gate_proj" else matrix[half:]
                output[name] = matrix.clone()
                continue
            raw = self._logical.get(name, name)
            expected = self._output_size(name) if name in self._logical else None
            if name in self._logical:
                output[name] = self.source.read_operation(
                    raw, columns=None if expected is None else slice(0, expected)
                )
            else:
                output[name] = self.source.read_tensors([raw])[raw]
        return output

    def read_columns(self, name: str, columns: slice) -> torch.Tensor:
        if name in self._stacked:
            return self.read_tensors([name])[name][columns, :]
        return self.source.read_operation(self._logical.get(name, name), columns)


class QuantizedArchitectureAdapter(ArchitectureAdapter):
    def __init__(self, reader: TensorReader) -> None:
        workflows = ("text", "vlm") if "text_config" in reader.config else ("text",)
        super().__init__(QuantizedSafetensorsReader(reader), reader.architecture, workflows)
