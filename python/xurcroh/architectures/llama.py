from __future__ import annotations

import torch

from xurcroh.architectures.base import ArchitectureAdapter, CanonicalReader
from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.formats import TensorReader

_SIMPLE = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
}
_LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
}
_EXPERTS = {
    "ffn_gate_exps.weight": "gate_proj",
    "ffn_up_exps.weight": "up_proj",
    "ffn_down_exps.weight": "down_proj",
}


class GGUFTransformerReader(CanonicalReader):
    def __init__(self, source: TensorReader) -> None:
        super().__init__(source)
        self._canonical: dict[str, str] = {}
        self._fused: dict[str, str] = {}
        self._experts: dict[str, tuple[str, str]] = {}
        for raw in source.tensor_names():
            if raw in _SIMPLE:
                self._canonical[raw] = _SIMPLE[raw]
            elif raw.startswith("blk.") and raw.endswith(".attn_qkv.weight"):
                self._fused[raw] = raw
            elif raw.startswith("blk.") and raw.split(".", 2)[-1] in _EXPERTS:
                self._experts[raw] = (
                    f"model.layers.{raw.split('.')[1]}.mlp",
                    raw.split(".", 2)[-1],
                )
            elif raw.startswith("blk."):
                parts = raw.split(".")
                suffix = ".".join(parts[2:])
                if len(parts) == 4 and suffix in _LAYER:
                    prefix = f"model.layers.{parts[1]}"
                    self._canonical[raw] = f"{prefix}.{_LAYER[suffix]}"
        if (
            "lm_head.weight" not in self._canonical.values()
            and "model.embed_tokens.weight" in self._canonical.values()
        ):
            self._canonical["output.weight"] = "lm_head.weight"
        self._names = (
            tuple(self._canonical.values())
            + tuple(
                f"model.layers.{layer}.self_attn.{projection}_proj.weight"
                for fused in self._fused
                for layer in [fused.split(".")[1]]
                for projection in ("q", "k", "v")
            )
            + tuple(
                f"{prefix}.experts.{expert}.{projection}.weight"
                for prefix, encoding in self._experts.values()
                for expert in range(int(source.config.get("num_experts", 0)))
                for projection in [_EXPERTS[encoding]]
            )
        )

    def tensor_names(self) -> tuple[str, ...]:
        self._ensure_open()
        return self._names

    def tensor_shapes(self, names: list[str]) -> dict[str, tuple[int, ...]]:
        self._ensure_open()
        shapes: dict[str, tuple[int, ...]] = {}
        for name in names:
            raw = next(
                (physical for physical, logical in self._canonical.items() if logical == name),
                None,
            )
            if raw is not None:
                shapes[name] = self.source.tensor_shapes([raw])[raw]
                continue
            fused = next(iter(self._fused), None)
            if fused is not None and ".self_attn." in name:
                config = self.config
                head = int(
                    config.get("head_dim", config["hidden_size"] // config["num_attention_heads"])
                )
                query = int(config["num_attention_heads"]) * head
                key_value = (
                    int(config.get("num_key_value_heads", config["num_attention_heads"])) * head
                )
                width = self.source.tensor_shapes([fused])[fused][1]
                shapes[name] = (
                    query if name.endswith(".q_proj.weight") else key_value,
                    width,
                )
                continue
            for raw_expert, (prefix, encoding) in self._experts.items():
                marker = f"{prefix}.experts."
                if name.startswith(marker) and name.endswith(f".{_EXPERTS[encoding]}.weight"):
                    stacked = self.source.tensor_shapes([raw_expert])[raw_expert]
                    if len(stacked) == 3:
                        shapes[name] = stacked[1:]
                    else:
                        shapes[name] = (stacked[0] // int(self.config["num_experts"]), stacked[1])
                    break
            if name not in shapes:
                raise ModelLoadError(f"unsupported GGUF logical tensor: {name}")
        return shapes

    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        output: dict[str, torch.Tensor] = {}
        for name in names:
            raw = next(
                (physical for physical, logical in self._canonical.items() if logical == name), None
            )
            if raw is not None:
                output[name] = self.source.read_tensors([raw])[raw]
                continue
            fused = next(iter(self._fused), None)
            if fused is not None and ".self_attn." in name:
                layer = fused.split(".")[1]
                for projection in ("q", "k", "v"):
                    if name == f"model.layers.{layer}.self_attn.{projection}_proj.weight":
                        output[name] = self._split_fused(fused, projection)
                        break
            if name in output:
                continue
            for raw_expert, (prefix, encoding) in self._experts.items():
                marker = f"{prefix}.experts."
                if name.startswith(marker) and name.endswith(f".{_EXPERTS[encoding]}.weight"):
                    expert = int(name[len(marker) :].split(".", 1)[0])
                    stacked = self.source.read_tensors([raw_expert])[raw_expert]
                    if stacked.ndim == 3:
                        output[name] = stacked[expert]
                    elif stacked.ndim == 2:
                        count = int(self.config["num_experts"])
                        rows = stacked.shape[0] // count
                        output[name] = stacked[expert * rows : (expert + 1) * rows]
                    else:
                        raise ModelLoadError(
                            f"GGUF expert tensor is not a matrix stack: {raw_expert}"
                        )
                    break
            if name not in output:
                raise ModelLoadError(f"unsupported GGUF logical tensor: {name}")
        return output

    def read_columns(self, name: str, columns: slice) -> torch.Tensor:
        raw = next(
            (physical for physical, logical in self._canonical.items() if logical == name), None
        )
        if raw is not None:
            return self.source.read_operation(raw, columns)
        return super().read_columns(name, columns)

    def _split_fused(self, raw: str, projection: str) -> torch.Tensor:
        fused = self.source.read_tensors([raw])[raw]
        config = self.config
        head = int(config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]))
        q = int(config["num_attention_heads"]) * head
        kv = int(config.get("num_key_value_heads", config["num_attention_heads"])) * head
        if fused.shape[0] != q + 2 * kv:
            raise ModelLoadError("GGUF fused QKV dimensions do not match the model config")
        start = {"q": 0, "k": q, "v": q + kv}[projection]
        stop = start + (q if projection == "q" else kv)
        return fused[start:stop]


class LlamaArchitectureAdapter(ArchitectureAdapter):
    def __init__(self, reader: TensorReader) -> None:
        super().__init__(GGUFTransformerReader(reader), reader.architecture, ("text", "vlm"))


def open_gguf_architecture(reader: TensorReader) -> ArchitectureAdapter:
    architecture = reader.architecture
    supported = {
        "llama",
        "qwen2",
        "qwen3",
        "qwen3moe",
        "gemma3",
        "gemma4",
        "deepseek2",
    }
    if architecture not in supported:
        raise CapabilityError(
            f"GGUF architecture {architecture!r} has no Fortress semantic adapter"
        )
    return LlamaArchitectureAdapter(reader)
