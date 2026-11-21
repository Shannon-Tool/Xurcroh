from __future__ import annotations

import re
from typing import Any

import torch

from xurcroh.architectures.base import ArchitectureAdapter, CanonicalReader
from xurcroh.architectures.gguf_families import (
    EXPERT_STACKS,
    FUSED_QKV,
    MEDIA_PREFIXES,
    UNSUPPORTED_MARKERS,
    GGUFFamily,
    gguf_family,
)
from xurcroh.errors import CapabilityError, ModelLoadError
from xurcroh.formats import TensorReader

_BLOCK = re.compile(r"blk\.(\d+)\.(.+)")
_EXPERT = re.compile(r"(model\.layers\.\d+\.mlp)\.experts\.(\d+)\.(\w+)\.weight")
_FUSED = re.compile(r"model\.layers\.(\d+)\.self_attn\.([qkv])_proj\.weight")
_QUERY_KEY = re.compile(r"model\.layers\.\d+\.self_attn\.([qk])_proj\.(?:weight|bias)")


def _unsupported_reason(raw: str) -> str:
    if raw.endswith(".bias"):
        return "bias tensors are not part of this architecture's verified layout"
    return next(
        (text for marker, text in UNSUPPORTED_MARKERS if marker in raw),
        "unrecognized GGUF tensor",
    )


class GGUFTransformerReader(CanonicalReader):
    """Maps a verified llama.cpp GGUF family onto canonical Transformers names.

    Tensor names, stored-value conventions and configuration all come from the
    family selected by ``general.architecture``; see
    :mod:`xurcroh.architectures.gguf_families`. A tensor that the family does not
    define is reported instead of being dropped, because a silently missing
    weight changes the model.
    """

    def __init__(self, source: TensorReader) -> None:
        super().__init__(source)
        self.family: GGUFFamily = gguf_family(source.architecture)
        family = self.family
        self._canonical: dict[str, str] = {}
        self._fused: dict[int, str] = {}
        self._experts: dict[tuple[str, str], str] = {}
        problems: list[str] = []
        raw_names = frozenset(source.tensor_names())
        for raw in source.tensor_names():
            if raw in family.simple:
                self._canonical[family.simple[raw]] = raw
                continue
            if raw in family.derived or raw.startswith(MEDIA_PREFIXES):
                continue
            block = _BLOCK.fullmatch(raw)
            if block is None:
                problems.append(f"{raw}: {_unsupported_reason(raw)}")
                continue
            layer, suffix = int(block.group(1)), block.group(2)
            prefix = f"model.layers.{layer}"
            if suffix in family.layer:
                self._canonical[f"{prefix}.{family.layer[suffix]}"] = raw
            elif suffix == FUSED_QKV and family.fused_qkv:
                self._fused[layer] = raw
            elif suffix in EXPERT_STACKS and family.experts:
                self._experts[(f"{prefix}.mlp", EXPERT_STACKS[suffix])] = raw
            else:
                problems.append(f"{raw}: {_unsupported_reason(raw)}")
        if problems:
            shown = problems[:12]
            if len(problems) > len(shown):
                shown.append(f"... and {len(problems) - len(shown)} more")
            raise CapabilityError(
                f"unsupported {source.architecture} GGUF layout:\n  " + "\n  ".join(shown)
            )
        if "model.embed_tokens.weight" not in self._canonical:
            raise CapabilityError("GGUF checkpoint has no recognized token embedding")
        if not any(_BLOCK.fullmatch(raw) for raw in raw_names):
            raise CapabilityError("GGUF checkpoint has no transformer blocks (blk.*)")
        self._config = family.config(
            source.metadata, f"{source.architecture}.", dict(source.config), raw_names
        )
        self._config["model_type"] = family.model_type
        self._config.pop("rope_dimension_count", None)
        if "vocab_size" not in self._config:
            embedding = self._canonical["model.embed_tokens.weight"]
            self._config["vocab_size"] = source.tensor_shapes([embedding])[embedding][0]
            if "hidden_size_per_layer_input" in self._config:
                self._config["vocab_size_per_layer_input"] = self._config["vocab_size"]
        expert_count = int(self._config.get("num_experts", 0))
        if self._experts and expert_count <= 0:
            raise ModelLoadError("GGUF expert tensors require expert_count metadata")
        self._names = (
            tuple(self._canonical)
            + tuple(
                f"model.layers.{layer}.self_attn.{projection}_proj.weight"
                for layer in self._fused
                for projection in ("q", "k", "v")
            )
            + tuple(
                f"{prefix}.experts.{expert}.{projection}.weight"
                for prefix, projection in self._experts
                for expert in range(expert_count)
            )
        )

    @property
    def config(self) -> dict[str, Any]:
        self._ensure_open()
        return self._config

    def tensor_names(self) -> tuple[str, ...]:
        self._ensure_open()
        return self._names

    def _head_rows(self, projection: str) -> tuple[int, int]:
        config = self._config
        heads = int(config["num_attention_heads"])
        head = int(config.get("head_dim", config["hidden_size"] // heads))
        if projection != "q":
            heads = int(config.get("num_key_value_heads", heads))
        return heads, heads * head

    def _fused_rows(self, projection: str) -> slice:
        _, query = self._head_rows("q")
        _, key_value = self._head_rows("k")
        start = {"q": 0, "k": query, "v": query + key_value}[projection]
        return slice(start, start + (query if projection == "q" else key_value))

    def _expert_shape(self, raw: str) -> tuple[int, int]:
        stacked = self.source.tensor_shapes([raw])[raw]
        count = int(self._config["num_experts"])
        if len(stacked) == 3 and stacked[0] == count:
            return (stacked[1], stacked[2])
        if len(stacked) == 2 and stacked[0] % count == 0:
            return (stacked[0] // count, stacked[1])
        raise ModelLoadError(f"GGUF expert tensor is not a stack of {count} matrices: {raw}")

    def tensor_shapes(self, names: list[str]) -> dict[str, tuple[int, ...]]:
        self._ensure_open()
        shapes: dict[str, tuple[int, ...]] = {}
        for name in names:
            raw = self._canonical.get(name)
            if raw is not None:
                shapes[name] = self.source.tensor_shapes([raw])[raw]
                continue
            fused = _FUSED.fullmatch(name)
            if fused is not None and int(fused.group(1)) in self._fused:
                raw = self._fused[int(fused.group(1))]
                rows = self._fused_rows(fused.group(2))
                shapes[name] = (rows.stop - rows.start, self.source.tensor_shapes([raw])[raw][1])
                continue
            expert = _EXPERT.fullmatch(name)
            if expert is not None and (expert.group(1), expert.group(3)) in self._experts:
                shapes[name] = self._expert_shape(self._experts[(expert.group(1), expert.group(3))])
                continue
            raise ModelLoadError(f"unsupported GGUF logical tensor: {name}")
        return shapes

    def _restore(self, name: str, tensor: torch.Tensor) -> torch.Tensor:
        family = self.family
        if family.norm_offset and tensor.ndim == 1 and "norm" in name.rsplit(".", 2)[-2]:
            tensor = tensor - family.norm_offset
        match = _QUERY_KEY.fullmatch(name)
        if family.permute_query_key and match is not None:
            heads, rows = self._head_rows(match.group(1))
            if tensor.shape[0] != rows or rows % (2 * heads) != 0:
                raise ModelLoadError(f"GGUF attention rows do not match the model config: {name}")
            tensor = (
                tensor.reshape(heads, rows // heads // 2, 2, *tensor.shape[1:])
                .swapaxes(1, 2)
                .reshape(tensor.shape)
            )
        return tensor

    def read_tensors(self, names: list[str]) -> dict[str, torch.Tensor]:
        self._ensure_open()
        output: dict[str, torch.Tensor] = {}
        stacked_name: str | None = None
        stacked: torch.Tensor | None = None
        for name in names:
            raw = self._canonical.get(name)
            if raw is not None:
                output[name] = self._restore(name, self.source.read_tensors([raw])[raw])
                continue
            fused = _FUSED.fullmatch(name)
            if fused is not None and int(fused.group(1)) in self._fused:
                raw = self._fused[int(fused.group(1))]
                weight = self.source.read_tensors([raw])[raw]
                _, query = self._head_rows("q")
                _, key_value = self._head_rows("k")
                if weight.shape[0] != query + 2 * key_value:
                    raise ModelLoadError("GGUF fused QKV dimensions do not match the model config")
                output[name] = self._restore(name, weight[self._fused_rows(fused.group(2))])
                continue
            expert = _EXPERT.fullmatch(name)
            if expert is not None and (expert.group(1), expert.group(3)) in self._experts:
                raw = self._experts[(expert.group(1), expert.group(3))]
                rows, columns = self._expert_shape(raw)
                index = int(expert.group(2))
                if index >= int(self._config["num_experts"]):
                    raise ModelLoadError(f"unsupported GGUF logical tensor: {name}")
                if stacked_name != raw or stacked is None:
                    stacked_name, stacked = raw, self.source.read_tensors([raw])[raw]
                output[name] = stacked.reshape(-1, rows, columns)[index]
                continue
            raise ModelLoadError(f"unsupported GGUF logical tensor: {name}")
        return output

    def read_columns(self, name: str, columns: slice) -> torch.Tensor:
        raw = self._canonical.get(name)
        if raw is not None and not (
            self.family.permute_query_key and _QUERY_KEY.fullmatch(name) is not None
        ):
            return self._restore(name, self.source.read_operation(raw, columns))
        return super().read_columns(name, columns)


class LlamaArchitectureAdapter(ArchitectureAdapter):
    def __init__(self, reader: TensorReader) -> None:
        super().__init__(GGUFTransformerReader(reader), reader.architecture, ("text", "vlm"))


def open_gguf_architecture(reader: TensorReader) -> ArchitectureAdapter:
    return LlamaArchitectureAdapter(reader)
