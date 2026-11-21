#!/usr/bin/env python3
"""Compare an official GGUF export against its source Transformers checkpoint.

Three independent checks, each of which must pass:

1. configuration: the Transformers configuration derived from GGUF metadata
   equals the source ``config.json`` on every graph-defining field;
2. tensors: every source tensor is exposed under its canonical name with the
   same shape and (within the encoding tolerance) the same values;
3. logits (``--logits``): the source graph loaded with the GGUF-decoded tensors
   produces the same logits as the source checkpoint.

This is trusted-client tooling. It never contacts a worker and prints only
aggregate numerical differences.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

# Fields that do not change the text graph or are not recorded in GGUF metadata.
_IGNORED_CONFIG = {
    "_name_or_path",
    "architectures",
    "dtype",
    "torch_dtype",
    "transformers_version",
    "use_cache",
    "initializer_range",
    "bos_token_id",
    "eos_token_id",
    "pad_token_id",
    "attention_dropout",
    "dropout",
    "output_router_logits",
    "router_aux_loss_coef",
    "id2label",
    "label2id",
    "problem_type",
    "is_encoder_decoder",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "chunk_size_feed_forward",
    "max_window_layers",
    "use_sliding_window",
    "model_type",
    "is_llama_config",
    "rope_interleaved",
    "transformers.js_config",
    "unsloth_fixed",
}


def _source_names(source: Path) -> dict[str, Path]:
    from safetensors import safe_open

    names: dict[str, Path] = {}
    for shard in sorted(source.glob("*.safetensors")):
        with safe_open(shard, framework="pt") as handle:
            for name in handle.keys():  # noqa: SIM118
                names[name] = shard
    if not names:
        raise SystemExit(f"no safetensors shards in {source}")
    return names


def _canonical(name: str) -> str | None:
    if name.startswith("model.language_model."):
        return "model." + name.removeprefix("model.language_model.")
    if name.startswith(("model.vision", "model.audio", "model.embed_vision", "model.embed_audio")):
        return None
    if name.startswith(("model.multi_modal_projector", "vision_", "audio_")):
        return None
    return name


def _config_differences(derived: dict[str, Any], source: Path) -> list[str]:
    from transformers import AutoConfig

    document = json.loads((source / "config.json").read_text())
    text = dict(document.get("text_config", document))
    model_type = str(derived["model_type"])
    text.pop("model_type", None)
    expected = AutoConfig.for_model(model_type, **text).to_dict()
    actual = AutoConfig.for_model(
        model_type, **{key: value for key, value in derived.items() if key != "model_type"}
    ).to_dict()
    differences = []
    for key in sorted(set(expected) | set(actual)):
        if key in _IGNORED_CONFIG:
            continue
        left, right = expected.get(key), actual.get(key)
        if isinstance(left, float) and isinstance(right, float):
            if abs(left - right) <= 1e-6 * max(1.0, abs(left)):
                continue
        elif left == right:
            continue
        differences.append(f"{key}: source={left!r} gguf={right!r}")
    return differences


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("source", type=Path, help="directory with config.json and safetensors")
    parser.add_argument("--tolerance", type=float, default=0.0, help="max absolute tensor error")
    parser.add_argument("--allow-missing-source", action="store_true")
    parser.add_argument("--logits", action="store_true")
    parser.add_argument("--logit-tolerance", type=float, default=1e-3)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    from safetensors import safe_open

    from xurcroh.architectures import open_architecture

    report: dict[str, Any] = {"gguf": arguments.gguf.name, "source": arguments.source.name}
    failures: list[str] = []
    with open_architecture(arguments.gguf) as adapter:
        reader = adapter.reader
        report["architecture"] = adapter.name
        report["model_type"] = reader.config["model_type"]
        report["encodings"] = sorted(reader.encodings)
        config_differences = _config_differences(reader.config, arguments.source)
        report["config_differences"] = config_differences
        failures.extend(f"config {item}" for item in config_differences)

        source_names = _source_names(arguments.source)
        expected = {
            canonical: name for name in source_names if (canonical := _canonical(name)) is not None
        }
        exposed = set(reader.tensor_names())
        tied = bool(reader.config.get("tie_word_embeddings"))
        missing = sorted(
            name for name in expected if name not in exposed and not (tied and "lm_head" in name)
        )
        extra = sorted(exposed - set(expected))
        report["missing_in_gguf"] = missing
        report["not_in_source"] = extra
        failures.extend(f"missing tensor {name}" for name in missing)
        if not arguments.allow_missing_source:
            failures.extend(f"tensor absent from source {name}" for name in extra)

        worst = 0.0
        worst_name = ""
        compared = 0
        shapes = reader.tensor_shapes(sorted(exposed & set(expected)))
        for name in sorted(exposed & set(expected)):
            with safe_open(source_names[expected[name]], framework="pt") as handle:
                reference = handle.get_tensor(expected[name])
            if tuple(reference.shape) != tuple(shapes[name]):
                failures.append(
                    f"shape {name}: source={tuple(reference.shape)} gguf={shapes[name]}"
                )
                continue
            actual = reader.read_tensors([name])[name]
            error = _max_abs_error(actual, reference)
            compared += 1
            if error > worst:
                worst, worst_name = error, name
            if error > arguments.tolerance:
                failures.append(f"values {name}: max_abs_error={error:.3e}")
            del actual, reference
        report["tensors_compared"] = compared
        report["max_abs_error"] = worst
        report["max_abs_error_tensor"] = worst_name

        if arguments.logits:
            report["logits"] = _logit_difference(reader, arguments.source)
            if report["logits"]["max_abs_error"] > arguments.logit_tolerance:
                failures.append(f"logits max_abs_error={report['logits']['max_abs_error']:.3e}")

    report["failures"] = failures[:50]
    report["failure_count"] = len(failures)
    report["passed"] = not failures
    text = json.dumps(report, indent=2)
    if arguments.output is not None:
        arguments.output.write_text(text + "\n")
    print(text)
    return 0 if not failures else 1


def _max_abs_error(actual: torch.Tensor, reference: torch.Tensor, rows: int = 8192) -> float:
    # Compared in row blocks so large embedding tables never need a 64-bit copy.
    if reference.numel() == 0:
        return 0.0
    if reference.ndim == 0:
        return float((actual.double() - reference.double()).abs())
    worst = 0.0
    for start in range(0, reference.shape[0], rows):
        block = actual[start : start + rows].double() - reference[start : start + rows].double()
        worst = max(worst, float(block.abs().max()))
    return worst


def _logit_difference(reader: Any, source: Path) -> dict[str, Any]:
    from transformers import AutoConfig, AutoModelForCausalLM

    config = dict(reader.config)
    model_type = str(config.pop("model_type"))
    model = AutoModelForCausalLM.from_config(AutoConfig.for_model(model_type, **config))
    model = model.to(torch.float32).eval()
    names = list(reader.tensor_names())
    state = {name: tensor.to(torch.float32) for name, tensor in reader.read_tensors(names).items()}
    if config.get("tie_word_embeddings") and "lm_head.weight" not in state:
        state["lm_head.weight"] = state["model.embed_tokens.weight"]
    result = model.load_state_dict(state, strict=False)
    reference = AutoModelForCausalLM.from_pretrained(
        source, dtype=torch.float32, local_files_only=True, trust_remote_code=False
    ).eval()
    generator = torch.Generator().manual_seed(20261121)
    vocabulary = int(model.get_input_embeddings().weight.shape[0])
    tokens = torch.randint(0, min(vocabulary, 50_000), (2, 48), generator=generator)
    with torch.inference_mode():
        actual = model(input_ids=tokens).logits.double()
        expected = reference(input_ids=tokens).logits.double()
    return {
        "missing_keys": sorted(result.missing_keys)[:20],
        "unexpected_keys": sorted(result.unexpected_keys)[:20],
        "max_abs_error": float((actual - expected).abs().max()),
        "argmax_agreement": float((actual.argmax(-1) == expected.argmax(-1)).double().mean()),
        "reference_logit_scale": float(expected.abs().max()),
    }


if __name__ == "__main__":
    sys.exit(main())
