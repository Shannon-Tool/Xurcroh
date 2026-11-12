#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey
from safetensors.torch import load_file, save_file

from xurcroh.adapters.vllm_v1_stage import create_vllm_stage, hidden_from_intermediate
from xurcroh.models.dense import (
    execute_dense_layer,
    execute_dense_layer_cached,
    layer_weights_from_dict,
)
from xurcroh.stage_worker import DenseStage


def optimized_outputs(root: Path, index: int, captured: dict[str, torch.Tensor]):
    private = X25519PrivateKey.from_private_bytes((root / "stage.identity").read_bytes())
    stage = create_vllm_stage(
        root / "worker-artifacts/manifest.json",
        index,
        private,
        checkpoint=root / "worker-checkpoint",
    )
    from vllm.sequence import IntermediateTensors

    actual = {}
    forwards = {}
    residuals = {}
    try:
        for token_index in (0, 1):
            stage_input = captured[f"input-{token_index}"].cuda()
            if index > 0:
                stage_input = IntermediateTensors(
                    {
                        "hidden_states": stage_input,
                        "residual": captured[f"residual-{token_index}"].cuda(),
                    }
                )
            result = stage.execute(stage_input, "fp32-stage-validation")
            actual[f"output-{token_index}"] = hidden_from_intermediate(result).detach().cpu()
            forwards[f"forward-{token_index}"] = result.tensors["hidden_states"].detach().cpu()
            residual = result.tensors.get("residual")
            if residual is not None:
                residuals[f"residual-{token_index}"] = residual.detach().cpu()
        full_history = stage.runner.model_runner.requests["fp32-stage-validation"].prompt_embeds
        full_intermediate = None
        if index > 0:
            full_intermediate = IntermediateTensors(
                {
                    "hidden_states": full_history,
                    "residual": torch.cat(
                        [captured["residual-0"], captured["residual-1"]], dim=0
                    ).cuda(),
                }
            )
        full_result = stage.runner.execute(
            full_history, "fp32-full-forward-validation", 0, full_intermediate
        )
        full_last = hidden_from_intermediate(full_result)[-1:].detach().cpu()
        error = (actual["output-1"] - full_last).abs().max().item()
        stage.drop_request("fp32-stage-validation")
        stage.drop_request("fp32-full-forward-validation")
        return actual, forwards, residuals, error
    finally:
        stage.close()


def reference_outputs(
    root: Path,
    index: int,
    captured: dict[str, torch.Tensor],
    actual,
    atol: float,
    rtol: float,
):
    private = X25519PrivateKey.from_private_bytes((root / "stage.identity").read_bytes())
    reference = DenseStage(root / "worker-artifacts/manifest.json", index, private)
    document = json.loads((root / "worker-artifacts/manifest.json").read_text())["stages"][index]
    inputs = {
        token_index: captured[f"input-{token_index}"].cuda().double() for token_index in (0, 1)
    }
    if index > 0:
        inputs = {
            token_index: value + captured[f"residual-{token_index}"].cuda().double()
            for token_index, value in inputs.items()
        }
    full_reference = torch.cat([inputs[0], inputs[1]], dim=0)
    pasts = {token_index: None for token_index in inputs}
    errors = {}
    reference_errors = []
    weights = {}
    try:
        for layer_index in document["layer_indices"]:
            prefix = f"model.layers.{layer_index}"
            weights = {
                name: value.cuda().double()
                for name, value in reference._tensors.items()
                if name.startswith(prefix + ".")
            }
            layer = layer_weights_from_dict(weights, prefix)
            next_inputs = {}
            next_pasts = {}
            for token_index, current_input in inputs.items():
                expected, past = execute_dense_layer_cached(
                    current_input,
                    layer,
                    query_heads=document["query_heads"],
                    kv_heads=document["kv_heads"],
                    past=None if token_index == 0 else next_pasts[0],
                    interleaved_rope=document["interleaved_rope"],
                    rope_base=document["rope_theta"],
                )
                next_inputs[token_index] = expected
                next_pasts[token_index] = past
            full_reference = execute_dense_layer(
                full_reference,
                layer,
                query_heads=document["query_heads"],
                kv_heads=document["kv_heads"],
                interleaved_rope=document["interleaved_rope"],
                rope_base=document["rope_theta"],
            )
            reference_errors.append((next_inputs[1] - full_reference[-1:]).abs().max().item())
            for past in pasts.values():
                if past is not None:
                    for value in past:
                        value.zero_()
            inputs = next_inputs
            pasts = next_pasts
            for value in weights.values():
                value.zero_()
            torch.cuda.synchronize()
        for token_index in inputs:
            actual_gpu = actual[f"output-{token_index}"].cuda().double()
            expected = inputs[token_index]
            errors[f"token-{token_index}"] = (actual_gpu - expected).abs().max().item()
            torch.testing.assert_close(actual_gpu, expected, atol=atol, rtol=rtol)
        return errors, max(reference_errors, default=0.0)
    finally:
        reference.close()
        for past in pasts.values():
            if past is not None:
                for value in past:
                    value.zero_()
        for value in weights.values():
            value.zero_()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    args = parser.parse_args()

    captured = load_file(str(args.input))
    errors = {}
    status = "passed"
    try:
        actual, forwards, residuals, vllm_error = optimized_outputs(args.root, args.index, captured)
        errors["vllm_decode_vs_full_forward"] = vllm_error
        reference_errors, cached_error = reference_outputs(
            args.root, args.index, captured, actual, args.atol, args.rtol
        )
        errors.update(reference_errors)
        errors["reference_decode_vs_full_forward"] = cached_error
    except Exception as error:
        status = "failed"
        errors["error_type"] = type(error).__name__
        raise
    finally:
        if "actual" in locals():
            save_file({**actual, **forwards, **residuals}, str(args.output))
        report = {
            "stage_index": args.index,
            "status": status,
            "max_abs_errors": errors,
            "atol": args.atol,
            "rtol": args.rtol,
            "hardware_validated": status == "passed",
            "production_accepted": False,
            "distributed_validated": False,
        }
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
