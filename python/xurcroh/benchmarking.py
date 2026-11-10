from __future__ import annotations

import base64
import importlib
import io
import json
import math
import os
import platform
import resource
import statistics
import time
import tracemalloc
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from PIL import Image

from Xurcroh import Xurcroh
from xurcroh.algebra.fwht import fwht, hadamard_matrix


def _environment() -> dict[str, Any]:
    accelerator = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "accelerator": accelerator,
        "hardware_validated": False,
    }


def _base_report(kind: str, parameters: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": kind,
        "created_unix": time.time(),
        "environment": _environment(),
        "parameters": parameters,
    }


def _milliseconds(started: float) -> float:
    return (time.perf_counter() - started) * 1000


def _statistics(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "minimum": ordered[0],
        "maximum": ordered[-1],
    }


def benchmark_chat(
    client: Xurcroh,
    *,
    model: str,
    messages: list[dict[str, str]],
    iterations: int = 5,
    max_tokens: int = 32,
    temperature: float = 0.0,
    timeout: float | None = None,
) -> dict[str, Any]:
    if iterations < 1 or max_tokens < 1:
        raise ValueError("iterations and max_tokens must be positive")
    samples: list[dict[str, Any]] = []
    for _ in range(iterations):
        started = time.perf_counter()
        first_chunk_ms: float | None = None
        intervals_ms: list[float] = []
        previous = started
        completion_tokens = 0
        prompt_tokens = 0
        finish_reason: str | None = None
        stream = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
            stream=True,
        )
        for chunk in stream:
            now = time.perf_counter()
            if first_chunk_ms is None:
                first_chunk_ms = (now - started) * 1000
            else:
                intervals_ms.append((now - previous) * 1000)
            previous = now
            choice = chunk.choices[0]
            if choice.finish_reason is not None:
                finish_reason = choice.finish_reason
            if chunk.usage is not None:
                completion_tokens = chunk.usage.completion_tokens
                prompt_tokens = chunk.usage.prompt_tokens
        duration_ms = _milliseconds(started)
        if first_chunk_ms is None or finish_reason is None or completion_tokens == 0:
            raise RuntimeError("benchmark stream ended without token usage and a finish reason")
        samples.append(
            {
                "time_to_first_chunk_ms": first_chunk_ms,
                "duration_ms": duration_ms,
                "completion_tokens": completion_tokens,
                "prompt_tokens": prompt_tokens,
                "tokens_per_second": completion_tokens * 1000 / duration_ms,
                "inter_chunk_ms": _statistics(intervals_ms) if intervals_ms else None,
                "finish_reason": finish_reason,
            }
        )
    report = _base_report(
        "chat",
        {
            "model": model,
            "iterations": iterations,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "timeout_seconds": timeout,
            "prompt_messages": len(messages),
        },
    )
    report["samples"] = samples
    report["aggregate"] = {
        key: _statistics([float(sample[key]) for sample in samples])
        for key in ("time_to_first_chunk_ms", "duration_ms", "tokens_per_second")
    }
    return report


def benchmark_fwht(*, dimension: int = 4096, iterations: int = 20, seed: int = 0) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn((128, dimension), dtype=torch.float64, generator=generator)
    direct = values @ hadamard_matrix(dimension) / math.sqrt(dimension)
    transformed = fwht(values)
    transform_error = float((transformed - direct).abs().max())
    inverse_error = float((fwht(transformed, inverse=True) - values).abs().max())
    started = time.perf_counter()
    for _ in range(iterations):
        fwht(values)
    duration = time.perf_counter() - started
    report = _base_report(
        "fwht",
        {"dimension": dimension, "rows": values.shape[0], "iterations": iterations, "seed": seed},
    )
    report.update(
        {
            "maximum_transform_error": transform_error,
            "maximum_inverse_error": inverse_error,
            "samples_per_second": values.shape[0] * iterations / duration,
        }
    )
    return report


def _native_transport() -> Any:
    try:
        return importlib.import_module("xurcroh_quic")
    except ModuleNotFoundError:
        return None


def benchmark_transport(*, payload_bytes: int = 65536, iterations: int = 100) -> dict[str, Any]:
    native = _native_transport()
    if native is None:
        raise RuntimeError(
            "Rust bindings are required; run maturin develop --manifest-path "
            "crates/transport/Cargo.toml --features python"
        )
    if payload_bytes < 1 or iterations < 1:
        raise ValueError("payload_bytes and iterations must be positive")
    master = os.urandom(32)
    sender = native.AeadSession(master, b"client", b"server", b"benchmark")
    receiver = native.AeadSession(master, b"server", b"client", b"benchmark")
    payload = bytes(index % 251 for index in range(payload_bytes))
    started = time.perf_counter()
    for _ in range(iterations):
        bytes(payload)
    copy_duration = time.perf_counter() - started
    started = time.perf_counter()
    verified = 0
    for sequence in range(iterations):
        encoded = native.encode_frame(
            bytes(16), 0, sequence, "tensor", "u8", [payload_bytes], payload
        )
        actual_sequence, nonce, ciphertext = sender.seal(b"benchmark", encoded)
        plaintext = receiver.open(b"benchmark", actual_sequence, nonce, ciphertext)
        decoded = native.decode_frame(plaintext)
        if decoded[-1] != payload:
            raise RuntimeError("transport benchmark failed its authenticated round trip")
        verified += 1
    secure_duration = time.perf_counter() - started
    master = bytes(len(master))
    report = _base_report(
        "transport-aead",
        {"payload_bytes": payload_bytes, "iterations": iterations},
    )
    report.update(
        {
            "verified_round_trips": verified,
            "aead_frames_per_second": iterations / secure_duration,
            "secure_megabytes_per_second": payload_bytes * iterations / secure_duration / 1e6,
            "copy_megabytes_per_second": payload_bytes * iterations / copy_duration / 1e6,
            "overhead_ratio": secure_duration / copy_duration,
            "scope": (
                "Rust AEAD, tensor framing, and memory copy on this host; not network throughput"
            ),
        }
    )
    return report


def image_quality(
    reference: np.ndarray[Any, Any], candidate: np.ndarray[Any, Any]
) -> dict[str, Any]:
    if reference.shape != candidate.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("image quality inputs must have matching height, width, and RGB channels")
    if reference.size == 0:
        raise ValueError("image quality inputs cannot be empty")
    left = reference.astype(np.float64)
    right = candidate.astype(np.float64)
    error = left - right
    mse = float(np.mean(error**2))
    return {
        "mean_absolute_error": float(np.mean(np.abs(error))),
        "mean_squared_error": mse,
        "psnr_db": None if mse == 0 else 10 * math.log10(255**2 / mse),
    }


def memory_snapshot(*, device: Literal["cpu", "cuda"] = "cpu") -> dict[str, Any]:
    with Path("/proc/self/statm").open() as statm:
        resident_pages = int(statm.read().split()[1])
    snapshot: dict[str, Any] = {
        "device": device,
        "resident_megabytes": resident_pages * os.sysconf("SC_PAGE_SIZE") / 1e6,
        "peak_rss_megabytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 1e6,
    }
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory profiling requires a CUDA device")
        torch.cuda.synchronize()
        snapshot.update(
            {
                "allocated_megabytes": torch.cuda.memory_allocated() / 1e6,
                "reserved_megabytes": torch.cuda.memory_reserved() / 1e6,
                "peak_allocated_megabytes": torch.cuda.max_memory_allocated() / 1e6,
            }
        )
    return snapshot


def benchmark_memory(
    *,
    dimension: int = 4096,
    rows: int = 128,
    iterations: int = 20,
    device: Literal["cpu", "cuda"] = "cpu",
) -> dict[str, Any]:
    if dimension < 1 or rows < 1 or iterations < 1:
        raise ValueError("dimension, rows, and iterations must be positive")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA memory profiling requires a CUDA device")
        torch.cuda.reset_peak_memory_stats()
    before = memory_snapshot(device=device)
    tracemalloc.start()
    started = time.perf_counter()
    for _ in range(iterations):
        values = torch.randn(rows, dimension, dtype=torch.float32, device=device)
        transformed = fwht(values)
        transformed.zero_()
        values.zero_()
        del transformed, values
    if device == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - started
    peak_python_bytes = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    report = _base_report(
        "memory-fwht",
        {"dimension": dimension, "rows": rows, "iterations": iterations, "device": device},
    )
    report.update(
        {
            "before": before,
            "after": memory_snapshot(device=device),
            "peak_python_bytes": peak_python_bytes,
            "samples_per_second": rows * iterations / duration,
            "scope": "Actual allocator counters; not proof of physical erasure",
        }
    )
    return report


def benchmark_image(
    client: Xurcroh,
    *,
    model: str,
    prompt: str,
    size: str = "1024x1024",
    steps: int = 30,
    seed: int | None = 0,
    reference: str | Path | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.images.generate(
        model=model, prompt=prompt, size=size, steps=steps, seed=seed, timeout=timeout
    )
    duration_ms = _milliseconds(started)
    encoded = base64.b64decode(response.data[0].b64_json, validate=True)
    with Image.open(io.BytesIO(encoded)) as source:
        pixels = np.asarray(source.convert("RGB"))
    report = _base_report(
        "image",
        {"model": model, "size": size, "steps": steps, "seed": seed, "timeout_seconds": timeout},
    )
    report.update(
        {
            "duration_ms": duration_ms,
            "encoded_bytes": len(encoded),
            "width": pixels.shape[1],
            "height": pixels.shape[0],
            "pixels_per_second": pixels.shape[0] * pixels.shape[1] * 1000 / duration_ms,
            "quality": None,
        }
    )
    if reference is not None:
        with Image.open(reference) as source:
            report["quality"] = image_quality(np.asarray(source.convert("RGB")), pixels)
    return report


def _decode_video(data: bytes) -> np.ndarray[Any, Any]:
    imageio = importlib.import_module("imageio.v3")
    frames = np.asarray(imageio.imread(io.BytesIO(data), extension=".mp4"))
    if frames.ndim != 4 or frames.shape[-1] != 3 or frames.shape[0] == 0:
        raise ValueError("decoded video must contain RGB frames")
    return frames


def benchmark_video(
    client: Xurcroh,
    *,
    model: str,
    prompt: str,
    size: str = "832x480",
    frames: int = 81,
    fps: int = 24,
    steps: int = 40,
    seed: int | None = 0,
    reference: str | Path | None = None,
    timeout: float | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.videos.generate(
        model=model,
        prompt=prompt,
        size=size,
        frames=frames,
        fps=fps,
        steps=steps,
        seed=seed,
        timeout=timeout,
    )
    duration_ms = _milliseconds(started)
    encoded = base64.b64decode(response.data[0].b64_json, validate=True)
    pixels = _decode_video(encoded)
    report = _base_report(
        "video",
        {
            "model": model,
            "size": size,
            "frames": frames,
            "fps": fps,
            "steps": steps,
            "seed": seed,
            "timeout_seconds": timeout,
        },
    )
    report.update(
        {
            "duration_ms": duration_ms,
            "encoded_bytes": len(encoded),
            "decoded_frames": pixels.shape[0],
            "width": pixels.shape[2],
            "height": pixels.shape[1],
            "generated_frames_per_second": pixels.shape[0] * 1000 / duration_ms,
            "quality": None,
        }
    )
    if reference is not None:
        imageio = importlib.import_module("imageio.v3")
        expected = np.asarray(imageio.imread(reference))
        if expected.ndim != 4:
            raise ValueError("reference video must contain RGB frames")
        report["quality"] = image_quality(
            expected.reshape(-1, *expected.shape[-2:]),
            pixels.reshape(-1, *pixels.shape[-2:]),
        )
    return report


def write_report(report: dict[str, Any], destination: str | Path) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path
