from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from xurcroh.errors import CompatibilityError
from xurcroh.models.parallelism import ParallelismConfig, validate_parallelism

__all__ = ["FrameworkAdapter", "ParallelismConfig"]


class FrameworkAdapter:
    def __init__(self, engine: str) -> None:
        if engine not in {"vllm", "sglang", "sglang-diffusion", "xdit"}:
            raise CompatibilityError(f"unsupported inference engine: {engine}")
        self.engine = engine

    def require_installed(self) -> str:
        package = {
            "vllm": "vllm",
            "sglang": "sglang",
            "sglang-diffusion": "sglang",
            "xdit": "xfuser",
        }[self.engine]
        try:
            return version(package)
        except PackageNotFoundError as error:
            raise CompatibilityError(
                f"{self.engine} adapter requires the {package} package and its native runtime"
            ) from error

    def worker_command(
        self,
        *,
        model: str,
        pipeline_parallel: int = 1,
        tensor_parallel: int = 1,
        expert_parallel: int = 1,
        parallelism: ParallelismConfig | None = None,
    ) -> str:
        if pipeline_parallel < 1 or tensor_parallel < 1 or expert_parallel < 1:
            raise CompatibilityError("parallel degrees must be positive")
        if self.engine == "vllm":
            return (
                "python3 -m vllm.entrypoints.openai.api_server "
                f"--model {model} --load-format xurcroh_sharded "
                "--no-enable-prefix-caching --no-async-scheduling "
                f"--pipeline-parallel-size {pipeline_parallel} "
                f"--tensor-parallel-size {tensor_parallel}"
            )
        if self.engine == "sglang":
            return (
                "python3 -m sglang.launch_server "
                f"--model-path {model} --pp-size {pipeline_parallel} "
                f"--tp-size {tensor_parallel} --ep-size {expert_parallel} "
                "--disable-radix-cache --chunked-prefill-size 4096 --enable-dynamic-chunking"
            )
        if self.engine == "sglang-diffusion":
            parallel = parallelism or ParallelismConfig(
                ring_degree=0,
                ulysses_degree=tensor_parallel,
                attention_backend="flash",
            )
            validate_parallelism(parallel)
            return (
                f"sglang serve --model-path {model} "
                f"--num-gpus {max(1, parallel.ulysses_degree)} "
                f"--ulysses-degree {parallel.ulysses_degree} "
                f"--ring-degree {parallel.ring_degree}"
            )
        parallel = parallelism or ParallelismConfig(
            ring_degree=0,
            ulysses_degree=tensor_parallel,
            attention_backend="flash",
        )
        validate_parallelism(parallel)
        command = f"xdit --model {model} --ulysses_degree {parallel.ulysses_degree}"
        if parallel.pipefusion:
            command += " --pipefusion_parallel_degree 2"
        if parallel.cfg_parallel:
            command += " --use_cfg_parallel"
        return command
