from __future__ import annotations

import threading
from typing import Any

import torch

from xurcroh.errors import CapabilityError


class ProtectedDiffusionPipeline:
    def __init__(
        self,
        pipeline: Any,
        basis: torch.Tensor,
        permutation: torch.Tensor,
    ) -> None:
        self.pipeline = pipeline
        self.basis = basis
        self.permutation = permutation
        self.scheduler_steps = 0
        self.model_calls = 0
        self._lock = threading.Lock()
        self._closed = False
        if basis.ndim != 2 or basis.shape[0] != basis.shape[1]:
            raise ValueError("latent basis must be square")
        if permutation.ndim != 1 or not torch.equal(
            torch.sort(permutation.cpu()).values,
            torch.arange(permutation.numel()),
        ):
            raise ValueError("latent permutation must contain every position exactly once")

    def _matrix(self, latent: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        if latent.ndim not in {4, 5} or latent.shape[1] != self.basis.shape[0]:
            raise ValueError("latent must be BCHW or BCFHW with basis-sized channels")
        positions = 1
        for dimension in latent.shape[2:]:
            positions *= dimension
        if positions != self.permutation.numel():
            raise ValueError("latent spatial size does not match the permutation")
        moved = latent.movedim(1, -1)
        return moved.reshape(-1, latent.shape[1]), tuple(moved.shape)

    def protect(self, latent: torch.Tensor) -> torch.Tensor:
        matrix, shape = self._matrix(latent)
        transformed = (
            matrix[self.permutation.to(latent.device)]
            @ self.basis.to(device=latent.device, dtype=latent.dtype).T
        )
        return transformed.reshape(shape).movedim(-1, 1)

    def recover(self, latent: torch.Tensor) -> torch.Tensor:
        matrix, shape = self._matrix(latent)
        recovered = matrix @ self.basis.to(device=latent.device, dtype=latent.dtype)
        inverse = torch.argsort(self.permutation).to(latent.device)
        return recovered[inverse].reshape(shape).movedim(-1, 1)

    @staticmethod
    def _last_timestep(scheduler: Any, timestep: Any, steps: int | None) -> bool:
        last = scheduler.timesteps[steps - 1] if steps is not None else scheduler.timesteps[-1]
        if torch.is_tensor(timestep):
            return bool((timestep == last).item())
        return bool(timestep == last)

    def generate(self, **parameters: Any) -> Any:
        if self._closed:
            raise CapabilityError("protected diffusion pipeline is closed")
        if parameters.get("latents") is None:
            raise CapabilityError(
                "protected diffusion requires client-prepared transformed latents"
            )
        scheduler = getattr(self.pipeline, "scheduler", None)
        model = getattr(self.pipeline, "unet", None) or getattr(self.pipeline, "transformer", None)
        if scheduler is None or model is None:
            raise CapabilityError("pipeline does not expose a scheduler and denoising model")
        steps = parameters.get("num_inference_steps")
        if steps is not None and steps > len(scheduler.timesteps):
            raise CapabilityError("requested denoising steps exceed the configured schedule")
        original_step = scheduler.step
        original_forward = model.forward

        def protected_step(
            model_output: Any, timestep: Any, sample: torch.Tensor, *args: Any, **kwargs: Any
        ) -> Any:
            self.scheduler_steps += 1
            recovered = self.recover(sample)
            output = original_step(model_output, timestep, recovered, *args, **kwargs)
            if not self._last_timestep(scheduler, timestep, parameters.get("num_inference_steps")):
                output.prev_sample = self.protect(output.prev_sample)
            return output

        def recovered_forward(sample: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
            self.model_calls += 1
            return original_forward(self.recover(sample), *args, **kwargs)

        with self._lock:
            scheduler.step = protected_step
            model.forward = recovered_forward
            try:
                return self.pipeline(**parameters)
            finally:
                scheduler.step = original_step
                model.forward = original_forward

    def close(self) -> None:
        if not self._closed:
            self.basis.zero_()
            self.permutation.zero_()
            self._closed = True
