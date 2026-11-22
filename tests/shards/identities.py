from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from xurcroh.kem import HybridKemKey
from xurcroh.shards.compiler import CompilationConfig, CompilationResult, XurcrohWeightCompiler
from xurcroh.shards.recipients import StageRecipient


@dataclass(frozen=True)
class StageIdentities:
    boundary_key: bytes
    private_keys: tuple[HybridKemKey, ...]
    recipients: tuple[StageRecipient, ...]

    @classmethod
    def create(cls, stage_count: int) -> StageIdentities:
        private_keys = tuple(HybridKemKey.generate() for _ in range(stage_count))
        recipients = tuple(
            StageRecipient.from_private_key(index, private_key)
            for index, private_key in enumerate(private_keys)
        )
        return cls(os.urandom(32), private_keys, recipients)

    def compile(
        self,
        checkpoint: Path,
        output_dir: Path,
        *,
        seed: int,
    ) -> CompilationResult:
        return XurcrohWeightCompiler(
            CompilationConfig(
                checkpoint=checkpoint,
                output_dir=output_dir,
                stage_count=len(self.private_keys),
                boundary_key=self.boundary_key,
                recipients=self.recipients,
                seed=seed,
            )
        ).compile()
