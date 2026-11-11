from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch


def scrub_sglang_request(worker: Any, request_index: int, token_indices: Iterable[int]) -> None:
    runner = worker.model_runner
    indices = list(token_indices)
    req_to_token = runner.req_to_token_pool.req_to_token
    if 0 <= request_index < req_to_token.shape[0]:
        req_to_token[request_index].zero_()
    for pool_name in ("k_buffer", "v_buffer"):
        pool = getattr(runner.token_to_kv_pool_allocator.kv_cache, pool_name, None)
        if pool is not None and indices:
            locations = torch.as_tensor(indices, dtype=torch.long, device=pool.device)
            pool.index_fill_(1, locations, 0)
    runner.req_to_token_pool.free_rows([request_index])
