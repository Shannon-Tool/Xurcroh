from __future__ import annotations

from pathlib import Path

import torch

from Xurcroh import Xurcroh
from xurcroh.requests import CompletionRequest
from xurcroh.runtime.generation import TokenSampler


def request(**settings: object) -> CompletionRequest:
    return CompletionRequest.model_validate(
        {"model": "m", "messages": [{"role": "user", "content": "x"}], **settings}
    )


def test_seeded_sampler_advances_between_tokens() -> None:
    sampler = TokenSampler(request(seed=7))
    logits = torch.zeros(1, 4096)

    tokens = [int(sampler(logits).item()) for _ in range(8)]

    assert len(set(tokens)) > 1


def test_seeded_sampler_is_reproducible_and_matches_one_generator() -> None:
    logits = torch.zeros(1, 4096)
    first = TokenSampler(request(seed=11))
    second = TokenSampler(request(seed=11))
    generator = torch.Generator().manual_seed(11)
    probabilities = torch.softmax(logits, dim=-1)

    expected = [
        int(torch.multinomial(probabilities, 1, generator=generator).item()) for _ in range(6)
    ]

    assert [int(first(logits).item()) for _ in range(6)] == expected
    assert [int(second(logits).item()) for _ in range(6)] == expected
    assert first.seed == 11


def test_unseeded_sampler_reports_the_seed_it_drew() -> None:
    logits = torch.zeros(1, 4096)
    sampler = TokenSampler(request())
    assert sampler.seed is not None

    replay = TokenSampler(request(seed=sampler.seed))

    assert [int(sampler(logits).item()) for _ in range(6)] == [
        int(replay(logits).item()) for _ in range(6)
    ]


def test_greedy_sampler_has_no_seed() -> None:
    sampler = TokenSampler(request(temperature=0, seed=5))
    logits = torch.tensor([[0.0, 3.0, 1.0]])

    assert int(sampler(logits).item()) == 1
    assert sampler.seed is None


def test_completion_reports_effective_seed_and_replays(tiny_model_path: Path) -> None:
    messages = [{"role": "user", "content": "hello"}]
    with Xurcroh(
        cluster={"endpoint": "local://cpu"}, model_path={"tiny": tiny_model_path}
    ) as client:
        first = client.chat.completions.create(model="tiny", messages=messages, max_tokens=6)
        seed = first.execution.seed
        assert seed is not None
        replay = client.chat.completions.create(
            model="tiny", messages=messages, max_tokens=6, seed=seed
        )
        greedy = client.chat.completions.create(
            model="tiny", messages=messages, max_tokens=2, temperature=0
        )

    assert replay.execution.seed == seed
    assert replay.choices[0].message.content == first.choices[0].message.content
    assert greedy.execution.seed is None
