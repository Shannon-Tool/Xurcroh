from __future__ import annotations

from xurcroh.models.scheduling import ContinuousBatchScheduler, dynamic_chunk_size


def test_dynamic_prefill_chunks_cover_prompt_and_shrink() -> None:
    first = dynamic_chunk_size(4096, prefix_tokens=0, smoothing_factor=0.75)
    second = dynamic_chunk_size(4096, prefix_tokens=first, smoothing_factor=0.75)
    assert first == 4096
    assert 64 <= second < first
    assert second % 64 == 0


def test_continuous_batch_scheduler_is_fair_and_cancels() -> None:
    scheduler = ContinuousBatchScheduler(max_running=2)
    scheduler.submit("a", 100)
    scheduler.submit("b", 100)
    scheduler.submit("c", 100)
    first = scheduler.next_batch()
    assert [chunk.request_id for chunk in first] == ["a", "b"]
    scheduler.complete("a")
    second = scheduler.next_batch()
    assert [chunk.request_id for chunk in second] == ["b", "c"]
    scheduler.cancel("c")
    assert scheduler.next_batch()[0].request_id == "b"
    scheduler.complete("b")
    assert scheduler.next_batch() == []
