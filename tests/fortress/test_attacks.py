from __future__ import annotations

import random
import time
from dataclasses import asdict

import pytest

from xurcroh.fortress.attacks import TimingClassifierReport, measure_timing_classifier


def _naive_timing_classifier(observations: list[tuple[float, bool]]) -> TimingClassifierReport:
    real_count = sum(not cover for _, cover in observations)
    cover_count = len(observations) - real_count
    correct = 0
    correct_real = 0
    correct_cover = 0
    for index, (duration, cover) in enumerate(observations):
        remaining = observations[:index] + observations[index + 1 :]
        real_mean = sum(value for value, label in remaining if not label) / sum(
            not label for _, label in remaining
        )
        cover_mean = sum(value for value, label in remaining if label) / sum(
            label for _, label in remaining
        )
        predicted_cover = abs(duration - cover_mean) < abs(duration - real_mean)
        correct += predicted_cover == cover
        correct_real += not cover and not predicted_cover
        correct_cover += cover and predicted_cover
    return TimingClassifierReport(
        real_samples=real_count,
        cover_samples=cover_count,
        leave_one_out_accuracy=correct / len(observations),
        leave_one_out_balanced_accuracy=(correct_real / real_count + correct_cover / cover_count)
        / 2,
        majority_baseline=max(real_count, cover_count) / len(observations),
    )


def test_timing_classifier_matches_naive_leave_one_out() -> None:
    rng = random.Random(731)
    for size in (4, 9, 31, 100):
        observations = [(rng.uniform(0.0, 10.0), index % 3 == 0) for index in range(size)]
        optimized = asdict(measure_timing_classifier(observations))
        naive = asdict(_naive_timing_classifier(observations))
        assert optimized == naive


def test_timing_classifier_is_linear_time() -> None:
    rng = random.Random(17)
    observations = [(rng.uniform(0.0, 1.0), index % 2 == 0) for index in range(50_000)]
    start = time.monotonic()
    report = measure_timing_classifier(observations)
    elapsed = time.monotonic() - start
    assert report.real_samples == 25_000
    assert elapsed < 10.0


def test_timing_classifier_single_sample_class_raises() -> None:
    observations = [(1.0, False), (2.0, True), (3.0, True)]
    with pytest.raises(ZeroDivisionError):
        measure_timing_classifier(observations)
