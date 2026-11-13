from __future__ import annotations

from dataclasses import dataclass

import torch

from xurcroh.fortress.arithmetic import combine_shares


@dataclass(frozen=True)
class ShareReconstructionReport:
    full_weight_collusion_exact: bool
    full_activation_collusion_exact: bool
    single_weight_share_reconstructs: bool
    single_activation_share_reconstructs: bool
    missing_weight_share_search_bits: int
    missing_activation_share_search_bits: int
    interpretation: str = (
        "Algebraic fixture results only. Full-role collusion reconstructs the secret; "
        "single-share mismatch is not an information-theoretic proof. Search bits assume "
        "independent uniform 2^32 shares and no additional structure or side information."
    )


def evaluate_share_reconstruction(
    weight: torch.Tensor,
    activation: torch.Tensor,
    weight_shares: list[torch.Tensor],
    activation_shares: list[torch.Tensor],
) -> ShareReconstructionReport:
    if len(weight_shares) < 2 or len(activation_shares) < 2:
        raise ValueError("reconstruction evaluation requires at least two shares")
    recovered_weight = combine_shares(weight_shares)
    recovered_activation = combine_shares(activation_shares)
    return ShareReconstructionReport(
        full_weight_collusion_exact=torch.equal(recovered_weight, weight.remainder(2**32)),
        full_activation_collusion_exact=torch.equal(
            recovered_activation, activation.remainder(2**32)
        ),
        single_weight_share_reconstructs=any(
            torch.equal(share, weight.remainder(2**32)) for share in weight_shares
        ),
        single_activation_share_reconstructs=any(
            torch.equal(share, activation.remainder(2**32)) for share in activation_shares
        ),
        missing_weight_share_search_bits=weight.numel() * 32,
        missing_activation_share_search_bits=activation.numel() * 32,
    )


@dataclass(frozen=True)
class TimingClassifierReport:
    real_samples: int
    cover_samples: int
    leave_one_out_accuracy: float
    leave_one_out_balanced_accuracy: float
    majority_baseline: float
    balanced_baseline: float = 0.5
    interpretation: str = (
        "Local leave-one-out nearest-mean timing result, not a general side-channel proof. "
        "Fixed-size frames and constant-rate cover do not eliminate scheduling, network, "
        "cache, power, or Whisper-Leak-class leakage."
    )


def measure_timing_classifier(observations: list[tuple[float, bool]]) -> TimingClassifierReport:
    if not observations or any(duration < 0 for duration, _ in observations):
        raise ValueError("timing observations must be nonempty and nonnegative")
    real_count = sum(not cover for _, cover in observations)
    cover_count = len(observations) - real_count
    if not real_count or not cover_count:
        raise ValueError("timing classification requires real and cover observations")
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
