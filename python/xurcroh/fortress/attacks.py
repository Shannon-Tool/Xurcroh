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
