"""Compute the original Smooth L1 and SSIM enhancement loss."""

import torch
from pytorch_msssim import ssim
from torch import nn


class ReconstructionLoss(nn.Module):
    def __init__(
        self,
        smooth_l1_weight: float = 1.0,
        smooth_l1_beta: float = 1.0,
        ssim_weight: float = 0.09,
        ssim_data_range: float = 1.0,
        ssim_window_size: int = 11,
    ) -> None:
        super().__init__()
        self.smooth_l1_weight = smooth_l1_weight
        self.ssim_weight = ssim_weight
        self.ssim_data_range = ssim_data_range
        self.ssim_window_size = ssim_window_size
        self.smooth_l1 = nn.SmoothL1Loss(beta=smooth_l1_beta)

    def forward(
        self, generated: torch.Tensor, reference: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        smooth_l1_value = self.smooth_l1(generated, reference)
        ssim_value = 1.0 - ssim(
            generated,
            reference,
            data_range=self.ssim_data_range,
            size_average=True,
            win_size=self.ssim_window_size,
        )
        total = self.smooth_l1_weight * smooth_l1_value + self.ssim_weight * ssim_value
        return (
            total,
            {"smooth_l1": smooth_l1_value.detach(), "ssim": ssim_value.detach()},
        )
