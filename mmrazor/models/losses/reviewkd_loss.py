# Copyright (c) OpenMMLab. All rights reserved.
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmrazor.registry import MODELS


@MODELS.register_module()
class HCLLoss(nn.Module):
    """Distilling Knowledge via Knowledge Review
    https://arxiv.org/pdf/2104.09044.pdf.

    Args:
        loss_weight (float, optional): loss weight. Defaults to 1.0.
        mul_factor (float, optional): multiply factor. Defaults to 1000.
    """

    def __init__(
        self,
        loss_weight: float = 1.0,
        pool_sizes: List[int] = [4, 2, 1],
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight
        self.pool_sizes = pool_sizes

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        _, _, h, _ = s_feature.shape
        loss = F.mse_loss(s_feature, t_feature, reduction='mean')
        weight = 1.0
        for size in self.pool_sizes:
            if size >= h:
                continue
            pooled_fs = F.adaptive_avg_pool2d(s_feature, (size, size))
            pooled_ft = F.adaptive_avg_pool2d(t_feature, (size, size))
            weight /= 2.0
            loss += F.mse_loss(pooled_fs, pooled_ft, reduction='mean') * weight
        loss = loss / len(self.pool_sizes)
        return loss
