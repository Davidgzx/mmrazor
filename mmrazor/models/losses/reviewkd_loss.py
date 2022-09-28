# Copyright (c) OpenMMLab. All rights reserved.
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
    ) -> None:
        super().__init__()
        self.loss_weight = loss_weight

    def forward(
        self,
        s_feature: torch.Tensor,
        t_feature: torch.Tensor,
    ) -> torch.Tensor:
        loss_all = 0.0
        _, _, h, _ = s_feature.shape
        loss = F.mse_loss(s_feature, t_feature, reduction='mean')
        cnt = 1.0
        tot = 1.0
        for size in [4, 2, 1]:
            if size >= h:
                continue
            pooled_fs = F.adaptive_avg_pool2d(s_feature, (size, size))
            pooled_ft = F.adaptive_avg_pool2d(t_feature, (size, size))
            cnt /= 2.0
            loss += F.mse_loss(pooled_fs, pooled_ft, reduction='mean') * cnt
            tot += cnt
        loss = loss / tot
        loss_all += loss
        return loss_all
