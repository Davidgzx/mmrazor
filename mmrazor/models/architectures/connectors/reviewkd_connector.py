# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmrazor.registry import MODELS
from .base_connector import BaseConnector


@MODELS.register_module()
class ABFConnector(BaseConnector):

    def __init__(
        self,
        in_channel: int,
        mid_channel: int,
        out_channel: int,
        residual: str = None,
        init_cfg: Optional[Dict] = None,
    ) -> None:
        super().__init__(init_cfg)
        super(ABFConnector, self).__init__()
        self.residual = residual
        self.conv1 = ConvModule(
            in_channel,
            mid_channel,
            kernel_size=1,
            bias=False,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None)

        self.conv2 = ConvModule(
            mid_channel,
            out_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            norm_cfg=dict(type='BN', requires_grad=True),
            act_cfg=None)

        if self.residual is not None:
            self.att_conv = ConvModule(
                mid_channel * 2,
                2,
                kernel_size=1,
                act_cfg=dict(type='Sigmoid'))
        else:
            self.att_conv = None

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> torch.Tensor:
        return self.forward_train(x, y)

    def forward_train(self, x: torch.Tensor, y: torch.Tensor = None):
        if y is None:
            return self.forward_no_fuse(x)
        else:
            return self.forward_fuse(x, y)

    def forward_no_fuse(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        return y, x

    def forward_fuse(self, x, y):
        n, _, h, w = x.shape
        # upsample residual features
        y = F.interpolate(y, (h, w), mode='nearest')
        # fusion
        z = torch.cat([x, y], dim=1)
        z = self.att_conv(z)
        x = (x * z[:, 0].view(n, 1, h, w) + y * z[:, 1].view(n, 1, h, w))
        y = self.conv2(x)
        return y, x
