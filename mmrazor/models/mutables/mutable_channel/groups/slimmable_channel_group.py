# Copyright (c) OpenMMLab. All rights reserved.

from typing import List, Union

import torch.nn as nn

from mmrazor.models.architectures.dynamic_ops.bricks import (
    DynamicConv2d, DynamicLinear, SwitchableBatchNorm2d)
from mmrazor.registry import MODELS
from ..mutable_channel_container import MutableChannelContainer
from .one_shot_mutable_channel_group import OneShotMutableChannelGroup


@MODELS.register_module()
class SlimmableChannelGroup(OneShotMutableChannelGroup):
    """A type of ``MutableChannelGroup`` to train several subnets together.

    Args:
        num_channels (int): The raw number of channels.
        candidate_choices (List[Union[int, float]], optional):
            A list of candidate width ratios. Each
            candidate indicates how many channels to be reserved.
            Defaults to [0.5, 1.0](candidate_mode='ratio').
        candidate_mode (str, optional): Mode of candidates.
            One of 'ratio' or 'number'. Defaults to 'number'.
        divisor (int, optional): Used to make choice divisible.
        min_value (int, optional): The minimal value used when make divisible.
        min_ratio (float, optional): The minimal ratio used when make
            divisible.
    """

    def __init__(self,
                 num_channels: int,
                 candidate_choices: List[Union[int, float]] = [],
                 candidate_mode='number',
                 divisor=1,
                 min_value=1,
                 min_ratio=0.9) -> None:
        super().__init__(num_channels, candidate_choices, candidate_mode,
                         divisor, min_value, min_ratio)

    def prepare_for_pruning(self, model: nn.Module):
        """Prepare for pruning."""
        self._replace_with_dynamic_ops(
            model, {
                nn.Conv2d: DynamicConv2d,
                nn.BatchNorm2d: SwitchableBatchNorm2d,
                nn.Linear: DynamicLinear
            })
        self.alter_candidates_of_switchbn(self.candidate_choices)
        self._register_channel_container(model, MutableChannelContainer)
        self._register_mutable_channel(self.mutable_channel)

    def alter_candidates_of_switchbn(self, candidates):
        """Change candidates of SwitchableBatchNorm2d."""
        for channel in self.output_related + self.input_related:
            if isinstance(channel.module, SwitchableBatchNorm2d) and \
                    len(channel.module.candidate_bn) == 0:
                channel.module.init_candidates(candidates)
        self.current_choice = self.max_choice