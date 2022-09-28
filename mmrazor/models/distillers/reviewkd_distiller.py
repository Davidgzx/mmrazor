# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Optional

from mmrazor.registry import MODELS
from ..algorithms.base import LossResults
from ..architectures.connectors import ABFConnector
from .configurable_distiller import ConfigurableDistiller


@MODELS.register_module()
class ReviewKDDistiller(ConfigurableDistiller):
    """Distiller for ``OverhaulFeatureDistillation``, inherited from
    ``ConfigurableDistiller``, add func:

    ``init_ofd_connectors`` to initialize margin.
    """

    def __init__(
            self,
            connectors,
            #  distill_losses,
            *args,
            **kwargs):
        super().__init__(connectors=connectors, *args, **kwargs)
        self.residual_buffer = {
            c.residual: None
            for c in self.connectors.values()
            if isinstance(c, ABFConnector) and c.residual is not None
        }

        # for name, connector in connectors.items():
        #     assert connector['type'] == 'ABFConnector'
        # for name, loss in distill_losses.items():
        #     assert loss['type'] == 'HCLLoss'

    def get_record(self,
                   recorder: str,
                   from_student: bool,
                   record_idx: int = 0,
                   data_idx: Optional[int] = None,
                   connector: Optional[str] = None,
                   connector_idx: Optional[int] = None) -> List:
        """According to each item in ``record_infos``, get the corresponding
        record in ``recorder_manager``."""

        if from_student:
            recorder_ = self.student_recorders.get_recorder(recorder)
        else:
            recorder_ = self.teacher_recorders.get_recorder(recorder)
        record_data = recorder_.get_record_data(record_idx, data_idx)
        if connector:
            connector_obj = self.connectors[connector]
            if isinstance(connector_obj, ABFConnector):
                if connector_obj.residual is not None:
                    record_data, residual = connector_obj(
                        record_data,
                        self.residual_buffer[connector_obj.residual])
                else:
                    record_data, residual = connector_obj(record_data)
                self.residual_buffer[connector] = residual
            else:
                record_data = self.connectors[connector](record_data)
        if connector_idx is not None:
            record_data = record_data[connector_idx]

        return record_data

    def compute_distill_losses(self) -> LossResults:
        """Compute distill losses automatically."""
        # Record all computed losses' results.
        losses = {}
        for loss_name, forward_mappings in self.loss_forward_mappings.items():
            forward_kwargs = {}
            for forward_key, record in forward_mappings.items():
                forward_var = self.get_record(**record)
                forward_kwargs[forward_key] = forward_var

            loss_module = self.distill_losses[loss_name]
            loss = loss_module(**forward_kwargs)  # type: ignore
            # add computed loss result.
            losses[loss_name] = loss

        return losses
