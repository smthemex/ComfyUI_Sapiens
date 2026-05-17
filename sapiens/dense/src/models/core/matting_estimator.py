# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Tuple, Union

import torch
from .....engine.models import BaseModel
from .....registry import MODELS
from torch import Tensor


@MODELS.register_module()
class MattingEstimator(BaseModel):
    def __init__(
        self,
        backbone: dict = None,
        decode_head: dict = None,
        init_cfg: dict = None,
    ):
        BaseModel.__init__(self, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.decode_head = MODELS.build(decode_head)

        ## Load init weights
        self.init_weights()

    def loss(
        self, outputs: Union[Tensor, Tuple[Tensor, ...]], data_samples: dict
    ) -> Tuple[dict, Tensor]:
        losses, fgr_pha = self.decode_head.loss(outputs, data_samples)
        parsed_losses, log_vars = self.parse_losses(losses)
        log_vars["outputs"] = fgr_pha
        return parsed_losses, log_vars

    def forward(self, inputs: Tensor) -> Tensor:
        ## backbone forward returns a list of tensors at different depths, 0 is the final layer
        if self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.backbone, inputs, use_reentrant=False
            )[0]
        else:
            x = self.backbone(inputs)[0]

        x = self.decode_head(x)  ## B x out_channels x H x W
        return x
