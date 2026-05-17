# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .....registry import MODELS


@MODELS.register_module()
class MattingHead(nn.Module):
    def __init__(
        self,
        in_channels: int = 768,
        upsample_channels: List[int] = [768, 384, 192, 96],
        conv_out_channels: Optional[Sequence[int]] = None,
        conv_kernel_sizes: Optional[Sequence[int]] = None,
        out_channels: int = 1,
        loss_decode=dict(type="L1Loss", loss_weight=1.0),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self._build_network(upsample_channels, conv_out_channels, conv_kernel_sizes)
        in_channels = (
            conv_out_channels[-1] if conv_out_channels else upsample_channels[-1]
        )
        self.conv_matting = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        if isinstance(loss_decode, dict):
            self.loss_decode = MODELS.build(loss_decode)
        elif isinstance(loss_decode, (list, tuple)):
            self.loss_decode = nn.ModuleList()
            for loss in loss_decode:
                self.loss_decode.append(MODELS.build(loss))
        else:
            raise TypeError(
                f"loss_decode must be a dict or sequence of dict,\
                but got {type(loss_decode)}"
            )

        self._init_weights()

    def _build_network(
        self,
        upsample_channels: List[int],
        conv_out_channels: Optional[Sequence[int]],
        conv_kernel_sizes: Optional[Sequence[int]],
    ) -> None:
        in_channels = self.in_channels

        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.InstanceNorm2d(in_channels),  # Normalize first
            nn.SiLU(inplace=True),
        )

        # Progressive upsampling blocks
        up_blocks = []
        cur_ch = in_channels
        for out_ch in upsample_channels:
            up_blocks.append(
                nn.Sequential(
                    nn.Conv2d(cur_ch, out_ch * 4, kernel_size=3, padding=1),
                    nn.PixelShuffle(2),  # ↑ spatial ×2
                    nn.InstanceNorm2d(out_ch),
                    nn.SiLU(inplace=True),
                )
            )

            cur_ch = out_ch
        self.upsample_blocks = nn.Sequential(*up_blocks)

        # optional extra conv layers
        conv_layers = []
        if conv_out_channels and conv_kernel_sizes:
            for out_ch, k in zip(conv_out_channels, conv_kernel_sizes):
                conv_layers.extend(
                    [
                        nn.Conv2d(cur_ch, out_ch, k, padding=(k - 1) // 2),
                        nn.InstanceNorm2d(out_ch),
                        nn.SiLU(inplace=True),
                    ]
                )
                cur_ch = out_ch

        self.conv_layers = nn.Sequential(*conv_layers)

    def _init_weights(self) -> None:
        """Initialize network weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                weight_dtype = m.weight.dtype
                weight = nn.init.kaiming_normal_(
                    m.weight.float(), mode="fan_out", nonlinearity="relu"
                )
                m.weight.data = weight.to(weight_dtype)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                weight_dtype = m.weight.dtype
                weight = nn.init.kaiming_normal_(
                    m.weight.float(), mode="fan_in", nonlinearity="linear"
                )
                m.weight.data = weight.to(weight_dtype)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Union[torch.Tensor, Tuple[torch.Tensor]]) -> torch.Tensor:
        x = self.input_conv(x)
        x = self.upsample_blocks(x)
        x = self.conv_layers(x)
        out = self.conv_matting(x)
        return out.sigmoid()

    def loss(
        self,
        outputs: torch.Tensor,
        data_samples: dict,
    ) -> dict:
        gt_alpha = data_samples["gt_alpha"]
        gt_foreground = data_samples["gt_foreground"]
        valid_mask = data_samples["mask"]

        if outputs.shape[2:] != gt_alpha.shape[2:]:
            outputs = F.interpolate(
                input=outputs,
                size=gt_alpha.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        # output could be rgba or alpha only
        pred_alpha = outputs[:, -1:, :, :]  ## [B, 1, H, W]
        pred_foreground = outputs[:, :3, :, :] if outputs.shape[1] == 4 else None

        loss = dict()

        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
        else:
            losses_decode = self.loss_decode

        def _accumulate_loss(
            loss_dict: Dict[str, Any],
            key: str,
            value: Any,
        ) -> None:
            if key in loss_dict:
                loss_dict[key] += value
            else:
                loss_dict[key] = value

        for loss_decode in losses_decode:
            loss_name = loss_decode.loss_name
            # Alpha loss
            _accumulate_loss(
                loss,
                loss_name + "_pha",
                loss_decode(pred_alpha, gt_alpha),
            )
            # Foreground loss - some data may not have GT foreground,
            # so we use mask to skip the loss calculation for semi-transparent areas
            if pred_foreground is not None:
                _accumulate_loss(
                    loss,
                    loss_name + "_fgr",
                    loss_decode(pred_foreground, gt_foreground, valid_mask=valid_mask),
                )

        return loss, outputs
