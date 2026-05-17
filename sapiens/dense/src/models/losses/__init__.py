# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .albedo_loss import AlbedoGradL1Loss
from .l1_loss import L1Loss
from .matting_loss import MattingGradLoss, MattingL1Loss, MattingLaplacianLoss
from .normal_loss import NormalCosineSimilarityLoss, NormalGradL1Loss
from .pointmap_loss import (
    PointmapIntrinsicsConsistencyLoss,
    PointmapNormalLoss,
    PointmapScaleL1Loss,
    PointmapShiftInvariantL1Loss,
)
from .seg_loss import CrossEntropyLoss, DiceLoss
from .silog_loss import SiLogLoss

__all__ = [
    "L1Loss",
    "PointmapIntrinsicsConsistencyLoss",
    "PointmapNormalLoss",
    "PointmapScaleL1Loss",
    "PointmapShiftInvariantL1Loss",
    "SiLogLoss",
    "CrossEntropyLoss",
    "DiceLoss",
    "NormalCosineSimilarityLoss",
    "NormalGradL1Loss",
    "AlbedoGradL1Loss",
    "MattingL1Loss",
    "MattingGradLoss",
    "MattingLaplacianLoss",
]
