# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .matting_base_dataset import MattingBaseDataset
from .matting_gss_dataset import MattingGSSDataset

__all__ = ["MattingBaseDataset", "MattingGSSDataset"]
