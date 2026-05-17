# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import os
from typing import List

import cv2
import numpy as np
from .....registry import DATASETS

from .matting_base_dataset import (
    MattingBaseDataset,
    _alpha_to_float,
    _read_image,
    _to_uint8,
)


##-----------------------------------------------------------------------
@DATASETS.register_module()
class MattingGSSDataset(MattingBaseDataset):
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.
        modify this function to load a list of all sample information in training

        Returns:
            list[dict]: All data info of dataset.
        """
        with open(self.ann_file, "r", encoding="utf-8") as f:
            image_paths = [line.strip() for line in f if line.strip()]

        data_list = []
        data_root = self.data_root or ""
        for file_path in image_paths:
            data_list.append(
                {
                    "image_path": os.path.join(data_root, file_path),
                }
            )

        print(
            "\033[92mDone! {}. Loaded total samples: {}. Test mode: {}\033[0m".format(
                self.__class__.__name__, len(data_list), self.test_mode
            )
        )

        return data_list

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        try:
            img = _read_image(data_info["image_path"], cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Error loading image {data_info}: {e}")
            return None

        if img.ndim != 3 or img.shape[-1] != 4:
            print(f"Expected BGRA image for {data_info}, got shape {img.shape}")
            return None

        bgr = _to_uint8(img[..., :3])

        mask = _alpha_to_float(img[:, :, 3])
        fgr = (bgr.astype(np.float32) * mask[..., None]).astype(np.uint8)

        data_info = {
            "img": bgr,
            "img_id": "",
            "img_path": data_info["image_path"],
            "alpha": mask,
            "fgr": fgr,
            "id": idx,
        }

        return data_info
