# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import os
from typing import List

import cv2
import numpy as np
from .....engine.datasets import BaseDataset
from .....registry import DATASETS


def _read_image(path: str, flags: int) -> np.ndarray:
    image = cv2.imread(path, flags)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


def _to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image
    if image.dtype == np.uint16:
        return np.round(image.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    if np.issubdtype(image.dtype, np.floating):
        image = np.nan_to_num(image.astype(np.float32), nan=0.0, posinf=1.0)
        if image.size and float(image.max()) <= 1.0:
            image = image * 255.0
        return image.clip(0, 255).astype(np.uint8)
    return image.clip(0, 255).astype(np.uint8)


def _alpha_to_float(alpha: np.ndarray) -> np.ndarray:
    alpha_float = alpha.astype(np.float32)
    if np.issubdtype(alpha.dtype, np.integer):
        alpha_float /= float(np.iinfo(alpha.dtype).max)
    elif alpha_float.size and float(alpha_float.max()) > 1.0:
        alpha_float /= 255.0
    return alpha_float.clip(0.0, 1.0)


##-----------------------------------------------------------------------
@DATASETS.register_module()
class MattingBaseDataset(BaseDataset):
    def __init__(self, ann_file, **kwargs) -> None:
        self.ann_file = ann_file
        super().__init__(**kwargs)
        return

    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.
        modify this function to load a list of all sample information in training

        Returns:
            list[dict]: All data info of dataset.
        """
        with open(self.ann_file, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        if not isinstance(raw_data, list):
            raise ValueError(f"Expected a JSON list in annotation file: {self.ann_file}")

        data_root = self.data_root or ""

        data_list = [
            {
                "image_path": os.path.join(data_root, sample["image"]),
                "mask_path": os.path.join(data_root, sample["mask"]),
            }
            for sample in raw_data
        ]

        print(
            "\033[92mDone! {}. Loaded total samples: {}. Test mode: {}\033[0m".format(
                self.__class__.__name__, len(data_list), self.test_mode
            )
        )

        return data_list

    def get_data_info(self, idx):
        data_info = copy.deepcopy(self.data_list[idx])

        try:
            img = _to_uint8(_read_image(data_info["image_path"], cv2.IMREAD_COLOR))
            mask = _read_image(data_info["mask_path"], cv2.IMREAD_UNCHANGED)
        except Exception as e:
            print(f"Error loading image/mask {data_info}: {e}")
            return None

        fgr = None
        if mask.ndim == 3 and mask.shape[-1] == 4:
            # cv2.IMREAD_UNCHANGED returns BGRA for 4-channel PNGs
            fgr = _to_uint8(mask[..., :3])  # bgr
            alpha = _alpha_to_float(mask[..., -1])

            fgr = (fgr.astype(np.float32) * alpha[..., None]).astype(np.uint8)
        elif mask.ndim == 3 and mask.shape[-1] in (1, 3):
            # mask could be HxWx3 with same values in each channel
            alpha = _alpha_to_float(mask[..., 0])
        elif mask.ndim == 2:
            alpha = _alpha_to_float(mask)
        else:
            print(f"Unexpected mask shape {mask.shape} for {data_info}")
            return None

        data_info = {
            "img": img,
            "img_id": "",
            "img_path": data_info["image_path"],
            "alpha": alpha,
            "id": idx,
        }

        if fgr is not None:
            data_info["fgr"] = fgr

        return data_info
