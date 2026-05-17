# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from pathlib import Path

import cv2
import numpy as np
import torch
from ....registry import VISUALIZERS
from torch import nn


@VISUALIZERS.register_module()
class MattingVisualizer(nn.Module):
    def __init__(
        self,
        output_dir: str,
        vis_interval: int = 100,
        vis_max_samples: int = 4,
        vis_image_width: int = 384,
        vis_image_height: int = 512,
    ):
        super().__init__()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.vis_max_samples = vis_max_samples
        self.vis_interval = vis_interval
        self.vis_image_width = vis_image_width
        self.vis_image_height = vis_image_height

    def add_batch(self, data_batch: dict, logs: dict, step: int):
        inputs = data_batch["inputs"].detach().cpu()  # B x 3 x H x W
        pred_alphas = logs["outputs"][:, -1, :, :].detach().cpu()  # B x H x W
        pred_fgrs = (
            logs["outputs"][:, 0:3, :, :].detach().cpu()
            if logs["outputs"].shape[1] == 4
            else None
        )  # B x 3 x H x W

        gt_alphas = (data_batch["data_samples"]["gt_alpha"].detach().cpu()).squeeze(
            dim=1
        )  # B x H x W
        if pred_fgrs is not None:
            gt_fgrs = data_batch["data_samples"]["gt_foreground"].detach().cpu()
            masks = data_batch["data_samples"]["mask"].detach().cpu().squeeze(dim=1)
        else:
            gt_fgrs = None
            masks = None

        if pred_alphas.dtype == torch.bfloat16:
            inputs = inputs.float()
            pred_alphas = pred_alphas.float()
            pred_fgrs = pred_fgrs.float() if pred_fgrs is not None else None

        batch_size = min(len(inputs), self.vis_max_samples)

        inputs = inputs[:batch_size]
        gt_alphas = gt_alphas[:batch_size].numpy()  # B x H x W
        pred_alphas = pred_alphas[:batch_size].numpy()  # B x H x W
        if pred_fgrs is not None:
            gt_fgrs = gt_fgrs[:batch_size].numpy()  # B x 3 x H x W
            pred_fgrs = pred_fgrs[:batch_size].numpy()  # B x 3 x H x W
            masks = masks[:batch_size].numpy()  # B x H x W

        prefix = os.path.join(self.output_dir, "train")
        suffix = str(step).zfill(6)
        vis_images = []

        for i, (input, gt_alpha, pred_alpha) in enumerate(
            zip(inputs, gt_alphas, pred_alphas)
        ):
            image = input.permute(1, 2, 0).cpu().numpy()  # BGR image
            image = np.ascontiguousarray(image.copy()).astype(np.uint8)

            gt_alpha_vis = (gt_alpha * 255).astype(np.uint8)
            pred_alpha_vis = (pred_alpha * 255).astype(np.uint8)
            pred_alpha_vis = np.stack([pred_alpha_vis] * 3, axis=-1)
            gt_alpha_vis = np.stack([gt_alpha_vis] * 3, axis=-1)

            error_alpha = np.abs(pred_alpha - gt_alpha)  # L1 error
            error_alpha_vis = (error_alpha * 255).astype(np.uint8)  # Scale to [0, 255]
            error_alpha_vis = cv2.applyColorMap(
                error_alpha_vis, cv2.COLORMAP_JET
            )  # Apply colormap for better visualization

            vis_list = [image, gt_alpha_vis, pred_alpha_vis, error_alpha_vis]

            if pred_fgrs is not None:
                gt_fgr_vis = (
                    (gt_fgrs[i] * 255).transpose((1, 2, 0))[:, :, ::-1].astype(np.uint8)
                )  # RGB to BGR
                pred_fgr_vis = (
                    (pred_fgrs[i] * 255)
                    .transpose((1, 2, 0))[:, :, ::-1]
                    .astype(np.uint8)
                )  # RGB to BGR

                error_fgr = np.abs(pred_alpha - gt_alpha) * masks[i]  # L1 error
                error_fgr_vis = (error_fgr * 255).astype(np.uint8)  # Scale to [0, 255]
                error_fgr_vis = cv2.applyColorMap(
                    error_fgr_vis, cv2.COLORMAP_JET
                )  # Apply colormap for better visualization
                vis_list += [gt_fgr_vis, pred_fgr_vis, error_fgr_vis]

            vis_image = np.concatenate(
                vis_list,
                axis=1,
            )
            vis_image = cv2.resize(
                vis_image,
                (len(vis_list) * self.vis_image_width, self.vis_image_height),
                interpolation=cv2.INTER_AREA,
            )
            vis_images.append(vis_image)

        grid_image = np.concatenate(vis_images, axis=0)
        grid_out_file = "{}_{}.jpg".format(prefix, suffix)
        cv2.imwrite(grid_out_file, grid_image)

        return
