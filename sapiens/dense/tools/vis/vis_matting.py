# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
from argparse import ArgumentParser

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ....pose.src import init_model
from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("config", help="Config file")
    parser.add_argument("checkpoint", help="Checkpoint file")
    parser.add_argument(
        "--input", required=True, help="Input image dir or path list file"
    )
    parser.add_argument("--output", required=True, help="Path to output dir")
    parser.add_argument(
        "--save_pred", action="store_true", help="Save alpha matte as .npy"
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for inference")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # Get image list
    if os.path.isdir(args.input):
        image_paths = [
            os.path.join(args.input, name)
            for name in sorted(os.listdir(args.input))
            if name.endswith((".jpg", ".png", ".jpeg"))
        ]
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            image_paths = [line.strip() for line in f if line.strip()]

    if not image_paths:
        return

    model = init_model(args.config, args.checkpoint, device=args.device)

    for image_path in tqdm(image_paths, total=len(image_paths)):
        image_name = os.path.basename(image_path)
        image = cv2.imread(image_path)  ## BGR
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        ##------------------------------------------
        data = model.pipeline(dict(img=image))  ## resize
        data = model.data_preprocessor(data)  ## normalize, add batch dim and cast
        inputs = data["inputs"]

        with torch.no_grad():
            outputs = model(inputs)  ## 1 x 4 x H x W: [fgr_rgb, alpha]

        outputs = F.interpolate(
            outputs,
            size=(image.shape[0], image.shape[1]),
            mode="bilinear",
            align_corners=False,
        )

        outputs = outputs.squeeze(0).float().cpu().numpy()  ## 4 x H x W
        fgr_rgb = outputs[0:3].clip(0, 1).transpose(1, 2, 0)  ## H x W x 3, RGB premult
        alpha = outputs[3].clip(0, 1)  ## H x W

        # ------------------------------------------
        alpha_vis = (alpha * 255).astype(np.uint8)
        alpha_vis_3ch = np.stack([alpha_vis] * 3, axis=-1)

        # Composite the predicted (pre-multiplied) foreground over a chroma-green
        # background.
        fgr_bgr = fgr_rgb[:, :, ::-1]  ## RGB -> BGR
        bg_bgr = np.array([64, 177, 0], dtype=np.float32) / 255.0  ## chroma green
        composite = fgr_bgr + (1 - alpha[..., None]) * bg_bgr
        composite = (composite.clip(0, 1) * 255).astype(np.uint8)

        vis_image = np.concatenate([image, alpha_vis_3ch, composite], axis=1)
        save_path = os.path.join(args.output, image_name)
        cv2.imwrite(save_path, vis_image)

        if args.save_pred:
            base = os.path.splitext(image_name)[0]
            np.save(os.path.join(args.output, f"{base}_alpha.npy"), alpha)


if __name__ == "__main__":
    main()
