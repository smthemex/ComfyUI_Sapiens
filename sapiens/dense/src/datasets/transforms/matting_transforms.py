# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import random
from typing import Dict, Final, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch
from .....engine.datasets import BaseTransform, to_tensor
from .....registry import TRANSFORMS


MAT_REC2020_TO_XYZ: Final[np.ndarray] = np.array(
    [
        [6.3695806e-01, 2.6270020e-01, 3.5174702e-10],
        [1.4461690e-01, 6.7799807e-01, 2.8072692e-02],
        [1.6888097e-01, 5.9301715e-02, 1.0609850e00],
    ],
    dtype=np.float32,
)

MAT_XYZ_TO_REC709: Final[np.ndarray] = np.array(
    [
        [3.2404542, -0.969266, 0.0556434],
        [-1.5371385, 1.8760108, -0.2040259],
        [-0.4985314, 0.041556, 1.0572252],
    ],
    dtype=np.float32,
)


@TRANSFORMS.register_module()
class MattingRandomFlip(BaseTransform):
    def __init__(self, prob=0.5) -> None:
        super().__init__()
        self.prob = prob

    def _flip(self, results: dict) -> None:
        """Flip images, masks, depth maps and adjust camera parameters."""
        # Flip image
        results["img"] = cv2.flip(results["img"], 1)  # 1 for horizontal flip
        results["alpha"] = cv2.flip(results["alpha"], 1)

        if "fgr" in results:
            results["fgr"] = cv2.flip(results["fgr"], 1)  # 1 for horizontal flip

    def transform(self, results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        if np.random.rand() < self.prob:
            self._flip(results)
        return results


@TRANSFORMS.register_module()
class MattingRandomCrop(BaseTransform):
    def __init__(self, crop_size: Tuple[int, int], prob: float = 0.5):
        super().__init__()

        assert len(crop_size) == 2 and crop_size[0] > 0 and crop_size[1] > 0, (
            f"Invalid crop size: {crop_size}"
        )

        self.crop_size = crop_size
        self.prob = prob

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(crop_size={self.crop_size}, prob={self.prob})"
        )

    def _get_crop_bbox(self, img: np.ndarray, crop_h: int, crop_w: int) -> tuple:
        """Randomly generate a crop bounding box for an image given target (h, w)."""
        h, w = img.shape[:2]

        # Ensure the target crop is not bigger than the image
        crop_h = min(crop_h, h)
        crop_w = min(crop_w, w)

        margin_h = h - crop_h
        margin_w = w - crop_w

        # Random top-left corner
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)

        y1, y2 = offset_h, offset_h + crop_h
        x1, x2 = offset_w, offset_w + crop_w
        return (y1, y2, x1, x2)

    def _crop_img(self, img: np.ndarray, crop_bbox: tuple) -> np.ndarray:
        """Crop image to bbox = (y1, y2, x1, x2)."""
        y1, y2, x1, x2 = crop_bbox
        return img[y1:y2, x1:x2, ...]

    def transform(self, results: dict) -> dict:
        # Decide whether to apply cropping
        if np.random.rand() >= self.prob:
            return results  # skip cropping

        img = results["img"]
        # Pick one (h, w) from the list of possible crop sizes

        crop_h, crop_w = self.crop_size

        # Generate the crop bounding box
        crop_bbox = self._get_crop_bbox(img, crop_h, crop_w)

        # Apply to the main image
        cropped_img = self._crop_img(img, crop_bbox)

        results["img"] = cropped_img
        results["img_shape"] = cropped_img.shape[:2]

        # Crop other maps if they exist
        for key in ["alpha", "fgr"]:
            if key in results:
                results[key] = self._crop_img(results[key], crop_bbox)

        return results


@TRANSFORMS.register_module()
class MattingResize(BaseTransform):
    def __init__(
        self,
        height=1024,
        width=768,
        keep_ratio=False,
        test_mode: bool = False,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.keep_ratio = keep_ratio
        self.test_mode = test_mode

    def transform(self, results: Dict) -> Dict:
        img = results["img"]
        h, w = img.shape[:2]

        target_height = self.height
        target_width = self.width

        if self.keep_ratio is True:
            scale_factor = min(target_width / w, target_height / h)
            new_w = int(round(w * scale_factor))
            new_h = int(round(h * scale_factor))

        else:
            new_w = target_width
            new_h = target_height

        dsize = (new_w, new_h)

        # Use INTER_AREA for shrinking and INTER_CUBIC for enlarging
        # to get antialiased results.
        img_interpolation = cv2.INTER_AREA if new_w < w else cv2.INTER_LINEAR

        # Update the results dictionary
        results["img"] = cv2.resize(img, dsize, interpolation=img_interpolation)
        results["img_shape"] = results["img"].shape[:2]

        # Resize gt alpha/foreground if training
        if "alpha" in results and self.test_mode is False:
            results["alpha"] = cv2.resize(
                results["alpha"], dsize, interpolation=cv2.INTER_LINEAR
            )
        if "fgr" in results and self.test_mode is False:
            results["fgr"] = cv2.resize(
                results["fgr"], dsize, interpolation=cv2.INTER_LINEAR
            )
        return results


@TRANSFORMS.register_module()
class MattingRandomResize(BaseTransform):
    def __init__(
        self,
        base_height=1024,
        base_width=768,
        ratio_range=(0.4, 2.0),
        keep_ratio=True,
    ):
        super().__init__()
        self.base_height = base_height
        self.base_width = base_width
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio
        self.resizer = MattingResize(
            height=self.base_height, width=self.base_width, keep_ratio=keep_ratio
        )

    def transform(self, results: dict) -> dict:
        min_ratio, max_ratio = self.ratio_range
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        self.resizer.height = int(self.base_height * ratio)
        self.resizer.width = int(self.base_width * ratio)
        return self.resizer.transform(results)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"base_height={self.base_height}, "
            f"base_width={self.base_width}, "
            f"ratio_range={self.ratio_range}, "
            f"keep_ratio={self.keep_ratio})"
        )


@TRANSFORMS.register_module()
class MattingRandomRotate(BaseTransform):
    def __init__(
        self,
        prob=0.5,
        degree=60,
        pad_val=0,
        seg_pad_val=0,
    ):
        super().__init__()
        self.prob = prob
        assert prob >= 0 and prob <= 1
        assert degree > 0, f"degree {degree} should be positive"
        self.degree = (-degree, degree)
        assert len(self.degree) == 2, (
            f"degree {self.degree} should be a tuple of (min, max)"
        )
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        degree = random.uniform(min(*self.degree), max(*self.degree))
        img = results["img"]
        alpha = results["alpha"]
        h, w = img.shape[:2]
        center = (w / 2, h / 2)

        M = cv2.getRotationMatrix2D(center, degree, 1.0)

        results["img"] = cv2.warpAffine(
            img, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=self.pad_val
        )

        # Rotate the alpha map and foreground
        results["alpha"] = cv2.warpAffine(
            alpha,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderValue=self.seg_pad_val,
        )
        if "fgr" in results:
            results["fgr"] = cv2.warpAffine(
                results["fgr"],
                M,
                (w, h),
                flags=cv2.INTER_LINEAR,
                borderValue=0,
            )
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(prob={self.prob}, "
            f"degree={self.degree}, "
            f"pad_val={self.pad_val}, "
            f"seg_pad_val={self.seg_pad_val}, "
        )
        return repr_str


@TRANSFORMS.register_module()
class MattingPhotoMetricDistortion(BaseTransform):
    def __init__(
        self,
        brightness_delta: int = 32,
        contrast_range: Sequence[float] = (0.5, 1.5),
        saturation_range: Sequence[float] = (0.5, 1.5),
        hue_delta: int = 18,
        prob: float = 0.5,
    ):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
        self.prob = prob

    def convert(
        self,
        img: np.ndarray | None,
        alpha: int = 1,
        beta: int = 0,
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Multiple with alpha and add beta with clip.

        Args:
            img (np.ndarray): The input image.
            alpha (int): Image weights, change the contrast/saturation
                of the image. Default: 1
            beta (int): Image bias, change the brightness of the
                image. Default: 0
            fgr (Optional[np.ndarray]): Optional foreground image to apply
                the same transformation. Default: None

        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: The transformed image,
                or tuple of (transformed img, transformed fgr) if fgr is provided.
        """
        if img is None:
            return None

        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        img = img.astype(np.uint8)

        return img

    def brightness(
        self, img: np.ndarray, fgr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Brightness distortion.

        Args:
            img (np.ndarray): The input image.
            fgr (Optional[np.ndarray]): Optional foreground image to apply
                the same transformation. Default: None
        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: Image after brightness change,
                or tuple of (transformed img, transformed fgr) if fgr is provided.
        """

        if random.randint(0, 1):
            beta = random.uniform(-self.brightness_delta, self.brightness_delta)
            img = self.convert(img, beta=beta)
            fgr = self.convert(fgr, beta=beta)

        return img, fgr

    def contrast(
        self, img: np.ndarray, fgr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Contrast distortion.

        Args:
            img (np.ndarray): The input image.
            fgr (Optional[np.ndarray]): Optional foreground image to apply
                the same transformation. Default: None
        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: Image after contrast change,
                or tuple of (transformed img, transformed fgr) if fgr is provided.
        """

        if random.randint(0, 1):
            alpha = random.uniform(self.contrast_lower, self.contrast_upper)
            img = self.convert(img, alpha=alpha)
            fgr = self.convert(fgr, alpha=alpha)
        return img, fgr

    def saturation(
        self, img: np.ndarray, fgr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Saturation distortion.

        Args:
            img (np.ndarray): The input image.
            fgr (Optional[np.ndarray]): Optional foreground image to apply
                the same transformation. Default: None
        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: Image after saturation change,
                or tuple of (transformed img, transformed fgr) if fgr is provided.
        """

        if random.randint(0, 1):
            alpha = random.uniform(self.saturation_lower, self.saturation_upper)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 1] = self.convert(img_hsv[:, :, 1], alpha=alpha)
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

            if fgr is not None:
                fgr_hsv = cv2.cvtColor(fgr, cv2.COLOR_BGR2HSV)
                fgr_hsv[:, :, 1] = self.convert(fgr_hsv[:, :, 1], alpha=alpha)
                fgr = cv2.cvtColor(fgr_hsv, cv2.COLOR_HSV2BGR)

        return img, fgr

    def hue(
        self, img: np.ndarray, fgr: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray | None]:
        """Hue distortion.

        Args:
            img (np.ndarray): The input image.
            fgr (Optional[np.ndarray]): Optional foreground image to apply
                the same transformation. Default: None
        Returns:
            np.ndarray or Tuple[np.ndarray, np.ndarray]: Image after hue change,
                or tuple of (transformed img, transformed fgr) if fgr is provided.
        """

        if random.randint(0, 1):
            delta = random.randint(-self.hue_delta, self.hue_delta)
            img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img_hsv[:, :, 0] = (img_hsv[:, :, 0].astype(int) + delta) % 180
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
            if fgr is not None:
                fgr_hsv = cv2.cvtColor(fgr, cv2.COLOR_BGR2HSV)
                fgr_hsv[:, :, 0] = (fgr_hsv[:, :, 0].astype(int) + delta) % 180
                fgr = cv2.cvtColor(fgr_hsv, cv2.COLOR_HSV2BGR)

        return img, fgr

    def transform(self, results: dict) -> dict:
        """Transform function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        fgr = results.get("fgr", None)

        # random brightness
        img, fgr = self.brightness(img, fgr)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(0, 1)
        if mode == 1:
            img, fgr = self.contrast(img, fgr)

        # random saturation
        img, fgr = self.saturation(img, fgr)

        # random hue
        img, fgr = self.hue(img, fgr)

        # random contrast
        if mode == 0:
            img, fgr = self.contrast(img, fgr)

        results["img"] = img

        if fgr is not None:
            fgr[results["alpha"] == 0] = 0
            results["fgr"] = fgr

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(brightness_delta={self.brightness_delta}, "
            f"contrast_range=({self.contrast_lower}, "
            f"{self.contrast_upper}), "
            f"saturation_range=({self.saturation_lower}, "
            f"{self.saturation_upper}), "
            f"hue_delta={self.hue_delta})"
        )
        return repr_str


@TRANSFORMS.register_module()
class MattingPackInputs(BaseTransform):
    def __init__(
        self,
        test_mode: bool = False,
        meta_keys=(
            "img_path",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "flip",
        ),
    ):
        super().__init__()
        self.test_mode = test_mode
        self.meta_keys = meta_keys

    def transform(self, results: dict) -> dict:
        packed_results = dict()
        if "img" in results:
            img = results["img"]
            if len(img.shape) < 3:
                img = np.expand_dims(img, -1)
            if not img.flags.c_contiguous:
                img = to_tensor(np.ascontiguousarray(img.transpose(2, 0, 1)))
            else:
                img = img.transpose(2, 0, 1)
                img = to_tensor(img).contiguous()
            packed_results["inputs"] = img

        data_sample = dict()

        if "alpha" in results:
            data_sample["gt_alpha"] = to_tensor(results["alpha"].squeeze()[None, ...])
        if "fgr" in results:
            gt_fgr = results["fgr"].astype(np.float32) / 255.0
            gt_fgr = np.ascontiguousarray(gt_fgr.transpose(2, 0, 1)[[2, 1, 0], ...])
            data_sample["gt_foreground"] = to_tensor(gt_fgr)
            data_sample["has_foreground"] = True
            data_sample["mask"] = torch.full(
                (1, data_sample["gt_alpha"].shape[1], data_sample["gt_alpha"].shape[2]),
                True,
                dtype=torch.bool,
            )
        else:
            if not self.test_mode:
                gt_fgr = torch.zeros(
                    3,
                    data_sample["gt_alpha"].shape[1],
                    data_sample["gt_alpha"].shape[2],
                    dtype=data_sample["gt_alpha"].dtype,
                )
                mask = data_sample["gt_alpha"][0] >= 0.996  # 254/255
                # assign colors for fully opaque foreground
                img_rgb = img[[2, 1, 0], ...].to(torch.float32) / 255.0
                gt_fgr[:, mask] = img_rgb[:, mask]

                data_sample["gt_foreground"] = gt_fgr
                # fully opaque or fully transparent areas
                data_sample["mask"] = (data_sample["gt_alpha"] >= 0.996) | (
                    data_sample["gt_alpha"] == 0.0
                )
            data_sample["has_foreground"] = False

        img_meta = {}
        for key in self.meta_keys:
            if key in results:
                if isinstance(results[key], (int, float)):
                    img_meta[key] = np.float32(results[key])
                elif isinstance(results[key], np.ndarray):
                    img_meta[key] = results[key].astype(np.float32)
                else:
                    img_meta[key] = results[key]
        data_sample["meta"] = img_meta
        packed_results["data_samples"] = data_sample

        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f"(meta_keys={self.meta_keys})"
        return repr_str


@TRANSFORMS.register_module()
class MattingRandomBackground(BaseTransform):
    """Random background transform for alpha matting.

    Composites the pre-multiplied foreground (`results["fgr"]`) over a random
    background image in linear color space.

    Args:
        background_images_root (Union[str, List[str]]): Root directory or list of
            directories containing background images.
        prob (float): Probability of applying the transformation. Defaults to 0.5.
    """

    def __init__(
        self,
        background_images_root: Union[str, List[str]] = [],
        prob: float = 0.5,
    ):
        super().__init__()

        self.prob = prob
        if isinstance(background_images_root, str):
            background_images_root = [background_images_root]

        self.background_images = []
        for root in background_images_root:
            self.background_images += [
                os.path.join(root, image_name)
                for image_name in os.listdir(root)
                if image_name.endswith(".jpg") or image_name.endswith(".png")
            ]

    @staticmethod
    def _srgb2linear(x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)

    @staticmethod
    def _linear2srgb(x: np.ndarray) -> np.ndarray:
        return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1.0 / 2.4) - 0.055)

    @staticmethod
    def _resize_background(
        background_image: np.ndarray, target_height: int, target_width: int
    ) -> np.ndarray:
        background_height, background_width = background_image.shape[:2]

        # Resize to match target height while preserving aspect ratio
        new_background_height = target_height
        new_background_width = int(
            new_background_height * background_width / background_height
        )
        background_image = cv2.resize(
            background_image, (new_background_width, new_background_height)
        )

        # Center-crop to target width
        if new_background_width > target_width:
            start_x = (new_background_width - target_width) // 2
            end_x = start_x + target_width
            background_image = background_image[:, start_x:end_x]

        # Final resize if dimensions don't match exactly
        if (
            background_image.shape[0] != target_height
            or background_image.shape[1] != target_width
        ):
            background_image = cv2.resize(
                background_image, (target_width, target_height)
            )

        return background_image

    def _composite(
        self, foreground: np.ndarray, background: np.ndarray, alpha: np.ndarray
    ) -> np.ndarray:
        fg_linear = self._srgb2linear(foreground / 255.0)
        bg_linear = self._srgb2linear(background / 255.0)
        composite_linear = fg_linear + (1 - alpha) * bg_linear
        composite_srgb = self._linear2srgb(composite_linear)
        return (composite_srgb * 255.0).astype(np.uint8)

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        image = results["fgr"]  # BGR image [0-255]
        alpha = np.expand_dims(results["alpha"], -1)
        image_height, image_width = image.shape[:2]

        background_image_path = random.choice(self.background_images)

        background_image = cv2.imread(background_image_path)
        background_image = self._resize_background(
            background_image, image_height, image_width
        )

        results["img"] = self._composite(image, background_image, alpha)

        return results


@TRANSFORMS.register_module()
class MattingRandomJPEGCompression(BaseTransform):
    """Simulate JPEG compression with random quality to introduce artifacts.

    Args:
        prob (float): Probability of applying the transform. Default 0.5.
        quality_range (tuple[int, int]): The min and max compression quality.
            Lower means heavier compression artifacts. E.g. (30, 60).
    """

    def __init__(self, prob=0.4, quality_range=(30, 60)):
        super().__init__()
        self.prob = prob
        self.quality_range = quality_range

    def transform(self, results: dict) -> dict:
        if np.random.rand() > self.prob:
            return results

        img = results["img"]
        q_min, q_max = self.quality_range
        quality = np.random.randint(q_min, q_max + 1)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        success, enc_img = cv2.imencode(".jpg", img, encode_param)
        if success:
            dec_img = cv2.imdecode(enc_img, cv2.IMREAD_COLOR)
            results["img"] = dec_img
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(prob={self.prob}, "
            f"quality_range={self.quality_range})"
        )


@TRANSFORMS.register_module()
class MattingCropAlphaBBox(BaseTransform):
    """Crop image and related maps to the bounding box of the alpha mask.

    This transform finds the bounding box of non-zero alpha regions and crops
    the image, alpha, and foreground (if present) to that region.

    Args:
        threshold (float): Alpha threshold for determining foreground pixels.
            Pixels with alpha >= threshold are considered foreground.
            Defaults to 0.0 (any non-zero alpha).
        padding_ratio (Tuple[float, float]): Range of random padding as a
            percentage of bbox width (min_ratio, max_ratio). For example,
            (0.0, 0.1) means padding will be randomly sampled between
            0% and 10% of the bbox width. Defaults to (0.0, 0.0) (no padding).
    """

    def __init__(
        self,
        threshold: float = 0.0,
        padding_ratio: Tuple[float, float] = (0.0, 0.1),
        padding_color: float = 255.0,
        prob: float = 0.5,
    ):
        super().__init__()
        self.threshold = threshold
        self.padding_ratio = padding_ratio
        self.padding_color = padding_color
        self.prob = prob

    def _get_alpha_bbox(self, alpha: np.ndarray) -> Tuple[int, int, int, int]:
        """Get bounding box of non-zero alpha region.

        Args:
            alpha: Alpha mask with values in [0, 1].

        Returns:
            Tuple of (x1, y1, x2, y2) defining the bounding box.
        """
        mask = alpha > self.threshold
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            h, w = alpha.shape[:2]
            return (0, 0, w, h)

        y1, y2 = np.where(rows)[0][[0, -1]]
        x1, x2 = np.where(cols)[0][[0, -1]]

        y2 += 1
        x2 += 1

        return (x1, y1, x2, y2)

    def _apply_padding(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        img_h: int,
        img_w: int,
    ) -> Tuple[int, int, int, int]:
        """Apply random padding based on bbox width percentage."""
        min_ratio, max_ratio = self.padding_ratio
        if max_ratio <= 0:
            return (x1, y1, x2, y2)

        bbox_w = x2 - x1
        ratio = random.uniform(min_ratio, max_ratio)
        padding = int(bbox_w * ratio)

        if padding > 0:
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(img_w, x2 + padding)
            y2 = min(img_h, y2 + padding)

        return (x1, y1, x2, y2)

    def _pad_to_aspect_ratio(
        self,
        data: np.ndarray,
        target_ar: float,
        pad_value: float = 255.0,
    ) -> np.ndarray:
        """Pad data to match target aspect ratio (width/height).

        Args:
            data: Input array of shape (H, W) or (H, W, C).
            target_ar: Target aspect ratio (width / height).
            pad_value: Value to use for padding. Defaults to 1.0 (white).

        Returns:
            Padded array with the target aspect ratio.
        """
        h, w = data.shape[:2]
        current_ar = w / h

        if abs(current_ar - target_ar) < 1e-6:
            return data

        if current_ar < target_ar:
            # Too tall, need to pad width
            new_w = int(h * target_ar)
            pad_total = new_w - w
            pad_left = pad_total // 2

            if data.ndim == 2:
                padded = np.full((h, new_w), pad_value, dtype=data.dtype)
                padded[:, pad_left : pad_left + w] = data
            else:
                c = data.shape[2]
                padded = np.full((h, new_w, c), pad_value, dtype=data.dtype)
                padded[:, pad_left : pad_left + w, :] = data
        else:
            # Too wide, need to pad height
            new_h = int(w / target_ar)
            pad_total = new_h - h
            pad_top = pad_total // 2

            if data.ndim == 2:
                padded = np.full((new_h, w), pad_value, dtype=data.dtype)
                padded[pad_top : pad_top + h, :] = data
            else:
                c = data.shape[2]
                padded = np.full((new_h, w, c), pad_value, dtype=data.dtype)
                padded[pad_top : pad_top + h, :, :] = data

        return padded

    def _crop(self, data: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Crop data to bounding box."""
        y1, y2, x1, x2 = bbox
        return data[y1:y2, x1:x2, ...]

    def transform(self, results: dict) -> dict:
        if random.random() > self.prob:
            return results

        alpha = results["alpha"]
        img = results["img"]
        img_h, img_w = alpha.shape[:2]
        target_ar = img_w / img_h

        x1, y1, x2, y2 = self._get_alpha_bbox(alpha)
        x1, y1, x2, y2 = self._apply_padding(x1, y1, x2, y2, img_h, img_w)
        bbox = (y1, y2, x1, x2)  # Convert to crop format (y1, y2, x1, x2)

        # Crop
        cropped_img = self._crop(img, bbox)
        cropped_alpha = self._crop(alpha, bbox)

        results["img"] = self._pad_to_aspect_ratio(
            cropped_img, target_ar, self.padding_color
        )
        results["alpha"] = self._pad_to_aspect_ratio(cropped_alpha, target_ar, 0.0)
        results["img_shape"] = results["img"].shape[:2]

        if "fgr" in results:
            cropped_fgr = self._crop(results["fgr"], bbox)
            results["fgr"] = self._pad_to_aspect_ratio(cropped_fgr, target_ar, 0.0)

        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(threshold={self.threshold}, "
            f"padding_ratio={self.padding_ratio})"
        )
