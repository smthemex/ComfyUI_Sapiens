# !/usr/bin/env python
# -*- coding: UTF-8 -*-
import time
from enum import Enum
from typing import List
import cv2
import numpy as np
import torch
from .common import pose_estimation_preprocessor, TaskType, download_hf_model
from .detector import Detector

from .pose_classes_and_palettes import (
    COCO_KPTS_COLORS,
    COCO_WHOLEBODY_KPTS_COLORS,
    GOLIATH_KPTS_COLORS,
    GOLIATH_SKELETON_INFO,
    GOLIATH_KEYPOINTS
)


class SapiensPoseEstimationType(Enum):
    POSE_ESTIMATION_03B = "sapiens_0.3b_goliath_best_goliath_AP_573.pth"
    POSE_ESTIMATION_06B = "ssapiens_0.6b_goliath_best_goliath_AP_609.pth"
    POSE_ESTIMATION_1B = "sapiens_1b_goliath_best_goliath_AP_639.pth"
    POSE_ESTIMATION_03B_T = "sapiens_0.3b_goliath_best_goliath_AP_575_torchscript.pt2"
    POSE_ESTIMATION_06B_T = "sapiens_0.6b_goliath_best_goliath_AP_609_torchscript.pt2"
    POSE_ESTIMATION_1B_T = "sapiens_1b_goliath_best_goliath_AP_640_torchscript.pt2"
    POSE_ESTIMATION_03B_16 = "sapiens_0.3b_goliath_best_goliath_AP_573_bfloat16.pt2"
    POSE_ESTIMATION_06B_16 = "sapiens_0.6b_goliath_best_goliath_AP_609_bfloat16.pt2"
    POSE_ESTIMATION_1B_16 = "sapiens_1b_goliath_best_goliath_AP_639_bfloat16.pt2"
    OFF="off"
    

class SapiensPoseEstimation:
    def __init__(self,
                 type: SapiensPoseEstimationType = SapiensPoseEstimationType.POSE_ESTIMATION_03B, local_pose="",pt_type="float32_torch",model_dir="",img_size=(1024, 768),use_torchscript=True,show_pose_object=False,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"),
                 dtype: torch.dtype = torch.float32):
        # Load the model
        self.local_pose = local_pose
        self.model_dir = model_dir
        self.pt_type = pt_type
        self.img_size = img_size
        self.use_torchscript = use_torchscript
        self.show_pose_object=show_pose_object
        if self.local_pose:
            path = self.local_pose
        else:
            path = download_hf_model(type.value, TaskType.POSE, self.model_dir, dtype=self.pt_type)
        self.device = device
        self.dtype = dtype
        if self.use_torchscript:
            model = torch.jit.load(path)
            model = model.eval()
            model.to(device).to(dtype)
        else:
            model = torch.export.load(path).module()
            dtype = torch.bfloat16 if self.dtype == "bfloat16" else torch.half
            model.to(dtype)
            model = torch.compile(model, mode="max-autotune", fullgraph=True)
        self.model = model
        self.preprocessor = pose_estimation_preprocessor(input_size=self.img_size)

        # Initialize the YOLO-based detector
        self.detector = Detector()

    def __call__(self, img: np.ndarray) -> (np.ndarray,):
        start = time.perf_counter()

        # Detect persons in the image
        bboxes = self.detector.detect(img)

        # Process the image and estimate the pose
        pose_result_image, keypoints = self.estimate_pose(img, bboxes)

        print(f"Pose estimation inference took: {time.perf_counter() - start:.4f} seconds")
        return pose_result_image, keypoints
    
    def enable_model_cpu_offload(self):
        self.model.to("cpu")
        torch.cuda.empty_cache()
        
    def move_to_cuda(self):
        self.model.to("cuda")

    @torch.inference_mode()
    def estimate_pose(self, img: np.ndarray, bboxes: List[List[float]]) -> (np.ndarray, List[dict]):
        all_keypoints = []
        result_img = img.copy()

        for bbox in bboxes:
            cropped_img = self.crop_image(img, bbox)
            tensor = self.preprocessor(cropped_img).unsqueeze(0).to(self.device).to(self.dtype)

            heatmaps = self.model(tensor).to(torch.float32)
            keypoints = self.heatmaps_to_keypoints(heatmaps[0].cpu().numpy())
            all_keypoints.append(keypoints)
            if not self.show_pose_object:
                # Draw black BG
                empty_cv = np.empty(result_img.shape, dtype=np.uint8)
                empty_cv[:] = (0, 0, 0)
                result_img = self.draw_keypoints(empty_cv, keypoints, bbox)
            else:
                # Draw the keypoints on the original image
                result_img = self.draw_keypoints(result_img, keypoints, bbox)
           
        return result_img, all_keypoints

    def crop_image(self, img: np.ndarray, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        return img[y1:y2, x1:x2]


    def heatmaps_to_keypoints(self, heatmaps: np.ndarray) -> dict:
        keypoints = {}
        for i, name in enumerate(GOLIATH_KEYPOINTS):
            if i < heatmaps.shape[0]:
                y, x = np.unravel_index(np.argmax(heatmaps[i]), heatmaps[i].shape)
                conf = heatmaps[i, y, x]
                keypoints[name] = (float(x), float(y), float(conf))
        return keypoints


    def draw_keypoints(self, img: np.ndarray, keypoints: dict, bbox: List[float]) -> np.ndarray:
        x1, y1, x2, y2 = map(int, bbox[:4])
        bbox_width, bbox_height = x2 - x1, y2 - y1
        img_copy = img.copy()

        # Draw keypoints on t1Bhe image
        for i, (name, (x, y, conf)) in enumerate(keypoints.items()):
            if conf > 0.3:  # Only draw confident keypoints
                x_coord = int(x * bbox_width / 192) + x1
                y_coord = int(y * bbox_height / 256) + y1
                cv2.circle(img_copy, (x_coord, y_coord), 3, GOLIATH_KPTS_COLORS[i], -1)

        # Optionally draw skeleton
        for _, link_info in GOLIATH_SKELETON_INFO.items():
            pt1_name, pt2_name = link_info['link']
            if pt1_name in keypoints and pt2_name in keypoints:
                pt1 = keypoints[pt1_name]
                pt2 = keypoints[pt2_name]
                if pt1[2] > 0.3 and pt2[2] > 0.3:
                    x1_coord = int(pt1[0] * bbox_width / 192) + x1
                    y1_coord = int(pt1[1] * bbox_height / 256) + y1
                    x2_coord = int(pt2[0] * bbox_width / 192) + x1
                    y2_coord = int(pt2[1] * bbox_height / 256) + y1
                    cv2.line(img_copy, (x1_coord, y1_coord), (x2_coord, y2_coord), GOLIATH_KPTS_COLORS[i], 2)

        return img_copy

    
 
       
       

