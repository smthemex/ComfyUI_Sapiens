# !/usr/bin/env python
# -*- coding: UTF-8 -*-
from tqdm import tqdm
from dataclasses import dataclass
import cv2
import numpy as np
import torch
from .depth import SapiensDepthType,  DepthSapiens
from .segmentation import  SapiensSegmentationType, SapiensSeg
from .normal import SapiensNormalType, NormalSapiens
from .pose import SapiensPoseEstimationType, SapiensPoseEstimation
from .common import aplly_seg


@dataclass
class SapiensConfig:
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    segmentation_type: SapiensSegmentationType = SapiensSegmentationType.OFF
    normal_type: SapiensNormalType = SapiensNormalType.OFF
    depth_type: SapiensDepthType = SapiensDepthType.OFF
    pose_type: SapiensPoseEstimationType = SapiensPoseEstimationType.OFF
    # detector_config: DetectorConfig = DetectorConfig()
    minimum_person_height: int = 0.5  # 50% of the image height
    local_seg_path: str = ""
    local_depth_path: str = ""
    local_normal_path: str = ""
    local_pose_path: str = ""
    pt_type: str = "float32_torch"
    model_dir = 'models'
    process_size = (1024, 768)
    detector = None
    remove_bg = True
    use_torchscript_seg = False
    use_torchscript_depth = False
    use_torchscript_pose = False
    use_torchscript_normal = False
    show_pose_object = False
    use_pellete = True
    
    def __str__(self):
        return f"SapiensConfig(dtype={self.dtype}\n" \
               f"device={self.device}\n" \
               f"segmentation_type={self.segmentation_type}\n" \
               f"normal_type={self.normal_type}\n" \
               f"depth_type={self.depth_type}\n" \
               f"minimum_person_height={self.minimum_person_height * 100}% of the image height"
        # f"detector_config={self.detector_config}\n" \


def filter_small_boxes(boxes: np.ndarray, img_height: int, height_thres: float = 0.1) -> np.ndarray:
    person_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        person_height = y2 - y1
        if person_height < height_thres * img_height:
            continue
        person_boxes.append(box)
    return np.array(person_boxes)


def expand_boxes(boxes: np.ndarray, img_shape: tuple[int, int], padding: int = 50) -> np.ndarray:
    expanded_boxes = []
    for box in boxes:
        x1, y1, x2, y2 = box
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(img_shape[1], x2 + padding)
        y2 = min(img_shape[0], y2 + padding)
        expanded_boxes.append([x1, y1, x2, y2])
    return np.array(expanded_boxes)


class SapiensPredictor:
    def __init__(self, config: SapiensConfig):
        self.has_normal = config.normal_type != SapiensNormalType.OFF
        self.has_depth = config.depth_type != SapiensDepthType.OFF
        self.has_pose = config.pose_type != SapiensPoseEstimationType.OFF
        self.has_seg = config.segmentation_type != SapiensSegmentationType.OFF
        
        self.minimum_person_height = config.minimum_person_height
        self.img_size = config.process_size
        
        self.local_seg = config.local_seg_path
        self.local_normal = config.local_normal_path
        self.local_depth = config.local_depth_path
        self.local_pose = config.local_pose_path
        
        self.use_torchscript_seg = config.use_torchscript_seg
        self.use_torchscript_depth = config.use_torchscript_depth
        self.use_torchscript_normal = config.use_torchscript_normal
        self.use_torchscript_pose = config.use_torchscript_pose
        
        self.show_pose_object = config.show_pose_object
        self.remove_bg = config.remove_bg
        
        self.pt_type = config.pt_type
        self.model_dir = config.model_dir
        
        self.pose_predictor = SapiensPoseEstimation(config.pose_type, self.local_pose, self.pt_type, self.model_dir,
                                                    self.img_size, self.use_torchscript_pose, self.show_pose_object,
                                                    config.device, config.dtype) if self.has_pose else None
        self.seg_pred = SapiensSeg(config.segmentation_type, self.local_seg, self.pt_type, self.model_dir,
                                   self.use_torchscript_seg,
                                   config.dtype) if self.has_seg  else None
        self.depth_pred = DepthSapiens(config.depth_type, self.local_depth, self.pt_type, self.model_dir,
                                       self.use_torchscript_seg,
                                       config.dtype) if self.has_depth  else None
        self.normal_pred = NormalSapiens(config.normal_type, self.local_normal, self.pt_type, self.model_dir,
                                         self.use_torchscript_seg,
                                         config.dtype) if self.has_normal  else None
        
        self.detector = None
    
    def __call__(self, img: np.ndarray, select_obj, RGB_BG):
        return self.predict(img, select_obj, RGB_BG)
    
    def enable_model_cpu_offload(self):
        if self.has_seg:
            self.seg_pred.enable_model_cpu_offload()
        if self.has_normal:
            self.normal_pred.enable_model_cpu_offload()
        if self.has_depth:
            self.depth_pred.enable_model_cpu_offload()
        if self.has_pose:
            self.pose_predictor.enable_model_cpu_offload()
    
    def move_to_cuda(self):
        if self.has_seg:        
            self.seg_pred.move_to_cuda()      
        if self.has_normal:     
            self.normal_pred.move_to_cuda()   
        if self.has_depth:
            self.depth_pred.move_to_cuda()
        if self.has_pose:
            self.pose_predictor.move_to_cuda()
    
    def predict(self, images, select_obj, RGB_BG):  # PIL.Image when use pellete ,np.ndarray
        seg_list,depth_list,normal_list,pose_list,mask_list=[],[],[],[],[]
        for img in tqdm(images, total=len(images)):
            if self.has_seg:
                seg_map, mask_map, seg_preds = self.seg_pred(img, select_obj, RGB_BG)
                seg_list.append(np2tensor(seg_map,False))
                mask_list.append(mask_map)
            seg_in = seg_preds if self.has_seg else None
            if self.has_depth:
                depth_map=self.depth_pred(img, self.has_seg, seg_in, select_obj,RGB_BG)
                depth_list.append(np2tensor(depth_map,False))
            if self.has_normal:
                normal_map=self.normal_pred(img, self.has_seg, seg_in,select_obj, RGB_BG)
                normal_list.append(np2tensor(normal_map,False))
            if self.has_pose:  # pil-cv-pil,keypoint can save
                filter_obj=select_obj.copy() if not self.has_seg and select_obj else None # only useful when no seg and select_obj
                pose_result_image, keypoints,box_size = self.pose_predictor(np.array(img),filter_obj)        
                if self.has_seg:
                    pose_result_image=aplly_seg(cv2.cvtColor(pose_result_image, cv2.COLOR_BGR2RGB),seg_in,select_obj,[0,0,0])
                pose_list.append(np2tensor(pose_result_image,False))        
        return seg_list , depth_list, normal_list, pose_list , mask_list
    


def np2tensor(img,need_convert=True):
    """
    Convert numpy array to comfy torch tensor.
    BGR-->RGB
    HWC-->BHWC
    """
    if need_convert:  
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    img = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
    return img