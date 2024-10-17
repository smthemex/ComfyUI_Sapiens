
from dataclasses import dataclass
import cv2
import numpy as np
import torch
from PIL import Image
from .depth import SapiensDepth, SapiensDepthType, draw_depth_map,DepthSapiens
from .segmentation import SapiensSegmentation, SapiensSegmentationType, draw_segmentation_map,SapiensSeg
from .normal import SapiensNormal, SapiensNormalType, draw_normal_map,NormalSapiens
from .pose import SapiensPoseEstimationType,SapiensPoseEstimation
from .detector import Detector
from .common import get_mask
@dataclass
class SapiensConfig:
    dtype: torch.dtype = torch.float32
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    segmentation_type: SapiensSegmentationType = SapiensSegmentationType.OFF
    normal_type: SapiensNormalType = SapiensNormalType.OFF
    depth_type: SapiensDepthType = SapiensDepthType.OFF
    pose_type: SapiensPoseEstimationType = SapiensPoseEstimationType.OFF
    #detector_config: DetectorConfig = DetectorConfig()
    minimum_person_height: int = 0.5  # 50% of the image height
    local_seg_path:str = ""
    local_depth_path: str = ""
    local_normal_path: str = ""
    local_pose_path: str = ""
    pt_type:str = "float32_torch"
    model_dir='models'
    process_size=(1024,768)
    detector=None
    remove_bg=True
    use_torchscript_seg=False
    use_torchscript_depth = False
    use_torchscript_pose = False
    use_torchscript_normal = False
    show_pose_object=False
    use_pellete=True
    
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
# !/usr/bin/env python
# -*- coding: UTF-8 -*-
class SapiensPredictor:
    def __init__(self, config: SapiensConfig):
        self.has_normal = config.normal_type != SapiensNormalType.OFF
        self.has_depth = config.depth_type != SapiensDepthType.OFF
        self.has_pose=config.pose_type != SapiensPoseEstimationType.OFF
        self.has_seg=config.segmentation_type != SapiensSegmentationType.OFF
        
        self.minimum_person_height = config.minimum_person_height
        self.img_size= config.process_size
        
        self.local_seg = config.local_seg_path
        self.local_normal = config.local_normal_path
        self.local_depth = config.local_depth_path
        self.local_pose=config.local_pose_path
        
        self.use_torchscript_seg=config.use_torchscript_seg
        self.use_torchscript_depth = config.use_torchscript_depth
        self.use_torchscript_normal = config.use_torchscript_normal
        self.use_torchscript_pose = config.use_torchscript_pose
        
        self.show_pose_object=config.show_pose_object
        self.remove_bg=config.remove_bg
        self.use_pellete=config.use_pellete
        
        self.pt_type=config.pt_type
        self.model_dir=config.model_dir
        
        self.normal_predictor = SapiensNormal(config.normal_type, self.local_normal ,self.pt_type,self.model_dir,self.img_size,self.use_torchscript_normal,config.device,
                                              config.dtype) if self.has_normal else None
        self.segmentation_predictor = SapiensSegmentation(config.segmentation_type,self.local_seg,self.pt_type,self.model_dir,self.img_size, self.use_torchscript_seg,config.device, config.dtype)if self.has_seg else None
        
        self.depth_predictor = SapiensDepth(config.depth_type, self.local_depth,self.pt_type,self.model_dir,self.img_size,self.use_torchscript_depth,config.device, config.dtype) if self.has_depth else None
        
        self.pose_predictor = SapiensPoseEstimation(config.pose_type, self.local_pose, self.pt_type, self.model_dir,
                                            self.img_size, self.use_torchscript_pose,self.show_pose_object,config.device, config.dtype) if self.has_pose else None
        
        self.seg_pred=SapiensSeg(config.segmentation_type,self.local_seg,self.pt_type,self.model_dir,self.use_torchscript_seg, config.dtype)if self.has_seg else None
        self.depth_pred =DepthSapiens(config.depth_type,self.local_depth,self.pt_type,self.model_dir,self.use_torchscript_seg, config.dtype)if self.has_depth else None
        self.normal_pred =NormalSapiens(config.normal_type,self.local_normal,self.pt_type,self.model_dir,self.use_torchscript_seg, config.dtype)if self.has_normal else None
        
        self.detector = config.detector  #Detector(config.detector_config) #TODO: Cropping seems to make the results worse

    def __call__(self, img: np.ndarray,select_obj,RGB_BG) :
        return self.predict(img,select_obj,RGB_BG)

    def predict(self, img,select_obj,RGB_BG) :# PIL.Image or np.ndarray
        if self.use_pellete:
            seg_list=[]
            depth_list=[]
            normal_list=[]
            pose_list=[]
            seg_mask=[]
            if self.has_seg:
                seg_map, mask_map,seg_preds = self.seg_pred(img, select_obj,RGB_BG)
                seg_list = [seg_map]
                seg_mask = [mask_map]
            seg_in = seg_preds if self.has_seg else None
            if self.has_depth:
                depth_list =[self.depth_pred(img, self.has_seg, seg_in,RGB_BG)]
            if self.has_normal:
                normal_list = [self.normal_pred(img, self.has_seg, seg_in,RGB_BG)]
            if self.has_pose: #pil-cv-pil,keypoint can save
                img_np=convert_from_image_to_cv2(img)
                pose_result_image, keypoints=self.pose_predictor(img_np)
                pose_list =[convert_from_cv2_to_image(pose_result_image)]
            return seg_list if seg_list else None,depth_list if depth_list else None,normal_list if normal_list else None,pose_list if pose_list else None,seg_mask if seg_list else None
        else:
            img_shape = img.shape  # 768, 1024, 3 np.ndarray
            if self.detector is not None:
                print("Detecting people...")
                person_boxes = self.detector.detect(img)
                person_boxes = filter_small_boxes(person_boxes, img_shape[0], self.minimum_person_height)
                
                if len(person_boxes) == 0:
                    # return img
                    raise "no person in img"
                
                person_boxes = expand_boxes(person_boxes, img_shape)
                print(f"{len(person_boxes)} people detected, predicting maps...")
            else:
                person_boxes = [[0, 0, img_shape[1], img_shape[0]], ]
            
            normal_maps = []
            segmentation_maps = []
            depth_maps = []
            pose_maps = []
            
            for box in person_boxes:
                crop = img[box[1]:box[3], box[0]:box[2]]
                if self.has_seg:
                    segmentation_maps.append(self.segmentation_predictor(crop))
                if self.has_normal:
                    normal_maps.append(self.normal_predictor(crop))
                if self.has_depth:
                    depth_maps.append(self.depth_predictor(crop))
                if self.has_pose:
                    pose_maps.append(self.pose_predictor(crop))  # tuple
            
            return self.draw_maps(img, person_boxes, normal_maps, segmentation_maps, depth_maps, pose_maps,RGB_BG)
        

    #TODO: Clean this up
    def draw_maps(self, img, person_boxes, normal_maps, segmentation_maps, depth_maps,pose_maps,RGB_BG):
        seg_list = []
        seg_mask=[]
        if self.has_seg:
            segmentation_img = img.copy()
            empty_cv = np.empty(segmentation_img.shape, dtype=np.uint8)
            empty_cv[:] = RGB_BG
            for segmentation_map, box in zip(segmentation_maps, person_boxes):
                mask = segmentation_map > 0
                origin = segmentation_img[box[1]:box[3], box[0]:box[2]]
                crop = empty_cv if self.remove_bg else origin
                segmentation_draw = draw_segmentation_map(segmentation_map)
                real_mask=get_mask(segmentation_draw)
                segmentation_img[box[1]:box[3], box[0]:box[2]] = origin * mask[..., None] + crop * ~mask[..., None]
            seg_list.append(segmentation_img)
            seg_mask.append(real_mask)
            
        normal_list=[]
        if self.has_normal:
            normal_img = img.copy()
            empty_cv = np.empty(normal_img.shape, dtype=np.uint8)
            empty_cv[:] = RGB_BG
            if self.has_seg:
                for i, (normal_map, box) in enumerate(zip(normal_maps, person_boxes)):
                    mask = segmentation_maps[i] > 0
                    crop = empty_cv if self.remove_bg else  normal_img[box[1]:box[3], box[0]:box[2]]
                    normal_draw = draw_normal_map(normal_map)
                    crop_draw = cv2.addWeighted(crop, 0.5, normal_draw, 0.7, 0)
                    normal_img[box[1]:box[3], box[0]:box[2]] = crop_draw * mask[..., None] + crop * ~mask[..., None]
                normal_list.append(normal_img)
            else:
                for i, (normal_map, box) in enumerate(zip(normal_maps, person_boxes)):
                    normal_img = draw_normal_map(normal_map)
                normal_list.append( normal_img)
            
        depth_list = []
        if self.has_depth:
            depth_img = img.copy()
            empty_cv = np.empty(depth_img.shape, dtype=np.uint8)
            empty_cv[:] = RGB_BG#(255, 255, 255)
            if self.has_seg:
                for i, (depth_map, box) in enumerate(zip(depth_maps, person_boxes)):
                    mask = segmentation_maps[i] > 0
                    crop = empty_cv if self.remove_bg else depth_img[box[1]:box[3], box[0]:box[2]]
                    depth_map[~mask] = 0
                    depth_draw = draw_depth_map(depth_map)
                    crop_draw = cv2.addWeighted(crop, 0.5, depth_draw, 0.7, 0)
                    depth_img[box[1]:box[3], box[0]:box[2]] = crop_draw * mask[..., None] + crop * ~mask[..., None]
                depth_list.append(depth_img)
            else:
                for i, (depth_map, box) in enumerate(zip(depth_maps, person_boxes)):
                    depth_img = draw_depth_map(depth_map)
                depth_list.append(depth_img)
                
        pose_list = []
        if self.has_pose:
            pose_img = img.copy()
            empty_cv = np.empty(pose_img.shape, dtype=np.uint8)
            empty_cv[:] =  RGB_BG#(0, 0, 0)
            if self.has_seg:
                for i, (pose_map, box) in enumerate(zip(pose_maps, person_boxes)):#pose_maps:tuple(result_img, all_keypoints)
                    mask = segmentation_maps[i] > 0
                    crop = empty_cv if self.remove_bg else pose_img[box[1]:box[3], box[0]:box[2]]
                    pose_map[0][~mask] = 0
                    pose_img[box[1]:box[3], box[0]:box[2]] = pose_map[0] * mask[..., None] + crop * ~mask[..., None]
                pose_list.append(pose_img)
            else:
                for i, (pose_map, box) in enumerate(zip(pose_maps, person_boxes)):
                    pose_img=pose_map[0]
                pose_list.append(pose_img)
        return  seg_list if seg_list else None,depth_list if depth_list else None,normal_list if normal_list else None,pose_list if pose_list else None,seg_mask if seg_list else None

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    # return np.asarray(img)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    # return Image.fromarray(img)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))